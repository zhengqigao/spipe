"""Active Mach-Zehnder modulator (``mzm``).

This is a *physical interferometer*, in contrast with :class:`~spipe.photonic.model.modm.ModM`
(paper Eq. 5), which is a variable-ratio coupler.  ``modm`` is kept unchanged -- it is a
legitimate abstraction of a tunable coupler and the published TCAD results depend on it -- but it
is not an MZM:

* its scatter matrix ``[[cos p, i sin p], [i sin p, cos p]]`` matches a physical push-pull MZM
  only under ``p = pi/2 - dphi/2``, i.e. its V_pi is a factor of two off and its bias point is
  offset by ``pi/2``;
* even then the two matrices differ by more than a global phase (a physical MZI has
  ``arg S11 = -arg S22``, while ``modm`` has ``S11 = S22``), so light that re-interferes
  downstream picks up the wrong relative phase;
* and it is *exactly* unitary at every drive, i.e. it has infinite extinction ratio and a bias
  independent insertion loss.  No real silicon modulator has either.

``mzm`` fixes all of that.  Its headline property is that **vpi is literally the voltage that
takes the device from full-on to full-off**.
"""

from typing import Optional, Tuple, Dict, Any, List
import re
import warnings
from math import pi

import torch

from spipe import config
from spipe.photonic.model.base import Device, complex_zeros, real_tensor
from spipe.photonic.func import FreeLightSpeed, neff
from ..func import collect_coeff, taylor

__all__ = ['MZM']

# Attributes that must be materialised at config['real_dtype'] before Device.__init__ wraps them.
# A bare ``torch.tensor(0.35)`` is float32, and torch.cos/torch.exp of such a value carries only
# ~1e-8 of accuracy no matter what config['complex_dtype'] says.
_NO_DRIVE_WARNED = False

_FLOAT_ATTR = ('vpi', 'vbias', 'er', 'il', 'chirp', 'tau', 'order',
               'kappa1', 'kappa2', 'act_l', 'wgu_l', 'wgl_l', 'alpha', 'neff', 'ng', 'wl')


def _one_pole(x: torch.Tensor, t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Causal single-pole lag ``tau * dy/dt + y = x`` sampled on the (possibly non-uniform) grid ``t``.

    The recursion is the *exact* solution of the ODE for a zero-order-hold input over each
    sample interval,

    ``y[k] = a[k] * y[k-1] + (1 - a[k]) * x[k]``,  ``a[k] = exp(-(t[k] - t[k-1]) / tau)``,

    started from ``y[0] = x[0]`` (the device is assumed to have settled before the first sample).
    It is unconditionally stable for ``tau > 0`` and reduces to ``y = x`` as ``tau -> 0``
    (``a -> 0``), which is why ``tau = 0`` is short-circuited by the caller rather than being
    allowed to divide by zero.
    """
    if x.shape[0] < 2:
        return x
    dt = t[1:] - t[:-1]
    a = torch.exp(-dt / tau)
    y = x[0]
    out = [y]
    for k in range(a.shape[0]):
        y = a[k] * y + (1.0 - a[k]) * x[k + 1]
        out.append(y)
    return torch.stack(out)


class MZM(Device):
    """A 2-in 2-out active Mach-Zehnder modulator.

    Netlist syntax (same convention as ``modm``: left nodes, right nodes, then the active node
    and the *electronic* model name, then keyword parameters)::

        mzm<name> <l1> <l2> <r1> <r2> <actnode> <elec_model> vpi=2.0 vbias=1.0 er=25 il=3.0

    Transfer matrix
    ---------------
    The device is built as an interferometer, not as a coupler::

        S_mzm = C(kappa2) @ diag(a1 * exp(j*phi1), a2 * exp(j*phi2)) @ C(kappa1)
        C(k)  = [[cos k, i sin k], [i sin k, cos k]]

    with ``kappa1 = kappa2 = pi/4`` (50:50) by default.  Written out, with
    ``ci = cos(kappa_i)``, ``si = sin(kappa_i)`` and ``Ai = ai * exp(j*phi_i)``::

        M[0,0] = c1*c2*A1 - s1*s2*A2          M[0,1] = j*(s1*c2*A1 + c1*s2*A2)
        M[1,0] = j*(c1*s2*A1 + s1*c2*A2)      M[1,1] = c1*c2*A2 - s1*s2*A1

    ``M[row, col]`` maps left port ``col`` to right port ``row``.  As in ``modm`` the 2x2 block is
    embedded in the 4x4 port structure with ``S[2:, :2] = M`` (forward) and ``S[:2, 2:] = M.T``
    (backward), which makes the whole device **reciprocal**: ``S == S.T`` exactly, entry by entry.

    Drive and chirp
    ---------------
    With ``dphi = pi * (v - vbias) / vpi``::

        phi1 = +0.5 * (1 + chirp) * dphi
        phi2 = -0.5 * (1 - chirp) * dphi

    so ``phi1 - phi2 = dphi`` for every ``chirp``: the Henry alpha-parameter ``chirp`` (default
    ``0.0``) moves only the *common-mode* phase and therefore never changes the transmission.
    ``chirp = 0`` is ideal chirp-free push-pull drive; ``chirp = +-1`` is single-arm drive.

    At 50:50 splitting and ``a1 = a2 = a`` this gives

    * bar port  ``|M[0,0]|^2 = a^2 * sin^2(dphi/2)``
    * cross port ``|M[1,0]|^2 = a^2 * cos^2(dphi/2)``

    i.e. the pair ``(sin^2, cos^2)`` of the specification with the roles of the two output ports
    exchanged -- the "consistent complement".  Either way the two ports sum to ``a^2`` and
    **vpi is exactly the voltage swing from full-on to full-off**: at ``v = vbias`` the cross port
    is at its maximum and the bar port extinguished, and at ``v = vbias + vpi`` they have swapped.

    Extinction ratio
    ----------------
    A lossless, perfectly balanced MZI has infinite extinction.  A finite ``er`` (in dB) is
    obtained from a *static* amplitude imbalance between the two arms::

        a1 = base * (1 + eps),   a2 = base * (1 - eps)

    The two arms interfere constructively at one bias and destructively at the other, so the
    measured on/off power ratio of either output port is

        ER_linear = ((a1 + a2) / (a1 - a2))^2 = ((1 + rho) / (1 - rho))^2 = 1 / eps^2

    where ``rho = a2 / a1 = (1 - eps) / (1 + eps)`` is the arm *amplitude ratio*.  (Both forms are
    the same number; ``((1+rho)/(1-rho))^2`` is the form quoted in terms of the ratio, ``1/eps^2``
    the form in terms of the symmetric imbalance actually stored.)  Inverting,

        eps = 10 ** (-er_dB / 20)

    so ``er = inf`` (the default) gives ``eps = 0``, perfectly balanced arms and infinite
    extinction, and e.g. ``er = 25`` gives ``eps = 0.0562`` and a measured 25.000 dB on/off ratio.

    Insertion loss
    --------------
    ``il`` (dB) enters as a common amplitude factor ``base = 10 ** (-il / 20)``.  Because the peak
    port transmission is ``((a1 + a2) / 2)^2 = base^2``, the peak *power* transmission is exactly
    ``10 ** (-il / 10)`` -- independently of ``er``.

    Loss modulation (free-carrier plasma dispersion)
    ------------------------------------------------
    In silicon, injecting or depleting carriers changes **both** the index and the absorption
    (Soref-Bennett): ``dn`` and ``dalpha`` move together.  The per-arm amplitude is therefore

        a_i = 10**(-il/20) * (1 +- eps) * exp(-0.5 * dalpha_i * act_l)

    with the excess (differential) power attenuation ``dalpha_i`` [1/m] given as a Taylor series in
    the *arm* drive, in the same style as the ``coeff1=, coeff2=, ...`` convention used by ``modm``
    and ``modp``::

        dalpha_i = sum_k dacoeff_k * u_i**k,
        u_1 = +0.5 * (1 + chirp) * (v - vbias),   u_2 = -0.5 * (1 - chirp) * (v - vbias)

    i.e. the same push-pull split that drives the phase also drives the loss, so that in a
    differential drive one arm gains loss while the other loses it *relative to the bias point*
    (the static loss at the bias point is what ``il`` describes).  All ``dacoeff*`` default to
    zero, which reproduces the lossless-modulation case exactly.

    Modulator response time
    -----------------------
    Optional ``tau`` (seconds, default ``0.0``) lifts the paper's Assumption 1 for this device:
    the phase then follows ``tau * dphi/dt + phi = pi * (v(t) - vbias) / vpi`` instead of tracking
    the drive instantaneously.  Because the phase is *linear* in the drive, the filter is applied
    once to ``v - vbias`` and everything (phase and loss modulation) is derived from the filtered
    drive; this is also the physically right order for the loss, where the carrier density lags
    the drive and the optical properties follow the carrier density.  ``order`` (default 1)
    cascades ``order`` identical single-pole sections, each with time constant ``tau``.
    ``tau = 0`` short-circuits the filter and reproduces the instantaneous result bit for bit.

    Parameters
    ----------
    Required: ``time``, ``omega`` (supplied by the simulator).

    Optional (netlist keyword arguments), with defaults:

    ``vpi=2.0``
        Half-wave voltage [V]: the drive swing that moves a port from full-on to full-off.
    ``vbias=0.0``
        Bias voltage [V]; ``dphi = 0`` there.
    ``er=inf``
        Extinction ratio [dB].  Use a large finite number (e.g. ``er=1e9``) in a netlist, where
        the parser cannot spell ``inf``.
    ``il=0.0``
        Insertion loss [dB].
    ``chirp=0.0``
        Henry alpha-parameter; ``0`` is chirp-free push-pull, ``+-1`` single-arm drive.
    ``kappa1=pi/4``, ``kappa2=pi/4``
        Input / output coupler angles [rad]; ``pi/4`` is 50:50.
    ``dacoeff0=, dacoeff1=, ...``
        Taylor coefficients of the drive-dependent excess loss [1/m, 1/(m*V), ...].
    ``act_l=1e-3``
        Active (phase-shifter) length [m], used by the loss modulation term.  1 mm is a
        representative lumped or travelling-wave depletion-mode silicon MZM segment.
    ``tau=0.0``, ``order=1``
        Modulator response time [s] and filter order.
    ``wgu_l=0.0``, ``wgl_l=0.0``, ``alpha=1.0``
        Optional *passive* upper/lower arm sections (arm-length imbalance), propagating with the
        mode index from ``.mode`` and amplitude ``alpha``.  Both zero by default, in which case
        the scatter matrix is exactly the expression above.
    ``neff=1.0``, ``ng=None``, ``wl=None``
        Mode parameters; normally supplied by the ``.mode`` line.  Only used by the passive arm
        sections.
    ``act``, ``an``
        Drive signal and the name of the electronic model driving it; supplied by the simulator
        for an active netlist element.  ``act`` defaults to ``0.0`` so that the model can also be
        instantiated, or parsed, without a drive.
    """

    _name = 'mzm'
    _required_attr = ['time', 'omega']
    _optional_attr = {
        'act': 0.0, 'an': None,
        'neff': 1.0, 'ng': None, 'wl': None,
        'vpi': 2.0, 'vbias': 0.0,
        'er': float('inf'), 'il': 0.0,
        'chirp': 0.0,
        'kappa1': pi / 4, 'kappa2': pi / 4,
        'dacoeff': None,
        'act_l': 1e-3,
        'tau': 0.0, 'order': 1.0,
        'wgu_l': 0.0, 'wgl_l': 0.0, 'alpha': 1.0,
    }
    _num_port = [2, 2]
    _active_port = 1

    def __init__(self, **kwargs):
        global _NO_DRIVE_WARNED
        if kwargs.get('act', None) is None and not _NO_DRIVE_WARNED:
            _NO_DRIVE_WARNED = True
            warnings.warn(
                "mzm was instantiated without a drive signal ('act'), so it is held at its bias "
                "point. In a netlist the drive is supplied for you -- by the circuit in a "
                "Circuit, or by Photonic.simulate(t, drive). When constructing the model "
                "directly, pass it: MZM(time=t, omega=w, act=v, vpi=..., vbias=...).",
                stacklevel=2)

        # Collect dacoeff0=, dacoeff1=, ... exactly the way modm/modp collect coeff0=, coeff1=,...
        # but without warning when the user simply did not ask for loss modulation.
        if any(re.match(r'^dacoeff\d+$', key) for key in kwargs.keys()):
            kwargs['dacoeff'] = collect_coeff(kwargs, 'dacoeff', delete=True)
        else:
            kwargs.setdefault('dacoeff', [0.0])
        if kwargs.get('dacoeff') is None:
            kwargs['dacoeff'] = [0.0]

        # Materialise every scalar at config['real_dtype'].  A python int reaching _param_wrap
        # would be stored as an int64 Parameter, and ``-er / 20`` on an int64 tensor promotes to
        # *float32*, silently throwing away half of the mantissa of a complex128 simulation.
        for key in _FLOAT_ATTR:
            if kwargs.get(key) is not None:
                kwargs[key] = float(kwargs[key])

        # Physically meaningless values used to pass silently: vpi=0 gave NaN photocurrents,
        # and a negative il is optical gain from a passive modulator.
        if kwargs.get('vpi') is not None and not kwargs['vpi'] > 0:
            raise ValueError(f"mzm: vpi must be > 0 volts, got {kwargs['vpi']}.")
        if kwargs.get('il') is not None and kwargs['il'] < 0:
            raise ValueError(f"mzm: il is an insertion loss in dB and must be >= 0, got "
                             f"{kwargs['il']} (a negative value would be optical gain).")
        if kwargs.get('er') is not None and kwargs['er'] < 0:
            raise ValueError(f"mzm: er is an extinction ratio in dB and must be >= 0, got "
                             f"{kwargs['er']}.")
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ helpers

    def _drive(self) -> torch.Tensor:
        """The drive deviation ``v(t) - vbias``, shaped ``(len(time),)`` and low-pass filtered
        with the modulator response time when ``tau > 0``."""
        num_t = len(self.params['time'])
        act = self.params['act']
        if not isinstance(act, torch.Tensor):
            act = real_tensor(act)
        if act.ndim == 0:
            act = act.reshape(1).expand(num_t)
        drive = act - self.params['vbias']

        tau = self.params['tau']
        order = int(self.params['order'])
        if float(tau) > 0.0 and order > 0:
            for _ in range(order):
                drive = _one_pole(drive, self.params['time'], tau)
        return drive

    def _arm_amplitudes(self, drive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(a1, a2)``: static insertion loss, the ER imbalance and the drive-dependent excess
        loss, per arm, shaped ``(len(time),)``."""
        # base**2 is the peak power transmission, hence base = 10**(-il/20).
        base = torch.pow(real_tensor(10.0), -self.params['il'] / 20.0)
        # eps = 10**(-er/20) makes the measured on/off power ratio exactly 10**(er/10).
        # er = inf -> eps = 0 -> perfectly balanced arms -> infinite extinction.
        eps = torch.pow(real_tensor(10.0), -self.params['er'] / 20.0)

        chirp = self.params['chirp']
        u1 = 0.5 * (1.0 + chirp) * drive
        u2 = -0.5 * (1.0 - chirp) * drive

        dacoeff = self.params['dacoeff']
        dalpha1 = taylor(dacoeff, u1)
        dalpha2 = taylor(dacoeff, u2)

        # The finite-ER imbalance is taken out of the WEAKER arm, never added to the stronger
        # one. Only the ratio a2/a1 = (1 - eps)/(1 + eps) sets the on/off ratio, so it stays
        # exactly 10**(er/10); and neither arm now exceeds `base`, so the device is passive.
        # (Writing the arms as base*(1 +/- eps) put amplitude 1 + eps on one arm and, at
        # il = 0, created light: total output 1.1 W from 1 W in at er = 10 dB.) The price is
        # physical: an imbalanced interferometer loses some light even at its peak, so with a
        # finite er the peak transmission is 10**(-il/10) / (1 + eps)**2, not 10**(-il/10).
        half_l = 0.5 * self.params['act_l']
        a1 = base * torch.exp(-half_l * dalpha1)
        a2 = base * (1.0 - eps) / (1.0 + eps) * torch.exp(-half_l * dalpha2)
        return a1, a2

    # ------------------------------------------------------------------ forward

    def forward(self, vari_set: Optional[set] = None) -> Tuple[torch.Tensor, Dict]:
        # The scatter matrix is (len(time), len(omega), 4, 4).  As for every active device, the
        # quasi-static assumption holds: scatter_matrix[i] depends only on params['time'][i]
        # (through the drive) -- except that with tau > 0 the drive itself is filtered, which is
        # precisely how the response time of the *device* is lifted out of that assumption.
        num_t, num_w = len(self.params['time']), len(self.params['omega'])

        drive = self._drive()                                     # (T,)
        dphi = pi * drive / self.params['vpi']                    # (T,)

        chirp = self.params['chirp']
        phi1 = 0.5 * (1.0 + chirp) * dphi                         # (T,)
        phi2 = -0.5 * (1.0 - chirp) * dphi                        # phi1 - phi2 == dphi always

        a1, a2 = self._arm_amplitudes(drive)                      # (T,)

        # Optional passive arm sections (arm-length imbalance).  Zero by default, in which case
        # the loop below adds nothing at all and the scatter matrix is frequency independent.
        phi1 = phi1.unsqueeze(-1).expand(num_t, num_w)
        phi2 = phi2.unsqueeze(-1).expand(num_t, num_w)
        a1 = a1.unsqueeze(-1).expand(num_t, num_w)
        a2 = a2.unsqueeze(-1).expand(num_t, num_w)

        wgu_l, wgl_l = self.params['wgu_l'], self.params['wgl_l']
        if float(wgu_l) != 0.0 or float(wgl_l) != 0.0:
            neff_func = neff(self.params['neff'], self.params['ng'], self.params['wl'])
            beta = self.params['omega'] / FreeLightSpeed * neff_func(self.params['omega'])  # (W,)
            phi1 = phi1 + (beta * wgu_l).unsqueeze(0)
            phi2 = phi2 + (beta * wgl_l).unsqueeze(0)
            alpha = self.params['alpha']
            if float(wgu_l) != 0.0:
                a1 = a1 * alpha
            if float(wgl_l) != 0.0:
                a2 = a2 * alpha

        big_a1 = a1 * torch.exp(1.j * phi1)                       # (T, W) complex
        big_a2 = a2 * torch.exp(1.j * phi2)

        cos1, sin1 = torch.cos(self.params['kappa1']), torch.sin(self.params['kappa1'])
        cos2, sin2 = torch.cos(self.params['kappa2']), torch.sin(self.params['kappa2'])

        # M = C(kappa2) @ diag(A1, A2) @ C(kappa1); M[row=right port, col=left port].
        m = complex_zeros((num_t, num_w, 2, 2))
        m[..., 0, 0] = cos1 * cos2 * big_a1 - sin1 * sin2 * big_a2
        m[..., 0, 1] = 1.j * (sin1 * cos2 * big_a1 + cos1 * sin2 * big_a2)
        m[..., 1, 0] = 1.j * (cos1 * sin2 * big_a1 + sin1 * cos2 * big_a2)
        m[..., 1, 1] = cos1 * cos2 * big_a2 - sin1 * sin2 * big_a1

        # Reciprocity by construction: the backward block is the transpose of the forward block,
        # so the full 4x4 satisfies S == S.T entry by entry (not merely to round-off).
        scatter_matrix = complex_zeros((num_t, num_w, 4, 4))
        scatter_matrix[..., 2:, :2] = m
        scatter_matrix[..., :2, 2:] = m.transpose(-1, -2)

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # No closed-form gradient is supplied; the base class falls back to automatic
            # differentiation, which handles the response-time recursion as well.
            grad = {}
            return scatter_matrix, grad
