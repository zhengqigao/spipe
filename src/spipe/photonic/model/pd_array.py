"""Photodetector array: responsivity, bandwidth, dark current and *physical* noise.

What this module used to do::

    clean = (x.abs() ** 2 * self.coeff).sum(dim=1)
    return clean * (1 + self.std * torch.randn_like(clean))

Two things were wrong with that, and both are fixed here.

**The noise was not physical.**  A *multiplicative* term with a constant relative sigma is
neither shot noise nor thermal noise: it makes sigma proportional to the current
(``d log sigma / d log I = 1``), whereas shot noise requires ``d log sigma / d log I = 0.5`` and
thermal noise requires ``0``.  Noise is now added **in current**, from the standard pair

* ``sigma_shot    = sqrt(2 * q * (I + I_dark) * B)``   -- signal dependent, ``sigma ~ sqrt(I)``
* ``sigma_thermal = sqrt(4 * k * T * B / R_load)``     -- signal independent
  (or ``sigma_thermal = i_n * sqrt(B)`` when the input-referred current noise density ``inoise``
  is given instead of the ``temp`` / ``rload`` pair),

and the draws come from a **local** ``torch.Generator`` seeded from ``config['seed']``, never from
the global RNG -- that global draw was the last thing standing between SPIPE and bitwise
reproducibility.

**The frequency axis was summed incoherently, silently.**  ``sum_n |p(omega_n)|^2`` is the right
answer only when the beat notes between the optical carriers fall *outside* the detector
bandwidth.  The general expression is

    I(t) = LPF_B { | sum_n sqrt(R(omega_n)) * p(omega_n, t) * exp(-i (omega_n - omega_ref) t) |^2 }

whose ``|.|^2`` expands into the incoherent sum ``sum_n R_n |p_n|^2`` plus beat terms at
``|omega_n - omega_m|``; if those are above ``bw`` the low-pass removes them and the expression
reduces **exactly** to the historical incoherent sum.  ``coherent=`` selects between the two, and
defaults to the incoherent sum so that existing results do not move.

Backward compatibility
----------------------
With none of the new parameters given (``bw``, ``idark``, ``temp``, ``rload``, ``inoise``,
``coherent``, ``dt``) the output is ``clean * (1 + std * randn)`` exactly as before, so a netlist
that uses only ``r0=``/``r1=``... and the deprecated ``std=`` produces the same numbers.  The one
deliberate difference is that the ``randn`` now comes from the local generator, so a ``std != 0``
netlist sees a *different realisation* of the same distribution (this is the reproducibility fix,
and it is exactly what makes two runs with the same ``config['seed']`` agree bit for bit).
"""

import warnings
from math import pi
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

from spipe import config
from spipe.photonic.func import FreeLightSpeed
from ..func import collect_coeff, taylor

__all__ = ['PDArray']

# CODATA 2018 exact values.
ELEMENTARY_CHARGE = 1.602176634e-19   # C
BOLTZMANN = 1.380649e-23              # J/K

_STD_DEPRECATION_WARNED = False


def init_coeff(pd_args: List[Dict], eval_omega) -> torch.Tensor:
    """Responsivity ``R(omega)`` of every detector, shape ``(len(eval_omega), len(pd_args))``.

    Built at ``config['real_dtype']``: the container used to be torch's float32 default, so a
    responsivity of ``1e-3`` was stored as ``1.0000000474974513e-03`` and every photocurrent in a
    complex128 simulation inherited that error.
    """
    dout_node = len(pd_args)
    coeff = torch.zeros((len(eval_omega), dout_node),
                        dtype=config['real_dtype'], device=config['device'])
    for i in range(coeff.shape[1]):
        if 'wl' not in pd_args[i].keys():
            if 'r0' not in pd_args[i].keys():
                raise RuntimeError(
                    f"photodetector {i + 1} has no responsivity: give r0= in A/W (and r1=, r2=, ... "
                    f"with wl= for a wavelength dependence)")
            else:
                coeff[:, i] = pd_args[i]['r0']
        else:
            omega0 = 2 * torch.pi * FreeLightSpeed / pd_args[i]['wl']
            coeff[:, i] = taylor(collect_coeff(pd_args[i], 'r'), eval_omega, omega0)
    return coeff


def _vector(pd_args: List[Dict], key: str, default: float) -> torch.Tensor:
    """Per-detector attribute as a 1-D tensor at ``config['real_dtype']``."""
    return torch.tensor([float(pd_arg.get(key, default)) for pd_arg in pd_args],
                        dtype=config['real_dtype'], device=config['device'])


def _lowpass(signal: torch.Tensor, time: torch.Tensor, bw: torch.Tensor) -> torch.Tensor:
    """Per-detector single-pole low-pass of ``signal`` (shape ``(T, D)``) at ``bw`` (shape ``(D,)``).

    A detector with ``bw <= 0`` is passed through untouched (infinite bandwidth, the historical
    behaviour).  The pole is the usual ``tau = 1 / (2 pi bw)``, discretised exactly for a
    zero-order-hold input on the -- possibly non-uniform -- grid ``time``.
    """
    active = bw > 0
    if signal.shape[0] < 2 or not bool(active.any()):
        return signal
    safe_bw = torch.where(active, bw, torch.ones_like(bw))
    tau = 1.0 / (2.0 * pi * safe_bw)                       # (D,)
    dt = (time[1:] - time[:-1]).unsqueeze(-1)              # (T-1, 1)
    decay = torch.where(active.unsqueeze(0),
                        torch.exp(-dt / tau.unsqueeze(0)),
                        torch.zeros_like(dt).expand(-1, bw.shape[0]))
    out = [signal[0]]
    running = signal[0]
    for k in range(decay.shape[0]):
        running = decay[k] * running + (1.0 - decay[k]) * signal[k + 1]
        out.append(running)
    return torch.stack(out)


class PDArray(nn.Module):
    """The array of photo detectors declared with ``pd`` lines in the photonic netlist.

    Netlist syntax::

        pd<name> <optical node> <electrical node> <elec_model> r0=1e-3 [r1=... wl=...]
                 [bw=40e9] [idark=1e-9] [temp=300] [rload=50] [inoise=1e-11]
                 [coherent=1] [dt=...] [std=0.05]

    Parameters
    ----------
    ``r0, r1, ... , wl``
        Taylor coefficients of the responsivity ``R(omega)`` about ``2*pi*c/wl`` [A/W].  Unchanged.
    ``bw``
        Detector / front-end bandwidth [Hz]: the 3 dB frequency of a single-pole low-pass.  The
        photocurrent passes through it, and so does the noise (see *Noise* below).  With ``bw``
        absent (the default) the detector has infinite bandwidth and adds no physical noise, which
        is the historical behaviour.
    ``noise``
        ``1`` (default) adds shot and thermal noise whenever ``bw`` is given; ``0`` keeps the
        bandwidth and drops the noise.
    ``idark``
        Dark current [A], default 0.  Added to the photocurrent and to the shot-noise argument.
    ``temp``, ``rload``
        Load temperature [K] (default 300) and resistance [Ohm] (default 50) for the thermal noise
        ``sqrt(4 k T B / R)``.
    ``inoise``
        Input-referred current noise density [A/sqrt(Hz)].  When given it *replaces* the
        ``temp``/``rload`` route: ``sigma_thermal = inoise * sqrt(B)``.  Use it to model a real
        TIA, whose input noise is nothing like the Johnson noise of its feedback resistor.
    ``coherent``
        ``0`` (default): the frequency axis is summed incoherently, ``sum_n R_n |p_n|^2``.  This
        is exact for WDM carriers whose beat notes lie outside ``bw``, and reproduces the
        historical result bit for bit.  ``1``: the carriers are summed *in field* with their
        relative optical phase ``exp(-i (omega_n - omega_ref) t)`` before squaring, so beats
        inside ``bw`` appear in the photocurrent.  ``omega_ref`` is the mean of the simulated
        frequency grid.  Coherent detection needs the time axis (see below) and a time step fine
        enough to resolve the beat: with ``N`` optical channels spaced by ``df``, beats appear up
        to ``(N-1)*df`` and are aliased if ``dt > 1/(2*(N-1)*df)``.
    ``dt``
        Sample interval [s].  Only needed as a *fallback* when the time axis is not otherwise
        available (see :meth:`set_time`); the low-pass and the coherent sum both need it.
    ``std``
        **Deprecated.**  The old multiplicative relative-noise term, kept so that existing
        netlists keep running.  It is applied after the physical noise, as
        ``I * (1 + std * randn)``, and warns once per session.

    Noise
    -----
    Shot and thermal noise are white at the photodiode, with one-sided density
    ``S = 2 q (I + I_dark) + 4 k T / R_load`` A^2/Hz (or ``inoise^2`` in place of the thermal
    term), and they reach the output through the same single pole as the signal.  The samples
    are those of that filtered continuous process, drawn exactly: an Ornstein-Uhlenbeck
    recursion ``n[k] = a n[k-1] + sqrt(var (1 - a^2)) w[k]`` with ``a = exp(-dt / tau)``
    and stationary variance ``var = S * (pi/2) * bw`` -- ``(pi/2) bw`` being the equivalent noise
    bandwidth of a single pole.  So the variance, the correlation between neighbouring samples
    and the spectrum are right at any time step, and a receiver's output noise does not depend
    on how finely ``.tran`` samples it once the grid resolves ``bw``.  Without a time axis the
    samples are independent, with the same variance.

    The time axis
    -------------
    ``PDArray`` is constructed by :class:`~spipe.photonic.photonic.Photonic` as
    ``PDArray(pd_args, omega)`` and called as ``pd_array(res)``, neither of which carries the
    transient time grid.  The bandwidth low-pass and the coherent sum are the only two features
    that need it, and they pick it up from, in order: the ``time`` argument of :meth:`forward`,
    a grid installed with :meth:`set_time`, the ``time`` argument of the constructor, or a
    uniform grid rebuilt from ``dt=`` on the ``pd`` line.  If none of those is available the
    low-pass and the coherent sum are skipped with a warning; the noise itself only needs ``bw``
    and is always applied.
    """

    def __init__(self, pd_args, eval_omega, time: Optional[torch.Tensor] = None):
        super().__init__()
        self.pd_args = pd_args
        self.eval_omega = eval_omega
        self.coeff = init_coeff(pd_args, eval_omega).to(config['device'])

        # Deprecated multiplicative relative noise, kept for backward compatibility.
        self.std = _vector(pd_args, 'std', 0.0)

        # Physical noise / bandwidth parameters, one entry per detector.
        self.bw = _vector(pd_args, 'bw', 0.0)
        self.idark = _vector(pd_args, 'idark', 0.0)
        self.temp = _vector(pd_args, 'temp', 300.0)
        self.rload = _vector(pd_args, 'rload', 50.0)
        self.inoise = _vector(pd_args, 'inoise', 0.0)
        self.has_inoise = torch.tensor([('inoise' in pd_arg) for pd_arg in pd_args],
                                       dtype=torch.bool, device=config['device'])
        self.coherent = torch.tensor([bool(float(pd_arg.get('coherent', 0.0))) for pd_arg in pd_args],
                                     dtype=torch.bool, device=config['device'])
        #: detectors that add noise: a bandwidth, and not switched off with noise=0
        self.noisy = (self.bw > 0) & torch.tensor(
            [bool(float(pd_arg.get('noise', 1.0))) for pd_arg in pd_args],
            dtype=torch.bool, device=config['device'])
        #: When set, the generator is reseeded from it at every call, so repeated evaluations
        #: (the iterations of a co-simulation fixed point, and its gradient) see one and the
        #: same noise realisation. Circuit sets it from its ``seed``.
        self.run_seed: Optional[int] = None

        self.time = None if time is None else time.to(config['device'])
        self._dt = None
        for pd_arg in pd_args:
            if 'dt' in pd_arg:
                self._dt = float(pd_arg['dt'])
                break

        self._no_time_warned = False

        # X: a *local* generator, so that the noise is reproducible from config['seed'] and the
        # caller's global torch RNG state is never touched nor depended upon.
        self.generator = torch.Generator(device=config['device'])
        self.reseed()

    # ------------------------------------------------------------------ plumbing

    def reseed(self, seed: Optional[int] = None) -> None:
        """Reseed the private noise generator (default: ``config['seed']``)."""
        self.generator.manual_seed(int(config['seed'] if seed is None else seed))

    def set_time(self, time: Optional[torch.Tensor]) -> None:
        """Install the transient time grid used by the bandwidth low-pass and the coherent sum."""
        self.time = None if time is None else time.to(config['device'])

    def _randn(self, shape) -> torch.Tensor:
        return torch.randn(shape, generator=self.generator,
                           dtype=config['real_dtype'], device=config['device'])

    def _time_axis(self, num_t: int, time: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        for candidate in (time, self.time):
            if candidate is not None and len(candidate) == num_t:
                return candidate.to(dtype=config['real_dtype'], device=config['device'])
        if self._dt is not None:
            return torch.arange(num_t, dtype=config['real_dtype'], device=config['device']) * self._dt
        return None

    # ------------------------------------------------------------------ physics

    def _photocurrent(self, x: torch.Tensor, time: Optional[torch.Tensor]) -> torch.Tensor:
        """Photocurrent before bandwidth limiting, shape ``(T, D)``.

        ``x`` is the complex optical amplitude, shape ``(T, len(omega), D)``.
        """
        incoherent = (x.abs() ** 2 * self.coeff).sum(dim=1)
        if not bool(self.coherent.any()):
            return incoherent

        if time is None:
            if not self._no_time_warned:
                warnings.warn(
                    "A photo detector asked for coherent=1 but the transient time axis is not "
                    "available to PDArray (Photonic builds it as PDArray(pd_args, omega) and "
                    "calls it as pd_array(res)).  Falling back to the incoherent sum.  Pass the "
                    "grid with PDArray.set_time(t), or give dt= on the pd line.")
                self._no_time_warned = True
            return incoherent

        # I(t) = | sum_n sqrt(R_n) p_n exp(-i (omega_n - omega_ref) t) |^2
        omega = self.eval_omega.to(dtype=config['real_dtype'], device=config['device'])
        omega_ref = omega.mean()
        phase = -(omega - omega_ref).unsqueeze(0) * time.unsqueeze(-1)      # (T, W)
        amplitude = x * self.coeff.clamp(min=0.0).sqrt()                    # (T, W, D)
        field = (amplitude * torch.exp(1.j * phase).unsqueeze(-1)).sum(dim=1)
        coherent = field.abs() ** 2                                         # (T, D)
        return torch.where(self.coherent.unsqueeze(0), coherent, incoherent)

    def _noise_variance(self, current: torch.Tensor) -> torch.Tensor:
        """Stationary variance of the filtered current noise, shape ``(T, D)``.

        One-sided white density ``S`` times the single pole's equivalent noise bandwidth
        ``(pi/2) bw``; zero for a detector that adds no noise.
        """
        density_shot = 2.0 * ELEMENTARY_CHARGE * current.clamp(min=0.0)
        density_johnson = 4.0 * BOLTZMANN * self.temp / self.rload
        density_thermal = torch.where(self.has_inoise, self.inoise ** 2, density_johnson)
        enbw = torch.where(self.noisy, 0.5 * pi * self.bw, torch.zeros_like(self.bw))
        return (density_shot + density_thermal.unsqueeze(0)) * enbw

    def _noise(self, current: torch.Tensor, time: Optional[torch.Tensor]) -> torch.Tensor:
        """One realisation of the detector noise, shape ``(T, D)`` (see *Noise* above)."""
        var = self._noise_variance(current)
        # sqrt has an infinite derivative at 0: keep a zero variance (a noiseless detector, or
        # zero light with no thermal term) out of it, or every gradient becomes NaN.
        positive = var > 0
        sigma = torch.where(positive, torch.sqrt(torch.where(positive, var, torch.ones_like(var))),
                            torch.zeros_like(var))
        draws = self._randn(current.shape)
        if time is None or current.shape[0] < 2:
            return sigma * draws
        tau = 1.0 / (2.0 * pi * torch.where(self.noisy, self.bw, torch.ones_like(self.bw)))
        dt = (time[1:] - time[:-1]).unsqueeze(-1)                          # (T-1, 1)
        a = torch.exp(-dt / tau.unsqueeze(0))                              # (T-1, D)
        innovation = torch.sqrt((1.0 - a ** 2).clamp(min=0.0))
        out = [sigma[0] * draws[0]]
        for k in range(a.shape[0]):
            out.append(a[k] * out[-1] + innovation[k] * sigma[k + 1] * draws[k + 1])
        return torch.stack(out)

    def forward(self, x: torch.Tensor, time: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x    shape (len(time), len(omega), len(dout_node))
        # out  shape (len(time), len(dout_node))
        # coeff shape (len(omega), len(dout_node))
        global _STD_DEPRECATION_WARNED

        if self.run_seed is not None:
            self.reseed(self.run_seed)
        time_axis = self._time_axis(x.shape[0], time)

        current = self._photocurrent(x, time_axis)

        # A real detector has a finite bandwidth; without bw= it keeps the historical infinite one.
        if bool((self.bw > 0).any()):
            if time_axis is None:
                if not self._no_time_warned:
                    warnings.warn(
                        "A photo detector declared bw= but the transient time axis is not "
                        "available to PDArray, so the bandwidth low-pass of the photocurrent is "
                        "skipped (the noise bandwidth itself is still honoured).  Pass the grid "
                        "with PDArray.set_time(t), or give dt= on the pd line.")
                    self._no_time_warned = True
            else:
                current = _lowpass(current, time_axis, self.bw)

        current = current + self.idark

        if bool(self.noisy.any()):
            current = current + self._noise(current, time_axis)

        if bool((self.std != 0).any()):
            if not _STD_DEPRECATION_WARNED:
                warnings.warn(
                    "The 'std=' attribute of a pd line is deprecated: it is a multiplicative term "
                    "with a constant *relative* sigma, which is neither shot noise (sigma ~ "
                    "sqrt(I)) nor thermal noise (sigma independent of I).  Use bw= together with "
                    "idark=/temp=/rload= or inoise= instead.  'std=' still works and is applied "
                    "on top of the physical noise.", DeprecationWarning, stacklevel=2)
                _STD_DEPRECATION_WARNED = True
            current = current * (1 + self.std * self._randn(current.shape))

        return current
