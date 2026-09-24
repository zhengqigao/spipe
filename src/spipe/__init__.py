# ---------------------------------------------------------------------------
# Import order matters: numpy and scipy MUST be imported before torch.
#
# On some installations (verified here on the `mp` env: torch 2.3.1 + scipy 1.8.0)
# importing torch first silently corrupts scipy.interpolate.interp1d(kind='cubic').
# Measured on a realistic HSPICE-shaped array (2400 non-uniform points, a duplicate
# timestamp, 9 columns): with numpy imported first the interpolation is accurate to
# 1.3e-07; with torch imported first it returns nan or inf, a DIFFERENT wrong value
# on each run. That function is `interp1d_warp`, which maps SPICE transient results
# back onto the simulation time grid, so the corruption reached every co-simulation
# result -- silently, with exit status 0.
#
# Importing them here, before torch, makes the safe order unconditional regardless
# of how the user's own script orders its imports.
import numpy          # noqa: F401  (import-order guard, see above)
import scipy.interpolate  # noqa: F401  (import-order guard, see above)

import torch

__version__ = "2.0.0"

#: Every setting SPIPE reads (docs/performance.md lists what they do). ``spipe.config`` is a
#: plain dict that warns when a key outside this set is written, so a typo such as
#: ``config['native_nsubb'] = 4`` does not pass silently as "no effect".
_KNOWN_CONFIG_KEYS = frozenset({
    'device', 'real_dtype', 'complex_dtype', 'lr', 'max_iter', 'atol', 'rtol', 'seed',
    'quasistatic_check', 'damping', 'min_damping', 'backoff_factor', 'divergence_factor',
    'anderson_depth', 'anderson_reg', 'fixed_point_stability_check',
    'coupling_tol', 'coupling_max_iter', 'coupling_dense_limit', 'coupling_cond_limit',
    'coupling_zero_rtol', 'coupling_probe_rtol', 'coupling_probe_linearity',
    'coupling_amplification_warn', 'photonic_solver', 'photonic_dense_budget',
    'photonic_jac_chunk', 'photonic_jac_bytes', 'native_nsub', 'native_adaptive',
    'native_uic', 'native_lte_reltol', 'fd_step', 'xyce_sens_photocurrent',
    'envelope_edge', 'envelope_adiabatic_ratio', 'envelope_tap_tol', 'envelope_max_bytes',
    'envelope_grad_max_bytes'})


def _check_config_value(key, value):
    """Type and range of the settings whose misuse gives a wrong answer rather than an error."""
    def fail(what):
        raise ValueError(f"spipe.config[{key!r}] = {value!r}: {what}")
    if key == 'real_dtype' and value not in (torch.float32, torch.float64):
        fail("must be torch.float64 (or torch.float32)")
    if key == 'complex_dtype' and value not in (torch.complex64, torch.complex128):
        fail("must be torch.complex128 (or torch.complex64); a real dtype discards the phase")
    if key == 'max_iter' and not (isinstance(value, int) and value >= 1):
        fail("must be a whole number of iterations >= 1")
    if key in ('atol', 'rtol', 'fd_step', 'photonic_dense_budget', 'envelope_max_bytes',
               'envelope_grad_max_bytes'):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0:
            fail("must be a number >= 0")
    if key == 'native_nsub' and value is not None and not (isinstance(value, int) and value >= 1):
        fail("must be None (adaptive) or a whole number >= 1")


class _Config(dict):
    def __setitem__(self, key, value):
        _check_config_value(key, value)
        if key not in _KNOWN_CONFIG_KEYS:
            import difflib
            import warnings
            close = difflib.get_close_matches(str(key), sorted(_KNOWN_CONFIG_KEYS), n=1)
            warnings.warn(f"spipe.config[{key!r}] is not a setting SPIPE reads, so it has no "
                          f"effect" + (f" (did you mean {close[0]!r}?)" if close else "")
                          + ". The settings are listed in docs/performance.md.", stacklevel=2)
        super().__setitem__(key, value)


config = _Config({'device': torch.device("cpu"), #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
          'real_dtype': torch.float64,
          'complex_dtype': torch.complex128,
          'lr': 1e-2,
          'max_iter': 100,
          'atol': 1e-3,
          'rtol': 1e-3,
          # Seed of the *local* random generator used to initialise the fixed-point iteration.
          # It is applied through a private torch.Generator so that the caller's global RNG
          # state is never touched.  Circuit.simulate(seed=...) overrides it for a single run.
          'seed': 0,
          # Emit a warning when the estimated optical group delay of the photonic network is no
          # longer negligible compared with the transient time step (the quasi-static assumption
          # the steady-state photonic solve relies on).  Set to False to disable the check.
          'quasistatic_check': True})

from .core import Circuit

from .photonic.register import reset as photonic_reset
from .photonic.register import register as photonic_register
from .photonic.photonic import Photonic

from .electronic import reset as electronic_reset
electronic_rest = electronic_reset  # misspelt name kept so older scripts still import
from .electronic import register as electronic_register
from .electronic import Electronic
