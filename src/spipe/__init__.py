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

config = {'device': torch.device("cpu"), #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
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
          'quasistatic_check': True}

from .core import Circuit

from .photonic.register import reset as photonic_reset
from .photonic.register import register as photonic_register
from .photonic.photonic import Photonic

from .electronic import reset as electronic_rest
from .electronic import register as electronic_register
from .electronic import Electronic
