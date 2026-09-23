import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
import os as _os, sys as _sys
# Make `import spipe` work from a source checkout without installing.
# Walks up to the repo root and puts `src/` on the path; a pip-installed
# spipe takes precedence because this only appends if the import fails.
try:
    import spipe as _probe  # noqa: F401
except ImportError:
    _here = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(6):
        _cand = _os.path.join(_here, 'src')
        if _os.path.isdir(_os.path.join(_cand, 'spipe')):
            _sys.path.insert(0, _cand)
            break
        _here = _os.path.dirname(_here)
from spipe import Photonic
from spipe import photonic_reset
from math import pi
import torch
from spipe.photonic import FreeLightSpeed
from spipe import config

# verify the calculation of gradients for a more general/complex active photonic circuit 2-by-2 mesh shown in
# https://github.com/zhengqigao/spode/blob/main/tutorials/lesson2_verify_2by2_mesh/main.ipynb
# Cautiously optimisitic: seems to work. The incremental gradient sometimes might be far from golden...
# I manually derived an anlytical solution in a special case (all modms in cross state) to compare it with the incremental.

photonic_reset()

config['real_dtype'] = torch.float64
config['complex_dtype'] = torch.complex64


freq_start, freq_end, freq_num = 200e12, 205e12, 6
mode_info = {'neff': 1.0, 'ng': 1.0, 'wl': FreeLightSpeed / 200.0e12}
omega = np.linspace(freq_start, freq_end, freq_num) * 2 * np.pi
prob_node = ['n23']
num_exp = 1

wg_u, wg_l, alpha = 0, 0, 1.0

coeff = 1.0
act_l = 10e-6

p_content = (
f"""


pd1 n17 m7 level1 r0=1.0 # n22 is a node connected to modm10
pd2 n23 m8 level1 r0=1.0 # n28 is a node connected to modm10

modm1 n1  n6  n2  n7   m1 level1 wgu_l={wg_u} wgl_l={wg_l} coeff1={coeff} act_l={act_l} alpha = {alpha}
modm4 n14 n20 n15 n21  m2 level1 wgu_l={wg_u} wgl_l={wg_l} coeff1={coeff} act_l={act_l} alpha = {alpha}
modm5 n24 n29 n25 n30  m3 level1 wgu_l={wg_u} wgl_l={wg_l} coeff1={coeff} act_l={act_l} alpha = {alpha}
modm6 n26 n31 n27 n32  m4 level1 wgu_l={wg_u} wgl_l={wg_l} coeff1={coeff} act_l={act_l} alpha = {alpha}
modm7 n6 n5 n12 n11    m5 level1 wgu_l={wg_u} wgl_l={wg_l} coeff1={coeff} act_l={act_l} alpha = {alpha}
modm9 n10 n9 n16 n15   m6 level1 wgu_l={wg_u} wgl_l={wg_l} coeff1={coeff} act_l={act_l} alpha = {alpha}
modm11 n20 n19 n26 n25 m7 level1 wgu_l={wg_u} wgl_l={wg_l} coeff1={coeff} act_l={act_l} alpha = {alpha}
modm12 n22 n21 n28 n27 m8 level1 wgu_l={wg_u} wgl_l={wg_l} coeff1={coeff} act_l={act_l} alpha = {alpha}


modm2  n3  n8  n4  n9  m9  level1 wgu_l={wg_u} wgl_l={wg_l} coeff1=1.0 act_l=10e-6 alpha = {alpha}
modm3  n12 n18 n13 n19 m10 level1 wgu_l={wg_u} wgl_l={wg_l} coeff1=1.0 act_l=10e-6 alpha = {alpha}
modm8  n8  n7  n14 n13 m11 level1 wgu_l={wg_u} wgl_l={wg_l} coeff1=1.0 act_l=10e-6 alpha = {alpha}
modm10 n18 n17 n24 n23 m12 level1 wgu_l={wg_u} wgl_l={wg_l} coeff1=1.0 act_l=10e-6 alpha = {alpha}

.mode neff={mode_info['neff']} ng={mode_info['ng']} wl={mode_info['wl']}
.freq {freq_start} {freq_end} {freq_num}
.source 1.0@n4
.prob n23 n17
"""
)

epsilon = 1e-5
start_time = time.time()
circuit = Photonic(p_content.split('\n'), need_grads=True)
end_time1 = time.time()


beta = 2 * pi * (freq_start + freq_end) / 2 * mode_info['neff'] / FreeLightSpeed
bound = 2 * pi / (beta * coeff * act_l)

torch.manual_seed(0)
for i in range(num_exp):
    t = torch.linspace(0,1,1).to(config['device'])
    v = (bound * torch.rand(len(t), len(circuit.mod_element.keys()))).to(config['device']).requires_grad_(True)

    # v = 3 / 80.0 * torch.ones(len(t), len(circuit.mod_element.keys()))
    # v[:,-1] = 1.0

    v = v.requires_grad_(True)

    vout, middle , _ = circuit.simulate(t, v)

    loss = (vout ** 2).sum()

    loss.backward()

    incre_grad = torch.zeros_like(v.grad)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            v_perturb = v.detach().clone()
            tmp = v_perturb[i, j].item()

            # Calculate the perturbations
            v_perturb[i, j] = (1 + 2 * epsilon) * tmp
            vout_perturb2h_plus, _, _ = circuit.simulate(t, v_perturb)

            v_perturb[i, j] = (1 + epsilon) * tmp
            vout_perturbh_plus, _, _ = circuit.simulate(t, v_perturb)

            v_perturb[i, j] = (1 - epsilon) * tmp
            vout_perturbh_minus, _, _ = circuit.simulate(t, v_perturb)

            v_perturb[i, j] = (1 - 2 * epsilon) * tmp
            vout_perturb2h_minus, _, _ = circuit.simulate(t, v_perturb)

            # Calculate the loss perturbations using the five-point stencil method
            loss_perturb = (
                    - (vout_perturb2h_plus ** 2).sum()
                    + 8 * (vout_perturbh_plus ** 2).sum()
                    - 8 * (vout_perturbh_minus ** 2).sum()
                    + (vout_perturb2h_minus ** 2).sum()
            )

            # Update the gradient
            incre_grad[i, j] = loss_perturb / (12 * epsilon * tmp)

    diff = torch.abs(incre_grad-v.grad)
    print("incre grad", incre_grad)
    print("bp grad", v.grad)
    print("diff", diff)
    print("relative diff", diff/ v.grad.abs())
    print("v", v)
    print(f"max diff: {diff.max().item()}, mean diff: {diff.mean().item()}, mean incre grad: {incre_grad.abs().mean().item()}")

end_time2 = time.time()
outward_ind = 1
print(f"spipe build circuit: {end_time1 - start_time:.3f} seconds")
print(f"spipe run time: {(end_time2 - end_time1)/num_exp:.3f} seconds")

plt.figure()
key = 'n23'
plt.plot(2 * np.pi * FreeLightSpeed / omega * 1e9, np.abs(middle[key][0, :, outward_ind]), 'blue', linestyle='-',
         label=f"Spipe middle freq --- {key}")
plt.xlabel('lambda in free space (nm)')
plt.legend()
plt.title('Abs()')
plt.show()



