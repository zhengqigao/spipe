import textwrap
import os
import torch
import sys
import argparse
import time
import numpy as np

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
import spipe
from spipe import Photonic, config
from torch import pi
import h5py
import matplotlib.pyplot as plt
from spipe.photonic import FreeLightSpeed

ng, neff, wl = 2.35, 2.35, FreeLightSpeed / 193.1e12

# the linspace in Interconnect and pytorch is different
# the following line is mainly for constructing exactly the same simulation frequency grid in SPIPE (which use pytorch linsapce)
# and Interconnect.
# NOTE: these two numerals are in THz -- the `.freq` line below writes them with an explicit
# `thz` unit suffix.  Anything that uses them as a physical frequency must convert to Hz first
# (see THZ below); using them raw makes beta 1e12 too small, which used to push the drive
# voltages to ~1e12 V and drown the finite-difference reference in round-off.
freq_start, freq_end, freq_num = 192.8, 193.4, 100
spipe_freq_end = freq_start + (freq_end - freq_start) / freq_num * (freq_num - 1)
THZ = 1e12  # THz -> Hz

theta, phi = 0.5, 0.0
alpha_value = 1.0
act_l = 250e-6
wg_u, wg_l = 0, 0
coeff = 1e-3

linewidth = 2
font_size = 18


epsilon = 1e-5

def run_spipe(num_row, num_col, source_in, prob_node):
    total = (num_row + 1) * num_col + (num_col + 1) * num_row


    count = 0
    final = ''
    for i in range(1, num_row + 1):
        for j in range(1, num_col + 2):
            device_name = f'modm_hori_{i}_{j} '
            ports = ''.join([f"n_hori_{i}_{j}_p{n} " for n in range(1, 5)])
            attri=f"wgu_l={wg_u} wgl_l={wg_l} coeff1={coeff} act_l={act_l} alpha = {alpha_value}"

            sentence = device_name + ports + attri + '\n'

            final += sentence
            count += 1

    for i in range(1, num_col + 2):
        for j in range(1, num_row + 1):
            device_name = f'modm_verti_{i}_{j} '
            ports = ''.join([f"n_verti_{i}_{j}_p{n} " for n in range(1, 5)])
            attri=f"wgu_l={wg_u} wgl_l={wg_l} coeff1={coeff} act_l={act_l} alpha = {alpha_value}"

            sentence = device_name + ports + attri + '\n'

            final += sentence
            count += 1

    # a trick to enforce node are the same
    cnt = 0
    for i in range(1, num_row + 1):
        for j in range(1, num_col + 1):
            d1 = f"wg_{cnt} n_verti_{i}_{j}_p1 n_hori_{i}_{j}_p2 l=0\n"
            d2 = f"wg_{cnt + 1} n_hori_{i}_{j}_p4 n_verti_{i + 1}_{j}_p2 l=0\n"
            d3 = f"wg_{cnt + 2} n_verti_{i + 1}_{j}_p4 n_hori_{i}_{j + 1}_p3 l=0\n"
            d4 = f"wg_{cnt + 3} n_hori_{i}_{j + 1}_p1 n_verti_{i}_{j}_p3 l=0\n"
            cnt += 4
            final += d1 + d2 + d3 + d4

    final += f"pd_{cnt} {prob_node} n_out level1 r0=1e-3\n"

    final += f".mode neff={neff} ng={ng} wl={wl}\n"
    final += f".freq {freq_start}thz {spipe_freq_end}thz {freq_num}\n"
    final += f".source 1.0@{source_in}\n"
    final += f".prob {prob_node}\n"

    start_time = time.time()
    t = torch.linspace(0, 1, 1).to(config['device'])

    beta = 2 * pi * (freq_start + freq_end) / 2 * THZ * neff / FreeLightSpeed
    bound = 2 * pi / (beta * coeff * act_l)

    photonic = Photonic(final.split('\n'), need_grads=True)
    v = (bound * torch.rand(len(t), len(photonic.mod_element.keys()))).to(config['device']).requires_grad_(True)
    v = v.requires_grad_(True)
    vout, middle, _ = photonic.simulate(t, v)


    loss = (vout ** 2).sum()

    loss.backward()

    incre_grad = torch.zeros_like(v.grad)


    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            v_perturb = v.detach().clone()
            tmp = v_perturb[i, j].item()

            # Calculate the perturbations
            v_perturb[i, j] = (1 + 2 * epsilon) * tmp
            vout_perturb2h_plus, _, _ = photonic.simulate(t, v_perturb)

            v_perturb[i, j] = (1 + epsilon) * tmp
            vout_perturbh_plus, _, _ = photonic.simulate(t, v_perturb)

            v_perturb[i, j] = (1 - epsilon) * tmp
            vout_perturbh_minus, _, _ = photonic.simulate(t, v_perturb)

            v_perturb[i, j] = (1 - 2 * epsilon) * tmp
            vout_perturb2h_minus, _, _ = photonic.simulate(t, v_perturb)

            # Calculate the loss perturbations using the five-point stencil method
            loss_perturb = (
                    - (vout_perturb2h_plus ** 2).sum()
                    + 8 * (vout_perturbh_plus ** 2).sum()
                    - 8 * (vout_perturbh_minus ** 2).sum()
                    + (vout_perturb2h_minus ** 2).sum()
            )

            # Update the gradient
            incre_grad[i, j] = loss_perturb / (12 * epsilon * tmp)
    run_time = time.time() - start_time
    diff = torch.abs(incre_grad - v.grad)

    return run_time, diff, vout



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_exp', type=int, default=1)
    parser.add_argument('--max_size', type=int, default=30,
                        help='largest mesh (N x N) in the 3, 6, 9, ... sweep; the default 30 takes '
                             'hours, because every one of the 1,860 phase shifters of a 30x30 '
                             'mesh costs four finite-difference solves. Try --max_size 9.')
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--save_plot', action='store_true', default = False)
    parser.add_argument('--gpu', type = int, default = -1)
    args = parser.parse_args()

    config['device'] = torch.device(f"cuda:{args.gpu}") if args.gpu >=0 else torch.device("cpu")

    num_row_list = list(range(3, args.max_size + 1, 3))
    num_col_list = list(range(3, args.max_size + 1, 3))

    run_time = np.empty((len(num_row_list), args.num_exp, 2))
    err = np.empty((len(num_row_list), args.num_exp))

    for j in range(len(num_row_list)):
        num_row, num_col = num_row_list[j], num_col_list[j]
        total = (num_row + 1) * num_col + (num_col + 1) * num_row
        print(f"--- {num_row}x{num_col} mesh, {total} devices", flush=True)

        in_key, in_i, in_j, in_p = 'hori', 1, 1, 1
        out_key, out_i, out_j, out_p = 'hori', 1, 1, 3

        avg_diff = 0
        for i in range(args.num_exp):
            run_time, diff, vout = run_spipe(num_row, num_col, source_in=f'n_{in_key}_{in_i}_{in_j}_p{in_p}',
                                              prob_node=f'n_{out_key}_{out_i}_{out_j}_p{out_p}')

            print("diff mean",diff.mean(), flush=True)
            print("vout",vout, flush=True)
            avg_diff += diff.mean()

        print(f"final avg diff: {avg_diff/args.num_exp}")

