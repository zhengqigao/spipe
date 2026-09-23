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
from spipe.photonic import FreeLightSpeed
from spipe import Photonic
from spipe import photonic_reset
from math import pi
import torch
import argparse
from spipe import config, Circuit
from matplotlib import pyplot as plt
import random
from typing import List

photonic_reset()

up_l, bottom_l = 0, 0  # 150e-6, 150e-6
act_l = 200e-6
neff = 2.35
ng = 2.35
wl = FreeLightSpeed / 193.1e12
freq_start, freq_end, freq_num = 193.1, 193.4, 1
coeff1 = 1e-3
num_t = 200
num_theta = 500


def write_component(input_port: str, output_port: str, inner: str, act_port, cnt):
    p_content = f'splitter1to2_{cnt} {input_port} {inner}_1 {inner}_2 \n'
    p_content += f'wg_{cnt + 1} {inner}_1 {inner}_3 l={up_l}\n'
    p_content += f'wg_{cnt + 2} {inner}_2 {output_port[1]} l={bottom_l}\n'
    p_content += f'modp_{cnt + 3} {inner}_3 {output_port[0]} {act_port} debug coeff1={coeff1} act_l={act_l}\n'
    return p_content, cnt + 4


def helper(d_input, vcontrol, vmax, bit, sorted=False):
    # Format each input as a binary string with length equal to 'bit'
    d_format = [f'{e:0{bit}b}' for e in d_input]

    v = []
    num_expect = len(d_format)  # Define num_expect based on the length of d_format

    for j in range(bit):
        cur_v = [0]
        for i in range(num_expect):
            if d_format[i][-1 - j] == '1':  # Corrected variable name from digital_input1_format
                cur_v.extend([vcontrol, vcontrol])
            else:
                cur_v.extend([0.0, 0.0])
        v.append(cur_v)

    v_value = [0.0]
    for i in range(num_expect):
        cur = d_input[i] * vmax / float(2 ** bit - 1)
        v_value.extend([cur, cur])

    if sorted:
        v_value, indices = torch.sort(torch.Tensor(v_value))
        v = torch.Tensor(v)[:,indices]
        return v, v_value
    else:
        return torch.Tensor(v), torch.Tensor(v_value)


def gen_exp(num_expect, t_interval, t_transit, vcontrol, vmax, bit):
    # Time grids for PWL source
    t = [0]
    for i in range(num_expect):
        t.extend([t_transit + t_interval * i, t_interval + t_interval * i])
    t = torch.Tensor(t)

    # Generate random inputs
    # d_input1 = [random.randint(0, 2 ** bit - 1) for _ in range(num_expect)]
    d_input1 = torch.linspace(0, 2 ** bit - 1, num_expect).numpy()
    d_input1 = [int(e) for e in d_input1]

    # Call helper to process the inputs
    v1, v1_value = helper(d_input1, vcontrol, vmax, bit, sorted=True)
    print(v1, v1_value)

    ## TODO: remove it
    # v1 = torch.zeros_like(v1)
    # v1_value = torch.zeros_like(v1_value)

    return t, v1, v1_value


def write_DAC(output_port, t, v, d):
    content = (
        f"""
    * DAC for operand fed to {output_port}
x{d}_1 vrefh net-_x{d}_1-pad2_ d{d}_0 d{d}_1 d{d}_2 d{d}_3 d{d}_4 d{d}_5 d{d}_6 vdd net-_x{d}_1-pad11_ s7bit_DAC
x{d}_2 net-_x{d}_1-pad2_ vrefl d{d}_0 d{d}_1 d{d}_2 d{d}_3 d{d}_4 d{d}_5 d{d}_6 vdd net-_x{d}_2-pad11_ s7bit_DAC
x{d}_3 d{d}_7 vdd net-_x{d}_1-pad11_ net-_x{d}_2-pad11_ {output_port} switch

Vd{d}_0 d{d}_0 0 PWL ({''.join([f'{t[i]} {v[0][i]} ' for i in range(len(t))])})
Vd{d}_1 d{d}_1 0 PWL ({''.join([f'{t[i]} {v[1][i]} ' for i in range(len(t))])})
Vd{d}_2 d{d}_2 0 PWL ({''.join([f'{t[i]} {v[2][i]} ' for i in range(len(t))])})
Vd{d}_3 d{d}_3 0 PWL ({''.join([f'{t[i]} {v[3][i]} ' for i in range(len(t))])})
Vd{d}_4 d{d}_4 0 PWL ({''.join([f'{t[i]} {v[4][i]} ' for i in range(len(t))])})
Vd{d}_5 d{d}_5 0 PWL ({''.join([f'{t[i]} {v[5][i]} ' for i in range(len(t))])})
Vd{d}_6 d{d}_6 0 PWL ({''.join([f'{t[i]} {v[6][i]} ' for i in range(len(t))])})
Vd{d}_7 d{d}_7 0 PWL ({''.join([f'{t[i]} {v[7][i]} ' for i in range(len(t))])})
"""
    )
    return content, d + 1


def generate(num_cols):
    p_content = ''
    e_content = ''

    num_expect = 5
    t_interval, t_transit = 10e-9, 1e-12
    vcontrol, vmax = 1.8, 3.3

    act_count = 0
    cnt = 0

    t_value, v1, v1_value = gen_exp(num_expect, t_interval, t_transit, vcontrol, vmax, bit=8)



    for i in range(1, 1 + num_cols):
        for j in range(2 ** (i - 1)):
            input_port = f'n_{i - 1}_{j}'
            output_port = [f'n_{i}_{2 * j}', f'n_{i}_{2 * j + 1}']
            inner = f'ni_{i}_{j}'
            act_port = f'm_{act_count}'
            act_count = act_count + 1
            p_content_new, cnt = write_component(input_port, output_port, inner, act_port, cnt)
            p_content += p_content_new

            cur, cnt = write_DAC(act_port, t_value, v1, cnt)
            e_content += cur

    p_content += f".mode neff={neff} ng={ng} wl={wl}\n"
    p_content += f".freq {freq_start}thz {freq_end}thz {freq_num}\n"
    p_content += f".source 1.0@n_0_0 power=0.005w eff=0.2\n"
    p_content += f".prob {''.join([f'n_{num_cols}_{2 * j} n_{num_cols}_{2 * j + 1} ' for j in range(2 ** (num_cols - 1))])}\n"

    e_content += ".include ./dac_model/8bit_DAC/switch.sub\n.include ./dac_model/8bit_DAC/7bit_DAC.sub\n"
    e_content += 'Vdd vdd 0 3.3\nVrefh vrefh 0 3.3\nVrefl vrefl 0 0\n\n'

    t_start, t_stop = 0, t_value[-1]

    e_content += f".tran {t_start} {t_stop} {num_t}\n"
    all_content = '.electronic\n' + e_content + '\n' + '.photonic\n' + p_content

    return all_content, p_content, e_content, t_start, t_stop, t_value, v1_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', type=int, default=2)
    parser.add_argument('--file_path', type=str, default='test14.sp')
    parser.add_argument('--sim', type=str, default='hspice')
    parser.add_argument('--save_plot', action='store_true', default=False)
    args = parser.parse_args()
    config['max_iter'] = args.max_iter

    num_cols = 4
    all_content, p_content, e_conent, t_start, t_stop, t_value, v1_value = generate(num_cols)
    print(p_content)

    with open(args.file_path, 'w') as f:
        f.write(all_content)

    start_time = time.time()
    circuit = Circuit(args.file_path,
                      spice_exe=os.environ.get('SPIPE_HSPICE', 'hspice') + ' ' if args.sim == 'hspice' else os.environ.get('SPIPE_XYCE', 'Xyce') + ' -quiet -hspice-ext all',
                      spice_file_name='tmp2.sp',
                      power_node=['vrefh', 'vrefl', 'vdd'])

    prob_e, prob_p, e_in, p_in, power_dict = circuit.simulate()
    run_time = time.time() - start_time

    d = 0.5 * FreeLightSpeed / (freq_start*1e12)
    theta = torch.linspace(-0.5 * torch.pi, 0.5 * torch.pi, num_theta)

    atenna_array = []
    for j in range(2 ** (num_cols - 1)):
        atenna_array.append(f'n_{num_cols}_{2 * j}')
        atenna_array.append(f'n_{num_cols}_{2 * j + 1}')

    added = 0
    k_vec = 2 * torch.pi * torch.linspace(freq_start, freq_end, freq_num) * 1e12 / FreeLightSpeed  # (freq,)

    kdtheta = k_vec.view(-1, 1) * d * torch.sin(theta).view(1, -1)  # (freq, num_theta)


    res = torch.zeros((num_t, freq_num, len(theta)), dtype=torch.complex64)
    for i in range(len(atenna_array)):
        prob = prob_p[atenna_array[i]]  # (time, freq, 2)
        signal = prob[..., 1]  # (time, freq)

        res += signal.unsqueeze(-1) * torch.exp(1.j * i * kdtheta).unsqueeze(0)

    af = res.abs() # (time, freq, theta)

    print(torch.any(torch.isnan(af[:,0,:])))

    plt.figure()
    plt.plot(theta / torch.pi * 180, af[0, 0, :])
    plt.title('AF')
    plt.xlabel('degree')
    plt.ylabel('response')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Meshgrid for X and Y
    X, Y = np.meshgrid(np.linspace(t_start, t_stop, num_t), theta.detach().cpu().numpy())
    Z = (af[:, 0, :]).detach().cpu().numpy()

    print("X.shape", X.shape)
    print("Y.shape", Y.shape)
    print("Z.shape", Z.shape)

    Z = Z.transpose()
    print("Z.shape", Z.shape)

    # Plot the surface with lighting and shading
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', rstride=1, cstride=1, alpha=0.8)

    # Add a color bar for reference
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Response Intensity', fontsize=12)

    # Set axis labels with larger fonts
    ax.set_xlabel('Time', fontsize=14, labelpad=10)
    ax.set_ylabel('Theta', fontsize=14, labelpad=10)
    ax.set_zlabel('Response', fontsize=14, labelpad=10)

    # Customize tick parameters for better readability
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Set axis limits for better control (optional)
    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    # ax.set_zlim([zmin, zmax])
    ax.grid(False)
    # Title and layout adjustments
    ax.set_title('3D Surface Plot of Response over Time and Theta', fontsize=16, pad=20)
    plt.tight_layout()

    print(p_in.shape)

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    x = t_value * 1e9
    y = v1_value
    x_sample_hold = np.repeat(np.concatenate(([0], x)), 2)[:-1]
    y_sample_hold = np.repeat(np.concatenate(([y[0]], y)), 2)[1:]
    # plt.scatter(x , y, s = 40, marker='o')
    linewidth, font_size = 2, 20
    plt.plot(x_sample_hold, y_sample_hold, '--', c='red', label='target', linewidth=linewidth)
    plt.plot(torch.linspace(t_start, t_stop, num_t) * 1e9, p_in[:,0], 'blue', label='simulated 1',
             linewidth=linewidth)
    plt.plot(torch.linspace(t_start, t_stop, num_t) * 1e9, p_in[:,1], 'blue', label='simulated 2',
             linewidth=linewidth)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    plt.legend(fontsize=font_size)
    plt.xlabel('Time (ns)', fontsize=font_size)
    plt.ylabel('Voltage (volts)', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)

    plt.show()
