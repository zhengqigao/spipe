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

# based on test11.py, let us add electronic part.

act_l = 200e-6
coeff1 = 1e-3
neff = 2.35
r0 = 1e-3
mag = 1.0
rload = 10e3

std = 0.05
freq_start, freq_end = 193.5e12, 193.5e12


def write_ddot_content(input_port: List[str], output_port: List[str], inner_node_prefix: str, device_count):
    n1, n2 = input_port
    n3, n4 = output_port
    content = (
        f"""
ps{device_count} {n2} {inner_node_prefix}_1 ps=-0.5pi
mzi{device_count + 1} {n1} {inner_node_prefix}_1 {inner_node_prefix}_2 {inner_node_prefix}_3 theta=0.25pi
pd{device_count + 2} {inner_node_prefix}_2 {n3} level2 r0={r0} std={std}
pd{device_count + 3} {inner_node_prefix}_3 {n4} level2 r0={r0} std={std}
"""
    )
    return content, device_count + 4


def write_modulator_content(input_port, output_port, active_port, inner_node_prefix, device_count):
    content = (
        f"""
wdm1to{len(active_port)}_{device_count} {input_port} {''.join([f'{inner_node_prefix}_b_{i} ' for i in range(len(active_port))])}
wdm1to{len(active_port)}_{device_count + 1} {inner_node_prefix}_bsp {''.join([f'{inner_node_prefix}_a_{i} ' for i in range(len(active_port))])}
splitter1to{len(output_port)}_{device_count + 2} {inner_node_prefix}_bsp {' '.join([str(n) for n in output_port])}
"""
    )

    device_count += 3
    for i in range(len(active_port)):
        cur_content = f"modm{device_count} {inner_node_prefix}_b_{i}  {inner_node_prefix}_dummy_{2 * i} {inner_node_prefix}_a_{i} {inner_node_prefix}_dummy_{2 * i + 1} {active_port[i]} level1 coeff1={coeff1} act_l={act_l} alpha = 1.0\n"
        content += cur_content
        device_count += 1

    return content, device_count


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


def helper(d_input, vcontrol, vmax, bit):
    # Format each input as a binary string with length equal to 'bit'
    d_format = [f'{e:0{bit}b}' for e in d_input]

    v = []
    num_expect = len(d_format)  # Define num_expect based on the length of d_format

    for j in range(bit):
        cur_v = [0]
        for i in range(num_expect):
            if d_format[i][-1-j] == '1':  # Corrected variable name from digital_input1_format
                cur_v.extend([vcontrol, vcontrol])
            else:
                cur_v.extend([0.0, 0.0])
        v.append(cur_v)

    v_value = [0.0]
    for i in range(num_expect):
        cur = d_input[i] * vmax / float(2 ** bit - 1)
        v_value.extend([cur, cur])

    return torch.Tensor(v), torch.Tensor(v_value)


def gen_exp(num_expect, t_interval, t_transit, vcontrol, vmax, bit):

    # Time grids for PWL source
    t = [0]
    for i in range(num_expect):
        t.extend([t_transit + t_interval * i, t_interval + t_interval * i])
    t = torch.Tensor(t)

    # Generate random inputs
    d_input1 = [random.randint(0, 2 ** bit - 1) for _ in range(num_expect)]
    d_input2 = [random.randint(0, 2 ** bit - 1) for _ in range(num_expect)]

    # Call helper to process the inputs
    v1, v1_value = helper(d_input1, vcontrol, vmax, bit)
    v2, v2_value = helper(d_input2, vcontrol, vmax, bit)

    return t, v1, v2, v1_value, v2_value

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', type=int, default=2)
    parser.add_argument('--file_path', type=str, default='test12.sp')
    parser.add_argument('--sim', type=str, default='hspice')
    args = parser.parse_args()

    # These decks instantiate the SkyWater SKY130 transistors, which are third-party and not
    # shipped. Without them HSPICE stops with "job aborted"; say what is missing instead.
    _sky130 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'dac_model', 'sky130_fd_pr', 'models', 'sky130.lib.spice')
    if not os.path.exists(_sky130):
        raise SystemExit(
            "This paper deck needs the SkyWater SKY130 transistor models, in HSPICE syntax, at\n"
            f"  {_sky130}\n"
            "They are third-party and not shipped with SPIPE: install them with "
            "scripts/fetch_sky130.sh (see examples/README.md).\n"
            "examples/derived/ rebuilds these circuits on level-1 devices and needs no models.")
    config['max_iter'] = args.max_iter

    random.seed(0)
    torch.manual_seed(0)
    config = (2, 2)
    num_channel = 2

    device_count = 0
    photonic_content = ''
    for i in range(config[0]):
        for j in range(config[1]):
            cur_content, device_count = write_ddot_content([f'n{i}{j}_up', f'n{i}{j}_do'],
                                                           [f'm{i}{j}_up', f'm{i}{j}_do'], f'nd{i}{j}', device_count)
            photonic_content += cur_content

    for i in range(config[0]):
        cur_content, device_count = write_modulator_content(f'h_in{i}',
                                                            [f'n{i}{j}_up' for j in range(config[1])],
                                                            [f'nac_h{i}_c{c}' for c in range(num_channel)],
                                                            f'nd_h{i}',
                                                            device_count)

        photonic_content += cur_content

    for j in range(config[1]):
        cur_content, device_count = write_modulator_content(f'v_in{j}',
                                                            [f'n{i}{j}_do' for i in range(config[0])],
                                                            [f'nac_v{j}_c{c}' for c in range(num_channel)],
                                                            f'nd_v{j}',
                                                            device_count)
        photonic_content += cur_content

    photonic_content += f"""
    .mode neff={neff}
    .freq {freq_start} {freq_end} {num_channel}
    .source {' '.join([f'{mag}@h_in{i}' for i in range(config[0])])} {' '.join([f'{mag}@v_in{j}' for j in range(config[1])])} power=0.005w eff=0.2
    .prob nd_h0_bsp n00_up 
    .end
    
    """

    electronic_content = ".include ./dac_model/8bit_DAC/switch.sub\n.include ./dac_model/8bit_DAC/7bit_DAC.sub\n"
    electronic_content += 'Vdd vdd 0 3.3\nVrefh vrefh 0 3.3\nVrefl vrefl 0 0\n\n'

    num_expect = 5
    t_interval, t_transit = 10e-9, 1e-12
    vcontrol, vmax = 1.8, 3.3

    v1 = torch.zeros((num_channel, config[0], 8, 2 * num_expect + 1))
    v2 = torch.zeros((num_channel, config[1], 8, 2 * num_expect + 1))
    v1_value = torch.zeros((num_channel, config[0], 2 * num_expect + 1))
    v2_value = torch.zeros((num_channel, config[1], 2 * num_expect + 1))

    for c in range(num_channel):
        for i in range(config[0]):
            for j in range(config[1]):
                t_value, v1[c,i,...], v2[c,j,...], v1_value[c,i,...], v2_value[c,j,...] = gen_exp(num_expect, t_interval, t_transit, vcontrol, vmax, bit=8)

    t_start, t_stop, num_t = 0, t_value[-1], 100

    for c in range(num_channel):
        for i in range(config[0]):
            cur, device_count = write_DAC(f'nac_h{i}_c{c}', t_value, v1[c,i,...], device_count)
            electronic_content += cur

        for j in range(config[1]):
            cur, device_count = write_DAC(f'nac_v{j}_c{c}', t_value, v2[c,j,...], device_count)
            electronic_content += cur

    for i in range(config[0]):
        for j in range(config[1]):
            electronic_content += f"r{device_count} m{i}{j}_up 0 {rload}\n"
            electronic_content += f"r{device_count+1} m{i}{j}_do 0 {rload}\n"
            device_count += 2

    electronic_content += f".tran {t_start} {t_stop} {num_t}\n"
    electronic_content += f".print tran {''.join([f'v(m{i}{j}_up) v(m{i}{j}_do) ' for i in range(config[0]) for j in range(config[1])])}"
    all = '.electronic\n' + electronic_content + '\n' + '.photonic' + photonic_content


    with open(args.file_path, 'w') as f:
        f.write(all)

    circuit = Circuit(args.file_path,
                      spice_exe=os.environ.get('SPIPE_HSPICE', 'hspice') + ' ' if args.sim == 'hspice' else os.environ.get('SPIPE_XYCE', 'Xyce') + ' -quiet -hspice-ext all',
                      power_node=['vrefh', 'vrefl', 'vdd'])

    prob_e, prob_p, e_in, p_in, power_dict = circuit.simulate()

    ind_i, ind_j = 0,0

    calculate = torch.zeros((num_t, config[0], config[1]))
    for i in range(config[0]):
        for j in range(config[1]):
            calculate[:,i,j] = prob_e[f'v(m{i}{j}_up)'] - prob_e[f'v(m{i}{j}_do)']

    freq = torch.linspace(freq_start, freq_end, num_channel)

    amplify = 2 * (1 / config[0]) ** 0.5 * (1 / config[1]) ** 0.5 * r0 * mag ** 2 * rload  # if pd uses debug model=rload, if pd uses level2 model=Rf.

    target = torch.zeros((len(t_value), config[0], config[1]))


    for j in range(config[0]):
        for k in range(config[1]):
            a = (2 * pi * freq * neff / FreeLightSpeed * act_l * coeff1).view(-1,1)
            phi1 = a * v1_value[:,j,:]
            phi2 = a * v2_value[:,k,:]
            target[:,j,k] += (amplify * torch.cos(phi1) * torch.cos(phi2)).sum(dim=0)


    font_size = 16
    linewidth = 3

    for ind_i in range(config[0]):
        for ind_j in range(config[1]):
            for ind_c in range(num_channel):
                disp_ind = [2 * i for i in range(1, 1 + num_expect)]


                plt.figure(figsize=(8, 6))
                plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
                plt.plot(t_value * 1e9, target[:,ind_i, ind_j], '--', c='red', label='target', linewidth=linewidth)
                # x = t_value[disp_ind] * 1e9
                # y = target[disp_ind, ind_i, ind_j]
                # x_sample_hold = np.repeat(np.concatenate(([0], x)), 2)[:-1]
                # y_sample_hold = np.repeat(np.concatenate(([y[0]], y)), 2)[1:]
                # plt.plot(x_sample_hold, y_sample_hold, '--', c='red', label='target', linewidth=linewidth)
                plt.plot(torch.linspace(t_start,t_stop,num_t) * 1e9, prob_e[f'v(m{ind_i}{ind_j}_up)'] - prob_e[f'v(m{ind_i}{ind_j}_do)'], 'blue', label='simulated', linewidth=linewidth)
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
                plt.title(f'Output of DDot ({ind_i},{ind_j})')

                plt.figure(figsize=(8, 6))
                plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
                plt.plot(torch.linspace(t_start,t_stop,num_t) * 1e9, p_in[:,ind_i * num_channel + ind_c],c='r', label =f'MOD control at row {ind_i}, channel {ind_c}', linewidth=linewidth)
                plt.plot(torch.linspace(t_start,t_stop,num_t) * 1e9, p_in[:,(config[0] + ind_j) * num_channel + ind_c], c='blue', label =f'MOD control at col {ind_j}, channel {ind_c}', linewidth=linewidth)
                # wrk1 = v1_value[ind_c, ind_i, disp_ind]
                # wrk1_sample_hold = np.repeat(np.concatenate(([wrk1[0]], wrk1)), 2)[1:]
                plt.plot(t_value * 1e9, v1_value[ind_c, ind_i,:], '--', label = f'ideal contr at row {ind_i}, channel {ind_c}')
                # wrk2 = v2_value[ind_c, ind_j, disp_ind]
                # wrk2_sample_hold = np.repeat(np.concatenate(([wrk2[0]], wrk2)), 2)[1:]
                plt.plot(t_value * 1e9, v2_value[ind_c, ind_j, :], '--', label = f'ideal contr at col {ind_j}, channel {ind_c}')
                plt.xlabel('Time (ns)', fontsize=font_size)
                plt.ylabel('Voltage (volts)', fontsize=font_size)
                plt.xticks(fontsize=font_size)
                plt.yticks(fontsize=font_size)
                plt.legend(fontsize=font_size)
                plt.title(f'ADCs for op1 (row={ind_i},c={ind_c}), op2 (col={ind_j},c={ind_c})', fontsize=font_size)


    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    all = 0
    for k,v in power_dict.items():
        if v is not None:
            all += v
            plt.plot(torch.linspace(t_start, t_stop, num_t) * 1e9, v * 1e3, label=f"power[{k}]", linewidth=linewidth)
    plt.plot(torch.linspace(t_start, t_stop, num_t) * 1e9, all * 1e3, label=f"all power", linewidth=linewidth)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    plt.legend(fontsize=font_size)
    plt.xlabel('Time (ns)', fontsize=font_size)
    plt.ylabel('Power (mW)', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)

    plt.show()
