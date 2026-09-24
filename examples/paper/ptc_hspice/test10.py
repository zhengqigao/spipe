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

# let's try to combine everything together DAC-photonic-ADC
# worked example. DAC is based on an open-source github repo using Sky 130nm process.
# overall, the output frequency is 0.1Ghz, but this work focuses is not design a PTC, but end-to-end simulate it.
# modulator uses the basic 'debug' model, otherwise, 'level1' might reduce the voltage fed into the photonic circuit.

photonic_reset()



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
    # [0, 8] # [0 for _ in range(num_expect)] # [random.randint(0, 2 ** bit - 1) for _ in range(num_expect)]
    d_input2 = [random.randint(0, 2 ** bit - 1) for _ in range(num_expect)]
    # [0, 8] # [0 for _ in range(num_expect)] # [random.randint(0, 2 ** bit - 1) for _ in range(num_expect)]

    # Call helper to process the inputs
    v1, v1_value = helper(d_input1, vcontrol, vmax, bit)
    v2, v2_value = helper(d_input2, vcontrol, vmax, bit)

    return t, v1, v2, v1_value, v2_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', type=int, default=2)
    parser.add_argument('--file_path', type=str, default='test10.sp')
    parser.add_argument('--sim', type=str, default='hspice')
    parser.add_argument('--save_plot', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default = 0)
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

    random.seed(args.seed)

    act_l, coeff1, neff, r0, mag, rload = 200e-6, 1e-3, 2.35, 1e-3, 1.0, 1e3

    freq = 193.5e12  # single channel, otherwise due to encoding/modulation dispersion, it will lead to product inaccuracy

    num_expect = 10
    t_interval, t_transit = 100e-9, 10e-12 # 10e-9, 1e-12
    vcontrol, vmax = 1.8, 3.3
    bit = 8

    t, v1, v2, v1_value, v2_value = gen_exp(num_expect, t_interval, t_transit, vcontrol, vmax, bit)

    # t shape (2 * num_expect + 1)
    # v1, v2 shape (bit, 2 * num_expect + 1)
    # v1_value, v2_value shape (2 * num_expect + 1)

    disp_ind = [2 * i for i in range(1, 1 + num_expect)]
    t_start, t_stop, num_t = 0, t[-1], 100


    content = (
    f"""
     a netlist with synthetic parameters so that it is easy for debugging purpose. It worked.
    
    .electronic # electronic components start here until encountering .end or .photonic

    .include ./dac_model/8bit_DAC/switch.sub
    .include ./dac_model/8bit_DAC/7bit_DAC.sub
    
    * m1 is the output node for DAC1, will be connected to MOD1
    
    x1_1 vrefh net-_x1_1-pad2_ d1_0 d1_1 d1_2 d1_3 d1_4 d1_5 d1_6 vdd net-_x1_1-pad11_ s7bit_DAC
    x1_2 net-_x1_1-pad2_ vrefl d1_0 d1_1 d1_2 d1_3 d1_4 d1_5 d1_6 vdd net-_x1_2-pad11_ s7bit_DAC
    x1_3 d1_7 vdd net-_x1_1-pad11_ net-_x1_2-pad11_ m1 switch
    
    Vdd vdd 0 3.3
    Vd1_0 d1_0 0 PWL ({''.join([f'{t[i]} {v1[0][i]} ' for i in range(len(t))])})
    Vd1_1 d1_1 0 PWL ({''.join([f'{t[i]} {v1[1][i]} ' for i in range(len(t))])})
    Vd1_2 d1_2 0 PWL ({''.join([f'{t[i]} {v1[2][i]} ' for i in range(len(t))])})
    Vd1_3 d1_3 0 PWL ({''.join([f'{t[i]} {v1[3][i]} ' for i in range(len(t))])})
    Vd1_4 d1_4 0 PWL ({''.join([f'{t[i]} {v1[4][i]} ' for i in range(len(t))])})
    Vd1_5 d1_5 0 PWL ({''.join([f'{t[i]} {v1[5][i]} ' for i in range(len(t))])})
    Vd1_6 d1_6 0 PWL ({''.join([f'{t[i]} {v1[6][i]} ' for i in range(len(t))])})
    Vd1_7 d1_7 0 PWL ({''.join([f'{t[i]} {v1[7][i]} ' for i in range(len(t))])})
    
    * m2 is the output node for DAC2, will be connected to MOD2
    
    x2_1 vrefh net-_x2_1-pad2_ d2_0 d2_1 d2_2 d2_3 d2_4 d2_5 d2_6 vdd net-_x2_1-pad11_ s7bit_DAC
    x2_2 net-_x2_1-pad2_ vrefl d2_0 d2_1 d2_2 d2_3 d2_4 d2_5 d2_6 vdd net-_x2_2-pad11_ s7bit_DAC
    x2_3 d2_7 vdd net-_x2_1-pad11_ net-_x2_2-pad11_ m2 switch
    
    Vd2_0 d2_0 0 PWL ({''.join([f'{t[i]} {v2[0][i]} ' for i in range(len(t))])})
    Vd2_1 d2_1 0 PWL ({''.join([f'{t[i]} {v2[1][i]} ' for i in range(len(t))])})
    Vd2_2 d2_2 0 PWL ({''.join([f'{t[i]} {v2[2][i]} ' for i in range(len(t))])})
    Vd2_3 d2_3 0 PWL ({''.join([f'{t[i]} {v2[3][i]} ' for i in range(len(t))])})
    Vd2_4 d2_4 0 PWL ({''.join([f'{t[i]} {v2[4][i]} ' for i in range(len(t))])})
    Vd2_5 d2_5 0 PWL ({''.join([f'{t[i]} {v2[5][i]} ' for i in range(len(t))])})
    Vd2_6 d2_6 0 PWL ({''.join([f'{t[i]} {v2[6][i]} ' for i in range(len(t))])})
    Vd2_7 d2_7 0 PWL ({''.join([f'{t[i]} {v2[7][i]} ' for i in range(len(t))])})
    
    Vrefh vrefh 0 3.3
    Vrefl vrefl 0 0


    ro1 n8 0 {rload}
    ro2 n9 0 {rload}

    .options timeint reltol=1e-06 abstol=1e-06
    .print Tran v(n8) v(n9)

    .tran {t_start} {t_stop} {num_t}


    .photonic # photonic components start below

    modm1 n1 n2 n3 n4 m1 debug coeff1={coeff1} act_l={act_l} alpha = 0.9
    ps1 n4 n5 ps=-0.5pi
    modm2 n3 n5 n6 n7 m2 debug coeff1={coeff1} act_l={act_l} alpha = 0.9
    pd1 n6 n8 level2 r0={r0}  
    pd2 n7 n9 level2 r0={r0} 

    .mode neff={neff}
    .freq {freq} {freq} 1
    .source {mag}@n1 power=0.05w eff=0.2
    .prob n6 n7

    .end
    """
    )
    
    # pd1 n6 n8 level2 r0={r0} std=0.05
    # pd2 n7 n9 level2 r0={r0} std=0.05
    
    # pd1 n6 n8 debug r0={r0} 
    # pd2 n7 n9 debug r0={r0}

    with open(args.file_path, 'w') as f:
        f.write(content)

    a = 2 * pi * freq * neff / FreeLightSpeed * act_l * coeff1
    phi1, phi2 = a * v1_value, a * v2_value
    R = 1e4 # rload # 1e4 # equals rload when pd uses debug model, equals RF when uses level2 model
    target = (r0 * R * mag ** 2) * torch.cos(2 * phi1) * torch.cos(2 * phi2)


    config['max_iter'] = args.max_iter

    for k, v in config.items():
        print(f"Config[{k}] = {v}")

    s1 = time.time()
    circuit = Circuit(args.file_path,
                      spice_exe=os.environ.get('SPIPE_HSPICE', 'hspice') + ' ' if args.sim == 'hspice' else os.environ.get('SPIPE_XYCE', 'Xyce') + ' -quiet -hspice-ext all',
                      power_node=['vrefh', 'vrefl', 'vdd'])
    s2 = time.time()

    prob_e, prob_p, e_in, p_in, power_dict = circuit.simulate()

    s3 = time.time()

    print(f"simulation time: {s3-s2:f}, setup time: {s2-s1:f}")

    font_size = 16
    linewidth = 3

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    for i in range(bit):
        plt.plot(t * 1e9, v1[i], label=f'bit-{i} control', linewidth=linewidth)
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
    if args.save_plot:
        plt.savefig('bit_control_1.png')

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    for i in range(bit):
        plt.plot(t * 1e9, v2[i], label=f'bit-{i} control', linewidth=linewidth)
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
    if args.save_plot:
        plt.savefig('./bit_control_2.png')

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    x = t[disp_ind] * 1e9
    y = target[disp_ind]
    x_sample_hold = np.repeat(np.concatenate(([0], x)), 2)[:-1]
    y_sample_hold = np.repeat(np.concatenate(([y[0]], y)), 2)[1:]
    # plt.scatter(x , y, s = 40, marker='o')
    plt.plot(x_sample_hold, y_sample_hold, '--', c='red', label='target', linewidth=linewidth)
    plt.plot(torch.linspace(t_start, t_stop, num_t) * 1e9, prob_e['v(n8)'] - prob_e['v(n9)'], 'blue', label='simulated', linewidth=linewidth)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    # plt.legend(fontsize=font_size)
    plt.xlabel('Time (ns)', fontsize=font_size)
    plt.ylabel('Voltage (volts)', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # plt.legend(fontsize=font_size)
    if args.save_plot:
        plt.savefig('output.png')

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    plt.plot(torch.linspace(t_start, t_stop, num_t) * 1e9, p_in[:,0], c='r', linewidth=linewidth)
    plt.plot(torch.linspace(t_start, t_stop, num_t) * 1e9, p_in[:, 1], c='blue', linewidth=linewidth)
    wrk1 = v1_value[disp_ind]
    wrk1_sample_hold = np.repeat(np.concatenate(([wrk1[0]], wrk1)), 2)[1:]
    plt.plot(x_sample_hold, wrk1_sample_hold, '--', linewidth=linewidth, c = 'green')
    wrk2 = v2_value[disp_ind]
    wrk2_sample_hold = np.repeat(np.concatenate(([wrk2[0]], wrk2)), 2)[1:]
    plt.plot(x_sample_hold, wrk2_sample_hold, '--', linewidth=linewidth, c= 'orange')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # plt.legend(fontsize=font_size)
    if args.save_plot:
        plt.savefig('./dac.png')

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    for k,v in power_dict.items():
        if v is not None:
            plt.plot(torch.linspace(t_start, t_stop, num_t) * 1e9, v * 1e3, linewidth=linewidth, label=f"power[{k}]")
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
    if args.save_plot:
        plt.savefig('./power.png')

    plt.show()