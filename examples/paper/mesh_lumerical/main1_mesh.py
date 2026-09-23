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
from math import pi
import h5py
import matplotlib.pyplot as plt
from spipe.photonic import FreeLightSpeed

ng, neff, wl = 2.35, 2.35, FreeLightSpeed / 193.1e12

# the linspace in Interconnect and pytorch is different
# the following line is mainly for constructing exactly the same simulation frequency grid in SPIPE (which use pytorch linsapce)
# and Interconnect.
freq_start, freq_end, freq_num = 192.8, 193.4, 100
spipe_freq_end = freq_start + (freq_end - freq_start) / freq_num * (freq_num - 1)

theta, phi = 0.5, 0.0
alpha_value = 1.0
uniform_wg_l = 250e-6

linewidth = 2
font_size = 18

file_path = os.environ.get('SPIPE_INTERCONNECT_DIR',
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'interconnect'))

#: The INTERCONNECT executable. It is lower case (`interconnect`), unlike Xyce. Check with
#: `which interconnect`; override with $SPIPE_INTERCONNECT if it is not on PATH.
INTERCONNECT_EXE = os.environ.get('SPIPE_INTERCONNECT', 'interconnect')

def run_spipe(num_row, num_col, param, source_in, prob_node):
    total = (num_row + 1) * num_col + (num_col + 1) * num_row

    if param.shape[0] != total or param.shape[1] != 2:
        raise ValueError('Param shape is not right')
    count = 0
    final = ''
    for i in range(1, num_row + 1):
        for j in range(1, num_col + 2):
            device_name = f'pbum_hori_{i}_{j} '
            ports = ''.join([f"n_hori_{i}_{j}_p{n} " for n in range(1, 5)])
            ps = f"theta={theta}pi phi={phi}pi "
            l = f"l={uniform_wg_l} "
            cp = f"cp_left={param[count, 0]} cp_right={param[count, 1]} "
            alpha = f"alpha={alpha_value}"

            sentence = device_name + ports + ps + l + cp + alpha + '\n'

            final += sentence
            count += 1

    for i in range(1, num_col + 2):
        for j in range(1, num_row + 1):
            device_name = f'pbum_verti_{i}_{j} '
            ports = ''.join([f"n_verti_{i}_{j}_p{n} " for n in range(1, 5)])
            ps = f"theta={theta}pi phi={phi}pi "
            l = f"l={uniform_wg_l} "
            cp = f"cp_left={param[count, 0]} cp_right={param[count, 1]} "
            alpha = f"alpha={alpha_value}"

            sentence = device_name + ports + ps + l + cp + alpha + '\n'

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

    final += f".mode neff={neff} ng={ng} wl={wl}\n"
    final += f".freq {freq_start}thz {spipe_freq_end}thz {freq_num}\n"
    final += f".source 1.0@{source_in}\n"
    final += f".prob {prob_node}\n"

    start_time = time.time()
    photonic = Photonic(final.split('\n'))

    res, middle_res, power_dict = photonic.simulate()
    run_time = time.time() - start_time

    return run_time, middle_res


def run_interconnect(num_row, num_col, param, source_device, source_node, prob_device, prob_node):
    total = (num_row + 1) * num_col + (num_col + 1) * num_row

    if param.shape[0] != total or param.shape[1] != 2:
        raise ValueError('Param shape is not right')


    text_lsf = f"""
clear; switchtodesign; deleteall;
file_prefix = '{file_path}{os.sep}';   # trailing separator: the lsf concatenates, it does not join

N = {num_row};
M = {num_col};

total = (N+1)*M+N*(M+1);

delta_x = 300;
delta_y = 300;

bias_x = -150;
bias_y = 150;

ng = {ng};
neff={neff};
wg_l = {uniform_wg_l:e};
loss = 1782.11; # corresponds to alpha=0.95 in our code

ps_init = [{theta}*pi, {phi}*pi];

bottom_cp_list = {{'C_1', 'C_2'}};
bottom_ps_list = {{'PHS_1', 'PHS_2'}};
bottom_wg_list = {{'WGD_1', 'WGD_2'}};


param = readdata(file_prefix + 'param.txt');
given_size = size(param);

if ((given_size(1) != total) | (given_size(2) != length(bottom_cp_list))){{
    print("error parameter shape");
    param = zeros(total, length(bottom_cp_list));    
}}


count = 1;

for (i=1:N){{
    for (j=1:M+1){{
        addelement("TBU");
        device_name = "compound_hori_" + num2str(i) + "_" + num2str(j);
        set("name", device_name);         
        set("x position", i*delta_x);
        set("y position", j*delta_y);
        
        for (m=1:length(bottom_wg_list)){{
             wrk_name = device_name + "::" + bottom_wg_list{{m}};
             select(wrk_name);
             set("length", wg_l);
             set("effective index 1", neff);
             set("group index 1", ng);
             {'#' if alpha_value == 1 else ''} set("loss 1", loss);
            }}
         for (m=1:length(bottom_ps_list)){{
             wrk_name = device_name + "::" + bottom_ps_list{{m}};
             select(wrk_name);
             set("phase shift", ps_init(m));
            }}
          
          for (m=1:length(bottom_cp_list)){{
             wrk_name = device_name + "::" + bottom_cp_list{{m}};
             select(wrk_name);
             set("coupling coefficient 1", param(count,m));
            }}
            count = count + 1;
        }}    
    }}
    
 for (i=1:N+1){{
    for (j=1:M){{
        addelement("TBU");
        device_name = "compound_verti_" + num2str(i) + "_" + num2str(j);
        set("name", device_name);         
        set("x position", bias_x + i*delta_x);
        set("y position", bias_y + j*delta_y);
        set("rotated", 1); # some magical number to put it in vertical
        
        for (m=1:length(bottom_wg_list)){{
             wrk_name = device_name + "::" + bottom_wg_list{{m}};
             select(wrk_name);
             set("length", wg_l);
             set("effective index 1", neff);
             set("group index 1", ng);
             {'#' if alpha_value == 1 else ''} set("loss 1", loss);
            }}
         for (m=1:length(bottom_ps_list)){{
             wrk_name = device_name + "::" + bottom_ps_list{{m}};
             select(wrk_name);
             set("phase shift", ps_init(m));
            }} 
         
         for (m=1:length(bottom_cp_list)){{
             wrk_name = device_name + "::" + bottom_cp_list{{m}};
             select(wrk_name);
             set("coupling coefficient 1", param(count,m));
            }}
            count = count + 1;   
        }}    
    }}
    
 
 # connect TBUs.
 
 for (i=1:N){{
    for (j=1:M){{
        connect("compound_verti_" + num2str(i) + "_" + num2str(j), "port 1", "compound_hori_" + num2str(i) + "_" + num2str(j), "port 2");
        connect("compound_hori_" + num2str(i) + "_" + num2str(j), "port 4", "compound_verti_" + num2str(i+1) + "_" + num2str(j), "port 2");        
        connect("compound_verti_" + num2str(i+1) + "_" + num2str(j), "port 4", "compound_hori_" + num2str(i) + "_" + num2str(j+1), "port 3");        
        connect("compound_hori_" + num2str(i) + "_" + num2str(j+1), "port 1", "compound_verti_" + num2str(i) + "_" + num2str(j), "port 3");        
        }}          
     }}

addelement("Optical Network Analyzer");
connect("ONA_1", "input 1", "{prob_device}", "{prob_node}");
connect("ONA_1", "output", "{source_device}", "{source_node}");
set("input parameter", "start and stop");
set("start frequency", {freq_start*1e12:e});
set("stop frequency", {freq_end*1e12:e});
set("number of points", {freq_num});

run;

wrk=getresult("ONA_1","input 1/mode 1/transmission");
t=getattribute(wrk,"TE transmission");
real_t = real(t);
image_t = imag(t);
write(file_prefix + "out_real.txt", num2str(real_t), "overwrite");
write(file_prefix + "out_imag.txt", num2str(image_t), "overwrite");
    """

    with open(os.path.join(file_path, 'param.txt'), 'w') as f:
        for i in range(param.shape[0]):
            f.write(f"{param[i, 0]} {param[i, 1]}\n")

    with open(os.path.join(file_path, 'photonic.lsf'), 'w') as f:
        f.write(text_lsf)

    out_real = os.path.join(file_path, 'out_real.txt')
    out_imag = os.path.join(file_path, 'out_imag.txt')

    # Delete last run's answers first. INTERCONNECT can fail (bad licence, a script error
    # it only prints, no display) and still exit 0, and then these files would be read as
    # though they were this run's result -- a comparison against stale data that looks
    # perfectly healthy. Removing them makes that failure loud.
    for stale in (out_real, out_imag):
        if os.path.exists(stale):
            os.remove(stale)

    start = time.time()
    cmd = f"{INTERCONNECT_EXE} {os.path.join(file_path, 'photonic.lsf')} -run -exit"
    status = os.system(cmd)
    run_time = time.time() - start

    missing = [p for p in (out_real, out_imag) if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"INTERCONNECT produced no output (exit status {status}).\n"
            f"  command : {cmd}\n"
            f"  expected: {', '.join(missing)}\n"
            f"Check that `which interconnect` finds it (override with $SPIPE_INTERCONNECT), "
            f"that a licence is available, and -- on a headless machine -- that "
            f"QT_QPA_PLATFORM=offscreen is set; INTERCONNECT otherwise dies with 'no Qt "
            f"platform plugin could be initialized'.")

    real_part = np.loadtxt(out_real)
    imag_part = np.loadtxt(out_imag)
    result = 1.j * imag_part + real_part
    return run_time, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_exp', type=int, default=10)
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--save_plot', action='store_true', default = False)
    parser.add_argument('--gpu', type = int, default = -1)
    args = parser.parse_args()

    config['device'] = torch.device(f"cuda:{args.gpu}") if args.gpu >=0 else torch.device("cpu")

    num_row_list = list(range(3,33,3)) # [2,4,8,16,20]  # [2,4,8]
    num_col_list = list(range(3,33,3)) # [2,4,8]

    run_time = np.empty((len(num_row_list), args.num_exp, 2))
    err = np.empty((len(num_row_list), args.num_exp))

    for j in range(len(num_row_list)):
        num_row, num_col = num_row_list[j], num_col_list[j]
        total = (num_row + 1) * num_col + (num_col + 1) * num_row
        param = torch.rand((total, 2)) * 0.5 * pi
        param_inte = torch.sin(param) ** 2 # the directional coupler model in Interconnect accepts one coupling ratio in [0,1]

        in_key, in_i, in_j, in_p = 'hori', 1, 1, 1
        out_key, out_i, out_j, out_p = 'hori', 1, 1, 3

        for i in range(args.num_exp):
            time_spipe, res_spipe = run_spipe(num_row, num_col, param, source_in=f'n_{in_key}_{in_i}_{in_j}_p{in_p}',
                                              prob_node=f'n_{out_key}_{out_i}_{out_j}_p{out_p}')

            res_spipe = res_spipe[f'n_{out_key}_{out_i}_{out_j}_p{out_p}'][0,:,1]


            time_inte, res_inte = run_interconnect(num_row, num_col, param_inte,
                                                   source_device=f'compound_{in_key}_{in_i}_{in_j}',
                                                   source_node=f'port {in_p}',
                                                   prob_device=f'compound_{out_key}_{out_i}_{out_j}',
                                                   prob_node=f'port {out_p}')

            err[j,i] = ((np.real(res_inte) - torch.real(res_spipe).cpu().detach().numpy()) ** 2).mean() + \
                       ((np.imag(res_inte) - torch.imag(res_spipe).cpu().detach().numpy()) ** 2).mean()

            run_time[j, i, 0] = time_spipe
            run_time[j, i, 1] = time_inte

            if i == args.num_exp - 1 and j == len(num_row_list) - 1 and (args.plot or args.save_plot):
                fig = plt.figure(figsize=[8, 6])
                plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
                freq = torch.linspace(freq_start, freq_end, freq_num)
                plt.plot(freq, torch.real(res_spipe) ** 2 + torch.imag(res_spipe) ** 2, 'blue', linestyle='-', label= 'SPIPE', linewidth = linewidth)
                plt.plot(freq, np.real(res_inte) ** 2 + np.imag(res_inte) ** 2, 'r', linestyle=':',  label='Interconnect', linewidth=linewidth)
                plt.xticks(fontsize=font_size)
                plt.yticks(fontsize=font_size)
                plt.ylabel(r'Response', fontdict={'family': 'Times New Roman', 'size': font_size, })
                plt.xlabel(r'freq',
                           fontdict={'family': 'Times New Roman', 'size': font_size, })
                plt.legend(prop={'family': 'Times New Roman', 'size': font_size, 'weight': 'normal'})
                ax = plt.gca()
                ax.spines['bottom'].set_linewidth(linewidth)
                ax.spines['left'].set_linewidth(linewidth)
                ax.spines['right'].set_linewidth(linewidth)
                ax.spines['top'].set_linewidth(linewidth)
                if args.save_plot:
                    plt.savefig(os.path.join(file_path, 'compare.png'))


        print(f"num_row = {num_row}, num_col = {num_col}, "
              f"run_time_spipe(avg)={np.mean(run_time[j, :, 0]):.4f}, "
              f"run_time_inte(avg)={np.mean(run_time[j, :, 1]):.4f}, "
              f"error(avg)={np.mean(err[j, :]):e}")

    fig = plt.figure(figsize=[8, 6])
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    settings = np.arange(1, run_time.shape[0] + 1)

    mean_spi = np.mean(run_time[:, :, 0], axis=1)
    std_spi = np.std(run_time[:, :, 0], axis=1)
    mean_inter = np.mean(run_time[:, :, 1], axis=1)
    std_inter = np.std(run_time[:, :, 1], axis=1)
    plt.plot(settings, mean_spi, color='blue', marker='o', markersize=10, label="SPIPE", linewidth=2)
    plt.plot(settings, mean_inter, color='r', marker='o', markersize=10, label="Interconnect", linewidth=2)
    plt.fill_between(settings, mean_spi - 3 * std_spi, mean_spi + 3 * std_spi,
                     color='blue', alpha=0.2)  # label='_nolegend_' to exclude from legend

    plt.fill_between(settings, mean_inter - 3 * std_inter, mean_inter + 3 * std_inter,
                     color='r', alpha=0.2)
    plt.xticks(list(range(1,1 + run_time.shape[0])), fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel(r'Run time (sec)', fontdict={'family': 'Times New Roman', 'size': font_size, })
    plt.xlabel(r'Setting',
               fontdict={'family': 'Times New Roman', 'size': font_size, })
    plt.legend(prop={'family': 'Times New Roman', 'size': font_size})
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    if args.save_plot:
        plt.savefig(os.path.join(file_path, 'run_time.png'))

    if args.plot:
        plt.show()

    print("err", np.mean(err,axis=1))
    print("run time saving", mean_inter/mean_spi)