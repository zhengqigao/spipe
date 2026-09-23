"""TB-08 : the photonic mesh against Lumerical INTERCONNECT.

This is the only bench whose ground truth is another *simulator* rather than closed-form
physics, and it earns its place: a programmable mesh of 2x2 couplers is exactly the case
where SPIPE's direct solve has to reproduce an infinite series of recirculating round
trips, and INTERCONNECT solves the same network by a completely different route.

It runs WITHOUT a Lumerical licence. `test/ref/lumerical_mesh.json` holds the field
transmission INTERCONNECT computed, together with the coupler settings that produced it,
so the comparison is a file read. If INTERCONNECT *is* on PATH the bench additionally
re-runs it live and checks the stored reference has not gone stale.

The agreement floor is ~1e-6: INTERCONNECT writes its results through `num2str()`, which
is 7 significant digits. Tolerances here are set from that, not from wishful thinking.
"""
import sys, os, json, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import TB, fresh_spipe, main, REPO
import numpy as np
import torch
import shutil

REF = os.path.join(REPO, 'test', 'ref', 'lumerical_mesh.json')
MESH_EXAMPLE = os.path.join(REPO, 'examples', 'paper', 'mesh_lumerical', 'main1_mesh.py')

# num2str() gives 7 significant digits, so nothing below this is meaningful.
TEXT_PRECISION = 1e-6
FIELD_TOL = 5e-6          # a few times the text precision
MSE_TOL = 1e-11           # (5e-6)^2 with room to spare


def _load_mesh_module():
    """Import the example's netlist builder.

    The test deliberately reuses `run_spipe` rather than re-deriving the mesh netlist:
    the thing under test is SPIPE's answer, the independent reference is INTERCONNECT's,
    and a second hand-written builder here would only add a third thing to be wrong.
    """
    spec = importlib.util.spec_from_file_location('_mesh_example', MESH_EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    spec.loader.exec_module(module)
    return module


def build():
    tb = TB('TB-08  photonic mesh vs Lumerical INTERCONNECT')
    fresh_spipe(complex_dtype=torch.complex128)

    if not os.path.exists(REF):
        tb.ok('LUM.reference_present', False,
              f'{os.path.relpath(REF, REPO)} is missing -- regenerate it with INTERCONNECT')
        return tb
    with open(REF) as handle:
        doc = json.load(handle)
    tb.ok('LUM.reference_present', True,
          f"{len(doc['cases'])} case(s) from {doc['source']}")

    ok_mod, module = tb.no_raise('LUM.example_importable', _load_mesh_module)
    if not ok_mod or module is None:
        return tb

    # The two tools must be asked about the SAME frequencies. torch.linspace and
    # INTERCONNECT's grid differ at the endpoint, which the example corrects for; if that
    # correction ever drifts the comparison silently becomes apples-to-oranges.
    expect_end = (doc['freq_start_THz']
                  + (doc['freq_end_THz'] - doc['freq_start_THz'])
                  / doc['freq_num'] * (doc['freq_num'] - 1))
    tb.close('LUM.frequency_grid_aligned', module.spipe_freq_end, expect_end, 1e-12,
             detail=f"SPIPE solves to {module.spipe_freq_end} THz so that its {doc['freq_num']} "
                    f"torch.linspace points coincide with INTERCONNECT's")

    for case in doc['cases']:
        nr, nc = case['num_row'], case['num_col']
        tag = f"{nr}x{nc}"
        param = torch.tensor(case['param'], dtype=torch.float64)
        reference = np.asarray(case['real']) + 1j * np.asarray(case['imag'])

        ok_run, got = tb.no_raise(
            f'LUM.{tag}.spipe_runs',
            lambda: module.run_spipe(nr, nc, param,
                                     source_in=case['source'], prob_node=case['probe']))
        if not ok_run or got is None:
            continue
        _, result = got
        field = result[case['probe']][0, :, 1].detach().cpu().numpy()

        tb.ok(f'LUM.{tag}.shape', field.shape == reference.shape,
              f"SPIPE {field.shape} vs INTERCONNECT {reference.shape} "
              f"({case['n_coupler']} couplers)")
        if field.shape != reference.shape:
            continue

        d_re = np.abs(field.real - reference.real).max()
        d_im = np.abs(field.imag - reference.imag).max()
        mse = ((field.real - reference.real) ** 2).mean() + \
              ((field.imag - reference.imag) ** 2).mean()
        d_pow = np.abs(np.abs(field) ** 2 - np.abs(reference) ** 2).max()

        tb.lt(f'LUM.{tag}.real_part', d_re, FIELD_TOL,
              f"max |Re(SPIPE) - Re(INTERCONNECT)| over {len(field)} frequency points")
        tb.lt(f'LUM.{tag}.imag_part', d_im, FIELD_TOL,
              f"max |Im(SPIPE) - Im(INTERCONNECT)|")
        tb.lt(f'LUM.{tag}.mse', mse, MSE_TOL, 'mean squared error of the complex field')
        tb.lt(f'LUM.{tag}.transmitted_power', d_pow, FIELD_TOL,
              'max difference in |E|^2, the quantity a detector actually sees')

        # The comparison is only meaningful if the response actually varies: agreeing on a
        # flat line would prove nothing about the recirculation.
        power = np.abs(field) ** 2
        spread = float(power.max() - power.min())
        tb.ok(f'LUM.{tag}.response_nontrivial', spread > 1e-3,
              f"|E|^2 varies by {spread:.4f} across the band, so the agreement above is "
              f"not vacuous")

        # A mesh of lossy waveguides cannot transmit more than was launched.
        tb.ok(f'LUM.{tag}.passive', float((np.abs(field) ** 2).max()) <= 1.0 + 1e-9,
              f"max |E|^2 = {float((np.abs(field) ** 2).max()):.6f} must not exceed 1")

    # ---- optional: is the stored reference still what INTERCONNECT says? -------------
    exe = os.environ.get('SPIPE_INTERCONNECT', 'interconnect')
    if shutil.which(exe) is None:
        tb.skip('LUM.reference_still_live',
                f"'{exe}' not on PATH, so the stored reference cannot be re-checked "
                f"(this is the normal case and not a failure)")
    else:
        case = doc['cases'][0]
        nr, nc = case['num_row'], case['num_col']
        param_inte = torch.sin(torch.tensor(case['param'], dtype=torch.float64)) ** 2
        ok_live, live = tb.no_raise(
            'LUM.interconnect_runs',
            lambda: module.run_interconnect(
                nr, nc, param_inte,
                source_device=f"compound_hori_1_1", source_node='port 1',
                prob_device=f"compound_hori_1_1", prob_node='port 3'))
        if ok_live and live is not None:
            _, fresh = live
            stored = np.asarray(case['real']) + 1j * np.asarray(case['imag'])
            drift = np.abs(fresh - stored).max()
            tb.lt('LUM.reference_still_live', drift, TEXT_PRECISION * 10,
                  f"a live INTERCONNECT run must still reproduce the committed reference "
                  f"for the {nr}x{nc} case")

    return tb


if __name__ == '__main__':
    main(build)
