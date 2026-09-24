"""Check SPIPE's photonic solver against Lumerical INTERCONNECT, on a recirculating mesh.

WHAT THIS DEMONSTRATES
----------------------
A programmable photonic mesh is a lattice of 2x2 couplers joined by waveguides. Light
entering one port does not simply pass through: it splits, recirculates, and interferes
with itself an unbounded number of times. That makes the mesh the sharpest test of the
claim in docs/scope.md that SPIPE "handles loops exactly" -- the direct solve of `A x = b`
sums the whole infinite series of round trips in one factorisation, with no iteration and
no truncation.

Lumerical INTERCONNECT solves the same network by an entirely different route, so it is a
genuinely independent check. They agree to ~5e-7 in field amplitude, which is not a
physics agreement at all -- it is the precision of INTERCONNECT's own text output
(`num2str()`, 7 significant digits). The two solvers agree as closely as it is possible to
measure through that file.

    SPIPE            0.03 s  for a 2x2 mesh (12 couplers), 100 frequency points
    INTERCONNECT     8.1  s  for the same thing

HOW TO RUN IT
-------------
    python examples/mesh_vs_lumerical.py              # stored reference, no licence
    python examples/mesh_vs_lumerical.py --live       # also re-run INTERCONNECT

The default needs nothing but SPIPE: `test/ref/lumerical_mesh.json` holds what
INTERCONNECT computed, together with the coupler settings that produced it.

`--live` needs Lumerical INTERCONNECT on PATH. Check with `which interconnect` -- note it
is lower case, unlike `Xyce`. On a machine with no display, set

    export QT_QPA_PLATFORM=offscreen

or INTERCONNECT exits with "no Qt platform plugin could be initialized".

WHERE THE PIECES LIVE
---------------------
`examples/paper/mesh_lumerical/main1_mesh.py` is the full study from the paper: it sweeps
the mesh from 3x3 up to 30x30, ten random parameter draws each, and plots runtime against
size. This script is the small, readable version of the same comparison, and
`test/tb/tb08_lumerical_mesh.py` is the automated regression form of it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys

import os as _os, sys as _sys
# Make `import spipe` work from a source checkout without installing.
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

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REFERENCE = os.path.join(REPO, 'test', 'ref', 'lumerical_mesh.json')
MESH_STUDY = os.path.join(HERE, 'paper', 'mesh_lumerical', 'main1_mesh.py')

# INTERCONNECT writes its results with num2str(), i.e. 7 significant digits. Nothing below
# this is a statement about physics.
TEXT_PRECISION = 1e-6


def load_mesh_builder():
    """Import `run_spipe` / `run_interconnect` from the paper study.

    They build the mesh netlist -- one `pbum` (programmable 2x2 coupler) per lattice site,
    plus the waveguides between them -- and are reused here rather than re-derived, so the
    two scripts cannot drift apart.
    """
    os.environ.setdefault('MPLBACKEND', 'Agg')
    # Run INTERCONNECT in a scratch copy of the project folder, never in the repository:
    # INTERCONNECT rewrites the project library it loads (interconnect/untitled.ich), which
    # otherwise modifies a tracked file on every live run. main1_mesh reads the folder from
    # $SPIPE_INTERCONNECT_DIR at import time, so it has to be set before the import.
    if not os.environ.get('SPIPE_INTERCONNECT_DIR'):
        import shutil, tempfile
        scratch = tempfile.mkdtemp(prefix='spipe_interconnect_')
        shutil.copytree(os.path.join(os.path.dirname(MESH_STUDY), 'interconnect'),
                        os.path.join(scratch, 'interconnect'))
        os.environ['SPIPE_INTERCONNECT_DIR'] = os.path.join(scratch, 'interconnect')
    spec = importlib.util.spec_from_file_location('_mesh_study', MESH_STUDY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--live', action='store_true',
                    help='also re-run Lumerical INTERCONNECT (needs a licence)')
    args = ap.parse_args()

    if not os.path.exists(REFERENCE):
        print(f"missing reference data: {REFERENCE}", file=sys.stderr)
        return 1
    with open(REFERENCE) as handle:
        doc = json.load(handle)
    module = load_mesh_builder()

    print()
    print('=' * 78)
    print('  SPIPE vs Lumerical INTERCONNECT -- programmable photonic mesh')
    print('=' * 78)
    print(f"  reference : {doc['source']}")
    print(f"  circuit   : {doc['what']}")
    print(f"  mode      : neff={doc['mode']['neff']}, ng={doc['mode']['ng']}, "
          f"alpha={doc['mode']['alpha']}, waveguide={doc['mode']['wg_length_m'] * 1e6:g} um")
    print(f"  band      : {doc['freq_start_THz']}-{doc['freq_end_THz']} THz, "
          f"{doc['freq_num']} points")
    print()
    print("  Note on the frequency grid: torch.linspace and INTERCONNECT do not place")
    print("  their endpoints the same way, so SPIPE is asked to stop at")
    print(f"  {doc['spipe_freq_end_THz']:.6f} THz. Without that the two tools would be")
    print("  compared at slightly different frequencies and the disagreement would look")
    print("  like a solver error.")
    print()

    worst = 0.0
    for case in doc['cases']:
        rows, cols = case['num_row'], case['num_col']
        param = torch.tensor(case['param'], dtype=torch.float64)
        reference = np.asarray(case['real']) + 1j * np.asarray(case['imag'])

        elapsed, result = module.run_spipe(rows, cols, param,
                                           source_in=case['source'],
                                           prob_node=case['probe'])
        # result[node] has shape (n_time, n_freq, 2); index 1 of the last axis is the
        # OUTWARD-travelling wave at that node, which is what a detector there would see.
        field = result[case['probe']][0, :, 1].detach().cpu().numpy()

        power = np.abs(field) ** 2
        d_field = np.abs(field - reference).max()
        d_power = np.abs(power - np.abs(reference) ** 2).max()
        worst = max(worst, d_field)

        print(f"  --- {rows}x{cols} mesh: {case['n_coupler']} couplers, "
              f"{case['source']} -> {case['probe']} ---")
        print(f"      SPIPE runtime                {elapsed:.3f} s")
        print(f"      max |E_spipe - E_lumerical|  {d_field:.3e}"
              f"   ({d_field / TEXT_PRECISION:.2f}x the text precision)")
        print(f"      max difference in |E|^2      {d_power:.3e}")
        print(f"      |E|^2 spans                  [{power.min():.6f}, {power.max():.6f}]"
              f"   (a flat response would make the agreement meaningless)")
        print()

        if args.live:
            exe = os.environ.get('SPIPE_INTERCONNECT', 'interconnect')
            if shutil.which(exe) is None:
                print(f"      --live: '{exe}' is not on PATH; skipping the live run.")
                print(f"              check with:  which interconnect")
            else:
                # INTERCONNECT's directional coupler is parameterised by a power coupling
                # ratio in [0, 1]; SPIPE's pbum takes the angle. sin^2 converts.
                ratio = torch.sin(param) ** 2
                secs, fresh = module.run_interconnect(
                    rows, cols, ratio,
                    source_device='compound_hori_1_1', source_node='port 1',
                    prob_device='compound_hori_1_1', prob_node='port 3')
                drift = np.abs(fresh - reference).max()
                print(f"      live INTERCONNECT runtime    {secs:.3f} s"
                      f"   ({secs / max(elapsed, 1e-9):.0f}x SPIPE)")
                print(f"      live vs stored reference     {drift:.3e}"
                      f"   (0 means the committed data is still exact)")
                print()

    print('-' * 78)
    print(f"  Worst field disagreement across every case: {worst:.3e}")
    print(f"  INTERCONNECT's own text output carries {TEXT_PRECISION:.0e}, so this is")
    print(f"  agreement to the limit of what the comparison can resolve.")
    print('=' * 78)
    print()
    print("  This example asserts nothing -- see test/tb/tb08_lumerical_mesh.py for the")
    print("  same comparison as a pass/fail regression check.")
    print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
