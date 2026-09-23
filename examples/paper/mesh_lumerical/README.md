# The Lumerical INTERCONNECT comparison

This is the paper's cross-check of SPIPE's photonic solver against a commercial tool. A
programmable mesh is the hardest case for the comparison — light recirculates through the
lattice indefinitely, so the two solvers have to agree on an infinite series of round trips.

**Start with [`examples/mesh_vs_lumerical.py`](../../mesh_vs_lumerical.py).** It is the
small, annotated version of this comparison and it needs no Lumerical licence, because the
INTERCONNECT results are stored in `test/ref/lumerical_mesh.json`. The automated form is
`test/tb/tb08_lumerical_mesh.py`.

## What is in here

| file | what it is |
|---|---|
| `main1_mesh.py` | the full study: SPIPE and INTERCONNECT on meshes from 3×3 to 30×30, ten random parameter draws each, plotting accuracy and runtime against size. Drives INTERCONNECT live. |
| `main2_dff.py` | gradient check on an active 2×2 mesh — SPIPE's analytic derivative against a finite-difference reference |
| `main3_diff.py` | differentiating a mesh response with respect to its phase shifters |
| `interconnect/` | the INTERCONNECT project `main1_mesh.py` drives. `circuit.lsf` is a standalone version of the generated script; `untitled.ich` is the compound-element library it instantiates. |
| `lumerical_files/` | the INTERCONNECT project behind the paper's figure, kept for reproduction: `circuit2.icp` is the schematic, `circuit2.lsf` the script that sweeps it, and `circuit2_real.txt` / `circuit2_imag.txt` the response it produced. No Python here reads them — they are archival, for opening in the INTERCONNECT GUI. |

Running `main1_mesh.py` writes `photonic.lsf`, `param.txt` and `out_*.txt` into
`interconnect/`. Those are regenerated every run and are gitignored.

## Running the full study

```bash
python examples/paper/mesh_lumerical/main1_mesh.py --num_exp 1
```

It needs Lumerical INTERCONNECT. Check with `which interconnect` — the executable is lower
case, unlike `Xyce`. On a machine with no display:

```bash
export QT_QPA_PLATFORM=offscreen
```

or INTERCONNECT exits with "no Qt platform plugin could be initialized". Set
`$SPIPE_INTERCONNECT` if it is not on `PATH`, and `$SPIPE_INTERCONNECT_DIR` to run
somewhere other than `interconnect/`.

Be aware that the default sweep runs INTERCONNECT a hundred times on meshes up to 30×30,
which takes hours. `--num_exp 1` is the quick version.

## A note on the frequency grid

`torch.linspace` and INTERCONNECT do not place their endpoints the same way. `main1_mesh.py`
therefore asks SPIPE to stop at `freq_start + (freq_end - freq_start)/N * (N-1)`, so that
its N points land exactly on INTERCONNECT's. Without that correction the two tools would be
compared at slightly different frequencies and the disagreement would look like a solver
error. `tb08_lumerical_mesh.py` checks this alignment explicitly.
