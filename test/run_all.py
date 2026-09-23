#!/usr/bin/env python3
"""Run the whole SPIPE acceptance suite and report a single pass/fail verdict.

This is the regression gate: run it against any new version of SPIPE before
trusting it. Exit status is 0 only if every check in every bench passed.

    python test/run_all.py                 # everything available
    python test/run_all.py --quick         # skip the slow benches
    python test/run_all.py --list          # show the benches and what they guard

Most benches need nothing but PyTorch. `tb05_native_crosstool` compares the
built-in engine against Xyce (free, open source) and HSPICE; it is skipped
automatically when neither is on PATH.

To see whether you have them, use `which` -- note the capital X on Xyce:

    which Xyce
    which hspice

If those print a path, this script finds them with no further setup. If not,
put them on PATH however your environment does it (that differs from site to
site, so nothing is assumed here), or set $SPIPE_CAD_SETUP to a shell snippet
that does it and the suite will run that first.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TB = os.path.join(HERE, 'tb')

# bench -> (what it guards, is it slow)
BENCHES = {
    'tb01_photonic_algebra':  ('photonic invariants: unitarity, reciprocity, energy, resonance', False),
    'tb02_eo_interface':      ('electro-optic interface: the mzm modulator vs textbook physics', False),
    'tb03_oe_interface':      ('opto-electronic interface: shot/thermal noise, bandwidth', False),
    'tb04_native_analytic':   ('built-in engine vs closed-form solutions', False),
    'tb05_native_crosstool':  ('built-in engine vs Xyce and HSPICE (needs one of them)', True),
    'tb06_native_gradients':  ('autograd vs adjoint vs finite difference', True),
    'tb07_fixedpoint':        ('fixed point: convergence, divergence, bistability', False),
    'tb10_bugfix_X':          ('regression guards on previously fixed defects', False),
    'tb11_envelope_P3':       ('optical memory / envelope propagation', False),
    'tb12_end_to_end_grad':   ('d|E|^2/dW through the whole chain', True),
}

SUMMARY = re.compile(r'^(TB-\S+)\s+(.*?)\s+(\d+) pass / (\d+) FAIL / (\d+) skip')


def cad_available():
    if os.environ.get('SPIPE_CAD_SETUP'):
        return True
    return bool(shutil.which('Xyce') or shutil.which('hspice'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='skip the slow benches')
    ap.add_argument('--list', action='store_true', help='list the benches and exit')
    ap.add_argument('--verbose', '-v', action='store_true', help='show each check, not just totals')
    args = ap.parse_args()

    if args.list:
        for name, (what, slow) in BENCHES.items():
            print(f"  {name:<26} {'[slow] ' if slow else '       '}{what}")
        return 0

    env = dict(os.environ, MPLBACKEND='Agg')
    have_cad = cad_available()
    total_pass = total_fail = total_skip = 0
    failed_benches = []
    t0 = time.time()

    print(f"SPIPE acceptance suite  ({sys.executable})")
    print('=' * 78)

    for name, (what, slow) in BENCHES.items():
        path = os.path.join(TB, name + '.py')
        if not os.path.exists(path):
            print(f"  {name:<26} MISSING"); failed_benches.append(name); continue
        if args.quick and slow:
            print(f"  {name:<26} skipped (--quick)"); continue
        if name == 'tb05_native_crosstool' and not have_cad:
            print(f"  {name:<26} skipped  -- needs Xyce or HSPICE; neither is on PATH")
            print(f"  {'':<26}            check with:  which Xyce   /   which hspice")
            continue

        r = subprocess.run([sys.executable, path], capture_output=True, text=True, env=env)
        m = None
        for line in r.stdout.splitlines():
            mm = SUMMARY.match(line.strip())
            if mm:
                m = mm
        if m is None:
            print(f"  {name:<26} ERROR (no summary line)")
            print('\n'.join(r.stdout.splitlines()[-6:]))
            print('\n'.join(r.stderr.splitlines()[-6:]))
            failed_benches.append(name)
            continue

        npass, nfail, nskip = int(m.group(3)), int(m.group(4)), int(m.group(5))
        total_pass += npass; total_fail += nfail; total_skip += nskip
        flag = 'ok  ' if nfail == 0 else 'FAIL'
        print(f"  {name:<26} {flag}  {npass:>3} pass  {nfail:>2} fail  {nskip:>2} skip   {what}")
        if nfail:
            failed_benches.append(name)
            for line in r.stdout.splitlines():
                if '[FAIL]' in line:
                    print(f"        {line.strip()}")
        elif args.verbose:
            for line in r.stdout.splitlines():
                if '[PASS]' in line:
                    print(f"        {line.strip()}")

    print('=' * 78)
    verdict = 'PASS' if not failed_benches else 'FAIL'
    print(f"  {verdict}: {total_pass} passing, {total_fail} failing, {total_skip} skipped "
          f"({time.time() - t0:.0f}s)")
    if failed_benches:
        print(f"  benches with failures: {', '.join(failed_benches)}")
    return 1 if failed_benches else 0


if __name__ == '__main__':
    raise SystemExit(main())
