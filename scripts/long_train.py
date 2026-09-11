#!/usr/bin/env python3
"""Long rogue-wave training with a checkpoint every 1000 rounds.

This is a thin wrapper around `fowd report train`. From the repo root:

    python3 scripts/long_train.py
    python3 scripts/long_train.py --rounds 5000 --save-every 1000
    python3 scripts/long_train.py --input catalogue.nc --rounds 8000
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fowd.cli import cli


def main(argv=None):
    args = ['report', 'train', '--synthetic']
    extra = list(sys.argv[1:] if argv is None else argv)
    # If the caller already chose a source, do not force --synthetic.
    source_flags = {'--input', '--cdip-folder', '--generic-infile', '--synthetic'}
    if any(item in source_flags or item.startswith('--input=') for item in extra):
        args = ['report', 'train']
    sys.argv = [sys.argv[0]] + args + extra
    cli(prog_name='long_train')


if __name__ == '__main__':
    main()
