"""Assemble the binary OTFS-vs-OFDM dataset from the MATLAB transmitter output.

The MATLAB scripts in ``transmitter/`` dump every sample into modulation-specific
folders such as ``otfs_rice/a1_ofdm_bpsk/`` and ``otfs_rice/b1_otfs_qpsk/``. For
the *binary* air-interface task we only care about the scheme (OFDM vs OTFS), so
this script collects every ``.mat`` into an ImageFolder-style layout::

    data/ofdm/*.mat   -> label 0
    data/otfs/*.mat   -> label 1

which ``utils.read_split_data`` consumes directly.

Example::

    # after running transmitter/otfs_syn.m and transmitter/ofdm_syn.m
    python prepare_data.py --raw ./otfs_rice --out ./data
    # or save disk space with symlinks:
    python prepare_data.py --raw ./otfs_rice --out ./data --symlink
"""

import argparse
import os
import shutil
from collections import Counter


def detect_scheme(path: str) -> str:
    """Return 'ofdm' or 'otfs' based on the deepest matching folder name."""
    parts = path.replace("\\", "/").split("/")
    for part in reversed(parts):
        low = part.lower()
        if "ofdm" in low:
            return "ofdm"
        if "otfs" in low:
            return "otfs"
    raise ValueError(f"cannot infer OFDM/OTFS scheme from path: {path}")


def main(args):
    os.makedirs(args.out, exist_ok=True)
    count = Counter()
    collisions = Counter()

    for dirpath, _, files in os.walk(args.raw):
        for fn in files:
            if not fn.endswith(".mat"):
                continue
            src = os.path.join(dirpath, fn)
            scheme = detect_scheme(src)
            dst_dir = os.path.join(args.out, scheme)
            os.makedirs(dst_dir, exist_ok=True)

            # Avoid name collisions across modulation folders.
            base = fn
            dst = os.path.join(dst_dir, base)
            while os.path.lexists(dst):
                collisions[base] += 1
                stem, ext = os.path.splitext(base)
                dst = os.path.join(dst_dir, f"{stem}_{collisions[base]}{ext}")

            if args.symlink:
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)
            count[scheme] += 1

    total = sum(count.values())
    print(f"organized {total} samples into {args.out}/")
    for k, v in sorted(count.items()):
        print(f"  {k}: {v}")
    if not total:
        print("[warn] no .mat files found -- did you run the MATLAB transmitters "
              f"under '{args.raw}'?")
    print("\nnext: python train.py --data-path", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Organize MATLAB output into class folders")
    p.add_argument("--raw", type=str, default="./otfs_rice",
                   help="root produced by transmitter/*.m (default ./otfs_rice)")
    p.add_argument("--out", type=str, default="./data",
                   help="ImageFolder-style output root (default ./data)")
    p.add_argument("--symlink", action="store_true",
                   help="symlink instead of copy to save disk space")
    main(p.parse_args())
