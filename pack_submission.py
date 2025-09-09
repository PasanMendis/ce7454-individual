#!/usr/bin/env python3
"""
Pack a submission zip with the required layout:
whatever.zip
├─ masks/
└─ solution/
   ├─ run.py
   ├─ requirements.txt
   └─ ckpt.pth  (you must place this file before packing)
"""
import argparse, os, zipfile, sys

def add_dir_to_zip(zf, folder, arc_prefix):
    for root, _, files in os.walk(folder):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, folder)
            zf.write(full, os.path.join(arc_prefix, rel))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_dir", required=True, help="Folder containing single-channel PNG masks")
    ap.add_argument("--solution_dir", required=True, help="Folder with run.py, requirements.txt, ckpt.pth")
    ap.add_argument("--out_zip", default="submission.zip", help="Output zip path")
    args = ap.parse_args()

    # Basic checks
    if not os.path.isdir(args.masks_dir):
        print("ERROR: masks_dir not found", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(os.path.join(args.solution_dir, "run.py")):
        print("ERROR: run.py missing in solution_dir", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(os.path.join(args.solution_dir, "requirements.txt")):
        print("ERROR: requirements.txt missing in solution_dir", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(os.path.join(args.solution_dir, "ckpt.pth")):
        print("ERROR: ckpt.pth missing in solution_dir", file=sys.stderr); sys.exit(1)

    with zipfile.ZipFile(args.out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        add_dir_to_zip(zf, args.masks_dir, "masks")
        add_dir_to_zip(zf, args.solution_dir, "solution")

    print(f"Packed: {args.out_zip}")

if __name__ == "__main__":
    main()
