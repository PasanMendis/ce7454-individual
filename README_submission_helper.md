# Mini-Challenge Submission Helper

This folder contains:
- `solution/run.py` — Inference entrypoint (single image) **exactly** as required:
  ```bash
  pip install -r requirements.txt
  python3 run.py --input /path/to/input-image.jpg --output /path/to/output-mask.png --weights ckpt.pth
  ```
- `solution/requirements.txt` — Minimal dependencies
- You must place your trained weights at `solution/ckpt.pth` (required by the grader)
- `pack_submission.py` — Helper script to produce the final `submission.zip`

## Steps
1. Train from scratch on the provided 1k images; export the best checkpoint as `solution/ckpt.pth`.
2. Generate test masks into a folder named `masks/` (filenames must mirror inputs; extension `.png`).
3. Create the final zip:
   ```bash
   python3 /mnt/data/pack_submission.py --masks_dir /path/to/masks --solution_dir /mnt/data/solution --out_zip /mnt/data/submission.zip
   ```

## Notes
- `run.py` saves a **single-channel** PNG (mode `L`), as required.
- The contained LiteUNet uses depthwise separable convolutions and stays under the parameter cap when trained with `base=64`.
