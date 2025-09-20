# solution/run.py
import argparse, os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from models.liteunet import LiteUNet

def load_model(weights_path, num_classes=None, base=None, device="cpu"):
    ckpt = torch.load(weights_path, map_location=device)
    # pull config from ckpt if present
    cfg = ckpt.get("config", {})
    ck_num_classes = cfg.get("num_classes", None)
    ck_base        = cfg.get("base", None)
    arch_sig       = ckpt.get("arch", None)  # optional

    # prefer cmd args if provided, else use ckpt values
    if num_classes is None: num_classes = ck_num_classes
    if base is None:        base        = ck_base

    assert num_classes is not None and base is not None, \
        f"num_classes/base not provided and not found in checkpoint config. Got: {cfg}."

    if arch_sig is not None:
        # update this string if you changed the model structure
        expected = "liteunet_v1_dw_bn_do"  # harmless tag; omit if you didn't save this
        if arch_sig != expected:
            print(f"[WARN] Checkpoint arch {arch_sig} != expected {expected}. Proceeding anyway.")

    model = LiteUNet(in_ch=3, num_classes=num_classes, base=base, dropout_p=0.0).to(device)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] Loaded with strict=False. Missing={len(missing)} Unexpected={len(unexpected)}")

    model.eval()
    return model, num_classes, base

def preprocess(img_path, size=512):
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = TF.to_tensor(img)
    x = TF.normalize(x, [0.5,0.5,0.5], [0.5,0.5,0.5])
    return x

def save_mask(mask_np, out_path):
    # single-channel, label IDs
    Image.fromarray(mask_np.astype(np.uint8), mode="L").save(out_path)

@torch.no_grad()
def infer_one(model, img_path, out_path, device="cpu", size=512, tta=False, num_classes=19):
    x = preprocess(img_path, size=size).unsqueeze(0).to(device)

    logits = model(x)
    if tta:
        x_flip = torch.flip(x, dims=[3])
        logits_flip = model(x_flip)
        logits = (logits + torch.flip(logits_flip, dims=[3])) / 2.0

    pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    save_mask(pred, out_path)

    # diagnostics
    probs = torch.softmax(logits, dim=1)
    maxprob = probs.max(dim=1)[0].mean().item()
    u, c = np.unique(pred, return_counts=True)
    print(f"[DEBUG] saved {out_path} | unique labels: {u.tolist()[:20]} | "
          f"freq(top5): {sorted([(int(uu), int(cc)) for uu,cc in zip(u,c)], key=lambda z:-z[1])[:5]} "
          f"| mean max prob: {maxprob:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, help="single input image")
    ap.add_argument("--output", required=False, help="single output mask")
    ap.add_argument("--images_dir", help="batch mode: dir of images")
    ap.add_argument("--out_dir", help="batch mode: dir to save masks")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--num_classes", type=int, default=None, help="if omitted, use checkpoint config")
    ap.add_argument("--base", type=int, default=None, help="if omitted, use checkpoint config")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tta", action="store_true", help="flip TTA")
    args = ap.parse_args()

    model, nc, base = load_model(args.weights, args.num_classes, args.base, args.device)

    if args.input and args.output:
        infer_one(model, args.input, args.output, device=args.device, size=args.size, tta=args.tta, num_classes=nc)
    else:
        assert args.images_dir and args.out_dir, "Provide --input/--output OR --images_dir/--out_dir"
        os.makedirs(args.out_dir, exist_ok=True)
        # collect jpg/png
        exts = (".jpg",".jpeg",".png",".JPG",".JPEG",".PNG")
        files = [f for f in sorted(os.listdir(args.images_dir)) if f.endswith(exts)]
        for fname in files:
            ip = os.path.join(args.images_dir, fname)
            op = os.path.join(args.out_dir, os.path.splitext(fname)[0] + ".png")
            infer_one(model, ip, op, device=args.device, size=args.size, tta=args.tta, num_classes=nc)

if __name__ == "__main__":
    main()