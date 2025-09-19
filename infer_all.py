import os, argparse
from glob import glob
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

from models.liteunet import LiteUNet

def load_model(weights, num_classes=19, base=64, device="cpu"):
    m = LiteUNet(in_ch=3, num_classes=num_classes, base=base).to(device)
    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    m.load_state_dict(state, strict=True)
    m.eval()
    return m

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", default="masks")
    ap.add_argument("--weights", default="solution/ckpt.pth")
    ap.add_argument("--num_classes", type=int, default=19)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.weights, args.num_classes, args.base, args.device)

    paths = sorted(glob(os.path.join(args.images_dir, "*.jpg")) + glob(os.path.join(args.images_dir, "*.png")))
    for p in paths:
        img = Image.open(p).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
        x = TF.to_tensor(img)
        x = TF.normalize(x, [0.5,0.5,0.5], [0.5,0.5,0.5])
        x = x.unsqueeze(0).to(args.device)
        logits = model(x)
        x_flip = torch.flip(x, dims=[3])
        logits_flip = model(x_flip)
        logits_flip = torch.flip(logits_flip, dims=[3])
        logits = (logits + logits_flip) / 2.0
        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        name = os.path.splitext(os.path.basename(p))[0] + ".png"
        Image.fromarray(pred, mode="L").save(os.path.join(args.out_dir, name))

if __name__ == "__main__":
    main()