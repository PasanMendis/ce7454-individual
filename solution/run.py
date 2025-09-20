# Usage:
# python3 run.py --input /path/to/input.jpg --output /path/to/output.png --weights ckpt.pth
import argparse, torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from models.liteunet import LiteUNet

def load_model(weights_path, device, num_classes=None, base=None):
    ck = torch.load(weights_path, map_location=device)
    cfg = ck.get("config", {})
    nc = num_classes if num_classes is not None else cfg.get("num_classes", 19)
    bs = base        if base        is not None else cfg.get("base", 64)
    m = LiteUNet(3, nc, base=bs).to(device)
    m.load_state_dict(ck.get("state_dict", ck), strict=True)
    m.eval()
    return m

def preprocess(path, size=512):
    img = Image.open(path).convert("RGB").resize((size,size), Image.BILINEAR)
    x = TF.normalize(TF.to_tensor(img), [0.5]*3, [0.5]*3).unsqueeze(0)
    return x

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--base", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=512)
    args=ap.parse_args()

    model = load_model(args.weights, args.device, args.num_classes, args.base)
    x = preprocess(args.input, args.size).to(args.device)
    pred = model(x).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    Image.fromarray(pred, mode="L").save(args.output)

if __name__=="__main__":
    main()