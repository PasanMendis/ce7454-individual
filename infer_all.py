import os, argparse
from glob import glob
from PIL import Image
import numpy as np, torch
import torchvision.transforms.functional as TF
from models.liteunet import LiteUNet

def load_model(weights, device="cpu"):
    ck = torch.load(weights, map_location=device)
    cfg = ck.get("config", {})
    nc  = cfg.get("num_classes", 19); base = cfg.get("base", 64)
    m = LiteUNet(3, nc, base=base).to(device)
    m.load_state_dict(ck.get("state_dict", ck), strict=True)
    m.eval()
    return m

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", default="masks")
    ap.add_argument("--weights", default="solution/ckpt.pth")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tta", action="store_true")
    args=ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.weights, args.device)

    paths = sorted(glob(os.path.join(args.images_dir,"*.jpg"))+glob(os.path.join(args.images_dir,"*.png")))
    for p in paths:
        img = Image.open(p).convert("RGB").resize((args.size,args.size), Image.BILINEAR)
        x = TF.normalize(TF.to_tensor(img), [0.5]*3, [0.5]*3).unsqueeze(0).to(args.device)
        logits = model(x)
        if args.tta:
            xf = torch.flip(x, [3]); lf = model(xf); lf = torch.flip(lf,[3])
            logits = (logits+lf)/2
        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        out = os.path.join(args.out_dir, os.path.splitext(os.path.basename(p))[0]+".png")
        Image.fromarray(pred, mode="L").save(out)

if __name__=="__main__":
    main()