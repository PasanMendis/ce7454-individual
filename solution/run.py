#!/usr/bin/env python3
import argparse
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# ------------------------------
# LiteUNet model (from-scratch)
# ------------------------------
class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class LiteUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=19, base=64):
        super().__init__()
        C = [in_ch, base, base*2, base*4, base*8]  # 3,64,128,256,512

        self.enc1 = DWConvBlock(C[0], C[1])  # 512
        self.pool1 = nn.MaxPool2d(2)         # 256
        self.enc2 = DWConvBlock(C[1], C[2])
        self.pool2 = nn.MaxPool2d(2)         # 128
        self.enc3 = DWConvBlock(C[2], C[3])
        self.pool3 = nn.MaxPool2d(2)         # 64

        self.bottleneck = DWConvBlock(C[3], C[4])

        self.up3 = nn.ConvTranspose2d(C[4], C[3], 2, stride=2)  # 128
        self.dec3 = DWConvBlock(C[3] + C[3], C[3])

        self.up2 = nn.ConvTranspose2d(C[3], C[2], 2, stride=2)  # 256
        self.dec2 = DWConvBlock(C[2] + C[2], C[2])

        self.up1 = nn.ConvTranspose2d(C[2], C[1], 2, stride=2)  # 512
        self.dec1 = DWConvBlock(C[1] + C[1], C[1])

        self.head = nn.Conv2d(C[1], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)          # 512
        x  = self.pool1(e1)        # 256
        e2 = self.enc2(x)
        x  = self.pool2(e2)        # 128
        e3 = self.enc3(x)
        x  = self.pool3(e3)        # 64

        x  = self.bottleneck(x)    # 64
        x  = self.up3(x)           # 128
        x  = torch.cat([x, e3], 1)
        x  = self.dec3(x)

        x  = self.up2(x)           # 256
        x  = torch.cat([x, e2], 1)
        x  = self.dec2(x)

        x  = self.up1(x)           # 512
        x  = torch.cat([x, e1], 1)
        x  = self.dec1(x)

        return self.head(x)

# ------------------------------
# Inference helpers
# ------------------------------
def load_model(weights_path, num_classes=19, base=64, device="cpu"):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"weights not found: {weights_path}")
    if os.path.getsize(weights_path) < 1024:
        raise EOFError(
            f"Checkpoint appears empty or truncated ({os.path.getsize(weights_path)} bytes). "
            "Overwrite the placeholder with your trained weights."
        )
    model = LiteUNet(in_ch=3, num_classes=num_classes, base=base).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # (optional) verify saved config matches
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def preprocess(img_path, size=512):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    x = TF.to_tensor(img)
    x = TF.normalize(x, [0.5,0.5,0.5], [0.5,0.5,0.5])
    return x

@torch.no_grad()
def infer_single(model, img_path, out_path, device="cpu", size=512):
    x = preprocess(img_path, size=size).unsqueeze(0).to(device)
    logits = model(x)
    pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Save SINGLE-CHANNEL PNG (mode "L") as required
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(pred, mode="L").save(out_path)

def main():
    parser = argparse.ArgumentParser(description="Face parsing inference runner (single image)")
    parser.add_argument("--input", required=True, help="/path/to/input-image.jpg")
    parser.add_argument("--output", required=True, help="/path/to/output-mask.png")
    parser.add_argument("--weights", required=True, help="ckpt.pth (must exist)")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.weights, num_classes=args.num_classes, base=args.base, device=args.device)
    infer_single(model, args.input, args.output, device=args.device, size=args.size)

if __name__ == "__main__":
    main()