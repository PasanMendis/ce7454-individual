import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def index_dir(d, exts):
    m = {}
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Directory not found: {d}")
    for ext in exts:
        for p in glob.glob(os.path.join(d, f"*{ext}")):
            m[os.path.splitext(os.path.basename(p))[0]] = p
    return m

class FaceParseDataset(Dataset):
    """
    Conservative loader: resize->(optional H-flip for train)->tensor->normalize.
    Masks must be single-channel PNG label IDs.
    """
    def __init__(self, img_dir, mask_dir, split="train", img_size=512):
        self.imgs  = index_dir(img_dir,  [".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"])
        self.masks = index_dir(mask_dir, [".png",".PNG"])
        self.bases = sorted(list(set(self.imgs.keys()) & set(self.masks.keys())))
        if len(self.bases) == 0:
            raise FileNotFoundError(f"No matching image/mask basenames between {img_dir} and {mask_dir}")
        self.split = split
        self.img_size = img_size

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, i):
        b = self.bases[i]
        img  = Image.open(self.imgs[b]).convert("RGB")
        mask = Image.open(self.masks[b])  # label IDs

        # resize first (512x512)
        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # only safe aug: horizontal flip
        if self.split == "train":
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img); mask = TF.hflip(mask)

        # to tensor + normalize
        img  = TF.to_tensor(img)
        img  = TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # (H,W)

        return img, mask