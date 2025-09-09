import os, glob, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

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
    Pairs images and masks by basename (case-insensitive).
    - Expects masks to be single-channel PNGs with label IDs.
    - Applies light augmentation when split="train".
    """
    def __init__(self, img_dir, mask_dir, split="train", img_size=512):
        self.imgs  = index_dir(img_dir,  [".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"])
        self.masks = index_dir(mask_dir, [".png",".PNG"])
        self.bases = sorted(list(set(self.imgs.keys()) & set(self.masks.keys())))
        if len(self.bases) == 0:
            raise FileNotFoundError(f"No matching image/mask basenames between {img_dir} and {mask_dir}")
        self.split = split
        self.img_size = img_size
        self.jitter = ColorJitter(0.2,0.2,0.2,0.1)

    def __len__(self): 
        return len(self.bases)

    def __getitem__(self, i):
        b = self.bases[i]
        img  = Image.open(self.imgs[b]).convert("RGB")
        mask = Image.open(self.masks[b])  # label IDs

        img  = img.resize((self.img_size,self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size,self.img_size), Image.NEAREST)

        if self.split == "train":
            if random.random() < 0.5:
                img = TF.hflip(img); mask = TF.hflip(mask)
            img = self.jitter(img)

        img  = TF.to_tensor(img)
        img  = TF.normalize(img, [0.5,0.5,0.5], [0.5,0.5,0.5])
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return img, mask