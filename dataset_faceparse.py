import os, glob, random
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

def _index(d, exts):
    m = {}
    for ext in exts:
        for p in glob.glob(os.path.join(d, f"*{ext}")):
            m[os.path.splitext(os.path.basename(p))[0]] = p
    return m

class FaceParseDataset(Dataset):
    """
    Face parsing dataset with safe augmentations (no flips).
    Augs: near-identity resized crop, color jitter, gaussian blur.
    """
    def __init__(self, img_dir, mask_dir, split="train", img_size=512):
        imgs  = _index(img_dir,  [".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"])
        masks = _index(mask_dir, [".png",".PNG"])
        self.basenames = sorted(set(imgs) & set(masks))
        if not self.basenames:
            raise FileNotFoundError("No matching basenames between images/ and masks/")
        self.imgs, self.masks = imgs, masks
        self.split, self.img_size = split, img_size
        self.jitter = ColorJitter(0.15, 0.15, 0.15, 0.05)  # mild

    def _random_resized_crop_params(self, w, h):
        scale = random.uniform(0.95, 1.0)
        ratio = random.uniform(0.98, 1.02)
        target_area = scale * w * h
        new_w = int(round((target_area * ratio) ** 0.5))
        new_h = int(round((target_area / ratio) ** 0.5))
        new_w = min(new_w, w); new_h = min(new_h, h)
        i = random.randint(0, h - new_h) if h > new_h else 0
        j = random.randint(0, w - new_w) if w > new_w else 0
        return i, j, new_h, new_w

    def __len__(self): return len(self.basenames)

    def __getitem__(self, i):
        b = self.basenames[i]
        img  = Image.open(self.imgs[b]).convert("RGB")
        mask = Image.open(self.masks[b])

        # resize to canonical size first
        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.split == "train":
            # near-identity resized crop
            if random.random() < 0.3:
                i0, j0, h0, w0 = self._random_resized_crop_params(self.img_size, self.img_size)
                img  = TF.resized_crop(img, i0, j0, h0, w0, (self.img_size, self.img_size), TF.InterpolationMode.BILINEAR)
                mask = TF.resized_crop(mask,i0, j0, h0, w0, (self.img_size, self.img_size), TF.InterpolationMode.NEAREST)

            # color jitter
            img = self.jitter(img)

            # occasional gaussian blur
            if random.random() < 0.2:
                img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # convert
        x = TF.normalize(TF.to_tensor(img), [0.5]*3, [0.5]*3)
        y = torch.from_numpy(np.array(mask, dtype=np.int64))
        return x, y