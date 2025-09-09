import os, argparse, random, numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models.liteunet import LiteUNet
from dataset_faceparse import FaceParseDataset
from utils_metrics import mean_iou

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_class_weights(ds, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    for _, masks in tqdm(loader, desc="Scanning class frequencies"):
        for m in masks:
            u, c = np.unique(m.numpy(), return_counts=True)
            for uid, cnt in zip(u, c):
                if 0 <= uid < num_classes: counts[uid] += cnt
    freq = counts / counts.sum()
    inv = 1.0 / (freq + 1e-8)
    inv = inv / inv.sum() * num_classes
    return torch.tensor(inv, dtype=torch.float32)

def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    loss_sum = 0.0; tot = 0; correct = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(imgs)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == masks).sum().item()
        tot += masks.numel()
    return loss_sum / len(loader.dataset), correct / tot

@torch.no_grad()
def validate(model, loader, device, num_classes):
    model.eval()
    preds_all, gts_all = [], []
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(1).cpu().numpy()
        preds_all.extend(list(preds))
        gts_all.extend([m.numpy() for m in masks])
    miou = mean_iou(preds_all, gts_all, num_classes)
    correct = sum([(p == g).sum() for p, g in zip(preds_all, gts_all)])
    total   = sum([g.size for g in gts_all])
    pixacc = correct / total
    return miou, pixacc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_images", required=True, help="path to train/images")
    ap.add_argument("--train_masks",  required=True, help="path to train/masks")
    ap.add_argument("--val_images",   default=None, help="path to val/images (optional)")
    ap.add_argument("--val_masks",    default=None, help="path to val/masks (optional)")
    ap.add_argument("--val_ratio",    type=float, default=0.1, help="if no val*, take this fraction from train")
    ap.add_argument("--num_classes",  type=int, default=19)
    ap.add_argument("--img_size",     type=int, default=512)
    ap.add_argument("--epochs",       type=int, default=80)
    ap.add_argument("--batch_size",   type=int, default=6)
    ap.add_argument("--lr",           type=float, default=5e-4)
    ap.add_argument("--wd",           type=float, default=1e-2)
    ap.add_argument("--base",         type=int, default=64)  # keep param cap
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--out_ckpt",     default="solution/ckpt.pth")
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)

    # Datasets / loaders with auto-split
    if args.val_images and args.val_masks and os.path.isdir(args.val_images) and os.path.isdir(args.val_masks):
        train_ds = FaceParseDataset(args.train_images, args.train_masks, split="train", img_size=args.img_size)
        val_ds   = FaceParseDataset(args.val_images,   args.val_masks,   split="val",   img_size=args.img_size)
    else:
        full_ds = FaceParseDataset(args.train_images, args.train_masks, split="train", img_size=args.img_size)
        n = len(full_ds)
        idx = list(range(n))
        random.shuffle(idx)
        n_val = max(1, int(n * args.val_ratio))
        val_idx = set(idx[:n_val])
        train_idx = [i for i in range(n) if i not in val_idx]
        train_ds = Subset(full_ds, train_idx)
        val_ds   = Subset(full_ds, list(val_idx))
        print(f"[AutoSplit] Using {len(train_ds)} train / {len(val_ds)} val from training pairs.")

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = LiteUNet(in_ch=3, num_classes=args.num_classes, base=args.base).to(args.device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Param count: {nparams:,}")
    assert nparams < 1_821_085, "Model exceeds parameter cap!"

    # Loss
    if args.use_class_weights:
        # compute on the underlying dataset
        base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        w = get_class_weights(base_ds, args.num_classes).to(args.device)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Optim & sched
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_miou = -1.0
    patience, bad_epochs = 15, 0

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, optimizer, scaler, criterion, args.device)
        miou, pixacc = validate(model, val_ld, args.device, args.num_classes)
        scheduler.step()

        print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | trPixAcc {tr_acc:.4f} | valPixAcc {pixacc:.4f} | valmIoU {miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "miou": best_miou,
                "config": {"num_classes": args.num_classes, "base": args.base}
            }, args.out_ckpt)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch} (no mIoU improvement for {patience} epochs).")
            break

    print(f"Best val mIoU: {best_miou:.4f}. Checkpoint saved to: {args.out_ckpt}")

if __name__ == "__main__":
    main()