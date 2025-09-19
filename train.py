import os, argparse, random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, SWALR  # SWA
from torch.nn.utils import clip_grad_norm_

from models.liteunet import LiteUNet
from dataset_faceparse import FaceParseDataset
from utils_metrics import mean_iou, compute_multiclass_fscore


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


# ------------------ CutMix for segmentation (hard box) ------------------
def rand_bbox(H, W, lam):
    # same as standard CutMix; lam is area proportion to keep
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_cutmix(imgs, masks, alpha=1.0):
    # imgs: (B,3,H,W), masks: (B,H,W)
    if imgs.size(0) < 2:  # need at least 2 samples
        return imgs, masks
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    x1,y1,x2,y2 = rand_bbox(imgs.size(2), imgs.size(3), lam)
    imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    masks[:, y1:y2, x1:x2]   = masks[idx, y1:y2, x1:x2]
    return imgs, masks
# -----------------------------------------------------------------------


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets, num_classes):
        probs = F.softmax(logits, dim=1)
        targets_1h = F.one_hot(targets, num_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        inter = (probs * targets_1h).sum(dims)
        card  = probs.sum(dims) + targets_1h.sum(dims)
        dice  = (2.0 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()


def train_one_epoch(model, loader, optimizer, scaler, ce_loss, dice_loss, device,
                    num_classes, cutmix_p=0.3, max_grad_norm=1.0, ema=None):
    model.train()
    loss_sum = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        # --- CutMix (probabilistic) ---
        if cutmix_p > 0 and random.random() < cutmix_p:
            imgs, masks = apply_cutmix(imgs, masks, alpha=1.0)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(imgs)
            loss = 0.7 * ce_loss(logits, masks) + 0.3 * dice_loss(logits, masks, num_classes)

        scaler.scale(loss).backward()
        # grad clipping
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            with torch.no_grad():
                ema.update_parameters(model)

        loss_sum += loss.item() * imgs.size(0)
    return loss_sum / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device, num_classes):
    if loader is None:
        return 0.0, 0.0
    model.eval()
    preds_all, gts_all = [], []
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(1).cpu().numpy()
        preds_all.extend(list(preds))
        gts_all.extend([m.numpy() for m in masks])
    # mean F-score across images
    f_scores = [compute_multiclass_fscore(g, p) for p, g in zip(preds_all, gts_all)]
    valF = float(np.mean(f_scores))
    # mean IoU
    valmIoU = mean_iou(preds_all, gts_all, num_classes)
    return valF, valmIoU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_images", required=True)
    ap.add_argument("--train_masks",  required=True)
    ap.add_argument("--val_images",   default=None)
    ap.add_argument("--val_masks",    default=None)
    ap.add_argument("--val_ratio",    type=float, default=0.1)  # use 0.0 for no val
    ap.add_argument("--num_classes",  type=int, default=19)
    ap.add_argument("--img_size",     type=int, default=512)
    ap.add_argument("--epochs",       type=int, default=100)     # a bit longer
    ap.add_argument("--batch_size",   type=int, default=6)
    ap.add_argument("--lr",           type=float, default=5e-4)
    ap.add_argument("--wd",           type=float, default=1e-2)
    ap.add_argument("--base",         type=int, default=64)
    ap.add_argument("--dropout",      type=float, default=0.0)   # enable e.g. 0.1
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--cutmix_p",     type=float, default=0.3)   # 0.3 is a good start
    ap.add_argument("--max_grad_norm",type=float, default=1.0)
    ap.add_argument("--use_swa",      action="store_true")       # optional SWA
    ap.add_argument("--swa_start",    type=int, default=80)      # start epoch for SWA
    ap.add_argument("--out_ckpt",     default="solution/ckpt.pth")
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)

    # ---------- datasets ----------
    val_ld = None
    if args.val_images and args.val_masks and os.path.isdir(args.val_images) and os.path.isdir(args.val_masks):
        train_ds = FaceParseDataset(args.train_images, args.train_masks, split="train", img_size=args.img_size)
        val_ds   = FaceParseDataset(args.val_images,   args.val_masks,   split="val",   img_size=args.img_size)
        val_ld   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    else:
        full_ds = FaceParseDataset(args.train_images, args.train_masks, split="train", img_size=args.img_size)
        if args.val_ratio > 0.0:
            n = len(full_ds)
            idx = list(range(n)); random.shuffle(idx)
            n_val = max(1, int(n * args.val_ratio))
            val_idx = set(idx[:n_val])
            train_idx = [i for i in range(n) if i not in val_idx]
            train_ds = Subset(full_ds, train_idx)
            val_ds   = Subset(full_ds, list(val_idx))
            val_ld   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
            print(f"[AutoSplit] Using {len(train_ds)} train / {len(val_ds)} val.")
        else:
            train_ds = full_ds
            print("[Info] No validation set (val_ratio=0.0). Training on all pairs.")

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # ---------- model ----------
    model = LiteUNet(in_ch=3, num_classes=args.num_classes, base=args.base, dropout_p=args.dropout).to(args.device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Param count: {nparams:,}")
    assert nparams < 1_821_085, "Model exceeds parameter cap!"

    # ---------- losses ----------
    if args.use_class_weights:
        base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        w = get_class_weights(base_ds, args.num_classes).to(args.device)
        ce = nn.CrossEntropyLoss(weight=w, label_smoothing=0.02)  # slightly lower LS helps edges
    else:
        ce = nn.CrossEntropyLoss(label_smoothing=0.02)
    dice = DiceLoss()

    # ---------- optim & sched ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # SWA setup (optional)
    swa_model = None
    if args.use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.5)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_f = -1.0
    patience, bad_epochs = 15, 0
    warmup_epochs = 5
    base_lr = args.lr

    for epoch in range(1, args.epochs + 1):
        # warmup LR
        if epoch <= warmup_epochs:
            for g in optimizer.param_groups:
                g["lr"] = base_lr * epoch / max(1, warmup_epochs)

        tr_loss = train_one_epoch(
            model, train_ld, optimizer, scaler,
            ce_loss=ce, dice_loss=dice, device=args.device,
            num_classes=args.num_classes, cutmix_p=args.cutmix_p,
            max_grad_norm=args.max_grad_norm,
            ema=None  # you can add EMA as an alternative to SWA if desired
        )

        # schedulers: switch to SWA after swa_start
        if args.use_swa and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # evaluate (use SWA weights if active & past start)
        if val_ld is None:
            print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | (no validation)")
            # still save latest
            torch.save({"state_dict": model.state_dict(),
                        "epoch": epoch,
                        "config": {"num_classes": args.num_classes, "base": args.base}},
                       args.out_ckpt)
            continue

        if args.use_swa and epoch >= args.swa_start:
            # Temporarily swap in SWA weights for eval
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(swa_model.module.state_dict(), strict=True)
            valF, valmIoU = validate(model, val_ld, args.device, args.num_classes)
            # restore
            model.load_state_dict(backup, strict=True)
        else:
            valF, valmIoU = validate(model, val_ld, args.device, args.num_classes)

        print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | valF {valF:.4f} | valmIoU {valmIoU:.4f}")

        # save best by F-score
        if valF > best_f:
            best_f = valF
            torch.save({
                "state_dict": (swa_model.module.state_dict() if (args.use_swa and epoch >= args.swa_start) else model.state_dict()),
                "epoch": epoch,
                "valF": best_f,
                "valmIoU": valmIoU,
                "config": {"num_classes": args.num_classes, "base": args.base, "dropout": args.dropout}
            }, args.out_ckpt)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch} (no F-score improvement for {patience} epochs).")
            break

    # If SWA enabled, you can optionally save final SWA weights:
    if args.use_swa:
        torch.save({
            "state_dict": swa_model.module.state_dict(),
            "epoch": args.epochs,
            "valF": best_f,
            "config": {"num_classes": args.num_classes, "base": args.base, "dropout": args.dropout}
        }, args.out_ckpt)
        print("[SWA] Saved final SWA-averaged weights.")

    print(f"Checkpoint saved to: {args.out_ckpt}")

if __name__ == "__main__":
    main()