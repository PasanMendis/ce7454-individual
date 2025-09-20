import os, argparse, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from models.liteunet import LiteUNet
from dataset_faceparse import FaceParseDataset
from utils_metrics import compute_multiclass_fscore, mean_iou

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def get_class_weights(ds, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    for _,m in tqdm(ld, desc="classfreq"):
        for t in m:
            u,c = np.unique(t.numpy(), return_counts=True)
            for uid, cnt in zip(u,c):
                if 0<=uid<num_classes: counts[uid]+=cnt
    f = counts/counts.sum()
    w = (1.0/(f+1e-8)); w = w/w.sum()*num_classes
    return torch.tensor(w, dtype=torch.float32)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.smooth=smooth
    def forward(self, logits, targets, num_classes):
        probs = F.softmax(logits,1)
        oneh = F.one_hot(targets, num_classes).permute(0,3,1,2).float()
        inter = (probs*oneh).sum((0,2,3))
        card  = probs.sum((0,2,3)) + oneh.sum((0,2,3))
        dice  = (2*inter + self.smooth)/(card + self.smooth)
        return 1.0 - dice.mean()

def train_one_epoch(model, loader, opt, scaler, ce, dice, device, num_classes, max_grad_norm=1.0):
    model.train(); loss_sum=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(x)
            loss = 0.7*ce(logits,y) + 0.3*dice(logits,y,num_classes)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(opt); scaler.update()
        loss_sum += loss.item()*x.size(0)
    return loss_sum/len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device, num_classes):
    if loader is None: return 0.0, 0.0
    model.eval(); preds,gts=[],[]
    for x,y in loader:
        x=x.to(device); logits=model(x)
        p=logits.argmax(1).cpu().numpy()
        preds.extend(list(p)); gts.extend([t.numpy() for t in y])
    valF = float(np.mean([compute_multiclass_fscore(g,p) for p,g in zip(preds,gts)]))
    valmIoU = mean_iou(preds,gts,num_classes)
    return valF, valmIoU

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_images", required=True)
    ap.add_argument("--train_masks",  required=True)
    ap.add_argument("--val_images",   default=None)
    ap.add_argument("--val_masks",    default=None)
    ap.add_argument("--val_ratio",    type=float, default=0.1)
    ap.add_argument("--num_classes",  type=int, default=19)
    ap.add_argument("--img_size",     type=int, default=512)
    ap.add_argument("--epochs",       type=int, default=100)
    ap.add_argument("--batch_size",   type=int, default=6)
    ap.add_argument("--lr",           type=float, default=5e-4)
    ap.add_argument("--wd",           type=float, default=1e-2)
    ap.add_argument("--base",         type=int, default=64)
    ap.add_argument("--dropout",      type=float, default=0.0)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--max_grad_norm",type=float, default=1.0)
    ap.add_argument("--out_ckpt",     default="solution/ckpt.pth")
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args=ap.parse_args()

    set_seed(args.seed); os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)

    # datasets
    val_ld=None
    if args.val_images and args.val_masks:
        tr = FaceParseDataset(args.train_images, args.train_masks, "train", args.img_size)
        va = FaceParseDataset(args.val_images,   args.val_masks,   "val",   args.img_size)
        val_ld = DataLoader(va, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    else:
        full = FaceParseDataset(args.train_images, args.train_masks, "train", args.img_size)
        if args.val_ratio>0.0:
            n=len(full); idx=list(range(n)); random.shuffle(idx)
            nval=max(1,int(n*args.val_ratio)); vset=set(idx[:nval])
            tr = Subset(full, [i for i in range(n) if i not in vset])
            va = Subset(full, list(vset))
            val_ld = DataLoader(va, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
            print(f"[split] {len(tr)} train / {len(va)} val")
        else:
            tr = full
            print("[split] using all pairs for training (no val)")

    tr_ld = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # model
    model = LiteUNet(3, args.num_classes, base=args.base, dropout_p=args.dropout).to(args.device)
    nparams=sum(p.numel() for p in model.parameters()); print(f"Params: {nparams:,}")
    assert nparams < 1_821_085, "Param cap exceeded"

    # losses
    if args.use_class_weights:
        base_ds = tr.dataset if isinstance(tr, Subset) else tr
        w = get_class_weights(base_ds, args.num_classes).to(args.device)
        ce = nn.CrossEntropyLoss(weight=w, label_smoothing=0.0)
    else:
        ce = nn.CrossEntropyLoss(label_smoothing=0.0)
    dice = DiceLoss()

    # optim & sched
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    warmup_epochs, base_lr = 5, args.lr

    best_f=-1.0; bad=0; patience=15
    for ep in range(1, args.epochs+1):
        if ep<=warmup_epochs:
            for g in opt.param_groups: g["lr"]=base_lr*ep/max(1,warmup_epochs)
        tr_loss = train_one_epoch(model, tr_ld, opt, scaler, ce, dice, args.device, args.num_classes, args.max_grad_norm)
        sched.step()
        if val_ld is None:
            print(f"Epoch {ep:03d} | loss {tr_loss:.4f} | (no val)")
            torch.save({"state_dict":model.state_dict(),
                        "epoch":ep, "config":{"num_classes":args.num_classes,"base":args.base}},
                       args.out_ckpt)
            continue
        valF, valmIoU = validate(model, val_ld, args.device, args.num_classes)
        print(f"Epoch {ep:03d} | loss {tr_loss:.4f} | valF {valF:.4f} | valmIoU {valmIoU:.4f}")

        if valF>best_f:
            best_f=valF; bad=0
            torch.save({"state_dict":model.state_dict(), "epoch":ep, "valF":best_f, "valmIoU":valmIoU,
                        "config":{"num_classes":args.num_classes,"base":args.base}}, args.out_ckpt)
        else:
            bad+=1
            if bad>=patience:
                print(f"Early stop at {ep}")
                break
    print(f"Saved: {args.out_ckpt}")

if __name__=="__main__":
    main()