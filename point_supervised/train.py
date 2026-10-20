#!/usr/bin/env python3
"""Point-supervised OxbowFlow training (centroid phi/alpha)."""
import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

import oxbow_lib as L

SUPERVISION = "point"


def run_epoch(model, loader, device, weights, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    agg = {"loss": 0.0, "fg": 0.0, "phi": 0.0, "v": 0.0, "div": 0.0, "curv": 0.0}
    n = 0
    torch.set_grad_enabled(train)
    for batch in loader:
        img = batch["img"].to(device); mask = batch["mask"].to(device)
        phi = batch["phi"].to(device); alpha = batch["alpha"].to(device)
        out = model(img)
        total, stats = L.compute_losses(out, mask, phi, alpha, weights)
        if train:
            optimizer.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        bs = img.size(0); n += bs
        for k in agg:
            agg[k] += stats[k] * bs
    torch.set_grad_enabled(True)
    for k in agg:
        agg[k] /= max(1, n)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="cpm17/train")
    ap.add_argument("--train-dir", default=None)
    ap.add_argument("--val-dir", default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work-dir", default="point_supervised/checkpoints")
    ap.add_argument("--arch", default="resunet34", choices=["resunet34", "resunet18", "unet"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--base-channels", type=int, default=32, help="only for --arch unet")
    ap.add_argument("--size", type=int, default=256, help="training patch size")
    ap.add_argument("--crops-per-image", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--w-fg", type=float, default=1.0)
    ap.add_argument("--w-phi", type=float, default=1.0)
    ap.add_argument("--w-v", type=float, default=1.0)
    ap.add_argument("--w-div", type=float, default=0.05)
    ap.add_argument("--w-curv", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Using device:", device, "| arch:", args.arch)

    train_root = Path(args.train_dir or args.data_dir)
    all_pairs = L.discover_pairs(train_root)
    if args.val_dir:
        train_pairs, val_pairs = all_pairs, L.discover_pairs(Path(args.val_dir))
        print(f"Train: {train_root} ({len(train_pairs)})  Val (explicit): {args.val_dir} ({len(val_pairs)})")
    else:
        train_pairs, val_pairs = L.split_pairs(all_pairs, args.val_frac, args.seed)
        print(f"Train/Val split of {train_root}: {len(train_pairs)}/{len(val_pairs)} images "
              f"(val_frac={args.val_frac}); test set untouched.")

    train_ds = L.PatchDataset(pairs=train_pairs, patch=args.size, crops_per_image=args.crops_per_image,
                              supervision=SUPERVISION, augment=True, seed=args.seed)
    val_ds = L.PatchDataset(pairs=val_pairs, patch=args.size, crops_per_image=2,
                            supervision=SUPERVISION, augment=False, seed=args.seed)
    nw = (0 if os.name == "nt" else 2) if args.num_workers is None else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    model = L.build_model(args.arch, pretrained=not args.no_pretrained, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / max(1, args.warmup)
        t = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    weights = {"fg": args.w_fg, "phi": args.w_phi, "v": args.w_v, "div": args.w_div, "curv": args.w_curv}
    out_dir = Path(args.work_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best = 1e9
    for epoch in range(1, args.epochs + 1):
        for g in optimizer.param_groups:
            g["lr"] = args.lr * lr_at(epoch - 1)
        tr = run_epoch(model, train_loader, device, weights, optimizer)
        va = run_epoch(model, val_loader, device, weights, None)
        print(f"[{datetime.now():%H:%M:%S}] Epoch {epoch}/{args.epochs} lr={optimizer.param_groups[0]['lr']:.2e} "
              f"TRAIN {tr['loss']:.4f} VAL {va['loss']:.4f} | "
              f"tr(fg={tr['fg']:.3f},phi={tr['phi']:.3f},v={tr['v']:.3f}) va(fg={va['fg']:.3f})")
        ckpt = {"net": model.state_dict(), "arch": args.arch, "base_channels": args.base_channels, "epoch": epoch}
        torch.save(ckpt, out_dir / f"oxbow_epoch{epoch:03d}.pth")
        if va["loss"] < best:
            best = va["loss"]; torch.save(ckpt, out_dir / "oxbow_best.pth")
    print("Done. Best val loss:", best)


if __name__ == "__main__":
    main()
