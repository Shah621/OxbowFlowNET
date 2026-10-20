#!/usr/bin/env python3
"""Fully-supervised OxbowFlow evaluation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import oxbow_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="fully_supervised/checkpoints/oxbow_best.pth")
    ap.add_argument("--test-dir", default="cpm17/test")
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--label-dir", default=None)
    ap.add_argument("--out-dir", default="fully_supervised/predictions")
    ap.add_argument("--raw-mask-dir", default=None)
    ap.add_argument("--arch", default=None, help="override; otherwise read from checkpoint")
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--separator", default="oxbow", choices=["oxbow", "watershed"])
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--tta-scales", default=None, help="comma list e.g. 0.85,1.0,1.15")
    ap.add_argument("--ensemble", default=None, help="comma list of extra checkpoints")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--smooth-sigma", type=float, default=2.0)
    ap.add_argument("--min-distance", type=int, default=4)
    ap.add_argument("--peak-rel", type=float, default=0.20)
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--min-size", type=int, default=0)
    ap.add_argument("--fill-alpha", type=float, default=0.4)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--supervision", default="fully", choices=["fully", "point"])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset_name = args.dataset_name or L.infer_dataset_name(args.test_dir)

    pairs = L.discover_pairs(Path(args.test_dir), args.image_dir, args.label_dir)
    print(f"Found {len(pairs)} image/label pairs for evaluation.")

    model, arch = L.load_model_from_checkpoint(args.checkpoint, device, args.arch, args.base_channels)
    models = None
    if args.ensemble:
        ckpts = [args.checkpoint] + [c.strip() for c in args.ensemble.split(",") if c.strip()]
        models = L.load_models_from_checkpoints(ckpts, device, args.arch, args.base_channels)
    scales = [float(s) for s in args.tta_scales.split(",")] if args.tta_scales else None
    sep_kwargs = {"smooth_sigma": args.smooth_sigma, "min_distance": args.min_distance,
                  "peak_rel": args.peak_rel, "beta": args.beta, "min_size": args.min_size}
    print(f"Loaded {arch} | separator={args.separator} | TTA={not args.no_tta} | "
          f"scales={scales or [1.0]} | ensemble={len(models) if models else 1} | sep={sep_kwargs}")

    metrics, per_image = L.evaluate(
        model, pairs, args.out_dir, device,
        separator=args.separator, threshold=args.threshold,
        tile=args.tile, stride=args.stride, tta=not args.no_tta,
        raw_mask_dir=args.raw_mask_dir, fill_alpha=args.fill_alpha,
        sep_kwargs=sep_kwargs, models=models, scales=scales,
    )
    print("\nEvaluation Results:")
    for k in ("DQ", "SQ", "PQ", "AJI"):
        print(f"  {k}:  {metrics[k]:.4f}")

    jp, cp = L.write_results(args.results_dir, args.supervision, dataset_name, metrics, per_image)
    print(f"\nSaved metrics      -> {jp}")
    print(f"Saved per-image CSV -> {cp}")
    print(f"Saved overlays      -> {args.out_dir}")


if __name__ == "__main__":
    main()
