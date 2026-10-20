#!/usr/bin/env python3
"""Point-supervised OxbowFlow evaluation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import oxbow_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="point_supervised/checkpoints/oxbow_best.pth")
    ap.add_argument("--test-dir", default="cpm17/test")
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--label-dir", default=None)
    ap.add_argument("--out-dir", default="point_supervised/predictions")
    ap.add_argument("--raw-mask-dir", default=None)
    ap.add_argument("--arch", default=None, help="override; otherwise read from checkpoint")
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--separator", default="oxbow", choices=["oxbow", "watershed"])
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.5, help="collision-ridge weight; use 0 for P-Geo")
    ap.add_argument("--fill-alpha", type=float, default=0.4)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--supervision", default="point", choices=["fully", "point"])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset_name = args.dataset_name or L.infer_dataset_name(args.test_dir)

    pairs = L.discover_pairs(Path(args.test_dir), args.image_dir, args.label_dir)
    print(f"Found {len(pairs)} image/label pairs for evaluation.")

    model, arch = L.load_model_from_checkpoint(args.checkpoint, device, args.arch, args.base_channels)
    print(f"Loaded {arch} | separator={args.separator} | TTA={not args.no_tta}")

    metrics, per_image = L.evaluate(
        model, pairs, args.out_dir, device,
        separator=args.separator, threshold=args.threshold,
        tile=args.tile, stride=args.stride, tta=not args.no_tta,
        raw_mask_dir=args.raw_mask_dir, fill_alpha=args.fill_alpha,
        sep_kwargs={"beta": args.beta},
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
