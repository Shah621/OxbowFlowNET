#!/usr/bin/env python3
"""Build an 80/20 NuInsSeg split in CPM-17 layout.

    python scripts/prepare_nuinseg.py
    python scripts/prepare_nuinseg.py --src nuinseg --dst nuinseg_prepared --val-frac 0.2 --seed 42 --force
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_SUBDIR = "tissue images"
LABEL_SUBDIR_CANDIDATES = ("label masks modify", "label masks")


def find_pairs(src: Path):
    pairs = []
    organs = sorted([d for d in src.iterdir() if d.is_dir()])
    skipped = []
    for organ in organs:
        img_dir = organ / IMAGE_SUBDIR
        if not img_dir.is_dir():
            skipped.append((organ.name, "no 'tissue images'"))
            continue
        label_dir = None
        for cand in LABEL_SUBDIR_CANDIDATES:
            if (organ / cand).is_dir():
                label_dir = organ / cand
                break
        if label_dir is None:
            skipped.append((organ.name, "no label folder"))
            continue

        organ_pairs = 0
        for img_path in sorted(img_dir.glob("*.png")):
            label_path = label_dir / (img_path.stem + ".tif")
            if not label_path.exists():
                alt = list(label_dir.glob(img_path.stem + ".*"))
                if not alt:
                    continue
                label_path = alt[0]
            pairs.append((img_path, label_path))
            organ_pairs += 1
        print(f"  {organ.name:45s} {organ_pairs:3d} pairs  (labels: {label_dir.name})")
    if skipped:
        print("\nSkipped organs:")
        for name, why in skipped:
            print(f"  {name}: {why}")
    return pairs


def copy_split(pairs, dst: Path, force: bool):
    for split in ("train", "test"):
        (dst / split / "Images").mkdir(parents=True, exist_ok=True)
        (dst / split / "Labels").mkdir(parents=True, exist_ok=True)

    for split, split_pairs in pairs.items():
        img_out = dst / split / "Images"
        lbl_out = dst / split / "Labels"
        existing = len(list(img_out.glob("*.png")))
        if existing and not force:
            print(f"[{split}] already has {existing} images; skipping (use --force to overwrite).")
            continue
        if force:
            for p in img_out.glob("*"):
                p.unlink()
            for p in lbl_out.glob("*"):
                p.unlink()
        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(label_path, lbl_out / label_path.name)
        print(f"[{split}] wrote {len(split_pairs)} image/label pairs -> {img_out.parent}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=str, default="nuinseg", help="Raw NuInsSeg root (31 organ folders)")
    ap.add_argument("--dst", type=str, default="nuinseg_prepared", help="Output root (CPM17-style)")
    ap.add_argument("--val-frac", type=float, default=0.2, help="Fraction held out as the test set")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    ap.add_argument("--force", action="store_true", help="Overwrite existing prepared data")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"Source directory not found: {src.resolve()}")

    print(f"Scanning {src.resolve()} ...")
    all_pairs = find_pairs(src)
    if not all_pairs:
        raise SystemExit("No (image, label) pairs found. Check the NuInsSeg folder layout.")

    rng = random.Random(args.seed)
    rng.shuffle(all_pairs)
    n_test = int(round(len(all_pairs) * args.val_frac))
    test_pairs = all_pairs[:n_test]
    train_pairs = all_pairs[n_test:]
    print(
        f"\nTotal pairs: {len(all_pairs)}  ->  train: {len(train_pairs)}  test: {len(test_pairs)}  "
        f"(seed={args.seed}, val_frac={args.val_frac})"
    )

    copy_split({"train": train_pairs, "test": test_pairs}, dst, args.force)
    print("\nDone. Use:")
    print(f"  --data-dir {dst}/train   (training)")
    print(f"  --test-dir {dst}/test    (evaluation)")


if __name__ == "__main__":
    main()
