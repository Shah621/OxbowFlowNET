# CPM-17

[Computational Precision Medicine](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6743190/) nuclear segmentation challenge (MICCAI 2017). 64 H&E histopathology tiles with 7,570 annotated nuclear boundaries, at $500\times500$ and $600\times600$.

OxbowFlow uses the **official 32 / 32 train/test split**. Three of the 32 training images are held out as validation at training time (`--val-frac 0.1`, seed 42). The test set is never seen during training.

## Download

The images and instance maps used here match the Hover-Net packaging of CPM-17:

**https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK**

Please cite the original challenge paper if you use this data:

```bibtex
@article{vu2019methods,
  title   = {Methods for Segmentation and Classification of Digital Microscopy Tissue Images},
  author  = {Vu, Quoc Dang and Graham, Simon and Kurc, Tahsin and To, Minh Nguyen Nhat
             and Shaban, Muhammad and Qaiser, Talha and Koohbanani, Navid Alemi
             and Khurram, Syed Ali and Kalpathy-Cramer, Jayashree and Zhao, Tianhao
             and others},
  journal = {Frontiers in Bioengineering and Biotechnology},
  volume  = {7},
  pages   = {53},
  year    = {2019}
}
```

## Expected layout

Place the official split under this folder. Overlays are optional and are not read by the code.

```
cpm17/
├── README.md
├── train/
│   ├── Images/     32 RGB tiles  (.png)
│   └── Labels/     32 instance maps  (.mat with key inst_map)
└── test/
    ├── Images/
    └── Labels/
```

`oxbow_lib.discover_pairs` looks for `Images/` + `Labels/` (case-insensitive). Labels may be `.mat`, `.npy`, or any common image format. For `.mat` files the loader accepts `inst_map`, `instance_map`, `stacked_mask`, `label`, `labels`, or `mask`.

## Train and evaluate

```bash
python fully_supervised/train.py --data-dir cpm17/train --arch resunet34 \
    --crops-per-image 12 --epochs 150 --batch-size 4

python fully_supervised/evaluation.py \
    --checkpoint fully_supervised/checkpoints/oxbow_best.pth \
    --test-dir cpm17/test --dataset-name CPM17 --separator oxbow
```

Point-supervised training uses the same folders; only the $\varphi/\alpha$ targets change (centroids instead of full masks):

```bash
python point_supervised/train.py --data-dir cpm17/train --arch resunet34 \
    --crops-per-image 12 --epochs 60 --batch-size 4
```
