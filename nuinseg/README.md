# NuInsSeg

[NuInsSeg](https://www.nature.com/articles/s41597-024-03117-2) is a fully annotated H&E nuclei dataset: 665 patches of $512\times512$ from 31 human and mouse organs, with 30,698 instance masks.

There is no official train/test split. Following Mudslide (CVPR 2024) we draw a **random 80 / 20 split with seed 42**, giving 532 training and 133 test images. 10% of the training images are then held out as validation. Because Mudslide did not publish its split seed, NuInsSeg comparisons are same-protocol, not same-split.

## Download

- Zenodo (CC BY 4.0): **https://doi.org/10.5281/zenodo.10518968**
- Kaggle: **https://www.kaggle.com/datasets/ipateam/nuinsseg**

Please cite:

```bibtex
@article{mahbod2024nuinseg,
  title     = {{NuInsSeg}: A Fully Annotated Dataset for Nuclei Instance Segmentation
               in {H\&E}-Stained Histological Images},
  author    = {Mahbod, Amirreza and Polak, Christine and Feldmann, Katharina
               and Khan, Rumsha and Gelles, Katharina and Dorffner, Georg
               and Woitek, Ramona and Hatamikia, Sepideh and Ellinger, Isabella},
  journal   = {Scientific Data},
  volume    = {11},
  pages     = {295},
  year      = {2024}
}
```

## Expected layout (raw)

Unpack the archive so that this folder contains one directory per organ:

```
nuinseg/
├── README.md
├── human bladder/
│   ├── tissue images/          RGB patches  (.png)
│   └── label masks modify/     instance maps  (.tif)
├── human brain/
│   └── ...
└── mouse thymus/
    └── ...
```

Only `tissue images/` and `label masks modify/` (fallback: `label masks/`) are used. Other NuInsSeg folders (binary masks, ROIs, ambiguous regions) can be left in place.

## Prepare the 80 / 20 split

```bash
python scripts/prepare_nuinseg.py --src nuinseg --dst nuinseg_prepared --val-frac 0.2 --seed 42
```

This copies files into a CPM-17-style layout (add `--force` to overwrite):

```
nuinseg_prepared/
├── train/
│   ├── Images/
│   └── Labels/
└── test/
    ├── Images/
    └── Labels/
```

`nuinseg_prepared/` is generated and is not part of the repository.

## Train and evaluate

```bash
python fully_supervised/train.py --data-dir nuinseg_prepared/train --arch resunet34 \
    --crops-per-image 4 --epochs 40 --batch-size 4 \
    --work-dir fully_supervised/checkpoints_nuinseg

python fully_supervised/evaluation.py \
    --checkpoint fully_supervised/checkpoints_nuinseg/oxbow_best.pth \
    --test-dir nuinseg_prepared/test --dataset-name NuInsSeg --separator oxbow \
    --out-dir fully_supervised/predictions_nuinseg
```

Point-supervised (paper: ResNet-18, 25 epochs):

```bash
python point_supervised/train.py --data-dir nuinseg_prepared/train --arch resunet18 \
    --crops-per-image 5 --epochs 25 --batch-size 4 \
    --work-dir point_supervised/checkpoints_nuinseg
```
