# OxbowFlow: Continuous Flow Decomposition for Nuclear Instance Segmentation

**[CVPR 2027]** Official implementation of OxbowFlow, a curvature-aware continuous-flow method for nuclear instance segmentation.

> Sibtain Syed<sup>1</sup>, Ali Ahmed Qureshi<sup>1</sup>, Maqbool Khan<sup>2</sup><br>
> <sup>1</sup>Comsat University Islamabad, Wah, Pakistan<br>
> <sup>2</sup>American University of Bahrain, Riffa, Bahrain

A U-Net with a pretrained ResNet encoder predicts three dense fields — foreground $f$, potential $\varphi$, and meander $\alpha$. Pixels then follow the continuous flow

$$
\tilde{v} = -\hat{n} + \alpha\, R\hat{n},\qquad v = \tilde{v}/\lVert\tilde{v}\rVert,
\qquad \hat{n}=\nabla\varphi/\lVert\nabla\varphi\rVert,
$$

where $R$ is a $90^\circ$ rotation. Trajectories meander, collide along the $\alpha$ ridge, and enclose each nucleus (an *oxbow*). We evaluate on **CPM17** and **NuInsSeg** under fully-supervised and point-supervised protocols, reporting DQ, SQ, PQ, and AJI.

## Results

Fully-supervised ($\mathrm{P}$-$\mathrm{Full}$):

| Method | CPM17 PQ | CPM17 AJI | NuInsSeg PQ | NuInsSeg AJI |
|---|---:|---:|---:|---:|
| Hover-Net | 0.697 | 0.705 | 0.568 | 0.611 |
| CellViT | 0.713 | 0.737 | 0.596 | 0.638 |
| Mudslide | **0.714** | **0.740** | **0.599** | **0.647** |
| **OxbowFlow (ours)** | 0.704 | 0.733 | 0.578 | 0.626 |

Point-supervised geometry ($\mathrm{P}$-$\mathrm{Geo}$, $\beta=0$):

| Method | CPM17 PQ | CPM17 AJI | NuInsSeg PQ | NuInsSeg AJI |
|---|---:|---:|---:|---:|
| Mudslide | 0.529 | 0.581 | 0.487 | 0.558 |
| **OxbowFlow (ours)** | **0.662** | **0.696** | **0.557** | **0.603** |

OxbowFlow attains the best CPM17 SQ among fully-supervised methods (0.814, tied with Hover-Net). Under $\mathrm{P}$-$\mathrm{Geo}$, $\varphi$ and $\alpha$ are synthesized from centroids; the foreground branch is still mask-supervised, so those numbers are an upper bound on strict point supervision. See the paper for the full tables, ablations, and protocol details.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+, PyTorch, and torchvision. The first training run downloads ImageNet ResNet weights (~85 MB). CUDA is used automatically when available; pass `--cpu` to force CPU.

```bash
python scripts/verify_oxbow.py   # optional: check that synthesized flow converges into basins
```

## Datasets

Images are **not** shipped with this repository. Place each dataset as described in its README, then train.

| Dataset | Images | Split | README |
|---|---|---|---|
| [CPM-17](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK) | 64 H&E tiles, 7,570 nuclei | official 32 / 32 | [`cpm17/README.md`](cpm17/README.md) |
| [NuInsSeg](https://doi.org/10.5281/zenodo.10518968) | 665 patches, 31 organs | random 80 / 20, seed 42 | [`nuinseg/README.md`](nuinseg/README.md) |

```bash
python scripts/prepare_nuinseg.py --src nuinseg --dst nuinseg_prepared --val-frac 0.2 --seed 42
```

## Training

```bash
# Fully supervised  (P-Full)
python fully_supervised/train.py --data-dir cpm17/train --arch resunet34 --crops-per-image 12 --epochs 150 --batch-size 4
python fully_supervised/train.py --data-dir nuinseg_prepared/train --arch resunet34 --crops-per-image 4 --epochs 40 --batch-size 4 \
    --work-dir fully_supervised/checkpoints_nuinseg

# Point supervised  (P-Geo: centroid phi/alpha)
python point_supervised/train.py --data-dir cpm17/train --arch resunet34 --crops-per-image 12 --epochs 60 --batch-size 4
python point_supervised/train.py --data-dir nuinseg_prepared/train --arch resunet18 --crops-per-image 5 --epochs 25 --batch-size 4 \
    --work-dir point_supervised/checkpoints_nuinseg
```

Validation is a random hold-out of the training images (`--val-frac 0.1`, seed 42). The test set is never seen during training. Checkpoints are written to `*/checkpoints/oxbow_best.pth`.

| Flag | Default | Notes |
|---|---|---|
| `--arch` | `resunet34` | `resunet18`, `resunet34`, `resunet50`, or `unet` |
| `--size` | 256 | native-resolution crop size |
| `--crops-per-image` | 16 | random crops per source image per epoch |
| `--lr` | `1e-3` | AdamW with linear warmup and cosine decay |
| `--w-fg --w-phi --w-v --w-div --w-curv` | 1, 1, 1, 0.05, 0.05 | loss weights |

## Evaluation

```bash
python fully_supervised/evaluation.py \
    --checkpoint fully_supervised/checkpoints/oxbow_best.pth \
    --test-dir cpm17/test --dataset-name CPM17 --separator oxbow

python point_supervised/evaluation.py \
    --checkpoint point_supervised/checkpoints/oxbow_best.pth \
    --test-dir cpm17/test --dataset-name CPM17 --separator oxbow --beta 0

python fully_supervised/evaluation.py \
    --checkpoint fully_supervised/checkpoints_nuinseg/oxbow_best.pth \
    --test-dir nuinseg_prepared/test --dataset-name NuInsSeg --separator oxbow \
    --out-dir fully_supervised/predictions_nuinseg
```

Sliding-window inference (`--tile 256 --stride 192`) with flip TTA is on by default. Colour+boundary overlays are saved under `*/predictions/`. Use `--raw-mask-dir <dir>` to also write instance maps as `.npy`. Metrics (DQ, SQ, PQ, AJI) are printed and written as JSON/CSV.

`--separator oxbow` is the collision-ridge method from the paper; `--separator watershed` is flow-seeded watershed without the ridge. Under $\mathrm{P}$-$\mathrm{Geo}$ set `--beta 0`, because the target $\alpha^\star \equiv 0$.

## Repository layout

```
OxbowFlowNET/
├── oxbow_lib.py                 shared library (model, losses, data, inference, separation, metrics)
├── fully_supervised/
│   ├── train.py
│   └── evaluation.py
├── point_supervised/
│   ├── train.py
│   └── evaluation.py
├── scripts/
│   ├── prepare_nuinseg.py       80/20 NuInsSeg split (seed 42)
│   └── verify_oxbow.py          sanity check that flow converges into basins
├── cpm17/README.md
├── nuinseg/README.md
├── requirements.txt
└── LICENSE
```

## Method (short)

1. **Terrain.** Each nucleus is a basin of $\varphi$ (low at the centroid, $\approx 0$ at the rim). Background is a shallow linear drift.
2. **Meander.** $\alpha$ is high on instance rims. The flow $v = -\hat{n} + \alpha\, R\hat{n}$ bends trajectories off steepest descent.
3. **Cutoff.** Neighbouring basins deflect in opposite senses, so trajectories collide on the $\alpha$ ridge and never cross it. The collision locus is a closed curve around each body.
4. **Separation.** Foreground pixels are integrated along $v$ (RK4). Density peaks become markers; the ridge $\beta|\alpha|$ is added to the watershed elevation so touching nuclei split rather than merge.

The encoder is a ResNet; the decoder is a U-Net head with a 3-channel output $(f, \varphi, \alpha)$. Training uses BCE+Dice on $f$, MSE+MSGE on $\varphi$, a vector-field match on $v$, and small divergence/curl regularisers. Crops are taken at native resolution with geometric and photometric augmentation.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{syed2027oxbowflow,
  title     = {OxbowFlow: Continuous Flow Decomposition for Nuclear Instance Segmentation},
  author    = {Syed, Sibtain and Qureshi, Ali Ahmed and Khan, Maqbool},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2027}
}
```

## Contact

- Sibtain Syed — sibtainshah621@gmail.com
- Ali Ahmed Qureshi — aliahmadqureshi041@gmail.com
- Maqbool Khan — maqbool.khan@aubh.edu.bh

## Acknowledgements

We thank the authors of Mudslide (CVPR 2024), [Hover-Net](https://github.com/vqdang/hover_net), CPM-17, and [NuInsSeg](https://github.com/masih4/NuInsSeg) for datasets and baselines.

## License

This repository is released under the [MIT License](LICENSE). Dataset licenses are those of their original publishers; see the dataset READMEs.
