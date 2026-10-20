# OxbowFlow: Continuous Flow Decomposition for Nuclear Instance Segmentation

**[CVPR 2027]** Official PyTorch implementation.

> Sibtain Syed<sup>1</sup>, Ali Ahmed Qureshi<sup>1</sup>, Maqbool Khan<sup>2</sup><br>
> <sup>1</sup>Comsat University Islamabad, Wah, Pakistan<br>
> <sup>2</sup>American University of Bahrain, Riffa, Bahrain

A ResNet-encoder U-Net predicts foreground $f$, potential $\varphi$, and meander $\alpha$. Pixels follow

$$
\tilde{v} = -\hat{n} + \alpha\, R\hat{n},\qquad v = \tilde{v}/\lVert\tilde{v}\rVert,
\qquad \hat{n}=\nabla\varphi/\lVert\nabla\varphi\rVert,
$$

where $R$ is a $90^\circ$ rotation. Trajectories meander, collide on the $\alpha$ ridge, and enclose each nucleus. Train fully-supervised (`fully_supervised/`) or point-supervised (`point_supervised/`) on **CPM17** and **NuInsSeg**; metrics are DQ, SQ, PQ, and AJI.

## Installation

```bash
pip install -r requirements.txt
python scripts/verify_oxbow.py   # optional sanity check
```

Python 3.9+, PyTorch, torchvision. CUDA is used when available; pass `--cpu` otherwise.

## Datasets

Images are not included. See [`cpm17/README.md`](cpm17/README.md) and [`nuinseg/README.md`](nuinseg/README.md).

```bash
python scripts/prepare_nuinseg.py --src nuinseg --dst nuinseg_prepared --val-frac 0.2 --seed 42
```

## Training

```bash
python fully_supervised/train.py --data-dir cpm17/train --arch resunet34 --crops-per-image 12 --epochs 150 --batch-size 4
python fully_supervised/train.py --data-dir nuinseg_prepared/train --arch resunet34 --crops-per-image 4 --epochs 40 --batch-size 4 \
    --work-dir fully_supervised/checkpoints_nuinseg

python point_supervised/train.py --data-dir cpm17/train --arch resunet34 --crops-per-image 12 --epochs 60 --batch-size 4
python point_supervised/train.py --data-dir nuinseg_prepared/train --arch resunet18 --crops-per-image 5 --epochs 25 --batch-size 4 \
    --work-dir point_supervised/checkpoints_nuinseg
```

Validation is a 10% split of train (`--val-frac 0.1`, seed 42). Best weights: `*/checkpoints/oxbow_best.pth`.

## Evaluation

```bash
python fully_supervised/evaluation.py --checkpoint fully_supervised/checkpoints/oxbow_best.pth \
    --test-dir cpm17/test --dataset-name CPM17 --separator oxbow

python point_supervised/evaluation.py --checkpoint point_supervised/checkpoints/oxbow_best.pth \
    --test-dir cpm17/test --dataset-name CPM17 --separator oxbow --beta 0
```

Sliding-window + flip TTA by default. Overlays go to `*/predictions/`; `--raw-mask-dir` writes instance maps. For $\mathrm{P}$-$\mathrm{Geo}$ use `--beta 0`.

## Citation

```bibtex
@inproceedings{syed2027oxbowflow,
  title     = {OxbowFlow: Continuous Flow Decomposition for Nuclear Instance Segmentation},
  author    = {Syed, Sibtain and Qureshi, Ali Ahmed and Khan, Maqbool},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2027}
}
```

MIT License. Dataset licenses are those of their original publishers.
