#!/usr/bin/env python3
"""OxbowFlow library: model, losses, data, inference, separation, and metrics."""

import colorsys
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.measure import label
from skimage.segmentation import find_boundaries, watershed
from skimage.feature import peak_local_max

import scipy.io as sio

# ----------------------------------------------------------------------------- constants
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MASK_EXTENSIONS = IMG_EXTENSIONS + (".mat", ".npy")
IMAGE_DIR_CANDIDATES = ("images", "Images", "imgs", "Imgs", "tissue images")
MASK_DIR_CANDIDATES = ("labels", "Labels", "label", "Label", "masks", "Masks",
                       "overlay", "Overlay", "label masks modify", "label masks")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================= IO / masks
def _resolve_dir(root: Path, candidates, kind: str, extensions) -> Path:
    root = Path(root)
    if root.is_dir():
        for ext in extensions:
            if any(root.glob(f"*{ext}")):
                return root
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate {kind} directory under {root}. Tried: {candidates}")


def _stack_to_instance_map(stack):
    """HxWxK binary stack -> HxW instance map."""
    stack = np.asarray(stack)
    present = stack.max(axis=2) > 0
    inst = (np.argmax(stack, axis=2) + 1).astype(np.int32)
    inst[~present] = 0
    return inst


def _load_instance_mask(path: Path, size=None):
    """Load an instance map; resize with NEAREST if ``size`` is set."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        mat = sio.loadmat(str(path))
        mask_np = None
        if "stacked_mask" in mat:
            mask_np = _stack_to_instance_map(mat["stacked_mask"])
        else:
            for key in ("inst_map", "instance_map", "label", "labels", "mask"):
                if key in mat:
                    mask_np = np.asarray(mat[key])
                    break
        if mask_np is None:
            raise KeyError(f"No instance map key found in {path.name}")
    elif suffix == ".npy":
        mask_np = np.load(path)
    else:
        with Image.open(path) as mask_img:
            mask_np = np.asarray(mask_img.convert("I"))
    mask_np = np.asarray(mask_np)
    if mask_np.ndim == 3:
        if mask_np.shape[2] > 4 or set(np.unique(mask_np)).issubset({0, 1}):
            mask_np = _stack_to_instance_map(mask_np)
        else:
            mask_np = mask_np[..., 0]
    mask_np = mask_np.astype(np.int32)
    if size is not None:
        mask_img = Image.fromarray(mask_np).resize(size, Image.NEAREST)
        mask_np = np.asarray(mask_img).astype(np.int32)
    return mask_np


def load_image_rgb(path, size=None):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize(size, Image.BILINEAR)
        return np.asarray(im).astype(np.uint8)


def build_pairs(image_dir: Path, label_dir: Path):
    image_dir, label_dir = Path(image_dir), Path(label_dir)
    image_paths = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS])
    label_map = {p.stem: p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS}
    pairs, missing = [], []
    for img in image_paths:
        lbl = label_map.get(img.stem)
        (pairs if lbl else missing).append((img, lbl) if lbl else img.name)
    if missing:
        print(f"Warning: {len(missing)} image(s) have no matching label. Examples: {missing[:3]}")
    if not pairs:
        raise RuntimeError(f"No image/label pairs found in {image_dir} / {label_dir}")
    return pairs


def discover_pairs(root, image_dir=None, label_dir=None):
    root = Path(root)
    img_dir = Path(image_dir) if image_dir else _resolve_dir(root, IMAGE_DIR_CANDIDATES, "image", IMG_EXTENSIONS)
    lbl_dir = Path(label_dir) if label_dir else _resolve_dir(root, MASK_DIR_CANDIDATES, "label/mask", MASK_EXTENSIONS)
    return build_pairs(img_dir, lbl_dir)


def normalize_to_chw(rgb_uint8):
    """uint8 HxWx3 -> ImageNet-normalized CxHxW float32."""
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# ============================================================================= phi / alpha
def synthesize_phi_alpha_from_mask(mask):
    """Full-mask targets: basin per instance (low at centroid), alpha high on rims."""
    H, W = mask.shape
    yy, xx = np.mgrid[:H, :W]
    xx_norm = xx / max(W, 1)
    phi = 2.0 - 0.5 * xx_norm
    alpha = np.zeros((H, W), dtype=np.float32)
    for lab in np.unique(mask):
        if lab == 0:
            continue
        m = (mask == lab)
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        cy, cx = ys.mean(), xs.mean()
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        radius = np.sqrt(len(ys) / np.pi)
        phi_obj = d / (radius + 1e-6) - 1.0
        phi = np.where(m, phi_obj, phi)
        bord = ndi.binary_dilation(m, iterations=3) & (~ndi.binary_erosion(m, iterations=1))
        alpha = np.where(bord, 0.6, alpha)
    return phi.astype(np.float32), alpha.astype(np.float32)


def synthesize_phi_alpha_from_points(mask, radius=15.0):
    """Centroid targets: basins around instance centres; alpha = 0."""
    H, W = mask.shape
    yy, xx = np.mgrid[:H, :W]
    phi = 2.0 - 0.5 * (xx / max(W, 1))
    alpha = np.zeros((H, W), dtype=np.float32)
    pts = []
    for lab in np.unique(mask):
        if lab == 0:
            continue
        ys, xs = np.nonzero(mask == lab)
        if len(ys):
            pts.append((ys.mean(), xs.mean()))
    if not pts:
        return phi.astype(np.float32), alpha
    dots = np.zeros((H, W), dtype=bool)
    for cy, cx in pts:
        yi, xi = int(cy), int(cx)
        if 0 <= yi < H and 0 <= xi < W:
            dots[yi, xi] = True
    dist = ndi.distance_transform_edt(~dots)
    phi = np.minimum(phi, dist / radius - 1.0)
    return phi.astype(np.float32), alpha


def compute_v_gt_from_phi(phi_np, alpha_np=None):
    gx = ndi.sobel(phi_np, axis=1, mode="nearest") / 8.0
    gy = ndi.sobel(phi_np, axis=0, mode="nearest") / 8.0
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
    nvx, nvy = gx / mag, gy / mag
    vx, vy = -nvx, -nvy
    if alpha_np is not None:
        vx, vy = vx + alpha_np * (-nvy), vy + alpha_np * nvx
        m = np.sqrt(vx ** 2 + vy ** 2) + 1e-8
        return vx / m, vy / m
    return vx, vy


# ============================================================================= models
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(i, o))

    def forward(self, x):
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.up = nn.ConvTranspose2d(i, i // 2, 2, stride=2)
        self.conv = DoubleConv(i, o)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OxbowUNet(nn.Module):
    """Small from-scratch U-Net; output channels are (fg, phi, alpha)."""
    def __init__(self, in_ch=3, base=32, out_ch=3):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = _Down(base, base * 2)
        self.down2 = _Down(base * 2, base * 4)
        self.up1 = _Up(base * 4, base * 2)
        self.up2 = _Up(base * 2, base)
        self.outconv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x = self.up1(x3, x2); x = self.up2(x, x1)
        return self.outconv(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__(); self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResUNet(nn.Module):
    """U-Net with an optional ImageNet-pretrained ResNet encoder. Output: (fg, phi, alpha)."""
    def __init__(self, encoder="resnet34", pretrained=True, out_ch=3):
        super().__init__()
        from torchvision import models
        weights = None
        if pretrained:
            try:
                wname = {"resnet18": "ResNet18_Weights", "resnet34": "ResNet34_Weights",
                         "resnet50": "ResNet50_Weights"}[encoder]
                weights = getattr(models, wname).IMAGENET1K_V1
            except Exception:
                weights = None
        net = getattr(models, encoder)(weights=weights)
        self.conv1, self.bn1, self.relu, self.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        with torch.no_grad():
            d = torch.zeros(1, 3, 64, 64)
            x1 = self.relu(self.bn1(self.conv1(d)))
            c1 = self.layer1(self.maxpool(x1)); c2 = self.layer2(c1)
            c3 = self.layer3(c2); c4 = self.layer4(c3)
            x1c, c1c, c2c, c3c, c4c = (t.shape[1] for t in (x1, c1, c2, c3, c4))
        self.dec4 = _DecoderBlock(c4c, c3c, c3c)
        self.dec3 = _DecoderBlock(c3c, c2c, c2c)
        self.dec2 = _DecoderBlock(c2c, c1c, c1c)
        self.dec1 = _DecoderBlock(c1c, x1c, 64)
        self.dec0 = DoubleConv(64, 32)
        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        in_hw = x.shape[-2:]
        x1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.layer1(self.maxpool(x1))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        d = self.dec4(c4, c3)
        d = self.dec3(d, c2)
        d = self.dec2(d, c1)
        d = self.dec1(d, x1)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec0(d)
        if d.shape[-2:] != in_hw:
            d = F.interpolate(d, size=in_hw, mode="bilinear", align_corners=False)
        return self.head(d)


def build_model(arch="resunet34", pretrained=True, base_channels=32):
    arch = arch.lower()
    if arch == "unet":
        return OxbowUNet(in_ch=3, base=base_channels, out_ch=3)
    if arch in ("resunet18", "resunet34", "resunet50"):
        return ResUNet(encoder=arch.replace("resunet", "resnet"), pretrained=pretrained, out_ch=3)
    raise ValueError(f"Unknown arch '{arch}'")


# ============================================================================= losses
def _sobel(device, dtype):
    sx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=dtype, device=device) / 8.0
    sy = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=dtype, device=device) / 8.0
    return sx, sy


def flow_from_phi_alpha(phi, alpha, eps=1e-8):
    sx, sy = _sobel(phi.device, phi.dtype)
    gx = F.conv2d(phi, sx, padding=1); gy = F.conv2d(phi, sy, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + eps); nx, ny = gx / mag, gy / mag
    vx = -nx + alpha * (-ny); vy = -ny + alpha * nx
    vmag = torch.sqrt(vx ** 2 + vy ** 2 + eps)
    return vx / vmag, vy / vmag


def _divergence(vx, vy):
    dxk = torch.tensor([[[[-1, 0, 1]]]], dtype=vx.dtype, device=vx.device)
    dyk = torch.tensor([[[[-1], [0], [1]]]], dtype=vy.dtype, device=vy.device)
    return F.conv2d(vx, dxk, padding=(0, 1)) + F.conv2d(vy, dyk, padding=(1, 0))


def _curl(vx, vy):
    dxk = torch.tensor([[[[-1, 0, 1]]]], dtype=vx.dtype, device=vx.device)
    dyk = torch.tensor([[[[-1], [0], [1]]]], dtype=vy.dtype, device=vy.device)
    return F.conv2d(vy, dxk, padding=(0, 1)) - F.conv2d(vx, dyk, padding=(1, 0))


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=[1, 2, 3]) + eps
    den = p.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3]) + eps
    return (1 - num / den).mean()


def bce_dice_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def mse_msge_loss(pred, target, w_grad=1.0):
    l = F.mse_loss(pred, target)
    sx, sy = _sobel(pred.device, pred.dtype)
    l += w_grad * (F.mse_loss(F.conv2d(pred, sx, padding=1), F.conv2d(target, sx, padding=1))
                   + F.mse_loss(F.conv2d(pred, sy, padding=1), F.conv2d(target, sy, padding=1)))
    return l


DEFAULT_WEIGHTS = {"fg": 1.0, "phi": 1.0, "v": 1.0, "div": 0.05, "curv": 0.05}


def compute_losses(out, mask, phi_gt, alpha_gt, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    fg, phi_p, al_p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    l_fg = bce_dice_loss(fg, mask)
    l_phi = mse_msge_loss(phi_p, phi_gt)
    with torch.no_grad():
        vgx, vgy = flow_from_phi_alpha(phi_gt, alpha_gt)
    vpx, vpy = flow_from_phi_alpha(phi_p, al_p)
    l_v = F.mse_loss(vpx, vgx) + F.mse_loss(vpy, vgy)
    l_div = _divergence(vpx, vpy).abs().mean()
    l_curv = _curl(vpx, vpy).abs().mean()
    total = (weights["fg"] * l_fg + weights["phi"] * l_phi + weights["v"] * l_v
             + weights["div"] * l_div + weights["curv"] * l_curv)
    stats = {"loss": total.item(), "fg": l_fg.item(), "phi": l_phi.item(),
             "v": l_v.item(), "div": l_div.item(), "curv": l_curv.item()}
    return total, stats


# ============================================================================= augmentation
def _augment_geometric(img, inst, rng):
    if rng.rand() < 0.5:
        img, inst = img[:, ::-1], inst[:, ::-1]
    if rng.rand() < 0.5:
        img, inst = img[::-1], inst[::-1]
    k = rng.randint(0, 4)
    if k:
        img, inst = np.rot90(img, k), np.rot90(inst, k)
    if rng.rand() < 0.5:
        img, inst = np.swapaxes(img, 0, 1), np.swapaxes(inst, 0, 1)
    return np.ascontiguousarray(img), np.ascontiguousarray(inst)


def _augment_photometric(img, rng):
    x = img.astype(np.float32)
    x *= rng.uniform(0.85, 1.15)
    m = x.mean(); x = (x - m) * rng.uniform(0.85, 1.15) + m
    g = x.mean(axis=2, keepdims=True); x = (x - g) * rng.uniform(0.8, 1.2) + g
    x *= rng.uniform(0.9, 1.1, size=3).astype(np.float32)
    if rng.rand() < 0.3:
        x += rng.normal(0, rng.uniform(2, 8), x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.rand() < 0.2:
        x = np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1))))
    return x


def _scale_crop(img, inst, patch, rng, train):
    if train:
        s = rng.uniform(0.8, 1.2)
        if abs(s - 1.0) > 1e-3:
            H, W = inst.shape
            nh, nw = max(patch, int(round(H * s))), max(patch, int(round(W * s)))
            img = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
            inst = np.asarray(Image.fromarray(inst).resize((nw, nh), Image.NEAREST)).astype(np.int32)
    H, W = inst.shape
    if H < patch or W < patch:
        py, px = max(0, patch - H), max(0, patch - W)
        img = np.pad(img, ((0, py), (0, px), (0, 0)), mode="reflect")
        inst = np.pad(inst, ((0, py), (0, px)), mode="constant")
        H, W = inst.shape
    if train:
        y0 = rng.randint(0, H - patch + 1); x0 = rng.randint(0, W - patch + 1)
    else:
        y0, x0 = (H - patch) // 2, (W - patch) // 2
    return img[y0:y0 + patch, x0:x0 + patch], inst[y0:y0 + patch, x0:x0 + patch]


def split_pairs(pairs, val_frac=0.1, seed=42):
    """Split (image, mask) pairs into train/val by image."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs)); rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * len(pairs)))) if len(pairs) > 1 else 0
    val_idx = set(idx[:n_val].tolist())
    train = [p for i, p in enumerate(pairs) if i not in val_idx]
    val = [p for i, p in enumerate(pairs) if i in val_idx]
    return train, val


class PatchDataset(torch.utils.data.Dataset):
    """Native-resolution random crops; phi/alpha are synthesized after cropping."""
    def __init__(self, root=None, patch=256, crops_per_image=16, supervision="full",
                 augment=True, seed=0, image_dir=None, label_dir=None, pairs=None):
        self.pairs = pairs if pairs is not None else discover_pairs(root, image_dir, label_dir)
        self.patch = patch
        self.crops = max(1, crops_per_image)
        self.augment = augment
        self.synth = synthesize_phi_alpha_from_mask if supervision == "full" else synthesize_phi_alpha_from_points
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.pairs) * self.crops

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx % len(self.pairs)]
        img = load_image_rgb(img_path)
        inst = _load_instance_mask(mask_path, size=None)
        if inst.shape[:2] != img.shape[:2]:
            inst = np.asarray(Image.fromarray(inst).resize((img.shape[1], img.shape[0]), Image.NEAREST)).astype(np.int32)
        img, inst = _scale_crop(img, inst, self.patch, self.rng, self.augment)
        if self.augment:
            img, inst = _augment_geometric(img, inst, self.rng)
            img = _augment_photometric(img, self.rng)
        phi, alpha = self.synth(inst)
        return {
            "img": torch.from_numpy(normalize_to_chw(img)).float(),
            "mask": torch.from_numpy((inst > 0).astype(np.float32)[None]).float(),
            "phi": torch.from_numpy(phi[None]).float(),
            "alpha": torch.from_numpy(alpha[None]).float(),
        }


# ============================================================================= inference
def _starts(length, tile, stride):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)
    return s


def _hann2d(tile):
    w = np.hanning(tile)
    w = np.outer(w, w).astype(np.float32)
    return w + 1e-3


def sliding_window_predict(model, rgb_uint8, device, tile=256, stride=192):
    """Full-resolution (fg, phi, alpha) via tiled, Hann-weighted blending."""
    H, W = rgb_uint8.shape[:2]
    pad_h, pad_w = max(0, tile - H), max(0, tile - W)
    if pad_h or pad_w:
        rgb_uint8 = np.pad(rgb_uint8, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    Hp, Wp = rgb_uint8.shape[:2]
    win = _hann2d(tile)
    acc = np.zeros((3, Hp, Wp), dtype=np.float32)
    wsum = np.zeros((Hp, Wp), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for y in _starts(Hp, tile, stride):
            for x in _starts(Wp, tile, stride):
                t = rgb_uint8[y:y + tile, x:x + tile]
                xt = torch.from_numpy(normalize_to_chw(t)).unsqueeze(0).to(device)
                o = model(xt)[0].cpu().numpy()
                fg = 1.0 / (1.0 + np.exp(-o[0]))
                acc[0, y:y + tile, x:x + tile] += fg * win
                acc[1, y:y + tile, x:x + tile] += o[1] * win
                acc[2, y:y + tile, x:x + tile] += o[2] * win
                wsum[y:y + tile, x:x + tile] += win
    acc /= np.maximum(wsum, 1e-6)[None]
    return acc[0, :H, :W], acc[1, :H, :W], acc[2, :H, :W]


_TTA = [
    ("id", lambda a: a, lambda a: a),
    ("hflip", lambda a: a[:, ::-1], lambda a: a[:, ::-1]),
    ("vflip", lambda a: a[::-1], lambda a: a[::-1]),
    ("rot180", lambda a: np.rot90(a, 2), lambda a: np.rot90(a, 2)),
]


def _resize_field(a, hw):
    if a.shape[:2] == tuple(hw):
        return a
    return ndi.zoom(a, (hw[0] / a.shape[0], hw[1] / a.shape[1]), order=1)


def predict_fields(model, rgb_uint8, device, tile=256, stride=192, tta=True, scales=None, models=None):
    """Average fg/phi/alpha over TTA, optional scales, and an optional ensemble."""
    model_list = models if models is not None else [model]
    transforms = _TTA if tta else [_TTA[0]]
    scales = scales or [1.0]
    fg_a = phi_a = al_a = None
    n = 0
    for mdl in model_list:
        for _, fwd, inv in transforms:
            timg = np.ascontiguousarray(fwd(rgb_uint8))
            th, tw = timg.shape[:2]
            for s in scales:
                if abs(s - 1.0) < 1e-6:
                    simg = timg
                else:
                    nh, nw = max(tile, int(round(th * s))), max(tile, int(round(tw * s)))
                    simg = np.asarray(Image.fromarray(timg).resize((nw, nh), Image.BILINEAR))
                fg, phi, al = sliding_window_predict(mdl, simg, device, tile, stride)
                if simg.shape[:2] != (th, tw):
                    fg, phi, al = _resize_field(fg, (th, tw)), _resize_field(phi, (th, tw)), _resize_field(al, (th, tw))
                fg, phi, al = inv(fg), inv(phi), inv(al)
                if fg_a is None:
                    fg_a, phi_a, al_a = fg.copy(), phi.copy(), al.copy()
                else:
                    fg_a += fg; phi_a += phi; al_a += al
                n += 1
    return fg_a / n, phi_a / n, al_a / n


# ============================================================================= separation
def integrate_streamlines(phi, alpha, mask_roi, step_size=0.5, max_steps=80):
    H, W = phi.shape
    ys, xs = np.nonzero(mask_roi)
    if len(ys) == 0:
        return np.zeros((2, 0), np.float32), (ys, xs)
    gx = ndi.sobel(phi, axis=1, mode="nearest") / 8.0
    gy = ndi.sobel(phi, axis=0, mode="nearest") / 8.0
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
    nx, ny = gx / mag, gy / mag
    vx, vy = -nx + alpha * (-ny), -ny + alpha * nx
    vmag = np.sqrt(vx ** 2 + vy ** 2) + 1e-8
    vx, vy = vx / vmag, vy / vmag
    pts = np.stack([ys, xs], 0).astype(np.float32)

    def vel(p):
        yi = np.clip(np.round(p[0]), 0, H - 1).astype(np.int32)
        xi = np.clip(np.round(p[1]), 0, W - 1).astype(np.int32)
        return np.stack([vy[yi, xi], vx[yi, xi]], 0)

    for _ in range(max_steps):
        k1 = vel(pts); k2 = vel(pts + 0.5 * step_size * k1)
        k3 = vel(pts + 0.5 * step_size * k2); k4 = vel(pts + step_size * k3)
        pts = pts + (step_size / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        pts[0] = np.clip(pts[0], 0, H - 1); pts[1] = np.clip(pts[1], 0, W - 1)
    return pts, (ys, xs)


def _density_and_seeds(fg, phi, alpha, threshold, max_steps, smooth_sigma, min_distance, peak_rel):
    H, W = fg.shape
    binary = fg > threshold
    if binary.sum() == 0:
        return binary, None, None
    ep, _ = integrate_streamlines(phi, alpha, binary, step_size=0.5, max_steps=max_steps)
    yi = np.clip(np.round(ep[0]).astype(np.int64), 0, H - 1)
    xi = np.clip(np.round(ep[1]).astype(np.int64), 0, W - 1)
    density = np.zeros((H, W), np.float32); np.add.at(density, (yi, xi), 1.0)
    density = ndi.gaussian_filter(density, smooth_sigma)
    if density.max() <= 0:
        return binary, density, None
    peaks = peak_local_max(density, min_distance=min_distance,
                           threshold_abs=peak_rel * float(density.max()), exclude_border=False)
    return binary, density, peaks


def _cleanup(inst, min_size=0):
    """Drop instances smaller than min_size, fill holes, and relabel. No-op if min_size <= 0."""
    if not min_size or min_size <= 0:
        return inst
    out = np.zeros_like(inst)
    nid = 0
    for i in np.unique(inst):
        if i == 0:
            continue
        m = inst == i
        if m.sum() < min_size:
            continue
        m = ndi.binary_fill_holes(m)
        nid += 1
        out[m] = nid
    return out


def separate_instances(fg, phi, alpha, threshold=0.5, max_steps=80,
                       smooth_sigma=2.0, min_distance=4, peak_rel=0.20, min_size=0, **_):
    """Marker-controlled watershed seeded by flow-convergence density."""
    binary, density, peaks = _density_and_seeds(fg, phi, alpha, threshold, max_steps,
                                                smooth_sigma, min_distance, peak_rel)
    if density is None:
        return np.zeros(fg.shape, np.int32)
    if peaks is None or len(peaks) == 0:
        return _cleanup(ndi.label(binary)[0].astype(np.int32), min_size)
    markers = np.zeros(fg.shape, np.int32)
    for i, (py, px) in enumerate(peaks, 1):
        markers[py, px] = i
    return _cleanup(watershed(-density, markers, mask=binary).astype(np.int32), min_size)


def oxbow_collapse_separate(fg, phi, alpha, threshold=0.5, max_steps=80,
                            smooth_sigma=2.0, min_distance=4, peak_rel=0.20, beta=1.5,
                            min_size=0, **_):
    """Oxbow-collision separation: flow-seeded watershed plus a beta-weighted alpha ridge."""
    binary, density, peaks = _density_and_seeds(fg, phi, alpha, threshold, max_steps,
                                                smooth_sigma, min_distance, peak_rel)
    if density is None:
        return np.zeros(fg.shape, np.int32)
    if peaks is None or len(peaks) == 0:
        return _cleanup(ndi.label(binary)[0].astype(np.int32), min_size)
    markers = np.zeros(fg.shape, np.int32)
    for i, (py, px) in enumerate(peaks, 1):
        markers[py, px] = i
    a = np.abs(alpha).astype(np.float32)
    a = ndi.gaussian_filter(a, 1.0)
    if a.max() > 1e-6:
        a = a / a.max()
    dnorm = density / (density.max() + 1e-8)
    elevation = -dnorm + beta * a
    return _cleanup(watershed(elevation, markers, mask=binary).astype(np.int32), min_size)


def extract_instances(fg, phi, alpha, method="oxbow", threshold=0.5, **kw):
    if method == "watershed":
        return separate_instances(fg, phi, alpha, threshold=threshold, **kw)
    return oxbow_collapse_separate(fg, phi, alpha, threshold=threshold, **kw)


# ============================================================================= metrics
def _prepare_label_map(mask):
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.dtype == np.bool_ or mask.max() <= 1:
        return label(mask)
    return mask.astype(np.int32)


def compute_pq(true_mask, pred_mask, iou_thresh=0.5):
    tl, pl = _prepare_label_map(true_mask), _prepare_label_map(pred_mask)
    tids = [x for x in np.unique(tl) if x != 0]
    pids = [x for x in np.unique(pl) if x != 0]
    tp, sum_iou = 0, 0.0
    matched_pred = set()
    for ti in tids:
        tm = tl == ti; best_iou, best = 0.0, None
        for pj in pids:
            pm = pl == pj
            inter = np.logical_and(tm, pm).sum()
            union = np.logical_or(tm, pm).sum()
            iou = inter / (union + 1e-6)
            if iou > best_iou:
                best_iou, best = iou, pj
        if best_iou > iou_thresh:
            tp += 1; sum_iou += best_iou; matched_pred.add(best)
    fp = len(pids) - len(matched_pred); fn = len(tids) - tp
    dq = tp / (tp + 0.5 * (fp + fn) + 1e-6)
    sq = sum_iou / (tp + 1e-6) if tp > 0 else 0.0
    return dq, sq, dq * sq


def compute_aji(true_mask, pred_mask):
    tl, pl = _prepare_label_map(true_mask), _prepare_label_map(pred_mask)
    tids = np.unique(tl); tids = tids[tids != 0]
    pids = np.unique(pl); pids = pids[pids != 0]
    if len(tids) == 0 and len(pids) == 0:
        return 1.0
    if len(tids) == 0 or len(pids) == 0:
        return 0.0
    iou = np.zeros((len(tids), len(pids)), np.float32)
    for i, t in enumerate(tids):
        tm = tl == t
        for j, p in enumerate(pids):
            pm = pl == p
            inter = np.logical_and(tm, pm).sum(); union = np.logical_or(tm, pm).sum()
            if union > 0:
                iou[i, j] = inter / union
    ri, ci = linear_sum_assignment(-iou)
    inter_tot = union_tot = 0
    mt, mp = set(), set()
    for i, j in zip(ri, ci):
        if iou[i, j] > 0:
            tm = tl == tids[i]; pm = pl == pids[j]
            inter_tot += np.logical_and(tm, pm).sum(); union_tot += np.logical_or(tm, pm).sum()
            mt.add(i); mp.add(j)
    ut = [tl == tids[i] for i in range(len(tids)) if i not in mt]
    up = [pl == pids[j] for j in range(len(pids)) if j not in mp]
    if ut:
        union_tot += np.stack(ut).sum()
    if up:
        union_tot += np.stack(up).sum()
    return inter_tot / union_tot if union_tot > 0 else 0.0


# ============================================================================= visualization
def _instance_colors(n, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        s = 0.55 + 0.35 * rng.rand(); v = 0.75 + 0.25 * rng.rand()
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((r * 255.0, g * 255.0, b * 255.0))
    return out


def overlay_instances(orig_rgb, inst_map, fill_alpha=0.4, boundary_color=(255, 255, 0)):
    out = orig_rgb.astype(np.float32).copy()
    ids = [i for i in np.unique(inst_map) if i != 0]
    if ids:
        colors = _instance_colors(len(ids))
        for idx, iid in enumerate(ids):
            m = inst_map == iid
            out[m] = (1.0 - fill_alpha) * out[m] + fill_alpha * np.asarray(colors[idx], np.float32)
        out[find_boundaries(inst_map, mode="inner")] = np.asarray(boundary_color, np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================================= evaluation driver
def infer_dataset_name(test_dir):
    s = str(test_dir).lower()
    if "nuinseg" in s:
        return "NuInsSeg"
    if "cpm" in s:
        return "CPM17"
    return Path(test_dir).name or "dataset"


def load_model_from_checkpoint(path, device, arch=None, base_channels=32, pretrained=False):
    ckpt = torch.load(path, map_location=device)
    arch = ckpt.get("arch", arch or "resunet34")
    base_channels = ckpt.get("base_channels", base_channels)
    model = build_model(arch, pretrained=pretrained, base_channels=base_channels)
    model.load_state_dict(ckpt["net"])
    return model.to(device), arch


def load_models_from_checkpoints(paths, device, arch=None, base_channels=32):
    out = []
    for p in paths:
        m, _ = load_model_from_checkpoint(p, device, arch, base_channels)
        m.eval()
        out.append(m)
    return out


def evaluate(model, pairs, out_dir, device, separator="oxbow", threshold=0.5,
             tile=256, stride=192, tta=True, raw_mask_dir=None, fill_alpha=0.4,
             sep_kwargs=None, models=None, scales=None):
    from tqdm import tqdm
    sep_kwargs = sep_kwargs or {}
    os.makedirs(out_dir, exist_ok=True)
    raw_path = Path(raw_mask_dir) if raw_mask_dir else None
    if raw_path:
        raw_path.mkdir(parents=True, exist_ok=True)
    dq_l, sq_l, pq_l, aji_l, per = [], [], [], [], []
    for ip, mp in tqdm(pairs, desc="Evaluating"):
        rgb = load_image_rgb(ip)
        fg, phi, al = predict_fields(model, rgb, device, tile, stride, tta, scales=scales, models=models)
        pred = extract_instances(fg, phi, al, method=separator, threshold=threshold, **sep_kwargs)
        gt = _load_instance_mask(mp, size=None)
        if gt.shape != pred.shape:
            gt = np.asarray(Image.fromarray(gt).resize((pred.shape[1], pred.shape[0]), Image.NEAREST)).astype(np.int32)
        dq, sq, pq = compute_pq(gt, pred); aji = compute_aji(gt, pred)
        dq_l.append(dq); sq_l.append(sq); pq_l.append(pq); aji_l.append(aji)
        per.append((ip.name, dq, sq, pq, aji))
        Image.fromarray(overlay_instances(rgb, pred, fill_alpha)).save(Path(out_dir) / (ip.stem + ".png"))
        if raw_path:
            np.save(raw_path / (ip.stem + ".npy"), pred.astype(np.int32))
    if not dq_l:
        raise RuntimeError("No samples evaluated.")
    metrics = {"DQ": float(np.mean(dq_l)), "SQ": float(np.mean(sq_l)),
               "PQ": float(np.mean(pq_l)), "AJI": float(np.mean(aji_l)), "n_images": len(dq_l)}
    return metrics, per


def write_results(results_dir, supervision, dataset_name, metrics, per_image, method="OxbowFlow"):
    rd = Path(results_dir); rd.mkdir(parents=True, exist_ok=True)
    rec = {"method": method, "dataset": dataset_name, "supervision": supervision, **metrics}
    jp = rd / f"{supervision}_{dataset_name}.json"
    jp.write_text(json.dumps(rec, indent=2))
    cp = rd / f"{supervision}_{dataset_name}_per_image.csv"
    with open(cp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "DQ", "SQ", "PQ", "AJI"])
        for name, dq, sq, pq, aji in per_image:
            w.writerow([name, f"{dq:.4f}", f"{sq:.4f}", f"{pq:.4f}", f"{aji:.4f}"])
    return jp, cp
