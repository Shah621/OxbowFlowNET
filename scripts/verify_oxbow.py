#!/usr/bin/env python3
"""Check that synthesized flow converges into a basin and that background drifts +x."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from oxbow_lib import synthesize_phi_alpha_from_mask, compute_v_gt_from_phi


def verify():
    H, W = 100, 100
    mask = np.zeros((H, W), dtype=np.int32)
    yy, xx = np.mgrid[:H, :W]
    mask[np.sqrt((yy - 50) ** 2 + (xx - 50) ** 2) <= 20] = 1

    phi, alpha = synthesize_phi_alpha_from_mask(mask)
    vx, vy = compute_v_gt_from_phi(phi, alpha)

    v_right = vx[50, 60]
    print(f"Velocity at (50, 60) [right of centre]: vx={v_right:.4f}")
    print("PASS: flow points left." if v_right < -0.5 else "FAIL: flow does not point towards the centre.")

    v_top = vy[40, 50]
    print(f"Velocity at (40, 50) [above centre]: vy={v_top:.4f}")
    print("PASS: flow points down." if v_top > 0.5 else "FAIL: flow does not point towards the centre.")

    v_bg_x, v_bg_y = vx[10, 10], vy[10, 10]
    print(f"Background velocity at (10, 10): vx={v_bg_x:.4f}, vy={v_bg_y:.4f}")
    print("PASS: background has +x drift." if v_bg_x > 0.1 else "FAIL: background drift missing or reversed.")


if __name__ == "__main__":
    verify()
