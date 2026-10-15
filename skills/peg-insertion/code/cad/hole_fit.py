"""Hole depth and precise orientation fits for the NIST ATB M1 board."""
import sys

import cv2
import numpy as np

from stl_info import read_stl
from stl_slice import chain, slice_mesh


def holes_at(board, h):
    loops = chain(slice_mesh(board, h))
    return [L for L in loops if 1.0 < (L.max(0) - L.min(0)).min() and (L.max(0) - L.min(0)).max() < 3.5]


def polygon_yaw(L, n):
    """Orientation of an n-fold symmetric polygon: angle (deg, mod 360/n) of the edge normals,
    from the 'n-th harmonic' of the outline's edge directions weighted by length."""
    e = np.roll(L, -1, 0) - L
    w = np.linalg.norm(e, axis=1)
    a = np.arctan2(e[:, 1], e[:, 0])
    z = np.sum(w * np.exp(1j * n * a))
    return np.degrees(np.angle(z) / n), abs(z) / w.sum()


if __name__ == '__main__':
    board = read_stl(sys.argv[1])
    for h in np.arange(0.05, 2.5, 0.1):
        print(f'z={h:.2f} holes={len(holes_at(board, h))}')
    L = holes_at(board, 2.4)
    cols = {'circle': -12.78, 'hexagon': -6.76, 'triangle': 0.0, 'rectangle': 6.09, 'L': 11.9}
    rows = {1: -9.5, 2: -3.87, 3: 1.25, 4: 6.5}
    for name, cx in cols.items():
        for lvl, cy in rows.items():
            cand = [l for l in L if abs(l[:, 0].mean() - cx) < 1.5 and abs(l[:, 1].mean() - cy) < 1.5]
            if not cand:
                continue
            l = cand[0]
            c = l.mean(0)
            # area centroid
            x, y = l[:, 0], l[:, 1]
            x1, y1 = np.roll(x, -1), np.roll(y, -1)
            cr = x * y1 - x1 * y
            A = cr.sum() / 2
            cxa = ((x + x1) * cr).sum() / (6 * A)
            cya = ((y + y1) * cr).sum() / (6 * A)
            rect = cv2.minAreaRect((l * 1000).astype(np.float32))
            (rx, ry), (rw, rh), ra = rect
            info = f'{name:9s} L{lvl} centroid=({cxa:7.3f},{cya:7.3f}) area={abs(A):.3f} minAreaRect=({rw/1000:.3f}x{rh/1000:.3f} @ {ra:.1f}deg)'
            if name == 'hexagon':
                info += ' edge_dir mod60=%.2f (q=%.2f)' % polygon_yaw(l, 6)
            if name == 'triangle':
                info += ' edge_dir mod120=%.2f (q=%.2f)' % polygon_yaw(l, 3)
            if name == 'rectangle':
                info += ' edge_dir mod90=%.2f (q=%.2f)' % polygon_yaw(l, 4)
            print(info)
