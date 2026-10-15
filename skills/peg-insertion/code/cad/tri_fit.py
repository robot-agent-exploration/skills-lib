"""Triangle apex directions (hole/peg) and peg end-chamfer check."""
import sys

import numpy as np

from hole_fit import holes_at
from stl_info import read_stl
from stl_slice import chain, slice_mesh


def corners(L, k=3):
    c = L.mean(0)
    d = L - c
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    r = np.linalg.norm(d, axis=1)
    out = []
    used = np.zeros(len(L), bool)
    for _ in range(k):
        i = np.argmax(np.where(used, -1, r))
        out.append((round(float(ang[i]), 1), round(float(r[i]), 3)))
        used |= np.abs(((ang - ang[i] + 180) % 360) - 180) < 40
    return c, sorted(out)


if __name__ == '__main__':
    board = read_stl(sys.argv[1])
    L = holes_at(board, 2.4)
    for cy in (-9.5, -3.87, 1.25, 6.5):
        l = [l for l in L if abs(l[:, 0].mean()) < 1.5 and abs(l[:, 1].mean() - cy) < 1.5][0]
        c, cs = corners(l)
        print('triangle hole y=%.2f centre=(%.3f,%.3f) corners (deg, r cm):' % (cy, *c), cs)
    for p in sys.argv[2:]:
        tri = read_stl(p)
        for h in (0.02, 0.1, 0.3, 2.5, 4.7, 4.9, 4.98):
            loops = chain(slice_mesh(tri, h))
            if not loops:
                print(p.split('/')[-1], h, 'no loop'); continue
            l = loops[0]
            ext = l.max(0) - l.min(0)
            c, cs = corners(l, 3)
            print(p.split('/')[-1], f'z={h}', 'bbox', np.round(ext, 3), 'centre', np.round(c, 3), 'corners' if 'tri' in p else '', cs if 'tri' in p else '')
