"""Slice an STL mesh at z=h and print closed cross-section loops (holes) with stats."""
import sys

import numpy as np

from stl_info import read_stl


def slice_mesh(tri, h):
    segs = []
    for t in tri:
        z = t[:, 2] - h
        pts = []
        for i in range(3):
            a, b = t[i], t[(i + 1) % 3]
            za, zb = z[i], z[(i + 1) % 3]
            if (za < 0) != (zb < 0):
                s = za / (za - zb)
                pts.append(a + s * (b - a))
        if len(pts) == 2:
            segs.append((pts[0][:2], pts[1][:2]))
    return segs


def chain(segs, tol=1e-4):
    segs = [tuple(map(tuple, np.round(s, 5))) for s in segs]
    from collections import defaultdict
    adj = defaultdict(list)
    for i, (a, b) in enumerate(segs):
        adj[a].append((i, b))
        adj[b].append((i, a))
    used = set()
    loops = []
    for i, (a, b) in enumerate(segs):
        if i in used:
            continue
        used.add(i)
        loop = [a, b]
        cur = b
        while True:
            nxt = None
            for j, p in adj[cur]:
                if j not in used:
                    nxt = (j, p)
                    break
            if nxt is None:
                break
            used.add(nxt[0])
            cur = nxt[1]
            if cur == loop[0]:
                break
            loop.append(cur)
        loops.append(np.array(loop))
    return loops


if __name__ == '__main__':
    path, h = sys.argv[1], float(sys.argv[2])
    tri = read_stl(path)
    loops = chain(slice_mesh(tri, h))
    print(len(loops), 'loops at z =', h)
    for L in sorted(loops, key=lambda L: (round(L[:, 1].mean(), 0), L[:, 0].mean())):
        c = L.mean(0)
        x = L - c
        cov = x.T @ x / len(x)
        w, V = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        ext = L.max(0) - L.min(0)
        # polygon area (shoelace)
        area = 0.5 * abs(np.dot(L[:, 0], np.roll(L[:, 1], 1)) - np.dot(L[:, 1], np.roll(L[:, 0], 1)))
        print(f'  n={len(L):4d} centre=({c[0]:7.3f},{c[1]:7.3f}) bbox=({ext[0]:.3f},{ext[1]:.3f}) area={area:.3f} major_axis_deg={ang:6.1f}')
