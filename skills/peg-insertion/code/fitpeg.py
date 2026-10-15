"""CLI/helper: fit a peg top face in a wrist frame, both polarities; save an overlay crop."""
import sys

import cv2
import numpy as np

from board import MODEL
from geom import load_frame, project, rgb
from pegvision import _outline_with_normals, fit_peg_edges
from plan import PERIOD


def fit_both(frame_id, shape, init_xy, z_top, span=0.006, out=None):
    f = load_frame(frame_id)
    o = MODEL['pegs'][shape]['outline']
    res = []
    for pol in (1.0, -1.0):
        r = fit_peg_edges(f, 'wrist', o, init_xy, z_top, PERIOD[shape], span_xy=span, polarity=pol)
        r['polarity'] = pol
        r['margin'] = r['score'] / max(r['alt_score'] or 1e-6, 1e-6)
        res.append(r)
    best = max(res, key=lambda r: r['score'])
    if out:
        W = rgb(f, 'wrist')
        pts, _ = _outline_with_normals(o)
        c, s = np.cos(best['yaw']), np.sin(best['yaw'])
        P = pts @ np.array([[c, -s], [s, c]]).T + best['centre']
        uv = project(f, 'wrist', np.column_stack([P, np.full(len(P), z_top)])).astype(int)
        for u, v in uv:
            if 0 <= v < W.shape[0] and 0 <= u < W.shape[1]:
                W[v, u] = (0, 0, 255)
        u0, v0 = np.clip(uv.mean(0).astype(int), 80, [1200, 640])
        cv2.imwrite(out, cv2.resize(W[max(0, v0 - 80):v0 + 80, u0 - 80:u0 + 80], None, fx=3, fy=3,
                                    interpolation=cv2.INTER_NEAREST))
    return best, res


if __name__ == '__main__':
    # run from the workspace root: python3 <skill>/code/fitpeg.py FRAME_ID SHAPE X Y Z_TOP
    # (the session ran it from code/peg_insertion and wrote to ../../output/inspect, the same folder)
    from pathlib import Path
    Path('output/inspect').mkdir(parents=True, exist_ok=True)
    fid, shape, x, y, z = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:6])
    best, res = fit_both(fid, shape, (x, y), z, out=f'output/inspect/fit_{shape}_{fid}.jpg')
    for r in res:
        print(f"pol {r['polarity']:+.0f}: centre ({r['centre'][0]:.5f},{r['centre'][1]:.5f}) yaw "
              f"{np.degrees(r['yaw']):.2f} score {r['score']:.1f} alt {r['alt_score']:.1f} @ {np.degrees(r['alt_yaw']):.0f}")
    print('BEST', best['polarity'], np.round(best['centre'], 5).tolist(), round(np.degrees(best['yaw']), 2))
