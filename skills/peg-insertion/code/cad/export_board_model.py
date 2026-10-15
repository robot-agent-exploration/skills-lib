"""Export the NIST ATB M1 board/peg geometry (metres, board frame) to JSON for the controller.

Board frame: origin at the board centre on its top face, x along the 320 mm side, y along the
250 mm side (engraved text at +y), z up. Holes: 5 shapes (columns) x 4 clearance levels (rows).

Offline tool (no robot). Input: nist_atb_m1.stl (units cm) of the ManipulationNet peg-in-hole board,
https://github.com/ManipulationNet/mnet_client, branch ros_2, assets/peg_in_hole/cad_files/
(2.7 MB, not shipped with the skill). Usage: python3 export_board_model.py nist_atb_m1.stl board_model.json
The shipped ../peg_insertion/board_model.json also has per-peg 'outline' / 'area_m2' entries, added in the session by a
one-off command that sliced each peg STL of the same folder at z = 2.5 cm (see SKILL.md, ## Code).
"""
import json
import sys

import cv2
import numpy as np

from hole_fit import holes_at, polygon_yaw
from stl_info import read_stl
from stl_slice import chain, slice_mesh
from tri_fit import corners

COLS = {'circle': -12.78, 'hexagon': -6.76, 'triangle': 0.0, 'rectangle': 6.09, 'lshape': 11.9}
ROWS = {1: -9.5, 2: -3.87, 3: 1.25, 4: 6.5}
CLEAR_MM = {1: 3.0, 2: 1.0, 3: 0.1, 4: 0.02}
SYM = {'hexagon': 6, 'triangle': 3, 'rectangle': 2, 'circle': 0, 'lshape': 1}


def area_centroid(l):
    x, y = l[:, 0], l[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2
    return np.array([((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)]), abs(A)


def simplify(l, eps_cm=0.01):
    c = (l * 1e4).astype(np.float32).reshape(-1, 1, 2)
    return cv2.approxPolyDP(c, eps_cm * 1e4, True).reshape(-1, 2) / 1e4


if __name__ == '__main__':
    board_path, out = sys.argv[1], sys.argv[2]
    board = read_stl(board_path)
    v = board.reshape(-1, 3)
    top = float(v[:, 2].max())
    L = holes_at(board, top - 0.05)
    deep = [h for h in np.arange(0.0, top, 0.02) if len(holes_at(board, h)) >= 20]
    model = {'units': 'm', 'frame': 'board: origin top-face centre, x along 320 mm side, y along 250 mm side (text at +y), z up',
             'size': [float(v[:, 0].max() - v[:, 0].min()) / 100, float(v[:, 1].max() - v[:, 1].min()) / 100, top / 100],
             'hole_depth': round((top - min(deep)) / 100, 4), 'holes': {}}
    for name, cx in COLS.items():
        for lvl, cy in ROWS.items():
            l = [l for l in L if abs(l[:, 0].mean() - cx) < 1.5 and abs(l[:, 1].mean() - cy) < 1.5][0]
            c, A = area_centroid(l)
            h = {'shape': name, 'level': lvl, 'clearance_mm': CLEAR_MM[lvl],
                 'centre': (c / 100).round(5).tolist(), 'area_m2': A / 1e4,
                 'outline': (simplify(l) / 100 - 0).round(5).tolist()}
            if name == 'hexagon':
                h['edge_dir_deg_mod60'] = round(polygon_yaw(l, 6)[0], 2)
            if name == 'rectangle':
                rect = cv2.minAreaRect((l * 1e4).astype(np.float32))
                h['rect_size'] = [rect[1][0] / 1e6, rect[1][1] / 1e6]
                h['edge_dir_deg_mod90'] = round(polygon_yaw(l, 4)[0], 2)
                # long-axis direction
                pts = l - c
                w, V = np.linalg.eigh(pts.T @ pts)
                h['long_axis_deg'] = round(float(np.degrees(np.arctan2(V[1, 1], V[0, 1])) % 180), 2)
            if name == 'triangle':
                h['edge_dir_deg_mod120'] = round(polygon_yaw(l, 3)[0], 2)
                cc, cs = corners(l)
                h['apex_dirs_deg'] = [a for a, r in cs]
            model['holes'][f'{name}_L{lvl}'] = h
    model['pegs'] = {
        'hexagon': {'across_flats': 0.01701, 'across_corners': 0.01932, 'length': 0.05},
        'rectangle': {'long': 0.016, 'short': 0.012, 'length': 0.05},
        'triangle': {'height': 0.014, 'side_bbox': 0.01586, 'inradius': 0.00497, 'length': 0.05},
        'cylinder': {'diameter': 0.012, 'length': 0.05},
    }
    json.dump(model, open(out, 'w'), indent=1)
    print('top', top, 'depth', model['hole_depth'], 'size', model['size'])
    for k, h in model['holes'].items():
        if h['level'] == 1:
            print(k, {kk: vv for kk, vv in h.items() if kk not in ('outline',)})
