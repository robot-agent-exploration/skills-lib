"""Board model (from public CAD) and board-frame <-> base-frame transforms."""
import json
from pathlib import Path

import numpy as np

MODEL = json.loads((Path(__file__).with_name('board_model.json')).read_text())


def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def board_to_base(pose, pts_b, z_top):
    """pose = (cx, cy, theta): board-frame xy points (N,2) -> base-frame (N,3) at height z_top."""
    cx, cy, th = pose
    pts_b = np.atleast_2d(np.asarray(pts_b, float))
    xy = pts_b @ rot2(th).T + np.array([cx, cy])
    return np.column_stack([xy, np.full(len(xy), z_top)])


def hole(name):
    return MODEL['holes'][name]


def hole_outline_base(pose, name, z_top):
    return board_to_base(pose, hole(name)['outline'], z_top)


def board_outline_base(pose, z_top):
    sx, sy = MODEL['size'][0] / 2, MODEL['size'][1] / 2
    return board_to_base(pose, [[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]], z_top)
