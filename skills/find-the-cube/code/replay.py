"""Replay a saved shuffle recording through shuffle.identify (no robot needed).

    python <skill>/code/replay.py <recording_dir> <observation_frame_id> [sheet.jpg]   (from the workspace root)

<recording_dir> holds wrist <t>.jpg (+ optional <t>_depth.png) and meta.json with t0, t1,
cube_px; the observation frame (runtime/frames/<id>) supplies the wrist camera K and pose.
"""
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.cwd()))                       # workspace root (runtime/frames/)
sys.path.insert(0, str(HERE))
import controller as C  # noqa: E402
import shuffle  # noqa: E402


def load_recording(d):
    d = Path(d)
    meta = json.loads((d / 'meta.json').read_text())
    rgb = sorted((int(p.stem), p) for p in d.glob('*.jpg'))
    depth = sorted((int(p.name.split('_')[0]), p) for p in d.glob('*_depth.png'))
    keep = lambda t: meta['t0'] <= t <= meta['t1']
    return meta, [f for f in rgb if keep(f[0])], [f for f in depth if keep(f[0])]


def camera(frame_id):
    m = json.loads((C.ROOT / 'runtime' / 'frames' / frame_id / 'meta.json').read_text())['cameras']['wrist']
    return np.array(m['K']), np.array(m['T_base_cam'])


def detector(K, T):
    return lambda image, depth: C.find_cube(dict(image=image, depth=depth, K=K, T=T))


def object_model(meta, frame_id):
    """The run's saved object model, or one rebuilt from its start observation."""
    if meta.get('model'):
        return meta['model']
    d = C.ROOT / 'runtime' / 'frames' / frame_id
    m = json.loads((d / 'meta.json').read_text())['cameras']['wrist']
    fr = dict(image=cv.imread(str(d / 'wrist.jpg')), K=np.array(m['K']), T=np.array(m['T_base_cam']),
              depth=cv.imread(str(d / 'wrist_depth.png'), cv.IMREAD_UNCHANGED).astype(float) / 1000.0)
    cube = C.find_cube(fr)
    return None if cube is None else {'size_mm': cube['size_mm'], 'area': cube['area'], 'lab': cube['lab'],
                                      'height': cube['top_z'] - cube['table_z']}


if __name__ == '__main__':
    rec, frame_id = sys.argv[1], sys.argv[2]
    meta, rgb, depth = load_recording(rec)
    K, T = camera(frame_id)
    model = object_model(meta, frame_id)
    n_cups = meta.get('n_cups') or C.count_cups(cv.imread(str(rgb[0][1])))
    print('model', model, 'cups', n_cups)
    t = time.time()
    r = shuffle.identify(rgb, depth, meta['cube_px'], detector(K, T), C.ROI, n_cups, model=model)
    print(json.dumps({k: v for k, v in r.items() if k != 'hist'}, default=str))
    print(f'{len(rgb)} rgb, {len(depth)} depth frames, {time.time() - t:.1f} s')
    if len(sys.argv) > 3:
        shuffle.save_sheet(rgb, r, sys.argv[3], C.ROI)
