"""Scene perception for the controller, from the two fixed exterior RGB-D cameras.

Measured 2026-10-10: exterior depth clusters give piece centres within ~3 mm in x and up to
~8 mm in y (depth fringes), and tops of rounded/thin pieces 5-15 mm low. So positions from
here are refined by the grasp itself (first grasp closes along y) and tower checks use
coarse presence thresholds, not exact heights.
"""
import base64
import json
import math

import cv2
import numpy as np

from . import exterior
from .motion import table_z_at
from .tower import BASE, SLOTS

from .params import HOME, MAX_JOINT_STEP, CLASS_SLACK, MAX_EXTENT, PRESENT_MIN_POINTS  # hyper-parameters: params.py

# piece height above the table (touch model) as seen by exterior depth (p97 of the consensus
# cluster), 2026-10-10 layouts: ring 13-20, barrel 38-42, urn 37-40, knob 49-51, goblet 55-62,
# dowel 70-104 mm (thin: its top is undersampled)
HEIGHT_CLASSES = [('ring', 0.008, 0.028), ('barrel|urn', 0.030, 0.0445), ('knob|slant', 0.0445, 0.054),
                  ('goblet|slant', 0.054, 0.066), ('dowel', 0.066, 0.140)]      # slant: 49-62 mm

# Presence at the tower slots: number of wood points within radius r of the slot axis that are
# higher than `level` above the base top. Calibrated on frames of every stack state (2026-10-10):
# present 246-3391 points, absent 0-4 (neighbour fringes stay below these levels).
PRESENCE = {'ring': (0.020, 0.008), 'barrel': (0.015, 0.025), 'goblet': (0.020, 0.040),
            'urn': (0.015, 0.076), 'dowel': (0.012, 0.050), 'knob': (0.012, 0.105),
            'slant': (0.015, 0.060)}     # slant: probed around its grasp axis, above the barrel top

def park(ctx, m):
    """Lift clear and go to the home joint pose (exterior cameras then see the whole table)."""
    s = m.status()
    if s['tcp_position'][2] < 0.28:
        m.lift(z=0.30)
        s = m.status()
    q = np.array(s['joints'])
    target = np.array(HOME)
    if np.max(np.abs(target - q)) < 0.02:
        return
    n = max(1, math.ceil(np.max(np.abs(target - q)) / MAX_JOINT_STEP))
    for i in range(1, n + 1):
        ctx.value('franka_move_joints', positions=[float(v) for v in q + (target - q) * i / n])


def snapshot(ctx):
    return ctx.value('franka_observe', cameras=['left', 'right'])['frame_id']


def wood_cloud(fid):
    return np.vstack([exterior.wood_points(fid, c) for c in ('left', 'right')])


def consensus_clusters(fid, cell=0.006, min_points=40):
    """Clusters of wood points in cells that both exterior cameras see (each camera's occupancy
    dilated by one cell). Stereo depth bridges pieces that touch in one camera's image (seen:
    dowel and barrel in line from the left camera merged into one 'lying' cluster); the other
    camera does not confirm the bridge, so it drops out here."""
    pts = [exterior.wood_points(fid, c) for c in ('left', 'right')]
    allp = np.vstack(pts)
    if len(allp) == 0:
        return []
    origin = allp[:, :2].min(0)
    ij_all = np.floor((allp[:, :2] - origin) / cell).astype(int)
    shape = ij_all.max(0) + 3
    occ = []
    for p in pts:
        g = np.zeros(shape, np.uint8)
        if len(p):
            ij = np.floor((p[:, :2] - origin) / cell).astype(int)
            np.add.at(g, (ij[:, 0], ij[:, 1]), 1)
        occ.append(cv2.dilate((g >= 2).astype(np.uint8), np.ones((3, 3), np.uint8)))
    both = occ[0] & occ[1]
    n, lab = cv2.connectedComponents(cv2.dilate(both, np.ones((2, 2), np.uint8)) & both, connectivity=8)
    labels = lab[ij_all[:, 0], ij_all[:, 1]]
    out = [allp[labels == k] for k in range(1, n)]
    return [c for c in out if len(c) >= min_points]

def in_base_area(xy, base=BASE, margin=0.015):
    dx, dy = xy[0] - base['center'][0], xy[1] - base['center'][1]
    a = base['axis']
    along, across = dx * a[0] + dy * a[1], -dx * a[1] + dy * a[0]
    return abs(along) < base['half_len'] + margin and abs(across) < base['half_wid'] + margin


def table_objects(fid, base=BASE):
    """Clusters standing on the table outside the base: xy, local table z, height, extent and
    candidate classes by height. (Exterior depth of the white table itself is 1-6 mm off, so
    the table height comes from the touch model.)"""
    out = []
    for c in consensus_clusters(fid):
        d = exterior.describe(c)
        xy = d['center_all']
        if in_base_area(xy, base):
            continue
        tz = table_z_at(xy)
        h = d['top_z'] - tz
        strict = [n for n, lo, hi in HEIGHT_CLASSES if lo <= h < hi]
        loose = [n for n, lo, hi in HEIGHT_CLASSES if lo - CLASS_SLACK <= h < hi + CLASS_SLACK]
        out.append(dict(xy=xy, table_z=tz, height=h, extent=d['extent_xy'], n=d['n'],
                        classes=strict[0].split('|') if strict else [],
                        possible=[p for c in loose for p in c.split('|')],
                        lying=max(d['extent_xy']) > MAX_EXTENT))
    return out


def slot_presence(cloud, state_xy, base=BASE):
    """{piece: (points above its level near its slot axis, present)}. state_xy gives each
    piece's tower xy (actual placement where known)."""
    out = {}
    for name in SLOTS:
        xy = state_xy[name]
        r, level = PRESENCE[name]
        d = np.hypot(cloud[:, 0] - xy[0], cloud[:, 1] - xy[1])
        n = int((cloud[d < r, 2] > base['top_z'] + level).sum())
        out[name] = (n, n >= PRESENT_MIN_POINTS)
    return out

def lean_pose(cloud, xy, table_z, grasp_h, r=0.035):
    """Axis xy at grasp height and lean direction (rad, bottom -> top) of a leaning piece from
    depth slices of both cameras (tested once: direction within ~20 deg)."""
    sel = cloud[np.hypot(cloud[:, 0] - xy[0], cloud[:, 1] - xy[1]) < r]
    top = np.percentile(sel[:, 2], 99)
    lo = sel[(sel[:, 2] > table_z + 0.008) & (sel[:, 2] < table_z + 0.022), :2]
    hi = sel[(sel[:, 2] > top - 0.020) & (sel[:, 2] < top - 0.004), :2]
    mid = sel[np.abs(sel[:, 2] - (table_z + grasp_h)) < 0.008, :2]
    d = np.median(hi, 0) - np.median(lo, 0)
    return [float(v) for v in np.median(mid, 0)], float(math.atan2(d[1], d[0]))


# ---------------------------------------------------------------- labels by Gemini
LABEL_PROMPT = """These numbered tiles show wooden toy pieces standing on a white table (each
tile: the same piece seen from two cameras). Name each numbered piece with exactly one of:
ring (short open tube, wider than tall), barrel (ribbed cylinder, flat closed top),
urn (vase with a narrow waist and an open top), goblet (egg cup: cup on a stem and a wide flat
foot), knob (ball on a thin neck over a small foot), dowel (long thin plain stick, about 11 cm),
slant (plain cylinder about 2.5 cm wide and 7 cm tall whose body leans to one side),
unknown (anything else, several pieces, or not a piece).
Answer JSON: {"labels": {"1": "...", ...}}"""


def crop_tiles(fid, objects, size=150, half=0.05):
    """One tile per object (left|right crop around its centre), labelled with its number."""
    meta = json.loads((exterior.FRAMES / fid / 'meta.json').read_text())
    imgs = {c: cv2.imread(str(exterior.FRAMES / fid / meta['cameras'][c]['rgb'])) for c in ('left', 'right')}
    tiles = []
    for i, o in enumerate(objects, 1):
        pair = []
        for c in ('left', 'right'):
            cam = meta['cameras'][c]
            K, T = np.array(cam['K']), np.array(cam['T_base_cam'])
            centre = np.array([o['xy'][0], o['xy'][1], o['table_z'] + max(0.02, o['height'] / 2), 1.0])
            P = np.linalg.inv(T) @ centre
            u, v = (K @ (P[:3] / P[2]))[:2]
            r = int(K[0, 0] * half / P[2])
            img = imgs[c]
            u0, v0 = max(0, int(u) - r), max(0, int(v) - r)
            crop = img[v0:int(v) + r, u0:int(u) + r]
            pair.append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC))
        tile = np.hstack(pair)
        cv2.putText(tile, str(i), (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        tiles.append(tile)
    cols = 2
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def gemini_labels(ctx, fid, objects, save=None):
    sheet = crop_tiles(fid, objects)
    if save:
        cv2.imwrite(save, sheet)
    ok, png = cv2.imencode('.png', sheet)
    reply = ctx.value('gemini_generate', messages=[{'role': 'user', 'content': [
        {'type': 'text', 'text': LABEL_PROMPT},
        {'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(png.tobytes()).decode('ascii')}]}],
        response_format={'type': 'json_object'}, reasoning={'effort': 'low'}, max_tokens=2048, temperature=0)
    if reply.get('finish_reason') == 'length':
        raise RuntimeError('gemini answer truncated')
    text = reply['message']['content']
    labels = json.loads(text)['labels']
    return [labels.get(str(i), 'unknown') for i in range(1, len(objects) + 1)]
