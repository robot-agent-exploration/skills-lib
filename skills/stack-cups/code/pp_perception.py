"""Object localization for pick & place: Gemini segmentation masks say which pixels are which object, depth gives
metres. No robot motion; every function takes a saved frame id, so it runs offline on recorded frames.

localize(ctx, frame_id, objects) with objects = {name: kind} -> {name: description}:
  'elongated' (banana-like, lying): ridge (top line) points, grasp point near the middle of the ridge where it is
              straight, axis_yaw there, top_z, thickness, support_z
  'box'       (upright box):         top rectangle centre, long / short side, long_yaw, top_z, support_z
  'round'     (bowl, bucket):        rim centre, radius r, rim_z, support_z
Kinds and their tested sizes are in the skill's SKILL.md."""
import importlib.util
import json
import math
from pathlib import Path

import _deps  # noqa: F401
import cv2
import numpy as np

import geom
import pp_vis

_spec = importlib.util.spec_from_file_location('vlm', geom.ROOT / '.agents' / 'skills' / 'gemini' / 'scripts' / 'vlm.py')
vlm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vlm)

OUT = geom.ROOT / 'output' / 'pick_place'
ELONG_STRAIGHT_DEG = 25      # grasp where the ridge turns less than this within ±3.5 cm
BOX_MAX_SHORT = 0.085        # an upright box thicker than this does not fit the 0.10 m opening with margin


def _norm(s):
    return ' '.join(w for w in str(s).lower().replace('_', ' ').replace('-', ' ').split() if w not in ('the', 'a'))


def _match(label, names):
    """Name a Gemini label belongs to: exact match, else the largest word overlap (None if tied or none)."""
    lab = _norm(label)
    for n in names:
        if _norm(n) == lab:
            return n
    words = set(lab.split())
    scores = sorted(((len(words & set(_norm(n).split())) / len(words | set(_norm(n).split())), n) for n in names),
                    reverse=True)
    if scores and scores[0][0] > 0 and (len(scores) == 1 or scores[0][0] > scores[1][0]):
        return scores[0][1]
    return None


def _points(obj):
    """All numeric 2-element lists inside a (possibly nested, label-prefixed) polygon answer."""
    if isinstance(obj, (list, tuple)):
        if obj and all(isinstance(v, (int, float)) for v in obj):
            if len(obj) == 2:
                return [list(obj)]
            if len(obj) % 2 == 0:                       # flat [y0, x0, y1, x1, ...]
                return [list(obj[i:i + 2]) for i in range(0, len(obj), 2)]
            return []
        return [p for o in obj for p in _points(o)]
    return []


def _parse_masks(ans, size, offset=(0, 0)):
    """Gemini segmentation answer -> [{label, box_px, poly_px}] in full-size pixels. Polygons come as [[y, x], ...],
    sometimes nested as [["label", [y, x], ...]]; the [y, x] / [x, y] order is decided by containment in box_2d."""
    if isinstance(ans, dict):
        ans = next((v for v in ans.values() if isinstance(v, list)), [ans])
    w, h = size
    x0, y0 = offset
    out = []
    for m in ans if isinstance(ans, list) else []:
        try:
            yb0, xb0, yb1, xb1 = [float(v) for v in m['box_2d']]
        except (KeyError, TypeError, ValueError):
            continue
        poly = np.array(_points(m.get('mask')), float)
        if len(poly) < 3:
            continue

        def inside(cols, rows):
            return float(np.mean((cols >= xb0 - 15) & (cols <= xb1 + 15) & (rows >= yb0 - 15) & (rows <= yb1 + 15)))
        yx = inside(poly[:, 1], poly[:, 0]) >= inside(poly[:, 0], poly[:, 1])
        cols, rows = (poly[:, 1], poly[:, 0]) if yx else (poly[:, 0], poly[:, 1])
        out.append({'label': str(m.get('label', '')),
                    'box_px': [xb0 / 1000 * w + x0, yb0 / 1000 * h + y0, xb1 / 1000 * w + x0, yb1 / 1000 * h + y0],
                    'poly_px': np.c_[cols / 1000 * w + x0, rows / 1000 * h + y0].astype(np.int32).tolist()})
    return out


def refine_mask(rgb, mask, box=None, grow=8, max_de=28.0):
    """Sharpen a coarse polygon mask with the object's own colour: pixels near the mask whose Lab colour is within
    max_de of the mask core's median, connected to the core. Falls back to the input when the result is not
    plausible (grey / black objects, or the area changes by more than 60 %)."""
    core = cv2.erode(mask, np.ones((9, 9), np.uint8))
    if core.sum() < 80:
        return mask
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = np.median(lab[core > 0], axis=0)
    if np.hypot(ref[1] - 128, ref[2] - 128) < 12:      # unsaturated: colour does not separate it
        return mask
    near = cv2.dilate(mask, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    if box is not None:                                 # stay inside Gemini's box (+ grow): same-colour neighbours
        u0, v0, u1, v1 = [int(round(b)) for b in box]
        clip = np.zeros_like(near)
        clip[max(0, v0 - grow):v1 + grow, max(0, u0 - grow):u1 + grow] = True
        near &= clip
    ok = (np.linalg.norm(lab - ref, axis=2) < max_de) & near
    ok = cv2.morphologyEx(ok.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(ok)
    if n < 2:
        return mask
    overlap = [((lab_cc == i) & (core > 0)).sum() for i in range(n)]
    k = int(np.argmax(overlap[1:])) + 1
    out = (lab_cc == k).astype(np.uint8)
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.drawContours(np.zeros_like(out), cnts, -1, 1, -1)       # fill holes (bowl insides, labels)
    if not 0.4 < out.sum() / max(1, mask.sum()) < 1.6:
        return mask
    return out


def segment_named(ctx, frame_id, names, hint='', refine=True, tries=2, log=print):
    """Gemini masks for the named objects in one frame -> {ok, masks: {name: uint8 mask}, boxes, overlay, why}.
    Answers vary between calls (an object is sometimes left out): names still missing are asked again once."""
    f = geom.load(frame_id)
    H, W = f['rgb'].shape[:2]
    block, size, off = vlm.image_block(f['rgb'])
    masks, boxes, found = {}, {}, []
    for _ in range(tries):
        ask = [n for n in names if n not in masks]
        if not ask:
            break
        what = ('each of these objects: ' + '; '.join(ask) + '. Label each mask with exactly one name from this '
                'list: ' + ', '.join(ask) + '. Leave out objects that are not visible.' + (' ' + hint if hint else ''))
        r, ans = vlm._call(ctx, 'segment', [{'type': 'text', 'text': vlm.SEG_PROMPT.format(what=what)}, block],
                           {'type': 'json_object'}, 'minimal', vlm.MAX_TOKENS)
        if not r.get('ok'):
            log(f"gemini segmentation failed: {r.get('why')}")
            continue
        got = _parse_masks(ans, size, off)
        found += got
        log(f"gemini segment {r.get('seconds')}s ${r.get('cost_usd')}: {[m['label'] for m in got]}")
        for m in got:
            n = _match(m['label'], ask)
            if n is None:
                continue
            mk = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mk, [np.array(m['poly_px'], np.int32)], 1)
            if n not in masks or mk.sum() > masks[n].sum():
                masks[n], boxes[n] = mk, m['box_px']
    if refine:
        masks = {n: refine_mask(f['rgb'], mk, boxes[n]) for n, mk in masks.items()}
    ov = f['rgb'].copy()
    for i, (n, mk) in enumerate(masks.items()):
        col = [(0, 255, 0), (255, 0, 255), (0, 200, 255), (255, 128, 0), (0, 0, 255), (255, 255, 0)][i % 6]
        cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, col, 2)
        x0, y0 = boxes[n][:2]
        cv2.putText(ov, n, (int(x0), max(12, int(y0) - 4)), 0, 0.6, col, 2)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'seg_{frame_id}.jpg'
    cv2.imwrite(str(path), ov)
    missing = [n for n in names if n not in masks]
    return {'ok': not missing, 'masks': masks, 'boxes': boxes, 'overlay': str(path),
            'labels': [m['label'] for m in found], 'why': f'not found: {missing}' if missing else ''}


# ---------- geometry per kind ----------
def elongated(f, mask):
    """Lying elongated object (banana): top ridge along its principal axis; grasp at the ridge point nearest the
    middle (by arc length) where the ridge is straight within ±3.5 cm."""
    P, _ = pp_vis.mask_points(f, mask)
    sz = pp_vis.support_z(f, mask)
    if sz is None or len(P) < 50:
        return {'ok': False, 'why': 'too few depth points'}
    P = P[P[:, 2] > sz + 0.006]
    xy = P[:, :2]
    mu = xy.mean(0)
    e1 = np.linalg.svd(xy - mu, full_matrices=False)[2][0]
    s = (xy - mu) @ e1
    ridge = []
    for lo in np.arange(s.min(), s.max(), 0.01):
        k = (s >= lo) & (s < lo + 0.01)
        if k.sum() < 10:
            continue
        zt = np.percentile(P[k, 2], 97)
        kk = k & (P[:, 2] > zt - 0.004)
        ridge.append([*P[kk, :2].mean(0), zt])
    ridge = np.array(ridge)
    if len(ridge) < 5:
        return {'ok': False, 'why': 'ridge too short'}
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ridge[:, :2], axis=0), axis=1))]
    L = arc[-1]

    def seg_dir(a0, a1):
        k = (arc >= a0) & (arc <= a1)
        if k.sum() < 2:
            return None
        q = ridge[k, :2]
        return np.linalg.svd(q - q.mean(0), full_matrices=False)[2][0]

    best = None
    for i in np.argsort(np.abs(arc - L / 2)):
        d1, d2 = seg_dir(arc[i] - 0.035, arc[i]), seg_dir(arc[i], arc[i] + 0.035)
        if d1 is None or d2 is None:
            continue
        turn = math.degrees(math.acos(min(1.0, abs(float(d1 @ d2)))))
        if turn < ELONG_STRAIGHT_DEG:
            best = i
            break
    if best is None:
        return {'ok': False, 'why': 'no straight ridge section'}
    d = seg_dir(arc[best] - 0.035, arc[best] + 0.035)
    return {'ok': True, 'kind': 'elongated', 'grasp_xy': ridge[best, :2].round(3).tolist(),
            'axis_yaw': round(math.atan2(d[1], d[0]), 3), 'top_z': round(float(ridge[best, 2]), 3),
            'thickness': round(float(ridge[:, 2].max() - sz), 3), 'length': round(float(L), 3),
            'support_z': round(sz, 3), 'centre': mu.round(3).tolist(), 'ridge': ridge.round(3).tolist()}


def box_upright(f, mask):
    """Upright box: rectangle of its top face (points within 1.2 cm of the top)."""
    P, _ = pp_vis.mask_points(f, mask)
    sz = pp_vis.support_z(f, mask)
    if sz is None or len(P) < 50:
        return {'ok': False, 'why': 'too few depth points'}
    zt = float(np.percentile(P[:, 2], 98))
    top = P[P[:, 2] > zt - 0.012]
    (cx, cy), (w, h), ang = cv2.minAreaRect((top[:, :2] * 1000).astype(np.float32))
    long_, short = max(w, h) / 1000, min(w, h) / 1000
    a = math.radians(ang) if w >= h else math.radians(ang + 90)
    a = math.atan2(math.sin(a), math.cos(a))
    return {'ok': short < BOX_MAX_SHORT, 'kind': 'box', 'centre': [round(cx / 1000, 3), round(cy / 1000, 3)],
            'top_z': round(zt, 3), 'long': round(long_, 3), 'short': round(short, 3), 'long_yaw': round(a, 3),
            'height': round(zt - sz, 3), 'support_z': round(sz, 3),
            'why': '' if short < BOX_MAX_SHORT else f'top {short:.3f} m thick: wider than the grip'}


def round_obj(f, mask):
    """Bowl / bucket: rim height from depth, centre and radius from the silhouette at that height."""
    P, _ = pp_vis.mask_points(f, mask)
    sz = pp_vis.support_z(f, mask)
    if len(P) < 50:
        return {'ok': False, 'why': 'too few depth points'}
    rim_z = float(np.percentile(P[:, 2], 98))
    r = pp_vis.round_from_silhouette(f, mask, rim_z)
    return dict(r, ok=True, kind='round', support_z=None if sz is None else round(sz, 3),
                height=None if sz is None else round(rim_z - sz, 3))


KINDS = {'elongated': elongated, 'box': box_upright, 'round': round_obj}


def localize(ctx, frame_id, objects, hint='', log=print):
    """objects {name: kind} -> {ok, objects: {name: description + frame}, overlay, why}."""
    seg = segment_named(ctx, frame_id, list(objects), hint=hint, log=log)
    f = geom.load(frame_id)
    out = {}
    for n, kind in objects.items():
        if n not in seg['masks']:
            out[n] = {'ok': False, 'why': 'not segmented'}
            continue
        try:
            d = KINDS[kind](f, seg['masks'][n])
        except Exception as e:      # geometry on a bad mask: report, do not act
            d = {'ok': False, 'why': f'geometry failed: {e}'}
        d['frame'] = frame_id
        d['mask_px'] = int(seg['masks'][n].sum())
        out[n] = d
    bad = [n for n, d in out.items() if not d.get('ok')]
    return {'ok': not bad, 'objects': out, 'overlay': seg.get('overlay'), 'labels': seg.get('labels'),
            'why': '; '.join(f"{n}: {out[n].get('why')}" for n in bad)}


# ---------- Gemini checkpoint questions (2-D, yes / no) ----------
def check(ctx, frame_id, question, log=print):
    """Yes / no / unsure on one saved frame -> {ok, yes, answer, reason}."""
    return vlm.check(ctx, str(geom.frame_dir(frame_id)), question, log=log, camera=geom.CAM)


def flags(ctx, frame_id, question, names, log=print):
    """One boolean per name (e.g. 'is inside the bucket') -> {ok, flags: {name: bool}, why}."""
    keys = {n: n.replace(' ', '_') for n in names}
    schema = {'type': 'object', 'additionalProperties': False, 'required': list(keys.values()) + ['reason'],
              'properties': dict({k: {'type': 'boolean', 'description': n} for n, k in keys.items()},
                                 reason={'type': 'string', 'description': 'one short sentence'})}
    block = vlm.image_block(str(geom.frame_dir(frame_id)), camera=geom.CAM)[0]
    r = vlm.ask(ctx, question, [block], schema, name='flags', log=log)
    if not r.get('ok'):
        return r
    return {'ok': True, 'flags': {n: bool(r['answer'][k]) for n, k in keys.items()}, 'reason': r['answer']['reason']}


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1, default=float))


def above_opening(ctx, frame_id, obj, bucket='orange bucket', margin=0.02, log=print):
    """Geometric release check (no parallax): share of the held object's depth points whose xy lie inside the
    bucket rim circle shrunk by `margin`, both from the same frame. -> {ok, inside (0-1), bucket, obj_xy, why}"""
    seg = segment_named(ctx, frame_id, [obj, bucket], log=log)
    if obj not in seg['masks'] or bucket not in seg['masks']:
        return {'ok': False, 'why': f"not segmented: {[n for n in (obj, bucket) if n not in seg['masks']]}",
                'overlay': seg.get('overlay')}
    f = geom.load(frame_id)
    b = round_obj(f, seg['masks'][bucket])
    P, _ = pp_vis.mask_points(f, seg['masks'][obj])
    P = P[P[:, 2] > b['rim_z'] + 0.01] if b.get('ok') else P          # the hanging object, not the bucket contents
    if not b.get('ok') or len(P) < 30:
        return {'ok': False, 'why': 'too few points', 'overlay': seg.get('overlay')}
    d = np.linalg.norm(P[:, :2] - np.array(b['centre']), axis=1)
    inside = float(np.mean(d < b['r'] - margin))
    return {'ok': True, 'inside': round(inside, 2), 'bucket': b['centre'], 'r': b['r'],
            'obj_xy': np.median(P[:, :2], axis=0).round(3).tolist(), 'obj_low_z': round(float(np.percentile(P[:, 2], 2)), 3),
            'overlay': seg.get('overlay')}


CUP_LIPS = {'near': 'the point of the rim of {cup} nearest to the camera',
            'far': 'the point of the rim of {cup} farthest from the camera',
            'left': 'the leftmost point of the rim of {cup} in the image',
            'right': 'the rightmost point of the rim of {cup} in the image'}


def cup(ctx, frame_id, name, log=print):
    """Upright open cup (thin wall, round rim): rim height from depth (top of its mask points), rim centre and
    radius from Gemini points on four rim lips ray-cast to that height. Silhouette extremes of a grey cup are biased
    (the polygon's side points lie on the body below the rim); lip points measured 2026-10-09 agreed within 1 cm.
    -> {ok, kind 'round', centre, r, rim_z, support_z, lips, frame}"""
    seg = segment_named(ctx, frame_id, [name], log=log)
    if name not in seg['masks']:
        return {'ok': False, 'why': f'{name}: not segmented', 'frame': frame_id}
    f = geom.load(frame_id)
    P, _ = pp_vis.mask_points(f, seg['masks'][name])
    sz = pp_vis.support_z(f, seg['masks'][name])
    if len(P) < 50:
        return {'ok': False, 'why': 'too few depth points', 'frame': frame_id}
    rim_z = float(np.percentile(P[:, 2], 98))
    pts = {k: v.format(cup=f'the {name}') for k, v in CUP_LIPS.items()}
    r = vlm.point(ctx, str(geom.frame_dir(frame_id)), f'Locate points on the rim (top edge) of the {name}.', pts,
                  log=log, camera=geom.CAM)
    if not r.get('ok') or any(r['px'].get(k) is None for k in pts):
        return {'ok': False, 'why': f"lip points: {r.get('why')}", 'frame': frame_id}
    L = {k: pp_vis.ray_plane(f, [r['px'][k]], rim_z)[0] for k in pts}
    c = np.mean([L[k][:2] for k in pts], axis=0)
    rad = (np.linalg.norm(L['near'][:2] - L['far'][:2]) + np.linalg.norm(L['left'][:2] - L['right'][:2])) / 4
    ok = 0.025 < rad < 0.08
    return {'ok': ok, 'kind': 'round', 'centre': c.round(3).tolist(), 'r': round(float(rad), 3),
            'rim_z': round(rim_z, 3), 'support_z': None if sz is None else round(sz, 3),
            'height': None if sz is None else round(rim_z - sz, 3), 'frame': frame_id,
            'lips': {k: v.round(3).tolist() for k, v in L.items()}, 'why': '' if ok else f'radius {rad:.3f} implausible'}
