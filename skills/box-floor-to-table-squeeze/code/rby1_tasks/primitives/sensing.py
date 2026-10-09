"""Sensing primitives: search for an object kind, re-localize a known object, find a support surface, verify that
an object rests on a surface. Detections are stored in the world (odometry frame)."""
import cv2
import numpy as np

from ..core import geom, robot
from ..core.runner import StepFailed
from ..perception import box as pbox
from ..perception import table as ptable

DETECTORS = {'box': pbox.find_box}         # kind -> detector(frame, near=, radius=, support_z=) -> {ok, center, ...}
SEARCH_LOOKS = [[2.0, 0.0, 0.1], [1.2, 0.0, 0.1], [3.0, 0.0, 0.1], [1.6, 0.8, 0.1], [1.6, -0.8, 0.1]]


FULL_SIZE = {'box': (0.33, 0.15)}       # a closer detection smaller than this is a partial view (occluded)
MOVED_TOL = 0.10                        # m: a re-detected object this far from its prediction moved (contact?)


def _partial(kind, d):
    lim = FULL_SIZE.get(kind)
    return bool(lim and np.hypot(*d['center']) < 2.5 and (d['size'][0] < lim[0] or d['size'][1] < lim[1]))


def locate(rt, kind, near=None, support_z=0.0, looks=None, radius=None, look_z=0.12):
    """Look (at `near` or along the search looks), detect, centre the head on it and detect again; a partial
    view (occluded, too small) is re-observed from the centred head and rejected if still partial."""
    det = DETECTORS[kind]
    looks = [[near[0], near[1], support_z + look_z]] if near is not None else (looks or SEARCH_LOOKS)
    partial = None
    for p in looks:
        fid = rt.observe(p)
        d = det(fid, near=near, radius=radius or (0.6 if near is not None else 1.5), support_z=support_z)
        if not d['ok']:
            continue
        for _ in range(2):
            fid = rt.observe([d['center'][0], d['center'][1], support_z + look_z])
            d2 = det(fid, near=d['center'], radius=0.5, support_z=support_z)
            if not d2['ok']:
                break
            d2.pop('candidates', None)
            if not _partial(kind, d2):
                return d2
            partial = d2
            d = d2
    if partial is not None:
        rt.log(f'{kind}: only partial views ({np.round(partial["size"], 3)} at {np.round(partial["center"], 2)})')
        raise StepFailed(f'{kind} seen only partially at {np.round(partial["center"], 2)} size '
                         f'{np.round(partial["size"], 3)}: move the robot / head for a clear view (checkpoint)')
    return None


def find_object(rt, name, kind, support_z=0.0, search_turn=1.0, max_turns=6, looks=None):
    """Search for an object of `kind` (turning in place between look sweeps) and register it as `name`."""
    for k in range(max_turns + 1):
        d = locate(rt, kind, support_z=support_z, looks=looks)
        if d is not None:
            rt.world.put(name, kind, d, rt.odom())
            rt.log(f"{name}: {kind} at {np.round(d['center'], 3)} size {np.round(d['size'], 3)} top {d['top_z']:.3f} "
                   f"yaw {d['yaw_short']:+.3f} frame {d['frame']}")
            return {name: d}
        if k < max_turns:
            rt.log(f'no {kind} in view: turning {search_turn:+.2f} to search')
            from . import navigation
            navigation.safe_drive(rt, [{'turn': search_turn}], tag='search_turn')
    raise StepFailed(f'{kind} not found after a full turn')


def relocate(rt, name, radius=0.6):
    """Re-detect a known object near where the world predicts it; update the world."""
    o = rt.world.objects[name]
    odom = rt.odom()
    pred = rt.world.in_base(name, odom)
    d = locate(rt, o['kind'], near=pred['center'], support_z=o.get('support_z', 0.0), radius=radius)
    if d is None:
        raise StepFailed(f'lost {name} near its predicted place {np.round(pred["center"], 2)}: moved? (checkpoint)')
    moved = float(np.hypot(*(np.array(d['center']) - pred['center'])))
    if moved > MOVED_TOL and rt.world.held != name:
        raise StepFailed(f'{name} is {moved:.2f} m from where it was: it moved - possible contact. Stop and look.')
    rt.world.put(name, o['kind'], d, rt.odom())
    d['moved_m'] = moved
    return d


def find_surface(rt, name, prior_odom=None, size=(1.0, 0.62), radius=1.3, max_score=0.9):
    """Find the dark table top `name`: turn towards the odometry prior if it is off to the side, look at it and
    around, keep the candidate closest in size (and position) to the expected top."""
    s = rt.status()
    prior = geom.odom_to_base(s['odometry'], prior_odom)[:2] if prior_odom is not None else np.array([1.5, 0.0])
    bearing = float(np.arctan2(prior[1], prior[0] - robot.AXLE_X))
    if abs(bearing) > 0.45:
        rt.log(f'turning {bearing:+.2f} towards the {name} prior')
        from . import navigation
        navigation.safe_drive(rt, [{'turn': bearing}], tag=f'turn_to_{name}')
        prior = geom.odom_to_base(rt.odom(), prior_odom)[:2]
    looks = [[prior[0], prior[1], 0.7], [1.5, 0.0, 0.7], [1.3, 0.55, 0.7], [1.3, -0.55, 0.7]]
    best, frames = None, []
    for p in looks:
        if not robot.look(rt.ctx, p):
            continue
        fid = rt.observe()
        frames.append(fid)
        t = ptable.find_table(fid, near=prior, radius=radius)
        if not t['ok']:
            continue
        dims = sorted(t['size'], reverse=True)
        score = abs(dims[0] - size[0]) + abs(dims[1] - size[1]) + 0.3 * float(np.hypot(*(np.array(t['center']) - prior)))
        rt.log(f"{name} candidate {np.round(t['center'], 2)} size {np.round(t['size'], 2)} z {t['top_z']:.3f} score {score:.2f}")
        if best is None or score < best[0]:
            best = (score, t)
    if best is None or best[0] > max_score:
        raise StepFailed(f'no {name} like {size} near {np.round(prior, 2)}: {best and best[1]}')
    t = best[1]
    t['support_z'] = t['top_z']
    rt.world.put(name, 'table', t, rt.odom())
    rt.world.notes[f'{name}_frames'] = frames
    from . import navigation
    pic = navigation.table_sides_picture(rt, name)
    return {name: {k: t[k] for k in ('center', 'size', 'top_z')}, 'sides_picture': pic}


def look_around(rt, points=((1.0, 0.0, 0.0), (0.9, 0.6, 0.0), (0.9, -0.6, 0.0), (2.0, 0.0, 0.0))):
    """Views ahead and to both sides (the camera never sees behind the robot)."""
    return {'frames': [rt.observe(list(p)) for p in points]}


def verify_on_surface(rt, name, surface, height=(0.21, 0.31), kind='box'):
    """The object rests on the surface: found on it (top height above the surface in range) and inside it."""
    tb = rt.world.in_base(surface, rt.odom())
    fid = rt.observe([0.65, 0.05, tb['top_z'] + 0.1])
    t = ptable.find_table(fid)
    table_z = t['top_z'] if t['ok'] else tb['top_z']
    d = DETECTORS[kind](fid, support_z=table_z - 0.02, exclude_self=False)
    if not d['ok']:
        raise StepFailed(f'{name} not found on {surface}')
    h = d['top_z'] - table_z
    corners = np.array(t['corners'] if t['ok'] else tb['corners'], np.float32)
    inside = cv2.pointPolygonTest(corners, (float(d['center'][0]), float(d['center'][1])), True) > 0.05
    rt.log(f'verify: {name} top {d["top_z"]:.3f} {surface} {table_z:.3f} height {h:.3f} inside {inside} frame {fid}')
    if not (height[0] <= h <= height[1] and inside):
        raise StepFailed(f'{name} not resting on {surface}: height {h:.3f} inside {inside}')
    d.pop('candidates', None)
    rt.world.put(name, kind, d, rt.odom())
    return {'height_above_surface': h, 'center': d['center'], 'frame': fid}
