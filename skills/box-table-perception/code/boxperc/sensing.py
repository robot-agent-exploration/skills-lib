"""Sensing primitives (Runtime level): find / re-localize a box, find a table top, verify a box rests on a surface
or on the floor, check that a box is held. Gemini-seeded (locate.py) with the depth-only detectors as fallback;
detections are stored in the world (odometry frame) so later steps can predict them after base moves.

Every primitive returns the frame ids it used ('frame' / 'frames') so the agent or a subagent can look."""
import cv2
import numpy as np

from rby1rt import frames as pframes
from rby1rt import robot, vlm
from rby1rt.runner import StepFailed

from . import box as pbox
from . import locate as ploc
from . import table as ptable

SEARCH_LOOKS = [[1.6, 0.0, 0.1], [1.2, 0.6, 0.1], [1.2, -0.6, 0.1], [2.5, 0.0, 0.1]]
TABLE_LOOKS = [[1.5, 0.0, 0.7], [1.3, 0.55, 0.7], [1.3, -0.55, 0.7]]
FULL_SIZE = {'box': (0.33, 0.15)}       # depth-only fallback: a closer detection smaller than this is partial
MOVED_TOL = 0.10                        # m: a re-detected object this far from its prediction moved (contact?)


def _depth_only(fid, near, radius, support_z):
    d = pbox.find_box(fid, near=near, radius=radius, support_z=support_z)
    if d['ok']:
        d.pop('candidates', None)
        lim = FULL_SIZE['box']
        d['fully_visible'] = not (np.hypot(*d['center']) < 2.5 and (d['size'][0] < lim[0] or d['size'][1] < lim[1]))
        d['method'] = 'depth-only'
    return d


def locate(rt, kind='box', near=None, support_z=0.0, looks=None, look_z=0.12, describe=None, support='floor',
           radius=0.6, exclude_self=True):
    """Look (at `near` or along the search looks), localize (Gemini-seeded; depth-only if Gemini fails), centre the
    head on it and localize again. A box that Gemini reports as not fully visible is re-observed once from the
    centred head and then refused (StepFailed + frame): partial views gave the wrong pose in crash A2."""
    describe = describe or ploc.BOX_DESC
    looks = [[near[0], near[1], support_z + look_z]] if near is not None else (looks or SEARCH_LOOKS)
    last = None
    for p in looks:
        fid = rt.observe(p)
        d = ploc.locate_box(rt.ctx, fid, describe=describe, support=support, support_z=support_z,
                            exclude_self=exclude_self, log=rt.log)
        if not d['ok'] and d.get('why', '').startswith('gemini:'):
            rt.log(f"gemini unavailable ({d['why'][:120]}): depth-only fallback")
            d = _depth_only(fid, near, radius, support_z)
        if not d['ok']:
            rt.log(f"{kind} not found in {fid}: {d.get('why')}")
            continue
        if near is not None and np.hypot(*(np.array(d['center']) - near[:2])) > radius:
            rt.log(f"{kind} found {np.round(d['center'], 2)} but > {radius} m from {np.round(near[:2], 2)}")
            continue
        for _ in range(2):                      # centre the head on it (best depth) and localize again
            fid2 = rt.observe([d['center'][0], d['center'][1], d['top_z']])
            d2 = ploc.locate_box(rt.ctx, fid2, describe=describe, support=support, support_z=support_z,
                                 exclude_self=exclude_self, log=rt.log)
            if not d2['ok']:
                break
            if d2['fully_visible']:
                return d2
            last = d2
            d = d2
        if d.get('fully_visible'):
            return d
        last = d
    if last is not None:
        raise StepFailed(f"{kind} seen only partially (frame {last['frame']}, centre {np.round(last['center'], 2)}): "
                         'move the robot / head for a clear view (checkpoint)')
    return None


def find_object(rt, name, kind='box', support_z=0.0, looks=None, describe=None, support='floor', on=None):
    """Look for the object along `looks` (head only; turning the base to search is the caller's choice) and
    register it as `name`. on: name of a known surface it stands on (its top height becomes support_z).
    -> {name: det, frame}"""
    if on is not None:
        support_z = rt.world.objects[on]['top_z'] - 0.01
    d = locate(rt, kind, support_z=support_z, looks=looks, describe=describe, support=support)
    if d is None:
        raise StepFailed(f'{kind} {name} not found (frames {rt.frames[-4:]})')
    o = rt.world.put(name, kind, d, rt.odom())
    o['support'] = support
    rt.log(f"{name}: {kind} at {np.round(d['center'], 3)} size {np.round(d['size'], 3)} top {d['top_z']:.3f} "
           f"yaw {d['yaw_short']:+.3f} frame {d['frame']}")
    return {name: {k2: d[k2] for k2 in ('center', 'size', 'top_z', 'yaw_short', 'fully_visible', 'vlm_corner_err_m')
                   if k2 in d}, 'frame': d['frame']}


def relocate(rt, name, radius=0.4):
    """Re-detect a known object near where the world predicts it; stop if it moved (possible contact)."""
    o = rt.world.objects[name]
    pred = rt.world.in_base(name, rt.odom())
    d = locate(rt, o['kind'], near=pred['center'], support_z=o.get('support_z', 0.0), radius=radius,
               support=o.get('support', 'floor'), look_z=o.get('top_z', 0.25) - o.get('support_z', 0.0) - 0.1)
    if d is None:
        raise StepFailed(f'lost {name} near its predicted place {np.round(pred["center"], 2)}: moved? (checkpoint)')
    moved = float(np.hypot(*(np.array(d['center']) - pred['center'])))
    if moved > MOVED_TOL and rt.world.held != name:
        raise StepFailed(f'{name} is {moved:.2f} m from where it was: it moved - possible contact. Stop and look '
                         f'(frame {d["frame"]}).')
    o2 = rt.world.put(name, o['kind'], d, rt.odom())
    o2['support'] = o.get('support', 'floor')
    d['moved_m'] = moved
    return d


def find_surface(rt, name, describe=None, looks=None):
    """Find the table top `name` (Gemini: which table and its corners; depth: top height and rectangle).
    Keeps the view with the most of the top visible. -> {name: {...}, frame, sides_picture}"""
    best = None
    for p in (looks or TABLE_LOOKS):
        if not robot.look(rt.ctx, p):
            continue
        fid = rt.observe()
        t = ploc.locate_table(rt.ctx, fid, describe=describe or ploc.TABLE_DESC, log=rt.log)
        if not t['ok']:
            continue
        score = (t['top_fully_visible'], -abs(np.hypot(*t['center']) - 1.3))
        if best is None or score > best[0]:
            best = (score, t)
        if t['top_fully_visible']:
            break
    if best is None:
        raise StepFailed(f'no {describe or ploc.TABLE_DESC} found (frames {rt.frames[-3:]})')
    t = best[1]
    t['support_z'] = t['top_z']
    rt.world.put(name, 'table', t, rt.odom())
    pic = table_sides_picture(rt, name)
    return {name: {k: t[k] for k in ('center', 'size', 'top_z', 'top_fully_visible', 'objects_on_top')},
            'frame': t['frame'], 'sides_picture': pic}


def verify_on_surface(rt, name, surface, height=(0.21, 0.31)):
    """The box rests on the table: depth height of its top above the table top in range, centre inside the top,
    and Gemini confirms (resting, not held). -> {height_above_surface, center, inside, gemini, frame}"""
    tb = rt.world.in_base(surface, rt.odom())
    fid = rt.observe([tb['center'][0], tb['center'][1], tb['top_z'] + 0.1])
    t = ptable.find_table(fid, near=tb['center'], radius=1.0)
    table_z = t['top_z'] if t['ok'] else tb['top_z']
    d = ploc.locate_box(rt.ctx, fid, support='black table top', support_z=table_z - 0.02, exclude_self=False,
                        log=rt.log)
    if not d['ok']:
        raise StepFailed(f"{name} not found on {surface} (frame {fid}): {d.get('why')}")
    h = d['top_z'] - table_z
    corners = np.array(t['corners'] if t['ok'] else tb['corners'], np.float32)
    inside = cv2.pointPolygonTest(corners, (float(d['center'][0]), float(d['center'][1])), True) > 0.05
    g = vlm.check(rt.ctx, fid, 'Robot head-camera image. Is the cardboard box standing upright on the black table '
                  'top, fully supported by it (not hanging over its edge), with both robot grippers clear of it?',
                  log=rt.log)
    rt.log(f'verify: {name} top {d["top_z"]:.3f} {surface} {table_z:.3f} height {h:.3f} inside {inside} '
           f'gemini {g.get("answer")} ({g.get("reason")}) frame {fid}')
    if not (height[0] <= h <= height[1] and inside):
        raise StepFailed(f'{name} not resting on {surface}: height {h:.3f} inside {inside} (frame {fid})')
    if g.get('ok') and not g['yes']:
        raise StepFailed(f'gemini disagrees: {g["reason"]} (frame {fid})')
    rt.world.put(name, 'box', d, rt.odom())
    return {'height_above_surface': round(h, 3), 'center': d['center'], 'inside': bool(inside),
            'gemini': g.get('answer'), 'gemini_reason': g.get('reason'), 'frame': fid}


def verify_on_floor(rt, name, height=(0.20, 0.31)):
    """The released box stands on the floor in front: depth top height in range and Gemini confirms."""
    fid = rt.observe([0.55, 0.0, 0.1])
    d = ploc.locate_box(rt.ctx, fid, support='floor', support_z=0.0, log=rt.log)
    if not d['ok']:
        raise StepFailed(f"{name} not found on the floor (frame {fid}): {d.get('why')}")
    g = vlm.check(rt.ctx, fid, 'Robot head-camera image. Is the cardboard box standing upright on the floor with '
                  'both robot grippers clear of it (not touching it)?', log=rt.log)
    rt.log(f"verify floor: top {d['top_z']:.3f} centre {np.round(d['center'], 3).tolist()} gemini {g.get('answer')} "
           f"({g.get('reason')})")
    if not height[0] <= d['top_z'] <= height[1]:
        raise StepFailed(f"{name} top at {d['top_z']:.3f}: not standing on the floor? (frame {fid})")
    if g.get('ok') and not g['yes']:
        raise StepFailed(f'gemini disagrees: {g["reason"]} (frame {fid})')
    o = rt.world.put(name, 'box', d, rt.odom())
    o['support'] = 'floor'
    return {'top_z': round(d['top_z'], 3), 'center': d['center'], 'yaw_short': d['yaw_short'],
            'gemini': g.get('answer'), 'gemini_reason': g.get('reason'), 'frame': fid}


def check_held(rt, name, look=(0.5, 0.0, 0.8)):
    """Gemini: is the box held between both grippers and off its support? -> {held, reason, frame}"""
    fid = rt.observe(list(look))
    g = vlm.check(rt.ctx, fid, 'Robot head-camera image. Is the cardboard box held between the two robot grippers '
                  '(one at each end), lifted clear of the surface below it and level (not tilting in the grip)?',
                  log=rt.log)
    return {'held': g.get('yes') if g.get('ok') else None, 'reason': g.get('reason') or g.get('why'), 'frame': fid}


def table_sides_picture(rt, name):
    """Camera image of the table view with its sides numbered (the agent / a subagent picks one)."""
    o = rt.world.objects[name]
    fid = o.get('frame')
    if not fid:
        return None
    _, _, rgb, meta = pframes.load(fid, stride=16)
    tb = rt.world.in_base(name, o['odom_at_detection'])
    z = o.get('top_z', 0.7)
    img = rgb.copy()
    for k, sd in enumerate(ptable.sides(tb['corners'])):
        a = sd['mid'] - sd['dir'] * sd['length'] / 2
        b = sd['mid'] + sd['dir'] * sd['length'] / 2
        ua, _ = pframes.project(meta, [a[0], a[1], z])
        ub, _ = pframes.project(meta, [b[0], b[1], z])
        um, _ = pframes.project(meta, [*(sd['mid'] + sd['normal'] * 0.25), z])
        cv2.line(img, tuple(int(x) for x in ua), tuple(int(x) for x in ub), (0, 0, 255), 3)
        cv2.putText(img, str(k), tuple(int(x) for x in um), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
    path = rt.out / f'table_sides_{fid}.jpg'
    cv2.imwrite(str(path), cv2.resize(img, (960, 540)))
    return str(path)
