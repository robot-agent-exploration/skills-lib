"""Base navigation primitives (differential drive; turns pivot about the wheel axle at base x = 0.228).

Perception when needed, agent at checkpoints: a plan is drawn on the live camera image (the robot footprint
swept along it) for the agent to review; while it runs, each segment first looks along the area it sweeps and
stops if depth points (robot's own body and a held object removed) lie inside the swept footprint. What the
camera cannot see (behind the robot) is the agent's call at the checkpoint; reversing is only allowed back
along the straight line the robot just drove (run A2 turned and reversed into a box and a cabinet)."""
import cv2
import numpy as np
from matplotlib.path import Path as MPath
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..core import footprint, geom, robot
from ..core import frames as pframes
from ..core.runner import StepFailed

from ..perception import locate as ploc
from ..perception import sensing
from ..perception import table as ptable

from ..params import MARGIN, SEG_FORWARD, SEG_TURN, MAX_HIT_POINTS, OBST_Z, BORDER_PX  # hyper-parameters: bimanual_box_pnp/params.py


def plan_turn_drive_turn(target_origin, heading):
    """Moves bringing the base origin to `target_origin` (base xy now) facing `heading` (rad, now)."""
    h = np.array([np.cos(heading), np.sin(heading)])
    a_goal = np.asarray(target_origin, float) + robot.AXLE_X * h
    a0 = np.array([robot.AXLE_X, 0.0])
    d = a_goal - a0
    dist = float(np.linalg.norm(d))
    if dist < 0.03:
        return [{'turn': geom.wrap(heading)}] if abs(geom.wrap(heading)) > 0.01 else []
    bearing = float(np.arctan2(d[1], d[0]))
    moves = []
    if abs(bearing) > 0.01:
        moves.append({'turn': geom.wrap(bearing)})
    moves.append({'forward': dist})
    final = geom.wrap(heading - bearing)
    if abs(final) > 0.01:
        moves.append({'turn': final})
    return moves


def posture_of(rt):
    return 'carry' if rt.world.held else 'ready'


def object_polygon(rt, name, margin=0.0):
    """Odometry-frame footprint of a known object (box: its top rectangle; table: its corners)."""
    o = rt.world.objects[name]
    if 'corners_odom' in o:
        return Polygon(o['corners_odom']).buffer(margin)
    c, la, sa = np.array(o['center_odom']), np.array(o['long_axis_odom']), np.array(o['short_axis_odom'])
    L, S = o['size'][0] / 2 + 0.01, o['size'][1] / 2 + 0.01
    return Polygon([c + la * L + sa * S, c - la * L + sa * S, c - la * L - sa * S, c + la * L - sa * S]).buffer(margin)


def swept(start_pose, moves, posture, margin):
    """[(odom polygon swept by a footprint layer, lowest z of that robot part)] along the moves."""
    out = []
    for poly, zlow in footprint.footprint_layers(posture, margin):
        parts, p = [], tuple(start_pose)
        for m in moves:
            poses = footprint.sample_move(p, m)
            parts += [footprint.place(poly, q) for q in poses[::2] + [poses[-1]]]
            p = poses[-1]
        parts = parts or [footprint.place(poly, start_pose)]
        out.append((unary_union(parts), zlow))
    return out


def _inside(g, xy):
    m = np.zeros(len(xy), bool)
    for part in (getattr(g, 'geoms', None) or [g]):
        m |= MPath(np.array(part.exterior.coords)).contains_points(xy)
    return m


def path_obstacles(rt, fid, moves, start_pose=None, ignore_names=(), margin=MARGIN, ignore_margin=0.03):
    """Depth points of frame `fid` inside the footprint swept along `moves` (from start_pose, odometry)."""
    pts, pix, _, meta = pframes.load(fid, stride=3, max_depth=4.0, drop_edges=True)
    w, h = meta['cameras']['head_camera'].get('width', 1280), meta['cameras']['head_camera'].get('height', 720)
    inner = (pix[:, 0] >= BORDER_PX) & (pix[:, 0] < w - BORDER_PX) & (pix[:, 1] >= BORDER_PX) & (pix[:, 1] < h - BORDER_PX)
    pts, pix = pts[inner], pix[inner]                 # ZED depth at the image border is unreliable
    keep = ~footprint.self_mask(pts, meta['joints'], rt.held_box_region())
    pts, pix = pts[keep], pix[keep]
    xy = footprint._to_odom(rt.frame_odom[fid], pts[:, :2])
    obst = (pts[:, 2] > OBST_Z[0]) & (pts[:, 2] < OBST_Z[1])
    floor = pts[:, 2] < 0.06
    ign = np.zeros(len(xy), bool)
    for n in ignore_names:
        if n in rt.world.objects:
            ign |= _inside(object_polygon(rt, n, ignore_margin), xy)
    hits, seen = np.zeros(len(xy), bool), np.zeros(len(xy), bool)
    for g, zlow in swept(start_pose or rt.pose(), moves, posture_of(rt), margin):
        ins = _inside(g, xy)
        hits |= ins & obst & (pts[:, 2] >= zlow) & ~ign
        seen |= ins & floor
    return {'obstacle_points': int(hits.sum()), 'floor_points': int(seen.sum()), 'hit_pix': pix[hits], 'frame': fid}


def draw_path(rt, fid, moves, tag, start_pose=None, hits=None, note=''):
    """The camera image of `fid` with the base footprint drawn at floor level along the moves."""
    _, _, rgb, meta = pframes.load(fid, stride=16)
    img = rgb.copy()
    odom = rt.frame_odom[fid]
    p = tuple(start_pose or rt.pose())
    cols = [(255, 160, 0), (0, 200, 0)]
    for k, m in enumerate(moves):
        poses = footprint.sample_move(p, m)
        for q in poses[::max(1, len(poses) // 6)] + [poses[-1]]:
            poly = footprint.place(footprint.BASE_POLY.buffer(MARGIN), q)
            uv = []
            for x, y in np.array(poly.exterior.coords)[::3]:
                b = geom.odom_to_base(odom, [x, y])
                u, depth = pframes.project(meta, [b[0], b[1], 0.0])
                uv.append(u if depth > 0.1 else [np.nan, np.nan])
            uv = np.array(uv)
            if np.isfinite(uv).all():
                cv2.polylines(img, [uv.astype(np.int32)], True, cols[k % 2], 2)
        p = poses[-1]
    for u, v in (hits if hits is not None else []):
        cv2.circle(img, (int(u), int(v)), 4, (0, 0, 255), -1)
    cv2.putText(img, f'{tag} {note}'[:110], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    path = rt.out / f'path_{tag}_{fid}.jpg'
    cv2.imwrite(str(path), cv2.resize(img, (960, 540)))
    return str(path)


def look_along(rt, move):
    """Frames covering what the next move sweeps: ahead (near and far) for straight moves, both sides for turns."""
    fids = []
    if 'forward' in move and move['forward'] > 0:
        for x in (0.75, min(0.75 + move['forward'] + 0.4, 2.5)):
            fids.append(rt.observe([x, 0.0, 0.0]))
    elif 'turn' in move:
        for side in (1, -1):
            fids.append(rt.observe([0.9, side * 0.6, 0.0]))
    return fids


def check_path(rt, moves, ignore_names=(), tag='check', fids=None, ignore_margin=0.03):
    """Look along the moves (unless frames are given) and count obstacle points in the swept footprint."""
    fids = fids if fids is not None else [f for m in moves for f in look_along(rt, m)]
    rows = [path_obstacles(rt, f, moves, ignore_names=ignore_names, ignore_margin=ignore_margin) for f in fids]
    hits = sum(r['obstacle_points'] for r in rows)
    worst = max(rows, key=lambda r: r['obstacle_points']) if rows else None
    pic = draw_path(rt, worst['frame'], moves, tag, hits=worst['hit_pix'],
                    note=f'hits {hits}') if worst is not None else None
    return {'ok': hits <= MAX_HIT_POINTS, 'obstacle_points': hits,
            'floor_points': sum(r['floor_points'] for r in rows), 'picture': pic, 'frames': fids}


def safe_drive(rt, moves, ignore_names=(), tag='drive', ignore_margin=0.03):
    """Execute moves in closed loop: per segment look along it, check the swept footprint, then move."""
    done = []
    for m in moves:
        key = 'forward' if 'forward' in m else 'turn'
        if key == 'forward' and m[key] < 0:
            raise StepFailed(f'{tag}: reverse moves go through retreat()')
        n = max(1, int(np.ceil(abs(m[key]) / (SEG_FORWARD if key == 'forward' else SEG_TURN))))
        for _ in range(n):
            part = {key: m[key] / n}
            chk = check_path(rt, [part], ignore_names, tag=f'{tag}_segment', ignore_margin=ignore_margin)
            if not chk['ok']:
                raise StepFailed(f'{tag}: {part} blocked by {chk["obstacle_points"]} depth points in the swept '
                                 f'footprint (picture {chk["picture"]})')
            rt.base(**part, slow=key == 'forward')
            done.append(part)
    return done


def stance_for(obj, d, y=0.0):
    """Base-origin target and heading facing the object's long side with its centre d ahead and y to the side."""
    c = np.array(obj['center'])
    s = np.array(obj['short_axis'])
    n = np.array([-s[1], s[0]])
    return c - s * d - n * y, float(np.arctan2(s[1], s[0]))


def pose_problems(obj, d, y, y_tol, x_tol=0.05, max_yaw=0.12):
    c = obj['center']
    la = np.array(obj['long_axis'])
    yaw = float(np.arctan2(-la[0], la[1]))
    p = []
    if abs(c[0] - d) > x_tol:
        p.append(f'centre x {c[0]:.3f} not {d}+-{x_tol}')
    if abs(c[1] - y) > y_tol:
        p.append(f'centre y {c[1]:+.3f} not {y:+.3f}+-{y_tol}')
    if abs(yaw) > max_yaw:
        p.append(f'yaw {yaw:+.3f} beyond +-{max_yaw}')
    return p


# ------------------------------------------------------------------ approach an object for a grasp

def plan_approach(rt, name, stance_d=0.47, stance_y=-0.025, y_tol=0.035, pre_d=1.05):
    """Plan the next approach leg to `name` (a pre-stance pre_d in front of its long side when far or badly
    aligned, else the grasp stance), check what the camera sees of its swept footprint, draw it for the agent."""
    obj = rt.world.in_base(name, rt.odom())
    probs = pose_problems(obj, stance_d, stance_y, y_tol)
    if not probs:
        rt.world.notes['plan'] = {'moves': [], 'purpose': f'{name} at the grasp stance'}
        return {'at_stance': True}
    c = np.array(obj['center'])
    target, h = stance_for(obj, stance_d, stance_y)
    moves = plan_turn_drive_turn(target, h)
    max_turn = max([abs(m.get('turn', 0.0)) for m in moves] + [0.0])
    gap = c[0] - obj['size'][1] / 2 - robot.BUMPER_X
    leg = 'stance'
    if c[0] > pre_d + 0.3 or (gap > 0.5 and max_turn > 0.6):
        target, h = stance_for(obj, pre_d, stance_y)
        moves = plan_turn_drive_turn(target, h)
        leg = 'pre-stance'
    fid = rt.observe([max(0.8, min(2.2, np.hypot(*target))), float(np.clip(target[1], -0.6, 0.6)), 0.0])
    chk = check_path(rt, moves, ignore_names=(name,), tag=f'plan_{leg}', fids=[fid])
    rt.world.notes['plan'] = {'moves': moves, 'purpose': f'{leg} for {name}', 'ignore': [name]}
    rt.log(f'plan {leg} for {name}: {moves}; visible hits {chk["obstacle_points"]}; picture {chk["picture"]}')
    return {'leg': leg, 'moves': moves, 'visible_obstacle_points': chk['obstacle_points'], 'picture': chk['picture'],
            'problems': probs}


def execute_plan(rt):
    """Run world.notes['plan'] with safe_drive (each segment re-checked by perception)."""
    plan = rt.world.notes.get('plan') or {}
    if not plan.get('moves'):
        return {'moved': []}
    done = safe_drive(rt, plan['moves'], ignore_names=plan.get('ignore', ()), tag=plan['purpose'].replace(' ', '_'))
    rt.world.notes['plan'] = None
    return {'moved': done}


def check_stance(rt, name, stance_d=0.47, stance_y=-0.025, y_tol=0.035, replan_step='plan_approach_box'):
    """Re-localize `name` (stops if it moved: contact) and loop back to planning until at the grasp stance."""
    obj = sensing.relocate(rt, name)
    probs = pose_problems(obj, stance_d, stance_y, y_tol)
    rt.log(f'{name} at {np.round(obj["center"], 3)} yaw {obj["yaw_short"]:+.3f}: {probs or "at the grasp stance"}')
    if probs:
        return {'goto': replan_step, 'problems': probs}
    return {'at_stance': True, 'center': obj['center']}


# ------------------------------------------------------------------ stance with a straight final leg

def known_clearance(rt, moves, names, start_pose=None):
    """Smallest distance (m) between the base footprint swept along `moves` (no margin) and the polygons of the
    known objects `names` (world, odometry frame); per object."""
    out = {}
    swept_base = swept(start_pose or rt.pose(), moves, 'none', 0.0)[0][0]
    for n in names:
        if n in rt.world.objects:
            out[n] = round(float(swept_base.distance(object_polygon(rt, n))), 3)
    return out


def facing_side(rt, name, surface):
    """Heading (rad, base frame now) that faces the side of `surface` the object `name` is closest to
    (perpendicular to that edge, pointing into the surface)."""
    tb = rt.world.in_base(surface, rt.odom())
    c = np.array(rt.world.in_base(name, rt.odom())['center'])
    sides = ptable.sides(tb['corners'])
    best = min(sides, key=lambda sd: abs((c - sd['mid']) @ sd['normal']))   # edge nearest to the object
    return float(np.arctan2(-best['normal'][1], -best['normal'][0]))


def plan_stance_straight(rt, name, stance_d=0.55, stance_y=0.0, straight=0.35, avoid=(), min_clear=0.03, face=None,
                         face_blend=1.0):
    """Plan to the stance in front of `name` (its centre stance_d ahead and stance_y to the side) so that the
    LAST leg is a straight forward drive of `straight` m: that leg is the path the robot may later reverse along
    (after a grasp at a table there is no other safe way out). Heading: perpendicular to the edge of surface
    `face` nearest the object (an object skewed on a table would otherwise put a front wheel corner at the
    table edge), else the object's short axis. Turns pivot about the wheel axle; the swept base must stay
    min_clear from the known objects `avoid` (e.g. the table under the box). Draws the plan on the camera image
    and counts the visible depth hits. -> {moves, heading, clearance, visible_obstacle_points, picture}"""
    obj = rt.world.in_base(name, rt.odom())
    h = float(obj['yaw_short'])
    if face:                                   # blend 1: perpendicular to the edge, 0: the object's axis
        h = h + face_blend * float(np.angle(np.exp(1j * (facing_side(rt, name, face) - h))))
    hv, nv = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
    target = np.array(obj['center']) - hv * stance_d - nv * stance_y
    pre = target - hv * straight
    moves = plan_turn_drive_turn(pre, h) + [{'forward': float(straight)}]
    clear = known_clearance(rt, moves, avoid)
    fid = rt.observe([float(np.clip(target[0] + 0.3, 0.8, 2.0)), float(np.clip(target[1], -0.6, 0.6)), 0.0])
    chk = check_path(rt, moves, ignore_names=tuple(avoid) + (name,), tag=f'plan_stance_{name}', fids=[fid])
    rt.world.notes['plan'] = {'moves': moves, 'purpose': f'stance for {name}', 'ignore': list(avoid) + [name]}
    rt.log(f'plan stance for {name}: heading {h:+.3f} moves {moves}; clearance {clear}; visible hits '
           f'{chk["obstacle_points"]}; picture {chk["picture"]}')
    bad = {k: v for k, v in clear.items() if v < min_clear}
    if bad:
        raise StepFailed(f'plan to {name} passes within {bad} m of known objects (picture {chk["picture"]})')
    return {'moves': moves, 'heading': h, 'clearance': clear, 'visible_obstacle_points': chk['obstacle_points'],
            'picture': chk['picture'], 'frame': fid}


# ------------------------------------------------------------------ approach a side of a table

def plan_table_approach(rt, name, side=None, stance_edge=0.385, pre=0.65):
    """Plan the drive to the pre-stance in front of side `side` (index from the find_table picture, chosen by the
    agent) of table `name`; check what the camera sees of it; draw it for the agent."""
    tb = rt.world.in_base(name, rt.odom())
    sides = ptable.sides(tb['corners'])
    rows = [{'side': k, 'mid': np.round(sd['mid'], 2).tolist(), 'len': round(sd['length'], 2)} for k, sd in enumerate(sides)]
    if side is None:
        raise StepFailed(f'choose a side of {name} (picture {sensing.table_sides_picture(rt, name)}): {rows}')
    sd = sides[int(side)]
    p = sd['mid'] + sd['normal'] * (stance_edge + pre)
    hs = float(np.arctan2(-sd['normal'][1], -sd['normal'][0]))
    moves = plan_turn_drive_turn(p, hs)
    fid = rt.observe([float(np.clip(np.hypot(*p), 0.8, 2.2)), float(np.clip(p[1], -0.6, 0.6)), 0.0])
    chk = check_path(rt, moves, tag=f'plan_table_side{side}', fids=[fid])
    rt.world.notes['plan'] = {'moves': moves, 'purpose': f'pre-stance at {name}', 'ignore': []}
    return {'side': int(side), 'moves': moves, 'visible_obstacle_points': chk['obstacle_points'],
            'picture': chk['picture']}


def measure_edge(rt, looks=((0.6, 0.25, 0.68), (0.6, -0.25, 0.68))):
    """Near edge of the table ahead from Gemini's segmentation mask of its top + depth (locate.table_near_edge),
    from the first view (left / right of a held box) that sees enough of the top. -> edge dict or None"""
    for lk in looks:
        fid = rt.observe(list(lk))
        e = ploc.table_near_edge(rt.ctx, fid, log=rt.log)
        if e['ok']:
            return e
        rt.log(f"edge view {fid}: {e.get('why')}")
    return None


def final_table_approach(rt, name, stance_edge=0.385, max_first=1.2, edge_x=None):
    """Drive straight to the table ahead until its near edge is stance_edge ahead of the base origin (bumper ~8 cm
    from it). The edge comes from Gemini's mask of the top (v1's far dark-pixel fit read 0.917 m for ~0.74 m and
    the robot drove too close; the per-segment depth check stopped it). The table is the target: 12 cm around its
    outline are ignored by the path check (drawer fronts / handles).
    edge_x: the agent's own measurement (base frame, m) of the near edge - pass it when the held box hides the near
    top or a second table is in view: 2026-10-09 the mask edge read 1.053 m (the farther of two cabinets) for a top
    0.45 m ahead (drawer face below the box at 0.47 m by pixel depth); the path check then blocked the drive."""
    if edge_x is not None:
        last = float(edge_x) - stance_edge
        rt.log(f'{name} edge {edge_x:.3f} (agent-measured): drive {last:+.3f}')
        if not (-0.05 <= last <= 0.25):
            raise StepFailed(f'agent edge {edge_x:.3f}: {last:+.3f} m to the stance is outside -0.05..0.25')
        if last > 0.01:
            rt.base(forward=last, slow=True)
        rt.world.notes[f'{name}_edge_x'] = float(edge_x) - max(last, 0.0)
        rt.world.notes.setdefault(f'{name}_top_z', 0.70)
        return {'edge_x': round(rt.world.notes[f'{name}_edge_x'], 3), 'top_z': rt.world.notes[f'{name}_top_z'],
                'source': 'agent'}
    e = measure_edge(rt)
    if e is None:
        raise StepFailed(f'{name} edge not measurable (frames {rt.frames[-2:]})')
    first = e['edge_x'] - stance_edge - 0.12
    rt.log(f"{name} edge {e['edge_x']:.3f} ahead: drive {first:+.3f} then measure again")
    if first > max_first:
        raise StepFailed(f"{name} edge {e['edge_x']:.2f} m ahead: too far for a straight final approach")
    if first > 0.02:
        safe_drive(rt, [{'forward': first}], ignore_names=(name,), tag=f'approach_{name}', ignore_margin=0.12)
        e = measure_edge(rt)
        if e is None:
            raise StepFailed(f'{name} edge lost after the approach (frames {rt.frames[-2:]})')
    last = e['edge_x'] - stance_edge
    if not (-0.05 <= last <= 0.25):
        raise StepFailed(f"{name} edge at {e['edge_x']:.3f} after the approach")
    edge = e['edge_x']
    if last > 0.01:
        rt.base(forward=last, slow=True)
        e2 = measure_edge(rt)
        edge = e2['edge_x'] if e2 else edge - last
    rt.world.notes[f'{name}_edge_x'] = edge
    rt.world.notes[f'{name}_top_z'] = e['top_z']
    return {'edge_x': round(edge, 3), 'top_z': round(e['top_z'], 3), 'frame': e['frame']}


def retreat(rt, dist=0.4, tol=0.03):
    """Reverse straight, only back along the straight line the robot just drove forward (the camera cannot see
    behind): allowed up to the odometry-measured forward distance of the latest base motions without a turn since
    (a request up to `tol` longer than that is shortened to it: odometry reads ~2 % short)."""
    back = rt.reversible()
    if dist > back + tol:
        raise StepFailed(f'reverse {dist} m refused: only {back:.2f} m of the path just driven is known (checkpoint)')
    dist = min(dist, back)
    rt.base(forward=-abs(dist), slow=True)
    return {'backed_m': round(dist, 3)}


def check_surface_stance(rt, name, stance_d=0.60, stance_y=0.0, x_tol=0.06, y_tol=0.05, max_yaw=0.15):
    """After the straight final leg: re-localize `name` (stops if it moved) and check it is stance_d ahead within
    tolerances (no replanning next to a table: turning there would sweep the base into it)."""
    obj = sensing.relocate(rt, name, radius=0.3)
    c = obj['center']
    yaw = float(np.arctan2(-obj['long_axis'][0], obj['long_axis'][1]))
    probs = []
    if abs(c[0] - stance_d) > x_tol:
        probs.append(f'centre x {c[0]:.3f} not {stance_d}+-{x_tol}')
    if abs(c[1] - stance_y) > y_tol:
        probs.append(f'centre y {c[1]:+.3f} not {stance_y:+.3f}+-{y_tol}')
    if abs(yaw) > max_yaw:
        probs.append(f'yaw {yaw:+.3f} beyond +-{max_yaw}')
    rt.log(f'{name} at {np.round(c, 3).tolist()} yaw {yaw:+.3f}: {probs or "at the stance"}')
    if probs:
        raise StepFailed(f'{name} not at the stance: {probs} (frame {obj["frame"]})')
    return {'center': c, 'yaw': yaw, 'top_z': obj['top_z'], 'frame': obj['frame']}


def look_around_torso(rt, yaw=1.9, pan=0.47, tilt=0.95, moves=None, ignore_names=()):
    """See beside and behind the robot without moving the base: turn the chest (torso_5) to -yaw and +yaw with the
    head panned further the same way, observe each side, turn back. With pan 0.47 and a ~100 deg camera the two
    views cover about 86-186 deg to each side (the front views cover up to ~77 deg), i.e. where the base's rear
    corners swing during in-place turns (0.62 m from the wheel axle). If `moves` is given (default: the stored
    plan), the depth points of these views inside the footprint swept along them are counted.
    -> {frames, obstacle_points (per frame), ok}"""
    s = rt.status()
    torso0, head0 = list(s['torso']), list(s['head'])
    frames, hits = [], {}
    moves = moves if moves is not None else (rt.world.notes.get('plan') or {}).get('moves')
    ignore = tuple(ignore_names) or tuple((rt.world.notes.get('plan') or {}).get('ignore', ()))
    try:
        for sgn in (-1, 1):
            t = list(torso0)
            t[5] = sgn * yaw
            robot.torso_to(rt.ctx, t, rt.log)              # split under the 1.2 rad per-call cap
            rt.v('rby1_move_joints', component='head', positions=[sgn * pan, tilt])
            fid = rt.observe()
            frames.append(fid)
            if moves:
                r = path_obstacles(rt, fid, moves, ignore_names=ignore)
                hits[fid] = r['obstacle_points']
                draw_path(rt, fid, moves, f'look_around_{"right" if sgn < 0 else "left"}', hits=r['hit_pix'],
                          note=f"hits {r['obstacle_points']}")
    finally:
        robot.torso_to(rt.ctx, torso0, rt.log)
        rt.v('rby1_move_joints', component='head', positions=head0)
    total = sum(hits.values())
    rt.log(f'look around (torso yaw +-{yaw}): frames {frames} swept-footprint hits {hits}')
    return {'frames': frames, 'obstacle_points': hits, 'ok': total <= MAX_HIT_POINTS}
