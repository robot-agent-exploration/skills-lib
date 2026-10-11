"""Step primitives for the bimanual bucket pick-and-place (rim pinch with OPEN fingers, both grippers).

Every step is fn(rt, **kw) -> dict and raises StepFailed when a precondition or completion check fails; rt is the
library runner's Runtime (bimanual-box-pnp core/runner.py): rt.ctx (guarded tools), rt.world (objects in the
odometry frame, held object, notes), rt.observe(point), rt.base(...), rt.log. Parameters measured on the robot
2026-10-09 (memory/TIPS.md): 5-gal Homer bucket (rim r 0.152, h 0.37, thin wall), black-top cabinet (top 0.67 m)."""
import json
import time

import numpy as np
from shapely.geometry import Polygon

from lib import BOXPNP_CODE, frames, vlm  # noqa: F401
from boxpnp.core import geom, robot
from boxpnp.core.runner import StepFailed

import grasp_plan as gp
import nav
import perception as pc

TORSO_MID = [0, 0.785, -1.571, 0.785, 0, 0]
TORSO_READY = [0, 0.524, -1.047, 0.524, 0, 0]
TORSO_ZERO = [0, 0, 0, 0, 0, 0]
TORSO_GRASP = [0, 1.047, -2.094, 1.247, 0, 0]          # low + lean 0.2 (lean steps 0.1)
TORSO_LIFT1 = [0, 0.85, -1.7, 0.95, 0, 0]
ARM_READY = {'right': [-0.7854, -0.5236, 0, -1.5708, 0, 0.7854, 0], 'left': [-0.7854, 0.5236, 0, -1.5708, 0, 0.7854, 0]}
STANCE_D = 0.595         # bucket centre ahead of the base origin at the grasp: pinch IK ok for 0.58-0.60
                         # (y -0.04..+0.045, rim 0.37), fails at 0.62; knee clearance >= 3 cm needs >= 0.58
TILT = 0.6               # finger axis tilted forward (rad)
PINCH_DEPTH = 0.05       # pinch point below the rim top (fingers reach ~0.2 m along -z_ee)
PRE_DZ = 0.10            # pre-grasp height above the grasp pose
CLOSE_A = 0.35           # gripper current (right first, then left)
PINCH_MAX_W = 0.015      # a pinch of the thin wall reads 3-11 mm
CARRY_STEPS = ([-0.03, 0, 0.10], [0, 0, 0.15], [0, 0, 0.04])   # hand moves at torso zero after the lift
HELD_DF_MIN = 6.0        # summed wrist force change after the lift (bucket ~1.5 kg)
EDGE_IN = 0.21           # bucket centre inside the cabinet edge when placing
BUMPER_X, BUMPER_GAP = 0.33, 0.06
TOUCH_DF = 4.0
LOAD_STOP = 140.0
BUCKET_R = 0.16


def pins(torso):
    return [{'joint': f'torso_{i}', 'position': float(v)} for i, v in enumerate(torso)]


def loads(s):
    return {k: float(v.rstrip('%')) for k, v in s['safety']['highest_torque_load'].items()}


def wrist(rt, n=4):
    acc = []
    for _ in range(n):
        s = rt.status()
        acc.append([s['wrist_force_n']['right'], s['wrist_force_n']['left']])
        time.sleep(0.05)
    return np.mean(acc, axis=0), s


def hands(rt, d, torso, speed=0.02):
    """Both hands by d (base frame, relative) in one controller Cartesian body command, torso pinned."""
    return rt.v('rby1_cartesian', component='body', linear_velocity=speed, joint_targets=pins(torso),
                targets=[{'link': 'ee_left', 'position': [float(v) for v in d], 'relative': True},
                         {'link': 'ee_right', 'position': [float(v) for v in d], 'relative': True}])


def arms_to(rt, goal, max_step=1.1):
    """Both arms to joint goals {'right': q, 'left': q} in equal sub-moves under the 1.2 rad per-call cap."""
    s = rt.status()
    cur = {k: np.array(s[f'{k}_arm']) for k in ('right', 'left')}
    diff = max(float(np.abs(np.array(goal[k]) - cur[k]).max()) for k in cur)
    n = max(1, int(np.ceil(diff / max_step)))
    for i in range(1, n + 1):
        rt.v('rby1_move_body', targets={f'{k}_arm': {'positions': (cur[k] + (np.array(goal[k]) - cur[k]) * i / n).tolist()}
                                         for k in cur})


def torso_to(rt, goal, max_step=1.1):
    cur = np.array(rt.status()['torso'])
    goal = np.array(goal, float)
    n = max(1, int(np.ceil(np.abs(goal - cur).max() / max_step)))
    for i in range(1, n + 1):
        rt.v('rby1_move_joints', component='torso', positions=(cur + (goal - cur) * i / n).tolist())


def base_pose(rt):
    o = rt.odom()
    return (o['x'], o['y'], o['yaw'])


# ------------------------------------------------------------------ prepare / perception

def prepare(rt):
    s = rt.status()
    if not s['motion']['armed'] or s['emo_pressed'] or s['safety']['fault']:
        raise StepFailed(f"robot not ready: armed {s['motion']['armed']} emo {s['emo_pressed']} fault {s['safety']['fault']}")
    for side in ('right', 'left'):
        if (s['grippers']['openings_m'][side] or 0) < 0.09:
            rt.v('rby1_gripper', side=side, action='open')
    if np.abs(np.array(s['torso']) - TORSO_MID).max() > 0.02:
        arms_to(rt, ARM_READY)
        torso_to(rt, TORSO_MID)
    arms_to(rt, ARM_READY)
    rt.world.held = None
    return {'odometry': rt.odom()}


def find_bucket(rt, looks=((1.0, 0.0, 0.1), (1.0, 0.45, 0.1), (1.0, -0.45, 0.1))):
    """Gemini picks the orange bucket (not the red one); silhouette + depth give the centre. Stores world 'bucket'."""
    for lk in looks:
        fid = rt.observe(list(lk))
        d = pc.locate_bucket(rt.ctx, fid, log=rt.log)
        if d.get('ok') and not d.get('touches_border'):
            odom = rt.frame_odom[fid]
            rt.world.objects['bucket'] = {'kind': 'bucket', 'center_odom': geom.base_to_odom(odom, d['center'])[:2].tolist(),
                                          'rim_z': d['rim_z'], 'height': d['height'], 'frame': fid, 'iou': d['iou']}
            pic = pc.draw_bucket(fid, d, rt.out / f'bucket_{fid}.jpg')
            return {'frame': fid, 'picture': pic, 'center_base': d['center'], 'rim_z': d['rim_z'], 'iou': d['iou'],
                    'upright': d.get('upright'), 'fully_visible': d.get('fully_visible')}
        rt.log(f'no bucket in {fid}: {d.get("why")}')
    raise StepFailed('bucket not found in the looks')


CAB_DESC = ('the black top surface of the white drawer cabinet nearest to the robot (a cardboard box may or may not '
            'stand on it)')   # v2: the top may be empty (compositional task, 2026-10-09)


def find_cabinet(rt, looks=((1.3, 0.0, 0.6), (1.3, 0.5, 0.6), (1.3, -0.5, 0.6))):
    """Gemini segments the cabinet's black top; depth points of the mask at table height -> rectangle (odom) and
    top height; Gemini points at the cardboard box on it -> box centre. Stores world 'cabinet' and 'box'."""
    for lk in looks:
        fid = rt.observe(list(lk))
        seg = vlm.segment(rt.ctx, fid, CAB_DESC, log=rt.log)
        if not seg['ok']:
            continue
        P, pix, rgb, meta = frames.load(fid, stride=2, max_depth=4.0, drop_edges=True)
        inm = seg['mask'][pix[:, 1], pix[:, 0]] > 0
        top = P[inm & (P[:, 2] > 0.5) & (P[:, 2] < 0.9)]
        if len(top) < 300:
            rt.log(f'{fid}: only {len(top)} top points')
            continue
        top_z = float(np.median(top[:, 2]))
        top = top[np.abs(top[:, 2] - top_z) < 0.04]
        import cv2
        rect = cv2.minAreaRect(top[:, :2].astype(np.float32))
        corners = cv2.boxPoints(rect)
        g = vlm.point(rt.ctx, fid, 'Robot head-camera image.', points={'box': 'centre of the top face of the cardboard box '
                      'standing on the black cabinet top'}, flags={'box_visible': 'a cardboard box stands on the black top'},
                      log=rt.log)
        odom = rt.frame_odom[fid]
        cab = {'kind': 'cabinet', 'corners_odom': [geom.base_to_odom(odom, c)[:2].tolist() for c in corners],
               'top_z': top_z, 'frame': fid, 'size': [float(min(rect[1])), float(max(rect[1]))]}
        rt.world.objects['cabinet'] = cab
        out = {'frame': fid, 'top_z': round(top_z, 3), 'size': cab['size'],
               'corners_base': np.round(corners, 3).tolist()}
        if g.get('ok') and g['flags'].get('box_visible') and g['px'].get('box'):
            p = frames.pixel_point(fid, *g['px']['box'], radius=6)
            if p is not None:
                rt.world.objects['box'] = {'kind': 'box', 'center_odom': geom.base_to_odom(odom, p[:2])[:2].tolist(),
                                           'top_z': float(p[2])}
                out['box_base'] = np.round(p, 3).tolist()
        return out
    raise StepFailed('cabinet top not found')


def _obstacles(rt, names=('bucket', 'cabinet')):
    obs = {}
    for n in names:
        o = rt.world.objects.get(n)
        if not o:
            continue
        if 'corners_odom' in o:
            obs[n] = Polygon(o['corners_odom']).buffer(0.02)
        else:
            obs[n] = nav.circle(o['center_odom'], BUCKET_R if n == 'bucket' else 0.25)
    return obs


def _draw_plan(rt, moves, tag, extra=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pose = base_pose(rt)
    odom = {'x': pose[0], 'y': pose[1], 'yaw': pose[2]}
    fig, ax = plt.subplots(figsize=(6, 6))
    tb = lambda xy: np.array([geom.odom_to_base(odom, q)[:2] for q in xy])
    g = nav.swept(pose, moves)
    for part in (getattr(g, 'geoms', None) or [g]):
        xy = tb(np.array(part.exterior.coords)); ax.fill(xy[:, 0], xy[:, 1], alpha=0.3)
    for k, o in _obstacles(rt, ('bucket', 'cabinet', 'box')).items():
        xy = tb(np.array(o.exterior.coords)); ax.plot(xy[:, 0], xy[:, 1], 'r'); ax.text(*xy.mean(0), k)
    ax.plot(0, 0, 'k^'); ax.set_aspect('equal'); ax.grid(); ax.set_title(f'{tag}: robot frame now')
    path = rt.out / f'plan_{tag}.png'
    plt.savefig(path, dpi=70); plt.close(fig)
    return str(path)


# ------------------------------------------------------------------ base motion

def plan_grasp_approach(rt, heading_offset=0.0, final_leg=0.5):
    """Stance with the bucket centre STANCE_D ahead, facing it along the current bearing rotated by heading_offset
    (rad, about the bucket; the agent sets +-pi/2 when the handle ears face the robot's sides)."""
    b = np.array(rt.world.objects['bucket']['center_odom'])
    pose = base_pose(rt)
    bb = geom.odom_to_base({'x': pose[0], 'y': pose[1], 'yaw': pose[2]}, b)[:2]
    if abs(heading_offset) < 1e-3 and STANCE_D - 0.05 <= bb[0] <= 1.3 and abs(bb[1]) < 0.04:
        rt.world.notes['plan'] = {'moves': [], 'purpose': 'grasp stance (already facing the bucket)'}
        return {'moves': [], 'bucket_base': np.round(bb, 3).tolist(), 'note': 'aligned: final_approach drives the rest'}
    h = float(np.arctan2(b[1] - pose[1], b[0] - pose[0])) + float(heading_offset)
    goal = b - STANCE_D * np.array([np.cos(h), np.sin(h)])
    obs = _obstacles(rt)
    grid = nav.ring_grid(b, radii=(0.8, 1.0, 1.25), n=24)
    r = None
    for fl in (final_leg, 0.4, 0.3):
        r = nav.plan_route(pose, goal, h, obs, min_clear=0.12, final_straight=fl, grid=grid, target='bucket',
                           target_clear=0.05)
        if r:
            break
    if not r:
        raise StepFailed('no collision-free route to the grasp stance')
    rt.world.notes['plan'] = {'moves': r['moves'], 'purpose': 'grasp stance', 'ignore': ['bucket']}
    pic = _draw_plan(rt, r['moves'], 'grasp_approach')
    return {'moves': r['moves'], 'clearance': r['clearance'], 'picture': pic}


def _look_along(rt, move):
    if 'forward' in move and move['forward'] > 0:
        return rt.observe([0.9 + min(move['forward'], 0.8), 0.0, 0.0])
    return rt.observe([0.9, 0.45 if move.get('turn', 0) > 0 else -0.45, 0.0])


def drive_plan(rt, max_hits=25, seg_forward=0.5, seg_turn=0.9):
    """Run world.notes['plan']: each segment is looked along first and refused when depth points (robot, held
    bucket and ignored objects removed) lie in its swept footprint."""
    plan = rt.world.notes.get('plan') or {}
    held = rt.world.held is not None
    done = []
    for m in plan.get('moves', []):
        key = 'forward' if 'forward' in m else 'turn'
        if key == 'forward' and m[key] < 0:
            raise StepFailed('plans never reverse')
        n = max(1, int(np.ceil(abs(m[key]) / (seg_forward if key == 'forward' else seg_turn))))
        for _ in range(n):
            part = {key: m[key] / n}
            fid = _look_along(rt, part)
            ign = [_obstacles(rt, (k,))[k] for k in plan.get('ignore', []) if _obstacles(rt, (k,))]
            if held:
                p = base_pose(rt)
                c = np.array(p[:2]) + 0.62 * np.array([np.cos(p[2]), np.sin(p[2])])
                ign.append(nav.circle(c, 0.30))
            hits, _ = nav.depth_hits(fid, rt.frame_odom[fid], base_pose(rt), [part], ignore=ign,
                                     zmax=0.6 if held else 1.8)
            rt.log(f'segment {part}: {hits} depth hits (frame {fid})')
            if hits > max_hits:
                raise StepFailed(f'segment {part} blocked: {hits} depth points in the swept footprint (frame {fid})')
            rt.base(**part, slow=True)
            done.append(part)
    rt.world.notes['plan'] = None
    return {'moved': done}


def final_approach(rt, tol_y=0.04, tol_x=0.015):
    """Re-localize the bucket and drive straight until it is STANCE_D ahead (two steps). Returns the close view for
    the ears checkpoint."""
    for _ in range(3):
        fid = rt.observe([0.9, 0.0, 0.2])
        d = pc.locate_bucket(rt.ctx, fid, ask_details=False, log=rt.log)
        if not d.get('ok'):
            raise StepFailed(f'bucket lost: {d.get("why")}')
        cx, cy = d['center']
        rt.world.objects['bucket'].update(center_odom=geom.base_to_odom(rt.frame_odom[fid], d['center'])[:2].tolist(),
                                          rim_z=d['rim_z'], frame=fid)
        if abs(cy) > tol_y:
            raise StepFailed(f'bucket {cy:+.3f} m off the base axis: replan the approach (frame {fid})')
        dx = cx - STANCE_D
        if abs(dx) <= tol_x:
            pic = pc.draw_bucket(fid, d, rt.out / f'stance_{fid}.jpg')
            return {'frame': fid, 'picture': pic, 'center_base': d['center'], 'rim_z': d['rim_z']}
        step = dx if abs(dx) < 0.25 else 0.6 * dx
        if step < -0.15:
            raise StepFailed(f'bucket too close ({cx:.2f} m)')
        rt.base(forward=float(step), slow=True)
    raise StepFailed('stance not reached')


# ------------------------------------------------------------------ grasp

def _measure_rim(rt):
    fid = rt.observe([STANCE_D, 0.0, 0.3])
    d = pc.locate_bucket(rt.ctx, fid, ask_details=False, log=rt.log)
    if not d.get('ok'):
        raise StepFailed(f'bucket not localized at the stance: {d.get("why")}')
    _, _, rgb, _ = frames.load(fid, stride=16)
    tm = pc.target_mask(rgb, d['box_px'])
    P, _ = pc.mask_points(fid, tm & pc.orange_mask(rgb), stride=1)
    r = pc.rim_from_points(P, support_z=d['floor_z'])
    c = np.array(d['center'])
    rim = d['rim_z']
    if r.get('ok') and r['fit'] == 'rim band' and r['fit_residual_m'] < 0.01 and np.linalg.norm(np.array(r['center']) - c) < 0.03:
        c, rim = (c + np.array(r['center'])) / 2, (rim + r['rim_z']) / 2
    return fid, c, rim


def pregrasp(rt):
    """Bucket at the stance: IK for the rim pinch (closing axis base y, fingers tilted TILT forward), arms to the
    pre-grasp at torso mid, then torso mid -> low -> lean +0.1 -> +0.2 with the arms fixed."""
    fid, c, rim = _measure_rim(rt)
    pins_ = gp.pinch_points(c, rim, depth=PINCH_DEPTH)
    q = {}
    for stage, dz in (('pre', PRE_DZ), ('grasp', 0.0)):
        for side in ('left', 'right'):
            R = gp.grip_rotation(side, TILT, flip=True)
            seeds = [q[('pre', side)]['q']] if stage == 'grasp' else None
            q[(stage, side)] = gp.solve_arm(TORSO_GRASP, side, gp.ee_for_pinch(pins_[side] + [0, 0, dz], R), R, seeds=seeds)
    bad = {f'{k[0]}_{k[1]}': (round(v['pos_err'], 4), round(v['margin'], 3)) for k, v in q.items()
           if v['pos_err'] > 0.002 or v['margin'] < 0.1}
    if bad:
        raise StepFailed(f'pinch IK fails: {bad} (bucket {np.round(c, 3)} rim {rim:.3f})')
    kc = gp.knee_clearance(TORSO_GRASP, c, rim)
    if kc < 0.03:
        raise StepFailed(f'knee clearance {kc:.3f} m: bucket too close')
    fk = rt.v('rby1_fk', joints={'torso': {'positions': TORSO_GRASP}, 'right_arm': {'positions': q[('pre', 'right')]['q']},
                                 'left_arm': {'positions': q[('pre', 'left')]['q']}})
    if fk['self_collisions']:
        raise StepFailed(f'pre-grasp self-collision {fk["self_collisions"]}')
    grasp_ee_z = float(rim - PINCH_DEPTH + gp.PINCH_D * np.cos(TILT))
    rt.world.grip = {'grasp_ee_z': grasp_ee_z, 'rim': rim, 'center': c.tolist()}
    arms_to(rt, {'right': q[('pre', 'right')]['q'], 'left': q[('pre', 'left')]['q']})
    for t in ([0, 1.047, -2.094, 1.047, 0, 0], [0, 1.047, -2.094, 1.147, 0, 0], TORSO_GRASP):
        rt.v('rby1_move_joints', component='torso', positions=t)
    s = rt.status()
    fid2 = rt.observe([STANCE_D, 0.0, rim])
    return {'frame': fid2, 'rim_z': round(rim, 3), 'center_base': np.round(c, 3).tolist(), 'knee_clearance': round(kc, 3),
            'ee_z': s['right_ee']['position'][2], 'grasp_ee_z': round(grasp_ee_z, 3), 'loads': loads(s)}


def relieve_pinch_load(rt, step=0.003, high=110.0):
    """Two pinches on one thin rim over-constrain the bucket: a few mm of mismatch loads the wrists (arm_5, rated
    10 N m) towards the 150 % guard. Move the hand of the most loaded wrist 3 mm inward (towards the other hand);
    keep it if that wrist's load drops, else go 3 mm outward instead (C1: left 3 mm inward 144 -> 118 %)."""
    def worst(s):
        L = {k: v for k, v in loads(s).items() if k in ('left_arm_5', 'right_arm_5')}
        return max(L.items(), key=lambda kv: kv[1]) if L else (None, 0.0)
    s = rt.status()
    joint, val = worst(s)
    if joint is None or val < high:
        return {'loads': loads(s)}
    side = joint.split('_')[0]
    inward = -1.0 if side == 'left' else 1.0
    for sign in (inward, -inward):
        rt.v('rby1_cartesian', component=f'{side}_arm', linear_velocity=0.01,
             targets=[{'link': f'ee_{side}', 'position': [0, sign * step, 0], 'relative': True}])
        s2 = rt.status()
        v2 = loads(s2).get(joint, 0.0)
        rt.log(f'relief: {side} hand {sign * step:+.3f} y: {joint} {val:.0f} -> {v2:.0f} %')
        if v2 < val:
            return {'loads': loads(s2), 'moved': {side: sign * step}}
        rt.v('rby1_cartesian', component=f'{side}_arm', linear_velocity=0.01,
             targets=[{'link': f'ee_{side}', 'position': [0, -sign * step, 0], 'relative': True}])
    return {'loads': loads(rt.status())}


def grasp_lift(rt):
    """Descend to the pinch height, close RIGHT then LEFT (CLOSE_A), then lift at once: torso up with the arms
    fixed (the low stretched stance loads the wrists ~100 %), torso zero, hands up to the carry height."""
    target = rt.world.grip['grasp_ee_z']
    F0, s = wrist(rt)
    z = s['right_ee']['position'][2]
    for dz in (min(0.06, z - target), z - 0.06 - target if z - target > 0.06 else 0.0):
        if dz > 0.003:
            try:
                hands(rt, [0, 0, -dz], TORSO_GRASP, speed=0.015)
            except RuntimeError as e:          # e.g. the wrist-shin self-collision guard near ee z 0.472
                rt.log(f'descent stopped: {str(e)[:200]}')
                break
        F, s = wrist(rt)
        dF = np.linalg.norm(F - F0, axis=1)
        if dF.max() > 8.0:
            hands(rt, [0, 0, 0.05], TORSO_GRASP)
            raise StepFailed(f'contact while descending (dF {np.round(dF, 1)}): a finger is on the rim; backed off 5 cm')
    widths = {}
    for side in ('right', 'left'):
        try:
            r = rt.v('rby1_gripper', side=side, action='close', current_a=CLOSE_A)
            widths[side] = r['measured_width_m']
        except RuntimeError as e:          # the 2nd pinch can trip a wrist torque guard (C1: left_arm_5 154 %)
            rt.log(f'{side} close stopped: {str(e)[:200]}')
            widths[side] = rt.status()['grippers']['openings_m'][side]
    s = rt.status()
    rt.log(f'pinch widths {widths} loads {loads(s)}')
    if max(widths.values()) > PINCH_MAX_W or min(widths.values()) < 0.002:
        for side in ('right', 'left'):
            rt.v('rby1_gripper', side=side, action='open')
        hands(rt, [0, 0, 0.10], TORSO_GRASP)
        raise StepFailed(f'pinch not as expected (widths {widths}); opened and lifted the hands')
    relieve_pinch_load(rt)
    Fp, _ = wrist(rt)
    rows = []
    for t in (TORSO_LIFT1, TORSO_READY, TORSO_ZERO):
        try:
            rt.v('rby1_move_joints', component='torso', positions=t)
        except RuntimeError as e:      # trial 2 (2026-10-10): left_arm_5 150 % stop half way LIFT1 -> READY; holding
            if '150%' not in str(e) and 'torque' not in str(e):    # still the load fell to 137 %, and re-sending the
                raise                  # same target stood up fine (operator: "just try lift it"): loads 61 % at zero
            rt.log(f'lift: torso move stopped ({str(e)[:160]}); re-sending the same target once')
            rt.v('rby1_move_joints', component='torso', positions=t)
        s = rt.status()
        rows.append(loads(s))
        rt.log(f'torso {t}: loads {loads(s)}')
    rt.world.held = 'bucket'
    for d in CARRY_STEPS:
        hands(rt, d, TORSO_ZERO, speed=0.03)
    F1, s = wrist(rt)
    dF = float(np.abs(F1[:, 2] - Fp[:, 2]).sum())
    fid = rt.observe([0.6, 0.0, 0.8])
    out = {'widths': widths, 'loads': loads(s), 'held_dFz': round(dF, 1), 'ee_z': s['right_ee']['position'][2],
           'frame': fid, 'lift_loads': rows}
    if dF < HELD_DF_MIN:
        out['warning'] = f'wrist z forces changed only {dF:.1f} N: is the bucket held?'
    return out


# ------------------------------------------------------------------ place

def plan_place(rt, final_leg=0.45, min_box_dist=0.42, edge_choice=None):
    """Target on the cabinet top EDGE_IN inside an edge, far enough from the box; stance facing that edge with the
    held bucket 0.62 ahead. Prefers the edge facing the robot. -> plan in world.notes."""
    cab = rt.world.objects['cabinet']
    C = np.array(cab['corners_odom'])
    box = rt.world.objects.get('box')
    pose = base_pose(rt)
    centre = C.mean(0)
    cands = []
    for k in range(4):
        a, b = C[k], C[(k + 1) % 4]
        e = b - a
        L = float(np.linalg.norm(e))
        u = e / L
        n_in = np.array([-u[1], u[0]])
        if (centre - a) @ n_in < 0:
            n_in = -n_in
        for t in np.linspace(0.22, L - 0.22, 7) if L > 0.44 else [L / 2]:
            p = a + u * t + n_in * EDGE_IN
            if box and np.linalg.norm(p - np.array(box['center_odom'])) < min_box_dist:
                continue
            stance = p - 0.62 * n_in
            dist = float(np.linalg.norm(stance - np.array(pose[:2])))
            cands.append((dist, k, p, n_in, stance))
    if edge_choice is not None:
        cands = [c for c in cands if c[1] == int(edge_choice)]
    if not cands:
        raise StepFailed('no free placement point on the cabinet top')
    dist, k, p, n_in, stance = min(cands, key=lambda c: c[0])
    h = float(np.arctan2(n_in[1], n_in[0]))
    obs = _obstacles(rt, ('cabinet',))
    r = nav.plan_route(pose, stance, h, obs, min_clear=0.10, final_straight=final_leg,
                       grid=nav.ring_grid(stance, radii=(0.6, 0.9, 1.2), n=24), target='cabinet', target_clear=0.04)
    if not r:
        raise StepFailed('no route to the placing stance')
    rt.world.notes['plan'] = {'moves': r['moves'], 'purpose': 'place stance', 'ignore': ['cabinet', 'box']}
    rt.world.notes['place'] = {'target_odom': p.tolist(), 'edge': k, 'heading': h}
    pic = _draw_plan(rt, r['moves'], 'place')
    return {'moves': r['moves'], 'edge': k, 'target_odom': np.round(p, 3).tolist(), 'picture': pic}


def _edge_ahead(rt, top_z):
    fid = rt.observe([0.9, 0.0, top_z])
    P, _, _, meta = frames.load(fid, stride=2, max_depth=2.5, drop_edges=True)
    from boxpnp.core import footprint
    P = P[~footprint.self_mask(P, meta['joints'])]
    sel = (np.abs(P[:, 2] - top_z) < 0.03) & (np.abs(P[:, 1]) < 0.22) & (P[:, 0] > 0.3) & (P[:, 0] < 1.6)
    if sel.sum() < 100:
        return None, fid
    return float(np.percentile(P[sel, 0], 3)), fid


def held_center(rt):
    """Base-frame xy of the held bucket's centre: mean of the two pinch points (gp.PINCH_D along -z_ee)."""
    from scipy.spatial.transform import Rotation
    s = rt.status()
    pts = [np.array(s[f'{k}_ee']['position']) - gp.PINCH_D * Rotation.from_quat(s[f'{k}_ee']['quaternion']).as_matrix()[:, 2]
           for k in ('right', 'left')]
    return np.mean(pts, axis=0)[:2]


def approach_edge(rt, bucket_x=None, edge_x=None):
    """Straight approach until the held bucket's centre is EDGE_IN past the top's near edge (depth at table
    height); the bumper keeps BUMPER_GAP from it. edge_x: the agent's pixel-depth measurement of the near edge (base
    frame) - v4 (trial 3, 2026-10-10): the held bucket hid the edge from the camera ('cabinet edge not seen ahead');
    the drawer face / top edge beside the bucket gave it. One straight drive, capped by the bumper gap."""
    top_z = rt.world.objects['cabinet']['top_z']
    if bucket_x is None:
        bucket_x = float(held_center(rt)[0])
    if edge_x is not None:
        d = min(float(edge_x) + EDGE_IN - bucket_x, float(edge_x) - BUMPER_X - BUMPER_GAP)
        rt.log(f'edge {float(edge_x):.3f} m ahead (agent) -> drive {d:+.3f}')
        if d < -0.02 or d > 0.8:
            raise StepFailed(f'agent edge {edge_x}: drive {d:+.3f} outside -0.02..0.8')
        if d > 0.02:
            rt.base(forward=float(d), slow=True)
        return {'edge_x': round(float(edge_x) - max(d, 0.0), 3), 'source': 'agent'}
    for _ in range(3):
        edge, fid = _edge_ahead(rt, top_z)
        if edge is None:
            raise StepFailed(f'cabinet edge not seen ahead (frame {fid})')
        d = edge + EDGE_IN - bucket_x
        d = min(d, edge - BUMPER_X - BUMPER_GAP)
        rt.log(f'edge {edge:.3f} m ahead -> drive {d:+.3f}')
        if abs(d) < 0.02:
            return {'edge_x': round(edge, 3), 'frame': fid}
        if d < 0:
            raise StepFailed(f'already past the stance (edge {edge:.2f})')
        rt.base(forward=float(min(d, 0.3)), slow=True)
    edge, fid = _edge_ahead(rt, top_z)
    return {'edge_x': edge, 'frame': fid}


def lower_to_touch(rt, t1_max=0.75, step=0.06):
    """Lower the held bucket by squatting the torso with the arms fixed (chest upright: the arm loads stay as they
    are; lowering with the arms at torso zero ran right_arm_5 into the 150 % guard twice). Stops at the first wrist
    force change > TOUCH_DF (touchdown). A tool stop is NOT a touchdown: it raises StepFailed."""
    F0, s = wrist(rt)
    t1 = float(s['torso'][1])
    while t1 < t1_max - 1e-6:
        t1 = min(t1_max, t1 + step)
        try:
            rt.v('rby1_move_joints', component='torso', positions=[0, t1, -2 * t1, t1, 0, 0])
        except RuntimeError as e:
            raise StepFailed(f'lowering stopped by a guard at torso_1 {t1:.2f}: {str(e)[:200]}')
        F, s = wrist(rt)
        dF = np.linalg.norm(F - F0, axis=1)
        rt.log(f'torso_1 {t1:.2f}: ee z {s["right_ee"]["position"][2]:.3f} dF {np.round(dF, 1)} loads {loads(s)}')
        if dF.max() > TOUCH_DF:
            fid = rt.observe([0.62, 0.0, 0.8])
            return {'touch_dF': np.round(dF, 1).tolist(), 'ee_z': s['right_ee']['position'][2], 'torso_1': t1, 'frame': fid}
    raise StepFailed(f'no touchdown down to torso_1 {t1_max}')


def release(rt):
    for side in ('left', 'right'):
        rt.v('rby1_gripper', side=side, action='open')
    rt.world.held = None
    hands(rt, [0, 0, 0.12], rt.status()['torso'], speed=0.03)
    arms_to(rt, ARM_READY)
    rt.base(forward=-0.3, slow=True)
    torso_to(rt, TORSO_MID)
    fid = rt.observe([1.0, 0.0, 0.7])
    return {'frame': fid}


def verify(rt):
    """Final view from torso ready (camera ~1.43 m) over the bucket to the far part of the top, where the box is:
    from the release posture (torso mid) the bucket hid the box and Gemini answered no (C1)."""
    torso_to(rt, TORSO_READY)
    fid = rt.observe([1.25, 0.0, 0.75])
    r = vlm.check(rt.ctx, fid, 'Robot head-camera image. Is an orange bucket standing upright on the black top of the '
                  'white cabinet, with a cardboard box also on that same black top (it may be partly behind the '
                  'bucket)?', log=rt.log)
    return {'frame': fid, 'gemini': r.get('answer'), 'reason': r.get('reason')}
