"""Air fryer task: open the drawer, drop the banana into the basket, close the drawer.

Run from the workspace root:  python <skill>/code/controller.py  [--steps open,banana,close]
(the library entry is run.py: main(ctx, ...) -> {ok, measured, why}).
Every step re-perceives the scene with the exterior cameras, plans with FR3 IK + a coarse
collision model, acts, verifies, and returns {'ok': ..., ...}. main() stops at the first failure.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collide import Fryer, clearance, clearance_holding, clearance_holding_ball, set_camera_obstacles  # noqa: E402
from geom import load_frame                                             # noqa: E402
from kin import fk, fk_all, ik, quat_tool, safe_margin, HOME, SAFE_MIN, SAFE_MAX  # noqa: E402
from motion import Robot, MotionError                                   # noqa: E402
import perception as per                                                # noqa: E402

# Run logs go to output/runs/ under the working directory (the workspace root, harness convention).
# Library import: was Path(__file__).parents[2], i.e. the session workspace's layout code/airfry/.
ROOT = Path('.')
Z = np.array([0.0, 0.0, 1.0])

# ---- geometry learned in manual run 1 (offsets along the drawer axis d, metres) ----------
FACE22_MODEL = 0.1674        # grip outer face at z=0.22 in the collision-model frame (closed)
GRASP_FROM_FACE = -0.0116    # handle grasp TCP along d, relative to face(z=0.22)
GRASP_BELOW_ARM_TOP = 0.009  # handle grasp TCP z below the arm top
HANDLE_TILT = np.deg2rad(5)  # approach axis tilted toward the fryer
BASKET_FROM_FACE = -0.1474   # basket rim centre (closed drawer) relative to face(z=0.22)
PROTRUSION_CLOSED = 0.065    # grip face - body front below this = drawer closed (closed ~0.056-0.059)
PULL = 0.125                 # drawer pull distance (tested 0.126; front tips ~2-3 cm on release)
PULL_STEP = 0.03
GRASP_FORCE_HANDLE = 50.0    # N
BANANA_FORCE = 20.0          # N
DROP_BETA = np.deg2rad(60)   # hand tilt about the banana axis at the drop
DROP_MID_FROM_BASKET = 0.015  # banana middle beyond basket centre along d
DROP_Z = 0.36                # TCP height at release (housing clears fryer top)
ROT_SPOT = (0.26, -0.05, 0.50)  # (along d, along t, z) where the held banana is re-oriented
FINGER_HALF = 0.00875        # closed-finger half thickness along hand x
PUSH_Z_BELOW_ARM_TOP = 0.0205
PERC_CAMS = ('left',)        # right exterior camera moved after a collision: unusable
VIEW_CAMS = ('wrist', 'left')
VIEW_QUAT = np.array([1.0, 0.0, 0.0, 0.0])   # wrist camera looks down and slightly forward
VIEW_DRAWER = (0.215, 0.0, 0.50)  # (along d, along t, z) wrist view of handle + open basket
TOP_VIEW_OFFSETS = [np.array(v) for v in ((-0.07, 0.06, 0.20), (-0.09, 0.05, 0.18), (-0.11, 0.04, 0.16))]  # TCP rel. lid
BALL_FORCE = 10.0            # N (ball may be hollow)
BALL_DROP_FROM_BASKET = 0.045  # ball released above basket centre + this along d (exposed part)
BALL_DROP_Z = 0.30           # TCP height at ball release (~3 cm above rim)
TABLE_VIEW = np.array([0.42, 0.26, 0.55])   # wrist view of the table left of the fryer
MIN_MARGIN = 0.15            # joint margin (rad) required for planned configurations
MIN_CLEAR = 0.008            # m, coarse model clearance required


class StepFailure(RuntimeError):
    pass


class Scene:
    """Fryer frame anchored on the measured handle face (closed drawer)."""

    def __init__(self, fr):
        self.d = np.asarray(fr['d'], float) / np.linalg.norm(fr['d'])
        self.t = np.array([-self.d[1], self.d[0]])
        self.face22 = fr['face'][0.22]
        self.face24 = fr['face'].get(0.24, self.face22 - 0.005)
        self.arm_top = fr['arm_top']
        self.grip_t = fr['grip_t']
        # model axis such that the face sits where it did in run 1's model
        self.c = np.asarray(fr['c'], float) + (self.face22 - FACE22_MODEL) * self.d
        self.face22_m = FACE22_MODEL
        self.face24_m = self.face24 - self.face22 + FACE22_MODEL

    def xy(self, sd, st=0.0):
        return self.c + sd * self.d + st * self.t

    def p(self, sd, st, z):
        return np.r_[self.xy(sd, st), z]

    def fryer_model(self, drawer_out=0.0):
        return Fryer(self.c, self.d, drawer_out=drawer_out)

    def handle_quat(self):
        zh = -np.sin(HANDLE_TILT) * np.r_[self.d, 0] - np.cos(HANDLE_TILT) * Z
        return quat_tool(zh, np.r_[self.t, 0])

    def handle_zaxis(self):
        return -np.sin(HANDLE_TILT) * np.r_[self.d, 0] - np.cos(HANDLE_TILT) * Z

    def drop_quat(self, s):
        zh = -np.cos(DROP_BETA) * Z - np.sin(DROP_BETA) * np.r_[self.d, 0]
        yh = s * (np.cos(DROP_BETA) * np.r_[self.d, 0] - np.sin(DROP_BETA) * Z)
        return quat_tool(zh, yh)


def perceive(robot, tag, cams=PERC_CAMS):
    fid = robot.observe(cameras=cams)
    robot.log('frame', tag=tag, frame_id=fid)
    return fid, load_frame(fid, cameras=cams)


def descend_to_grasp(robot, grasp, quat, label, d=None):
    """Final approach onto a handle. Tolerates a residual that cannot hurt the grasp: vertical
    < 6 mm (no resting on top), lateral (finger axis) < 8 mm, along the drawer axis < 15 mm
    (pads still on the 5 cm long handle arm). Slow settling in folded postures leaves ~1 cm."""
    ok, rep = robot.try_move_pose(grasp, quat, tol_mm=3, timeout_s=10)
    if ok:
        return
    e = np.array(robot.status()['tcp_position']) - grasp
    if d is not None:
        dd = np.r_[d, 0.0]; tt = np.array([-d[1], d[0], 0.0])
        e_d, e_t, e_z = abs(e @ dd), abs(e @ tt), abs(e[2])
        acceptable = e_z < 0.006 and e_t < 0.008 and e_d < 0.015
    else:
        e_d = e_t = np.nan; e_z = abs(e[2])
        acceptable = np.linalg.norm(e) < 0.008
    robot.log('descent_residual', label=label, err_mm=float(np.linalg.norm(e) * 1000),
              along_d=float(e_d * 1000), lateral=float(e_t * 1000), vertical=float(e_z * 1000), rep=rep)
    if not acceptable:
        robot.move_pose([0, 0, 0.08], relative=True, timeout_s=15)
        raise StepFailure(f'{label}: descent residual too large (d {e_d*1000:.1f}, t {e_t*1000:.1f}, z {e_z*1000:.1f} mm)')


def view_from(robot, tcp, tag, fryer=None):
    """Move the (empty) hand to a viewing pose with a checked joint move, observe wrist+left."""
    q = plan_ik(tcp, VIEW_QUAT, seeds=[HOME, robot.joints()], prefer=HOME, fryer=fryer, label=f'view {tag}')
    if fryer is not None and check_joint_path(robot.joints(), q, fryer) < MIN_CLEAR:
        robot.move_pose([0, 0, 0.05], relative=True, timeout_s=10)
    robot.move_joints(q)
    return perceive(robot, tag, cams=VIEW_CAMS)


def elbow_offset(q, pos):
    """Distance (m) of the elbow from the vertical plane through the base axis and the target."""
    e = fk_all(q)[3][:3, 3]
    b = np.asarray(pos[:2], float)
    nb = np.linalg.norm(b)
    if nb < 1e-6:
        return 0.0
    u = b / nb
    return float(abs(e[0] * u[1] - e[1] * u[0]))


def natural(q, pos):
    return elbow_offset(q, pos) < 0.10 and abs(q[2]) < 1.0


def plan_ik(pos, quat, seeds, prefer=None, fryer=None, holding=False, label=''):
    """IK restricted to natural (elbow-up, in-plane) postures with joint margin and clearance."""
    pos = np.asarray(pos, float)
    bearing = float(np.arctan2(pos[1], pos[0]))
    base = [np.array([bearing, -0.3, 0.0, -2.3, 0.0, 2.0, j7]) for j7 in (-2.0, -0.8, 0.4, 1.6)]
    rng = np.random.default_rng(0)
    cands = list(seeds) + base
    cands += [c + rng.normal(0, 0.25, 7) for c in list(cands) for _ in range(2)]
    best, best_score, why = None, -np.inf, 'no IK solution'
    for sd in cands:
        q, ok, _ = ik(pos, quat, np.clip(sd, SAFE_MIN + 0.05, SAFE_MAX - 0.05), iters=200)
        if not ok:
            continue
        if not natural(q, pos):
            why = 'only non-natural postures'
            continue
        m = safe_margin(q)
        if m < MIN_MARGIN:
            why = f'joint margin {m:.2f}'
            continue
        cl = 0.05
        if fryer is not None:
            c = clearance_holding(q, fryer) if holding else clearance(q, fryer, skip_fingers=True)
            if c[0] < MIN_CLEAR:
                why = f'clearance {c[0] * 1000:.0f} mm ({c[1]})'
                continue
            cl = min(c[0], 0.05)
        score = m + 4 * cl - (0.2 * np.linalg.norm(q - prefer) if prefer is not None else 0)
        if score > best_score:
            best, best_score = q, score
    if best is None:
        raise StepFailure(f'{label}: no acceptable IK at {np.round(pos, 3).tolist()} ({why})')
    return best


def check_joint_path(q0, q1, fryer, holding=False, n=40):
    worst = 9.0
    for s in np.linspace(0, 1, n):
        q = q0 + s * (q1 - q0)
        cl = clearance_holding(q, fryer) if holding else clearance(q, fryer)
        worst = min(worst, cl[0])
    return worst


def cart_path_ok(q_start, p0, p1, quat, fryer, holding=False, n=12):
    q = np.array(q_start, float)
    worst, wm = 9.0, 9.0
    for s in np.linspace(0, 1, n):
        q, ok, _ = ik(p0 + s * (p1 - p0), quat, q)
        if not ok:
            return False, worst, wm, q
        cl = clearance_holding(q, fryer) if holding else clearance(q, fryer, skip_fingers=True)
        worst, wm = min(worst, cl[0]), min(wm, safe_margin(q))
    return worst > MIN_CLEAR and wm > MIN_MARGIN * 0.7, worst, wm, q


# ======================================================================================
def step_prepare(robot):
    robot.require_armed()
    g = robot.status()['gripper']
    if g['is_grasped']:
        raise StepFailure('gripper reports a held object at start; not built for that')
    robot.open()
    st = robot.status()
    if st['tcp_position'][2] < 0.30:
        robot.move_pose([0, 0, 0.12], relative=True, timeout_s=15)
    robot.home()
    return {'ok': True}


def scene_from_frames(F, F2, need_banana=True):
    """Scene + banana from the initial left frame F and the wrist top view F2 of the fryer."""
    fr = per.locate_fryer(F, cams=PERC_CAMS)
    ban = per.locate_banana(F, cams=PERC_CAMS, exclude=(fr['c'], 0.20)) if need_banana else None
    top = per.locate_fryer_top(F2, fr['c'], cams=('wrist',))
    if np.degrees(abs(np.angle(complex(*top['d']) / complex(*fr['d'])))) > 12:
        raise StepFailure(f'top-view pull direction disagrees with side view ({top["d"]} vs {fr["d"]})')
    # faces from the home-pose side view F (at the top-view pose the arm occludes/mixes into
    # the handle in the left image); arm top + lateral offset from the wrist top view F2
    hs = per.handle_state(F, top['c_lid'], top['d'], cams=PERC_CAMS)
    if 0.22 not in hs['face']:
        raise StepFailure(f'handle face not measured: {hs["face"]}')
    hs_top = per.handle_state(F2, top['c_lid'], top['d'], cams=PERC_CAMS, top_cams=('wrist',),
                              face_ref=hs['face'][0.22])
    hs['arm_top'] = hs_top['arm_top'] if hs_top['arm_top'] is not None else hs['arm_top']
    hs['grip_t'] = hs_top['grip_t'] if hs_top['grip_t'] is not None else hs['grip_t']
    body_front = fr['body_front'] + float((fr['c'] - top['c_lid']) @ top['d']) if fr['body_front'] else None
    prot = hs['face'][0.22] - body_front if body_front is not None else fr['protrusion']
    fr2 = dict(c=top['c_lid'], d=top['d'], face=hs['face'], arm_top=hs['arm_top'] or top['arm_top'],
               grip_t=hs['grip_t'] if hs['grip_t'] is not None else 0.0, protrusion=prot)
    return Scene(fr2), fr2, ban


def locate_objects(robot, F, fr, want_ball):
    """Banana (+ ball) from the left frame; wrist table view if the left camera misses them."""
    ban = ball = None
    try:
        ban = per.locate_banana(F, cams=PERC_CAMS, exclude=(fr['c'], 0.20))
        if not (0.13 < ban['chord'] < 0.21):
            ban = None
    except ValueError:
        pass
    if want_ball:
        try:
            ball = per.locate_ball(F, cams=PERC_CAMS, exclude=(fr['c'], 0.20))
        except ValueError:
            pass
    if ban is None or (want_ball and ball is None):
        fid, Ft = view_from(robot, TABLE_VIEW, 'table_view', fryer=Fryer(fr['c'], fr['d']))
        if ban is None:
            ban = per.locate_banana(Ft, cams=('wrist',), exclude=(fr['c'], 0.20))
        if want_ball and ball is None:
            ball = per.locate_ball(Ft, cams=('wrist',), exclude=(fr['c'], 0.20))
    return ban, ball


def step_perceive(robot, want_ball=False):
    fid, F = perceive(robot, 'initial')
    set_camera_obstacles(F)
    fr = per.locate_fryer(F, cams=PERC_CAMS)
    # top view of lid + handle with the wrist camera: axis, pull direction, arm top
    fid2 = F2 = None
    for off in TOP_VIEW_OFFSETS:
        try:
            fid2, F2 = view_from(robot, np.r_[fr['c'], 0.41] + off, 'top_view', fryer=Fryer(fr['c'], fr['d']))
            break
        except StepFailure as e:
            robot.log('top_view_unreachable', offset=off, error=str(e))
    if F2 is None:
        raise StepFailure('no reachable top-view pose over the fryer (fryer too far?)')
    sc, fr2, _ = scene_from_frames(F, F2, need_banana=False)
    ban, ball = locate_objects(robot, F, fr, want_ball)
    if not (0.13 < ban['chord'] < 0.21):
        raise StepFailure(f'banana chord {ban["chord"]:.3f} m not banana-like')
    if fr2['protrusion'] is None or fr2['protrusion'] > PROTRUSION_CLOSED:
        raise StepFailure(f'drawer does not look closed at start (protrusion {fr2["protrusion"]})')
    if sc.arm_top is None or not (0.24 < sc.arm_top < 0.29):
        raise StepFailure(f'handle arm top z {sc.arm_top} outside tested range 0.24-0.29')
    robot.home()
    return {'ok': True, 'frame': fid, 'top_frame': fid2, 'scene': sc, 'fryer': fr2, 'banana': ban, 'ball': ball}


def plan_handle(sc, drawer_out=0.0, arm_top=None, face22=None, grip_t=None):
    """Handle grasp pose + pre-grasp for the drawer pulled out by drawer_out."""
    arm_top = sc.arm_top if arm_top is None else arm_top
    face22 = sc.face22_m + drawer_out if face22 is None else face22
    st = sc.grip_t if grip_t is None else grip_t
    grasp = sc.p(face22 + GRASP_FROM_FACE, st, arm_top - GRASP_BELOW_ARM_TOP)
    pre = grasp - 0.10 * sc.handle_zaxis()
    return grasp, pre


def step_open(robot, sc):
    quat = sc.handle_quat()
    fryer0 = sc.fryer_model(0.0)
    grasp, pre = plan_handle(sc)
    q_pre = plan_ik(pre, quat, seeds=[HOME, np.array([-0.29, -0.23, 0.08, -2.27, -0.03, 2.12, 1.27])],
                    prefer=np.array([-0.29, -0.23, 0.08, -2.27, -0.03, 2.12, 1.27]), fryer=fryer0, label='handle pre-grasp')
    ok, w, wm, q_g = cart_path_ok(q_pre, pre, grasp, quat, fryer0)
    if not ok:
        raise StepFailure(f'handle descent not feasible (clear {w:.3f}, margin {wm:.2f})')
    # pull path feasibility
    q = q_g
    for s in np.arange(PULL_STEP, PULL + 1e-6, PULL_STEP):
        q, okp, _ = ik(grasp + s * np.r_[sc.d, 0], quat, q)
        if not okp or safe_margin(q) < 0.12:
            raise StepFailure(f'pull path infeasible at {s:.2f} m (margin {safe_margin(q):.2f}); fryer too close/far')
    q_now = robot.joints()
    if check_joint_path(q_now, q_pre, fryer0) < MIN_CLEAR:
        raise StepFailure('joint path to handle pre-grasp too close to fryer')
    robot.move_joints(q_pre)
    robot.move_pose(grasp + 0.04 * (pre - grasp) / 0.10, quat, timeout_s=15)
    descend_to_grasp(robot, grasp, quat, 'open grasp', sc.d)
    g = robot.grasp(0.025, GRASP_FORCE_HANDLE)
    if not g['is_grasped'] or not (0.026 < g['width'] < 0.045):
        robot.open()
        robot.move_pose(pre, quat, timeout_s=15)
        raise StepFailure(f'handle grasp failed (width {g["width"]:.4f}, grasped {g["is_grasped"]})')
    p_start = np.array(robot.status()['tcp_position'])
    z_pull = p_start[2] - 0.006
    pulled = 0.0
    for s in np.arange(PULL_STEP, PULL + 1e-6, PULL_STEP):
        s = min(s, PULL)
        tgt = np.r_[p_start[:2] + s * sc.d, z_pull]
        ok, rep = robot.try_move_pose(tgt, tol_mm=5, timeout_s=15)
        p = np.array(robot.status()['tcp_position'])
        pulled = float((p[:2] - p_start[:2]) @ sc.d)
        g = robot.status()['gripper']
        if not ok or not g['is_grasped']:
            robot.open()
            robot.move_pose([0, 0, 0.10], relative=True, timeout_s=15)
            raise StepFailure(f'pull stopped at {pulled:.3f} m ({rep if not ok else "grasp lost"})')
    if PULL - pulled > 0.004:
        tgt = np.r_[p_start[:2] + PULL * sc.d, z_pull]
        robot.try_move_pose(tgt, tol_mm=5, timeout_s=10)
        pulled = float((np.array(robot.status()['tcp_position'])[:2] - p_start[:2]) @ sc.d)
    release_tcp = np.array(robot.status()['tcp_position'])
    robot.open()
    robot.move_pose([0, 0, 0.12], relative=True, timeout_s=15)
    fid, F = view_from(robot, sc.p(*VIEW_DRAWER), 'after_open', fryer=sc.fryer_model(pulled))
    hs = per.handle_state(F, sc.c, sc.d, cams=PERC_CAMS, expect_sd=sc.face22_m + pulled, top_cams=('wrist',))
    face = hs['face'].get(0.22) or hs['face'].get(0.20)
    opened = (face - sc.face22_m) if face is not None else None
    if opened is None or opened < PULL - 0.03:
        raise StepFailure(f'drawer not open enough after pull (measured {opened})')
    rim_c, rim_r, n_rim = per.locate_rim(F, sc.c, sc.d, expect_sd=sc.face22_m + BASKET_FROM_FACE + opened, cams=VIEW_CAMS)
    robot.home()
    return {'ok': True, 'pulled_tcp': pulled, 'opened': float(opened), 'handle_after': hs,
            'grasp_tcp': p_start.tolist(), 'release_tcp': release_tcp.tolist(), 'frame': fid,
            'rim': (None if rim_c is None else rim_c.tolist(), rim_r, n_rim)}


def step_banana(robot, sc, ban, opened, rim):
    out = sc.face22_m + BASKET_FROM_FACE + opened      # basket centre along d (model frame)
    basket = sc.xy(out, 0.0)
    if rim[0] is not None and np.linalg.norm(np.array(rim[0]) - basket) < 0.025 and 0.085 < rim[1] < 0.12:
        basket = np.array(rim[0])
    mid = basket + DROP_MID_FROM_BASKET * sc.d
    fryer_open = sc.fryer_model(opened)
    pick_z = max(0.105, ban['top_z'] - 0.02)
    pick = np.r_[ban['mid'], pick_z]
    n = np.r_[ban['n_convex'], 0.0]
    spot = np.r_[sc.xy(ROT_SPOT[0], ROT_SPOT[1]), ROT_SPOT[2]]
    drop_hi, drop = np.r_[mid, DROP_Z + 0.10], np.r_[mid, DROP_Z]
    plans, errors = [], []
    for s in (1, -1):
        try:
            qp = quat_tool(-Z, s * n)
            q_pick_hi = plan_ik(pick + [0, 0, 0.10], qp, seeds=[HOME], prefer=HOME, label=f'pick-above s={s}')
            q_pick, okk, _ = ik(pick, qp, q_pick_hi)
            if not okk or safe_margin(q_pick) < MIN_MARGIN:
                raise StepFailure('pick pose margin')
            qd = sc.drop_quat(s)
            seed_d = np.array([-0.02, -0.40, -0.15, -2.6, -1.72, 2.67, -1.97 if s == 1 else 1.17])
            q_drop = plan_ik(drop, qd, seeds=[seed_d], prefer=seed_d,
                             fryer=fryer_open, holding=True, label=f'drop s={s}')
            ok1, w1, m1, q_dhi = cart_path_ok(q_drop, drop, drop_hi, qd, fryer_open, holding=True)
            ok2, w2, m2, q_spotA = cart_path_ok(q_dhi, drop_hi, spot, qd, fryer_open, holding=True)
            if not (ok1 and ok2):
                raise StepFailure(f'drop approach infeasible (clear {min(w1, w2):.3f}, margin {min(m1, m2):.2f})')
            # the approach spot -> drop-above -> drop is executed in joint space (the service's
            # Cartesian IK refused a near-singular step there in run 8): check those joint paths
            wj = min(check_joint_path(q_spotA, q_dhi, fryer_open, holding=True),
                     check_joint_path(q_dhi, q_drop, fryer_open, holding=True))
            if wj < MIN_CLEAR:
                raise StepFailure(f'joint-space drop approach too close ({wj:.3f})')
            # reaching the spot with the pick orientation, then rotating in joint space
            q_lift, okl, _ = ik(np.r_[pick[:2], 0.42], qp, q_pick)
            okc, wc, mc, q_spotB = cart_path_ok(q_lift, np.r_[pick[:2], 0.42], spot, qp, fryer_open, holding=True)
            if not (okl and okc):
                raise StepFailure('carry to rotation spot infeasible')
            wr = check_joint_path(q_spotB, q_spotA, fryer_open, holding=True)
            if wr < 0.03:
                raise StepFailure(f'rotation sweep too close ({wr:.3f})')
            score = min(safe_margin(q) for q in (q_pick, q_pick_hi, q_drop, q_spotA, q_spotB))
            plans.append(dict(s=s, qp=qp, qd=qd, q_pick_hi=q_pick_hi, q_spotA=q_spotA, q_dhi=q_dhi, q_drop=q_drop, score=score))
        except StepFailure as e:
            errors.append(f's={s}: {e}')
    if not plans:
        raise StepFailure('no feasible banana plan: ' + '; '.join(errors))
    P = max(plans, key=lambda x: x['score'])
    robot.log('banana_plan', s=P['s'], score=P['score'], errors=errors, mid=mid, pick=pick)
    if check_joint_path(robot.joints(), P['q_pick_hi'], fryer_open) < MIN_CLEAR:
        raise StepFailure('joint path to banana pick-above too close to fryer')
    robot.move_joints(P['q_pick_hi'])
    robot.move_pose(pick, P['qp'], timeout_s=20)
    g = robot.grasp(0.02, BANANA_FORCE)
    if not g['is_grasped'] or g['width'] < 0.015:
        robot.open()
        robot.move_pose(pick + [0, 0, 0.10], P['qp'], timeout_s=15)
        raise StepFailure(f'banana grasp failed (width {g["width"]:.4f})')
    robot.move_pose(np.r_[pick[:2], 0.42], P['qp'], timeout_s=25)
    if not robot.status()['gripper']['is_grasped']:
        raise StepFailure('banana slipped out while lifting')
    robot.move_pose(spot, P['qp'], timeout_s=25)
    wr = check_joint_path(robot.joints(), P['q_spotA'], fryer_open, holding=True)
    if wr < 0.03:
        raise StepFailure(f'rotation sweep from actual config too close ({wr:.3f}); holding banana at spot')
    robot.move_joints(P['q_spotA'])
    robot.move_joints(P['q_dhi'])
    robot.move_joints(P['q_drop'])
    p_now = np.array(robot.status()['tcp_position'])
    if np.linalg.norm(p_now - drop) > 0.015:
        raise StepFailure(f'drop pose off by {np.linalg.norm(p_now - drop) * 1000:.0f} mm; holding banana')
    if not robot.status()['gripper']['is_grasped']:
        raise StepFailure('banana lost before the drop point')
    robot.open(speed=0.1)
    robot.move_joints(P['q_dhi'])
    if check_joint_path(robot.joints(), HOME, fryer_open) < MIN_CLEAR:
        robot.move_joints(P['q_spotA'])
    robot.home()
    fid, F = view_from(robot, sc.p(*VIEW_DRAWER), 'after_drop', fryer=fryer_open)
    inside = per.banana_inside(F, basket, 0.10, cams=VIEW_CAMS)
    if inside['inside'] < 150 or inside['frac'] < 0.6 or (inside['zmax'] is not None and inside['zmax'] > 0.235):
        raise StepFailure(f'banana not verified inside the basket: {inside}')
    return {'ok': True, 's': P['s'], 'inside': inside, 'frame': fid, 'basket': basket.tolist()}


def step_ball(robot, sc, ball, opened, basket):
    """Pick the ball top-down, release it ~3 cm above the rim over the exposed basket part."""
    fryer_open = sc.fryer_model(opened)
    r = ball['radius']
    pick = np.r_[ball['center'], max(0.105, per.TABLE_Z + r)]
    drop_xy = np.asarray(basket) + BALL_DROP_FROM_BASKET * sc.d
    drop = np.r_[drop_xy, BALL_DROP_Z]
    best = None
    for sgn in (1, -1):              # hand long axis along t at the drop (clear of body and front wall)
        quat = quat_tool(-Z, sgn * np.r_[sc.t, 0])
        try:
            q_hi = plan_ik(pick + [0, 0, 0.10], quat, seeds=[HOME], prefer=HOME, fryer=fryer_open, label='ball pick-above')
            q_dr = plan_ik(drop, quat, seeds=[HOME, q_hi], prefer=q_hi, fryer=fryer_open, label='ball drop')
            if clearance_holding_ball(q_dr, fryer_open, radius=r)[0] < MIN_CLEAR:
                raise StepFailure('ball drop clearance')
            ok1, w1, m1, _ = cart_path_ok(q_hi, pick + [0, 0, 0.10], np.r_[pick[:2], 0.42], quat, fryer_open)
            ok2, w2, m2, _ = cart_path_ok(q_dr, drop, np.r_[drop_xy, 0.42], quat, fryer_open)
            if not (ok1 and ok2):
                raise StepFailure('ball carry path')
            score = min(safe_margin(q_hi), safe_margin(q_dr))
            if best is None or score > best[0]:
                best = (score, quat, q_hi, q_dr)
        except StepFailure as e:
            robot.log('ball_plan_reject', sign=sgn, error=str(e))
    if best is None:
        raise StepFailure('no feasible ball plan')
    _, quat, q_hi, q_dr = best
    if check_joint_path(robot.joints(), q_hi, fryer_open) < MIN_CLEAR:
        raise StepFailure('joint path to ball pick-above too close')
    robot.move_joints(q_hi)
    robot.move_pose(pick, quat, timeout_s=20)
    g = robot.grasp(0.02, BALL_FORCE)
    if not g['is_grasped'] or g['width'] < 1.2 * r:
        robot.open()
        robot.move_pose(pick + [0, 0, 0.10], quat, timeout_s=15)
        raise StepFailure(f'ball grasp failed (width {g["width"]:.4f})')
    robot.move_pose(np.r_[pick[:2], 0.42], quat, timeout_s=25)
    robot.move_pose(np.r_[drop_xy, 0.42], quat, timeout_s=30)
    robot.move_pose(drop, quat, timeout_s=20)
    if not robot.status()['gripper']['is_grasped']:
        raise StepFailure('ball lost before the drop point')
    robot.open(speed=0.1)
    robot.move_pose(np.r_[drop_xy, 0.42], quat, timeout_s=20)
    robot.home()
    fid, F = view_from(robot, sc.p(*VIEW_DRAWER), 'after_ball', fryer=fryer_open)
    orange = per.locate_ball(F, cams=('wrist',), zmax=0.30)
    inside = np.linalg.norm(orange['center'] - np.asarray(basket)) < 0.10 and orange['top_z'] < 0.235
    if not inside:
        raise StepFailure(f'ball not verified inside: centre {orange["center"]}, top {orange["top_z"]:.3f}')
    ban_in = per.banana_inside(F, basket, 0.10, cams=VIEW_CAMS)
    return {'ok': True, 'ball_center': orange['center'].tolist(), 'ball_top': orange['top_z'],
            'banana_still_inside': ban_in, 'frame': fid}


def step_close(robot, sc, opened):
    quat = sc.handle_quat()
    robot.open()
    fid, F = view_from(robot, sc.p(*VIEW_DRAWER), 'before_close', fryer=sc.fryer_model(opened))
    hs = per.handle_state(F, sc.c, sc.d, cams=PERC_CAMS, expect_sd=sc.face22_m + opened, top_cams=('wrist',))
    if hs['arm_top'] is None or 0.22 not in hs['face'] and 0.20 not in hs['face']:
        raise StepFailure(f'handle not found before closing: {hs}')
    face22 = hs['face'].get(0.22, hs['face'].get(0.20, 0) - 0.004)
    fryer_open = sc.fryer_model(face22 - sc.face22_m)
    grasp, pre = plan_handle(sc, arm_top=hs['arm_top'], face22=face22,
                             grip_t=hs['grip_t'] if hs['grip_t'] is not None else sc.grip_t)
    q_pre = plan_ik(pre, quat, seeds=[HOME, np.array([-0.12, -0.32, 0.08, -2.74, -0.04, 2.49, 1.45])],
                    prefer=np.array([-0.12, -0.32, 0.08, -2.74, -0.04, 2.49, 1.45]), fryer=fryer_open, label='close pre-grasp')
    if check_joint_path(robot.joints(), q_pre, fryer_open) < MIN_CLEAR:
        raise StepFailure('joint path to close pre-grasp too close to fryer')
    robot.move_joints(q_pre)
    robot.move_pose(grasp + 0.04 * (pre - grasp) / 0.10, quat, timeout_s=15)
    descend_to_grasp(robot, grasp, quat, 'close grasp', sc.d)
    g = robot.grasp(0.025, GRASP_FORCE_HANDLE)
    if not g['is_grasped']:
        robot.open()
        raise StepFailure('could not grasp handle to close')
    # friction push with lift back to the closed grasp height, stopping 2 cm before closed
    p0 = np.array(robot.status()['tcp_position'])
    closed_tcp = sc.p(sc.face22_m + GRASP_FROM_FACE, sc.grip_t, sc.arm_top - GRASP_BELOW_ARM_TOP)
    remaining = float((p0[:2] - closed_tcp[:2]) @ sc.d)
    stop_short = 0.02
    n = max(1, int(np.ceil((remaining - stop_short) / PULL_STEP)))
    for k in range(1, n + 1):
        f = k / n
        tgt_sd = remaining - f * (remaining - stop_short)
        tgt = np.r_[closed_tcp[:2] + tgt_sd * sc.d, p0[2] + f * (closed_tcp[2] - p0[2])]
        ok, rep = robot.try_move_pose(tgt, tol_mm=6, timeout_s=12)
        if not ok:
            robot.log('push_blocked', at=k, rep=rep)
            break
    robot.open()
    robot.move_pose([0, 0, 0.10], relative=True, timeout_s=15)
    # fingertip push on the grip's outer face
    robot.close_fingers()
    robot.home()
    fid, F = perceive(robot, 'before_fingertip_push')
    hs = per.handle_state(F, sc.c, sc.d, cams=PERC_CAMS, expect_sd=sc.face22_m + 0.03)
    f24 = hs['face'].get(0.24)
    if f24 is None:
        f24 = hs['face'].get(0.22, sc.face22_m + 0.03) - 0.005
    gt = hs['grip_t'] if hs['grip_t'] is not None else sc.grip_t
    zp = sc.arm_top - PUSH_Z_BELOW_ARM_TOP
    start = sc.p(f24 + FINGER_HALF + 0.007, gt, zp)
    above = start + [0, 0, 0.075]
    end = sc.p(sc.face24_m + FINGER_HALF - 0.004, gt, zp)
    fryer_mid = sc.fryer_model(max(0.0, f24 - sc.face24_m))
    q_ab = plan_ik(above, quat, seeds=[HOME, q_pre], prefer=q_pre, fryer=fryer_mid, label='fingertip push above')
    if check_joint_path(robot.joints(), q_ab, fryer_mid) < MIN_CLEAR:
        raise StepFailure('joint path to fingertip-push pose too close to fryer')
    robot.move_joints(q_ab)
    robot.move_pose(start, quat, timeout_s=15)
    ok, rep = robot.try_move_pose(end, quat, tol_mm=5, timeout_s=6)
    robot.log('fingertip_push', ok=ok, rep=rep)
    back = -0.02 * np.r_[sc.d, 0]
    robot.move_pose(np.array(robot.status()['tcp_position']) - back, quat, timeout_s=10)
    robot.move_pose([0, 0, 0.15], relative=True, timeout_s=15)
    robot.open()
    robot.home()
    fid, F = perceive(robot, 'after_close')
    hs = per.handle_state(F, sc.c, sc.d, cams=PERC_CAMS, expect_sd=sc.face22_m)
    face_err = hs['face'].get(0.22, np.nan) - sc.face22_m
    fr = per.locate_fryer(F, cams=PERC_CAMS)
    if np.isfinite(face_err):
        closed = abs(face_err) < 0.006
    else:
        closed = fr['protrusion'] is not None and fr['protrusion'] < PROTRUSION_CLOSED
    if not closed:
        raise StepFailure(f'drawer not closed: face shift {face_err:.4f} m, protrusion {fr["protrusion"]}')
    return {'ok': True, 'protrusion': fr['protrusion'], 'face_shift': float(face_err), 'frame': fid}


# ======================================================================================
def resume_state(prev_run):
    """Scene, banana and drawer opening from a previous run directory (same fryer pose)."""
    prev = ROOT / 'output' / 'runs' / prev_run
    frames = {}
    for line in (prev / 'calls.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['kind'] == 'frame':
            frames[r['tag']] = r['frame_id']
    F = load_frame(frames['initial'], cameras=PERC_CAMS)
    F2 = load_frame(frames['top_view'], cameras=VIEW_CAMS)
    sc, fr2, ban = scene_from_frames(F, F2)
    rep = json.loads((prev / 'report.json').read_text())
    opened = rep.get('steps', {}).get('open', {}).get('opened', PULL)
    rim = rep.get('steps', {}).get('open', {}).get('rim', (None, None, 0))
    return {'ok': True, 'frame': frames['initial'], 'top_frame': frames['top_view'], 'scene': sc,
            'fryer': fr2, 'banana': ban, 'opened': opened, 'rim': rim, 'resumed_from': prev_run}


def main(ctx, steps=('open', 'banana', 'close'), run_dir=None, resume=None):
    run_dir = Path(run_dir or ROOT / 'output' / 'runs' / time.strftime('%Y%m%dT%H%M%S'))
    robot = Robot(ctx, run_dir)
    report = {'run_dir': str(run_dir), 'steps': {}, 'ok': False}
    t0 = time.time()
    try:
        report['steps']['prepare'] = step_prepare(robot)
        if resume:
            P = resume_state(resume)
            _, Fr = perceive(robot, 'resume_check')
            set_camera_obstacles(Fr)
            if 'banana' in steps:      # banana may have moved since the original frame
                P['banana'] = per.locate_banana(Fr, cams=PERC_CAMS, exclude=(P['scene'].c, 0.20))
        else:
            P = step_perceive(robot, want_ball='ball' in steps)
        sc = P['scene']
        report['steps']['perceive'] = {'ok': True, 'frame': P['frame'],
                                       'fryer_c': sc.c.tolist(), 'd_deg': float(np.degrees(np.arctan2(sc.d[1], sc.d[0]))),
                                       'protrusion': P['fryer']['protrusion'], 'arm_top': sc.arm_top,
                                       'banana_mid': P['banana']['mid'].tolist(), 'banana_chord': P['banana']['chord']}
        opened = P.get('opened', PULL)
        rim = P.get('rim', (None, None, 0))
        if 'open' in steps:
            r = step_open(robot, sc)
            report['steps']['open'] = {k: v for k, v in r.items() if k != 'handle_after'}
            opened, rim = r['opened'], r['rim']
        if 'banana' in steps:
            report['steps']['banana'] = step_banana(robot, sc, P['banana'], opened, rim)
        if 'ball' in steps:
            if P.get('ball') is None:
                raise StepFailure('ball step requested but no ball found')
            basket = report['steps'].get('banana', {}).get('basket') or sc.xy(sc.face22_m + BASKET_FROM_FACE + opened).tolist()
            report['steps']['ball'] = step_ball(robot, sc, P['ball'], opened, basket)
        if 'close' in steps:
            report['steps']['close'] = step_close(robot, sc, opened)
        report['ok'] = True
    except (StepFailure, MotionError, ValueError) as e:
        report['error'] = f'{type(e).__name__}: {e}'
        robot.log('failure', error=report['error'])
        try:
            st = robot.status()
            report['final_state'] = {'tcp': st['tcp_position'], 'gripper': st['gripper']}
        except Exception as e2:   # noqa: BLE001
            report['final_state'] = f'status unavailable: {e2}'
    report['seconds'] = round(time.time() - t0, 1)
    (run_dir / 'report.json').write_text(json.dumps(report, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    return report


if __name__ == '__main__':
    import argparse
    sys.path.insert(0, str(Path.cwd()))     # the workspace root holds the harness's api/ package
    from api.runtime import Context
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', default='open,banana,close')
    ap.add_argument('--resume', default=None, help='previous run dir name: reuse its scene (fryer not moved)')
    a = ap.parse_args()
    with Context() as session:
        print(json.dumps(main(session, steps=tuple(a.steps.split(',')), resume=a.resume), indent=1, default=str))
