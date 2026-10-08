"""Plans for the squeeze: base stance, chest yaw, arm waypoints, torso sequences (offline, no robot).

Tested values (2026-10-08, box 0.40 x 0.22 x 0.25 m, ~1.4 kg, on the floor):
  stance D = 0.46 m (box centre ahead of the base origin), torso lowest + lean 0.07,
  gripper axis tilt 0.5 rad forward-down, ball centre z 0.18, x = box centre x,
  standoff 5 cm outside the estimated end faces, inward steps 1 cm, extra squeeze 5 mm.
"""
import numpy as np

from kinematics import AXLE_X, BALL_R, arm_margin, solve_ball, torso_pose

STANCE_D = 0.46          # box centre ahead of the base origin (near face ~4-5 cm from the bumper)
PRE_TURN = 0.35          # do the last turn this far before the stance (box corners clear the base)
LEAN = 0.07              # torso_3 = torso_1 + LEAN keeps torso_3 0.2 rad inside its limit at lowest
TILT = 0.5               # gripper axis from vertical, fingers forward-down
BALL_Z = 0.18            # ball centre height for the squeeze (box top was ~0.25)
STANDOFF = 0.05          # start this far outside the estimated end face
FACE_UNCERTAINTY = 0.02  # widen depth-estimated ends by this much each side before planning
STEP = 0.01              # inward step while touching
MAX_YAW = 0.15           # chest yaw used to centre an off-centre box (-0.10 worked for 5 cm right)
# pre-grasp arm shapes used on the robot (balls ~5 cm outside a 0.40 m box, z 0.29 at the squat)
GOOD_PRE_HIGH = {'right': [-0.2977, -0.6478, 0.8091, -1.291, -0.2784, 0.6682, 0.7304],
                 'left': [-0.2959, 0.6898, -0.809, -1.2703, 0.3357, 0.6207, -0.7686]}


def squat_sequence(yaw=0.0):
    return [torso_pose(a, LEAN, yaw) for a in (0.785, 1.047, 1.2)]


def lift_sequence(yaw=0.0):
    """Torso-only lift with the arms fixed (the squeeze is preserved); last pose is drivable."""
    return [torso_pose(0.9, LEAN, yaw), torso_pose(0.6, LEAN, yaw), torso_pose(0.524, 0.0, yaw)]


def stance_moves(box):
    """Base moves from the current pose: turn1, forward, turn2 end PRE_TURN short of the stance, facing
    the box's long face; then `approach` metres straight ahead (in short re-localized steps) put the
    box centre STANCE_D ahead. Plans the wheel-axle point: turning in place pivots about it."""
    c, n = np.array(box['center']), np.array(box['short_axis'])
    h = float(np.arctan2(n[1], n[0]))
    a_target = c - STANCE_D * n + AXLE_X * n
    a_pre = a_target - PRE_TURN * n
    a0 = np.array([AXLE_X, 0.0])
    d = a_pre - a0
    if d[0] < 0.05 or np.linalg.norm(d) < 0.08:
        # already close: no detour behind the robot; face the box, then approach straight (the chest
        # yaw in prepare() absorbs up to ~7 cm of lateral offset; more needs a reposition)
        lateral = float(np.cross(n, a_target - a0))
        return {'turn1': 0.0, 'forward': 0.0, 'turn2': h, 'approach': float(max(0.0, n @ (a_target - a0))),
                'heading_to_face': h, 'lateral_after': lateral}
    bearing = float(np.arctan2(d[1], d[0]))
    return {'turn1': bearing, 'forward': float(np.linalg.norm(d)), 'turn2': float(np.angle(np.exp(1j * (h - bearing)))),
            'approach': PRE_TURN, 'heading_to_face': h, 'lateral_after': 0.0}


def chest_yaw(box):
    c = np.array(box['center'])
    return float(np.clip(np.arctan2(c[1], c[0]), -MAX_YAW, MAX_YAW))


def grasp_waypoints(x, y_right_face, y_left_face, yaw=0.0, z=BALL_Z, seeds=None, depth_to=0.03):
    """Per arm: pre_high (z 0.29), mid (0.235), low (z) at STANDOFF outside the face, then inward STEP
    waypoints to `depth_to` inside the estimated face. One IK branch along the path (no restarts)."""
    torso = torso_pose(1.2, LEAN, yaw)
    faces = {'right': y_right_face, 'left': y_left_face}
    inward = {'right': 1.0, 'left': -1.0}
    plan = {}
    for side in ('right', 'left'):
        y_out = faces[side] - inward[side] * (BALL_R + STANDOFF)
        wps = [('pre_high', y_out, 0.29), ('mid', y_out, 0.235), ('low', y_out, z)]
        y = y_out
        while inward[side] * ((faces[side] + inward[side] * depth_to) - y) > 1e-9:
            y += inward[side] * STEP
            wps.append(('in', y, z))
        q_prev = None if not seeds else np.array(seeds[side])
        seq = []
        for name, yy, zz in wps:
            centre = [x, yy, zz]
            if q_prev is None:            # start from the arm shape that worked on the robot
                q, err, _ = solve_ball(side, torso, centre, TILT, q0=GOOD_PRE_HIGH[side], tries=0, w_reg=0.0)
                if err > 0.003:
                    q, err, _ = solve_ball(side, torso, centre, TILT, q0=GOOD_PRE_HIGH[side], tries=8, w_reg=0.0)
            else:
                q, err, _ = solve_ball(side, torso, centre, TILT, q0=q_prev, tries=0, w_reg=0.02)
            seq.append({'name': name, 'y': round(yy, 4), 'z': zz, 'q': [round(v, 4) for v in q],
                        'err': round(err, 4), 'margin': round(arm_margin(side, q), 3),
                        'dq': 0.0 if q_prev is None else round(float(np.abs(q - q_prev).max()), 3)})
            q_prev = q
        plan[side] = seq
    return {'torso': torso, 'plan': plan}


def check_plan(p, max_err=0.008, max_err_outside=0.012, max_dq=0.6):
    """Problems that make a plan unsafe to run: missed targets or joint jumps between waypoints.
    Descent waypoints outside the box tolerate more error (8.4 mm worked on the robot)."""
    bad = []
    for side, seq in p['plan'].items():
        for k, w in enumerate(seq):
            if w['err'] > (max_err if w['name'] == 'in' else max_err_outside):
                bad.append(f"{side} {w['name']}#{k} misses its target by {w['err'] * 1000:.0f} mm")
            if w['dq'] > max_dq:
                bad.append(f"{side} {w['name']}#{k} jumps {w['dq']:.2f} rad")
    return bad
