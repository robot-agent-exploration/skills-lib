"""Bimanual ball-squeeze grasp of a box on the floor, lifted with the torso (RB-Y1).

From the workspace root:
    python3 <skill>/scripts/grasp.py --dry-run --frame <frame_id>   # plan only, no robot calls
    python3 <skill>/scripts/grasp.py                                # locate, align, grasp, torso lift
    python3 <skill>/scripts/grasp.py --at-stance                    # already facing the box at ~0.46 m
Stages: locate -> align -> prepare -> squat -> touch_squeeze -> torso_lift. The run stops at the first
failed stage and writes output/runs/<time>_box_squeeze/report.json. Preconditions: armed, no fault,
torso `ready` (or zero), arms `ready`, box visible on the floor ahead, nothing between the grippers.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORKSPACE = next(p for p in HERE.parents if (p / 'api' / 'runtime.py').exists())
sys.path.insert(0, str(WORKSPACE))

import perception                                    # noqa: E402
import plan as P                                     # noqa: E402
from kinematics import torso_pose                    # noqa: E402
from touch import squeeze_delta, step_until_contact, wrist_forces   # noqa: E402

LOOK_FLOOR = [1.2, 0.0, 0.1]      # first look when no box estimate is given
APPROACH_STEP = 0.15              # final straight approach per call (m)
MIN_SQUEEZE_N = 10.0              # weaker than this after the extra squeeze: do not lift
MAX_SQUEEZE_N = 32.0              # stronger: ease off (the right wrist arm_5 is the bottleneck)
HOLD_MIN_N = 6.0                  # during the lift: below this the grip is lost


class StageFailed(Exception):
    pass


SKILL_DIR = HERE.parent


def frame_dir(frame):
    """A frame id from runtime/frames/, or a path to a frame folder (e.g. evidence/frames/<id>)."""
    for p in (Path(frame), SKILL_DIR / frame):
        if (p / 'meta.json').exists():
            return p
    return WORKSPACE / 'runtime' / 'frames' / frame


def look_and_observe(ctx, point):
    try:
        ctx.value('rby1_look_at', point=list(map(float, point)))
    except RuntimeError as e:          # the head often stops ~1 deg short: a handled stop, nothing else moved
        if 'head did not reach' not in str(e):
            raise
    return ctx.value('rby1_observe')['frame_id']


def locate(ctx, log, near=None):
    s = ctx.value('rby1_status')
    if s['torso'][1] > 0.8:
        raise StageFailed('localize with the torso at ready/zero: squatting puts the knee in the depth')
    point = LOOK_FLOOR if near is None else [near[0], near[1], 0.12]
    fid = look_and_observe(ctx, point)
    box = perception.find_box(frame_dir(fid), near=near, radius=0.6 if near else 1.0)
    if not box['ok']:
        raise StageFailed('no box found: ' + box['why'])
    fid = look_and_observe(ctx, [box['center'][0], box['center'][1], 0.12])      # centre it, look again
    box = perception.find_box(frame_dir(fid), near=box['center'], radius=0.4)
    if not box['ok']:
        raise StageFailed('box lost after centring: ' + box['why'])
    log(f"box centre {np.round(box['center'], 3)} top {box['top_z']:.3f} size {np.round(box['size'], 3)} frame {fid}")
    if not (0.30 <= box['size'][0] <= 0.50 and 0.15 <= box['top_z'] <= 0.35):
        raise StageFailed(f'box outside the tested range (long side 0.38-0.42, top ~0.25): {box["size"]}, {box["top_z"]:.2f}')
    return box


def base_move(ctx, log, forward=0.0, turn=0.0):
    r = ctx.value('rby1_base_move', forward=float(forward), turn=float(turn), max_linear=0.1, max_angular=0.2)
    log(f"base_move fwd {forward:.3f} turn {turn:.3f} -> odom {r['odometry']}")
    return r


def corridor_clear(fid, length, box):
    """Rough obstacle check along the straight path: object points in front of the bumper, excluding the box."""
    pts, _, _ = perception.load_points(frame_dir(fid))
    m = (pts[:, 2] > 0.05) & (pts[:, 2] < 1.2) & (pts[:, 0] > 0.36) & (pts[:, 0] < 0.36 + length) & (np.abs(pts[:, 1]) < 0.4)
    if box is not None:
        m &= np.hypot(pts[:, 0] - box['center'][0], pts[:, 1] - box['center'][1]) > 0.35
    return int(m.sum()) < 200


def align(ctx, log, box):
    mv = P.stance_moves(box)
    log(f'stance plan {mv}')
    if abs(mv['turn1']) > 0.02:
        base_move(ctx, log, turn=mv['turn1'])
    remaining = mv['forward']
    while remaining > 0.02:
        step = min(0.5, remaining)
        fid = look_and_observe(ctx, [1.5, 0.0, 0.3])
        if not corridor_clear(fid, step, None):
            raise StageFailed(f'path not clear (frame {fid})')
        base_move(ctx, log, forward=step)
        remaining -= step
    if abs(mv['turn2']) > 0.02:
        base_move(ctx, log, turn=mv['turn2'])
    box = locate(ctx, log, near=[P.STANCE_D + P.PRE_TURN, 0.0])
    for _ in range(6):                                        # straight approach, re-localized each step
        err = float(np.angle(np.exp(1j * box['yaw_short'])))
        if abs(err) > 0.03:
            base_move(ctx, log, turn=err)
            box = locate(ctx, log, near=box['center'])
        gap = box['center'][0] - P.STANCE_D
        if gap < 0.015:
            break
        base_move(ctx, log, forward=min(APPROACH_STEP, gap))
        box = locate(ctx, log, near=[box['center'][0] - min(APPROACH_STEP, gap), box['center'][1]])
    if abs(box['center'][0] - P.STANCE_D) > 0.03 or box['near_face_x'] < 0.33:
        raise StageFailed(f"stance not reached: box centre {np.round(box['center'], 3)}, near face {box['near_face_x']:.3f}")
    return box


def prepare(ctx, log, box):
    yaw = P.chest_yaw(box) if abs(box['center'][1]) > 0.02 else 0.0
    (_, y_r), (_, y_l) = box['ends']
    plan = P.grasp_waypoints(box['center'][0], y_r - P.FACE_UNCERTAINTY, y_l + P.FACE_UNCERTAINTY, yaw)
    bad = P.check_plan(plan)
    if bad:
        raise StageFailed('grasp plan unsafe: ' + '; '.join(bad))
    for side in ('right', 'left'):
        g = ctx.value('rby1_gripper', side=side, action='close', current_a=0.5)
        if g['measured_width_m'] > 0.01:
            raise StageFailed(f'{side} gripper did not close into a ball (width {g["measured_width_m"]})')
    r = ctx.value('rby1_move_body', targets={
        'torso': {'positions': torso_pose(0.524, P.LEAN, yaw)},
        'right_arm': {'positions': plan['plan']['right'][0]['q']},
        'left_arm': {'positions': plan['plan']['left'][0]['q']}})
    if not r.get('settled'):
        raise StageFailed('pre-grasp posture not settled')
    log(f'prepared: chest yaw {yaw:.3f}, faces {y_r:.3f} / {y_l:.3f} (+-{P.FACE_UNCERTAINTY})')
    return {'yaw': yaw, 'plan': plan}


def squat(ctx, log, yaw):
    for t in P.squat_sequence(yaw):
        r = ctx.value('rby1_move_body', targets={'torso': {'positions': t}})
        s = ctx.value('rby1_status')
        log(f"squat torso {r['torso']} loads {s['safety']['highest_torque_load']}")
        if not r.get('settled'):
            raise StageFailed('squat not settled')


def _interp(qa, qb, t):
    return (np.array(qa) + t * (np.array(qb) - np.array(qa))).round(4).tolist()


def touch_squeeze(ctx, log, prep, box, retry=True):
    plan = prep['plan']['plan']
    baseline, _ = wrist_forces(ctx)
    W = {s: [w['q'] for w in plan[s][1:]] for s in plan}
    try:
        res = step_until_contact(ctx, W, baseline, log=log)
    except RuntimeError as e:
        side = re.search(r'(left|right) wrist force', str(e))
        if not (retry and side):
            raise StageFailed('touch stopped: ' + str(e)[:300])
        side = side.group(1)                     # descended onto the box edge: the face is further out
        log(f'{side} ball hit the box while descending; backing off 3 cm out, 2 cm up and re-planning')
        out = 0.03 if side == 'left' else -0.03
        ctx.value('rby1_move_eef', arm=side, position=[0.0, out, 0.02], relative=True)
        s = ctx.value('rby1_status')
        (_, y_r), (_, y_l) = box['ends']
        y_r, y_l = y_r - P.FACE_UNCERTAINTY, y_l + P.FACE_UNCERTAINTY
        y_r, y_l = (y_r - 0.03, y_l) if side == 'right' else (y_r, y_l + 0.03)
        new = P.grasp_waypoints(box['center'][0], y_r, y_l, prep['yaw'], seeds={'right': s['right_arm'], 'left': s['left_arm']})
        if P.check_plan(new):
            raise StageFailed('re-plan after contact unsafe: ' + '; '.join(P.check_plan(new)))
        ctx.value('rby1_move_body', targets={'right_arm': {'positions': new['plan']['right'][0]['q']},
                                             'left_arm': {'positions': new['plan']['left'][0]['q']}})
        prep = dict(prep, plan=new)
        return touch_squeeze(ctx, log, prep, box, retry=False)
    if not all(res['contact'].values()):
        raise StageFailed(f"no contact on {[s for s, c in res['contact'].items() if not c]} (box narrower/moved?)")
    targets = {}
    for s in plan:
        i = res['index'][s]
        nxt = W[s][min(i + 1, len(W[s]) - 1)]
        targets[f'{s}_arm'] = {'positions': _interp(W[s][i], nxt, 0.5)}         # +5 mm inward
    ctx.value('rby1_move_body', targets=targets)
    delta, s = squeeze_delta(ctx, baseline)
    log(f"squeeze dF {delta} loads {s['safety']['highest_torque_load']}")
    if max(delta.values()) > MAX_SQUEEZE_N:
        for s_ in plan:
            i = res['index'][s_]
            targets[f'{s_}_arm'] = {'positions': _interp(W[s_][i], W[s_][min(i + 1, len(W[s_]) - 1)], 0.25)}
        ctx.value('rby1_move_body', targets=targets)
        delta, s = squeeze_delta(ctx, baseline)
        log(f'eased squeeze dF {delta}')
    if min(delta.values()) < MIN_SQUEEZE_N:
        raise StageFailed(f'squeeze too weak to lift: {delta}')
    return {'baseline': {k: v.tolist() for k, v in baseline.items()}, 'delta': delta, 'contact': res, 'prep': prep}


def torso_lift(ctx, log, yaw, grip, box):
    before = ctx.value('rby1_observe')['frame_id']
    for t in P.lift_sequence(yaw):
        try:
            ctx.value('rby1_move_body', targets={'torso': {'positions': t}})
        except RuntimeError as e:
            raise StageFailed('torso lift stopped (torque guard? box held, ask the operator): ' + str(e)[:300])
        delta, s = squeeze_delta(ctx, grip['baseline'])
        log(f"lift torso {np.round(t, 3).tolist()} dF {delta} loads {s['safety']['highest_torque_load']}")
        if min(delta.values()) < HOLD_MIN_N:
            raise StageFailed(f'grip lost during the torso lift: {delta}')
    after = ctx.value('rby1_observe')['frame_id']
    chk = perception.moved_with_hands(frame_dir(before), frame_dir(after), (box['center'][0], box['center'][1], box['top_z']))
    log(f'hold check {chk}')
    if chk['held'] is False:
        raise StageFailed(f'box did not rise with the hands: {chk}')
    return {'frames': [before, after], 'check': chk, 'image_check': 'held' if chk['held'] else 'inconclusive (forces only)'}


def set_down(ctx, log, yaw):
    """Abort path (UNTESTED): lower the torso back to the squat so the box rests on the floor, move both
    balls 3 cm outward, raise them 10 cm, then stand up to `ready`. Run it with the box over clear floor."""
    for t in P.squat_sequence(yaw):                       # a 40 N wrist rise stops it if the box lands early
        ctx.value('rby1_move_body', targets={'torso': {'positions': t}})
    ctx.value('rby1_move_eef', arm='right', position=[0.0, -0.03, 0.0], relative=True)
    ctx.value('rby1_move_eef', arm='left', position=[0.0, 0.03, 0.0], relative=True)
    for side in ('right', 'left'):
        ctx.value('rby1_move_eef', arm=side, position=[0.0, 0.0, 0.10], relative=True)
    for t in P.squat_sequence(yaw)[::-1][1:] + [torso_pose(0.524, 0.0, yaw)]:
        ctx.value('rby1_move_body', targets={'torso': {'positions': t}})
    log('set down and released (verify with an image)')


def main(ctx, at_stance=False, log=print):
    report = {'stages': [], 'ok': False}
    try:
        box = locate(ctx, log, near=[P.STANCE_D, 0.0] if at_stance else None)
        report['stages'].append({'locate': box})
        if not at_stance:
            box = align(ctx, log, box)
            report['stages'].append({'align': box})
        prep = prepare(ctx, log, box)
        report['stages'].append({'prepare': {'yaw': prep['yaw']}})
        squat(ctx, log, prep['yaw'])
        report['stages'].append({'squat': 'ok'})
        grip = touch_squeeze(ctx, log, prep, box)
        report['stages'].append({'touch_squeeze': {'delta': grip['delta'], 'index': grip['contact']['index']}})
        lift = torso_lift(ctx, log, prep['yaw'], grip, box)
        report['stages'].append({'torso_lift': lift})
        report['ok'] = True
    except (StageFailed, RuntimeError) as e:
        report['failed'] = str(e)[:600]
        log('FAILED: ' + report['failed'])
    return report


def dry_run(frame_id, log=print):
    box = perception.find_box(frame_dir(frame_id))
    if not box['ok']:
        log('no box: ' + box['why'])
        return {'ok': False}
    log(f"box centre {np.round(box['center'], 3)} top {box['top_z']:.3f} size {np.round(box['size'], 3)} yaw_short {box['yaw_short']:.3f}")
    at_stance = abs(box['center'][0] - P.STANCE_D) < 0.05
    log(f"{'frame is at the stance' if at_stance else 'stance moves'} {'' if at_stance else P.stance_moves(box)}")
    cy = box['center'][1] if at_stance else 0.0               # align() centres the box on the heading
    at = dict(box, center=[P.STANCE_D, cy])
    yaw = P.chest_yaw(at) if abs(cy) > 0.02 else 0.0
    y_r, y_l = cy - box['size'][0] / 2, cy + box['size'][0] / 2
    plan = P.grasp_waypoints(P.STANCE_D, y_r - P.FACE_UNCERTAINTY, y_l + P.FACE_UNCERTAINTY, yaw)
    for side, seq in plan['plan'].items():
        log(f"{side}: " + ', '.join(f"{w['name']} y{w['y']:+.3f} err{w['err'] * 1000:.0f}mm m{w['margin']:.2f} dq{w['dq']:.2f}" for w in seq))
    bad = P.check_plan(plan)
    log('plan problems: ' + ('; '.join(bad) if bad else 'none'))
    return {'ok': not bad, 'box': box, 'yaw': yaw, 'plan': plan}


def selftest():
    """Offline regression on the bundled evidence (no robot): 4 dry-run plans + 2 hold checks."""
    ev = json.loads((SKILL_DIR / 'evidence.json').read_text())['offline_checks']
    ok = True
    for f in ev['dry_run_frames_ok']:
        r = dry_run(f, log=lambda m: None)
        print(f'dry-run {f}: {"ok" if r["ok"] else "PROBLEM"}')
        ok &= r['ok']
    cases = {'static_box_squat_step': ((0.47, 0.014, 0.255), False), 'held_box_torso_lift': ((0.47, 0.014, 0.41), True)}
    for name, (point, expected) in cases.items():
        c = ev['hold_check'][name]
        r = perception.moved_with_hands(frame_dir(c['before']), frame_dir(c['after']), point)
        print(f'hold check {name}: held={r["held"]} (expected {expected})')
        ok &= r['held'] is expected
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--frame', help='frame id in runtime/frames/ or a frame folder path, for --dry-run')
    ap.add_argument('--selftest', action='store_true', help='offline checks on the bundled evidence')
    ap.add_argument('--at-stance', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        good = selftest()
        print(json.dumps({'selftest_ok': bool(good)}))
        sys.exit(0 if good else 1)
    if args.dry_run:
        out = dry_run(args.frame)
        print(json.dumps({'ok': out['ok']}))
        sys.exit(0)
    from api.runtime import Context
    run_dir = WORKSPACE / 'output' / 'runs' / (time.strftime('%Y%m%dT%H%M%S') + '_box_squeeze')
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)
    with Context() as ctx:
        rep = main(ctx, at_stance=args.at_stance, log=log)
    rep['log'] = lines
    (run_dir / 'report.json').write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({'ok': rep['ok'], 'failed': rep.get('failed'), 'report': str(run_dir / 'report.json')}))
