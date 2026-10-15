"""push_chair task script: step-wise controller to grasp an office chair by its backrest and tuck it into a desk.

Run one step at a time from the workspace root, with <skill>/code on PYTHONPATH, and look at the returned frames
before the next step:

    python -m office_chair_tuck.run_tuck measure            # hands wide, bar profile, desk front, chair yaw
    python -m office_chair_tuck.run_tuck square             # turn by the chair yaw and lane-change to centre on it
    python -m office_chair_tuck.run_tuck grasp              # two-hand tangent-yaw grasp of the backrest top bar
    python -m office_chair_tuck.run_tuck pull 0.35          # pull the chair back (right hand caged)
    python -m office_chair_tuck.run_tuck turn_to <yaw>      # absolute odometry yaw, holding the chair
    python -m office_chair_tuck.run_tuck drive <m>          # straight, holding the chair
    python -m office_chair_tuck.run_tuck tuck               # push until bar-to-desk-edge = 0.42 m (stops on force)
    python -m office_chair_tuck.run_tuck release            # open, back out, hands up
    python -m office_chair_tuck.run_tuck verify             # measure the tuck (gap, centring, yaw) + frame

`main(ctx, step, *args)` returns {ok, measured, why, frames, checkpoint}. State between steps (the last measurement,
the desk-facing yaw) is kept in output/push_chair_state.json.
"""
import json
import math
import sys

import numpy as np

from . import chair_ctl as C

ROOT = C.geom.ROOT
STATE = ROOT / 'output' / 'push_chair_state.json'
from .params import TUCK_GAP, SQUARE_TOL, CENTRE_TOL  # hyper-parameters: params.py


def _load():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def main(ctx, step, *args, log=C.log_default):
    st = _load()
    if step == 'measure':
        m = C.measure(ctx, log=log)
        st['last'] = {k: m[k] for k in ('frame', 'profile', 'desk', 'chair_yaw')}
        if m['desk'].get('ok'):
            st['desk_yaw'] = C.status(ctx)['odometry']['yaw']   # robot is square to the desk when measuring it
        _save(st)
        prof = m['profile']
        return {'ok': m['ok'], 'why': m['why'], 'frames': [m['frame']],
                'measured': {'bar_centre_y': prof.get('centre_y'), 'bar_x': prof.get('x_at_centre'),
                             'bar_z': prof.get('z_top'), 'chair_yaw': m['chair_yaw'],
                             'desk_front_x': m['desk'].get('x_front_median')},
                'checkpoint': {'subagent': 'haiku', 'frames': [f"runtime/frames/{m['frame']}/head_camera.jpg"],
                               'question': 'Is the office chair directly in front of the robot with its backrest '
                                           'towards the camera, and is anything (person, object) between the robot '
                                           'and the chair?',
                               'pass_if': 'chair in front, backrest towards camera, nothing in between'}}
    if step == 'square':
        prof = st['last']['profile']
        yaw = st['last'].get('chair_yaw') or 0.0
        out = {}
        if abs(yaw) > SQUARE_TOL:
            out['turn'] = C.base_turn(ctx, yaw, step=0.2, speed=0.1, log=log)
        m = C.measure(ctx, log=log)
        st['last'] = {k: m[k] for k in ('frame', 'profile', 'desk', 'chair_yaw')}
        _save(st)
        dy = m['profile'].get('centre_y', 0.0)
        if m['ok'] and abs(dy) > CENTRE_TOL:
            out['lane'] = C.lane_change(ctx, dy, log=log)
            m = C.measure(ctx, log=log)
            st['last'] = {k: m[k] for k in ('frame', 'profile', 'desk', 'chair_yaw')}
            _save(st)
        return {'ok': m['ok'], 'measured': {'moves': out, 'bar_centre_y': m['profile'].get('centre_y'),
                                            'chair_yaw': m['chair_yaw']}, 'frames': [m['frame']], 'why': m['why']}
    if step == 'grasp':
        r = C.grasp_bar(ctx, st['last']['profile'], log=log)
        if r.get('ok'):
            st['wrist_base'] = r['wrist_base']
            _save(st)
        fid = C.observe(ctx, head=(0.0, 0.75))
        r['frames'] = [fid]
        r['checkpoint'] = {'subagent': 'haiku', 'frames': [f'runtime/frames/{fid}/head_camera.jpg'],
                           'question': 'Are both robot grippers closed around the top bar of the chair backrest?',
                           'pass_if': 'yes for both (widths 15-30 mm also say so)'}
        r.pop('targets', None)
        return r
    if step == 'pull':
        return C.pull(ctx, float(args[0]), log=log)
    if step == 'turn_to':
        return C.base_turn_to(ctx, float(args[0]) if args else st['desk_yaw'], log=log)
    if step == 'drive':
        return C.base_drive(ctx, float(args[0]), step=0.15, speed=0.05, log=log)
    if step == 'tuck':
        # optional agent-measured args (trial 2: holding the chair, `measure` would move the hands; the stored desk
        # front was from another place): tuck <desk_front_x> [bar_x], both base frame, robot square to the desk
        desk_x = float(args[0]) if args else st['last']['desk'].get('x_front_median')
        bar_x = float(args[1]) if len(args) > 1 else st['last']['profile'].get('x_at_centre')
        if desk_x is None or bar_x is None:
            return {'ok': False, 'why': 'measure the desk and the bar first (robot square to the desk)'}
        dist = desk_x - bar_x - TUCK_GAP
        log(f'  tuck: desk {desk_x:.3f} bar {bar_x:.3f} -> push {dist:.3f}')
        if dist <= 0:
            return {'ok': True, 'moved': 0.0, 'why': 'already tucked'}
        far = max(0.0, dist - 0.10)
        r = C.push(ctx, far, step=0.10, log=log) if far > 0 else {'ok': True, 'moved': 0.0}
        if r['ok']:
            r2 = C.push(ctx, dist - r['moved'], step=0.03, speed=0.03, log=log)
            r = {'ok': r2['ok'], 'moved': r['moved'] + r2['moved'], 'why': r2.get('why', '')}
        return r
    if step == 'find':
        ch = C.find_chair(ctx, log=log)
        if ch.get('ok'):
            ch['odom'] = C.status(ctx)['odometry']
            st['chair'] = {k: ch[k] for k in ('column', 'facing_rad', 'bar_centre', 'bar_z', 'odom', 'frame_id')}
            _save(st)
        return {'ok': bool(ch.get('ok')), 'why': ch.get('why', ''), 'frames': [ch.get('frame_id', '')],
                'measured': {k: ch.get(k) for k in ('column', 'facing_rad', 'bar_width', 'column_from')},
                'checkpoint': {'subagent': 'haiku', 'frames': [f"runtime/frames/{ch.get('frame_id', '')}/head_camera.jpg"],
                               'question': 'Is there an office chair with a mesh backrest in the image, and is the '
                                           'floor between the robot and the chair free of people and objects?',
                               'pass_if': 'chair visible, path free'}}
    if step == 'approach':
        ch = st['chair']
        odom = C.status(ctx)['odometry']
        col = C.from_odom(odom, C.to_odom(ch['odom'], ch['column']))
        facing = ch['facing_rad'] + ch['odom']['yaw'] - odom['yaw']
        pl = C.plan_approach({'column': col, 'facing_rad': facing})
        log('  approach plan', pl)
        if args and args[0] == 'plan':
            return {'ok': True, 'measured': pl}
        r = C.execute_moves(ctx, pl['moves'], log=log)
        r['plan'] = pl
        return r
    if step == 'desk':
        m = C.measure(ctx, log=log)
        d = m['desk']
        if not d.get('ok') or d.get('centre_y') is None:
            return {'ok': False, 'why': f"desk front not measured: {d.get('why', d)}", 'frames': [m['frame']]}
        bins = [b for b in d['bins'] if abs(b[1] - d['x_front_median']) < 0.06]
        slope = float(np.polyfit([b[0] for b in bins], [b[1] for b in bins], 1)[0]) if len(bins) > 2 else 0.0
        odom = C.status(ctx)['odometry']
        st['desk'] = {'front_centre_odom': C.to_odom(odom, [d['x_front_median'], d['centre_y']]),
                      'yaw_odom': odom['yaw'] - math.atan(slope), 'height': d['height'], 'frame': m['frame']}
        st['desk_yaw'] = st['desk']['yaw_odom']
        st['last'] = {k: m[k] for k in ('frame', 'profile', 'desk', 'chair_yaw')}
        _save(st)
        return {'ok': True, 'measured': st['desk'], 'frames': [m['frame']]}
    if step == 'transport':
        dk = st['desk']
        odom = C.status(ctx)['odometry']
        E = C.from_odom(odom, dk['front_centre_odom'])
        yaw_rel = math.atan2(math.sin(dk['yaw_odom'] - odom['yaw']), math.cos(dk['yaw_odom'] - odom['yaw']))
        pl = C.plan_transport(E, yaw_rel, pull_back=float(args[0]) if args else 0.30)
        log('  transport plan', pl)
        if args and args[-1] == 'plan':
            return {'ok': True, 'measured': pl}
        wb = C.wrist(C.status(ctx))
        for kind, val in pl['moves']:
            if kind == 'drive' and val < 0:
                r = C.pull(ctx, -val, log=log)
            elif kind == 'drive':
                r = C.base_drive(ctx, val, step=0.15, speed=0.05, wrist_base=wb, log=log)
            elif kind == 'turn':
                r = C.base_turn(ctx, val, step=0.15, speed=0.1, wrist_base=wb, log=log)
            else:
                r = C.base_turn_to(ctx, dk['yaw_odom'], wrist_base=wb, log=log)
            if not r['ok']:
                return {'ok': False, 'why': r.get('why'), 'plan': pl}
        fid = C.observe(ctx, head=(0.0, 0.55))
        return {'ok': True, 'plan': pl, 'frames': [fid]}
    if step == 'release':
        return C.release(ctx, log=log)
    if step == 'verify':
        m = C.measure(ctx, log=log)
        desk_x = m['desk'].get('x_front_median')
        bar_x = m['profile'].get('x_at_centre')
        gap = None if desk_x is None or bar_x is None else round(desk_x - bar_x, 3)
        ok = gap is not None and abs(gap - TUCK_GAP) < 0.06 and abs(m['profile'].get('centre_y', 1)) < 0.10
        return {'ok': ok, 'frames': [m['frame']],
                'measured': {'bar_to_desk_edge': gap, 'bar_centre_y': m['profile'].get('centre_y'),
                             'chair_yaw': m['chair_yaw']},
                'checkpoint': {'subagent': 'haiku', 'frames': [f"runtime/frames/{m['frame']}/head_camera.jpg"],
                               'question': 'Is the office chair pushed under the light-wood desk, facing it squarely '
                                           '(backrest parallel to the desk edge), with its armrests at or under the '
                                           'desk edge, and roughly centred on the desk?',
                               'pass_if': 'yes'}}
    return {'ok': False, 'why': f'unknown step {step}'}


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    from api.runtime import Context
    with Context() as ctx:
        print(json.dumps(main(ctx, sys.argv[1], *sys.argv[2:]), indent=1, default=str))
