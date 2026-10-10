"""Entry of oc-hand-reach: move one or both hands to base-frame poses with the controller's whole-body optimal
control (see oc_reach.py for the contract). Example:

    main(ctx, targets={'right': {'pos': [0.52, -0.10, 1.05], 'yaw': -0.17, 'pitch': -0.785},
                       'left':  {'pos': [0.52, 0.10, 1.05], 'yaw': 0.17, 'pitch': -0.785}})
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oc_reach as O  # noqa: E402


def main(ctx, targets, torso=None, posture='natural', w_posture=0.2, step_m=0.20, step_rad=0.35, tol_m=0.005,
         log=print):
    """targets: {side: {'pos': [x, y, z], and 'quat': [x, y, z, w] or 'yaw' + 'pitch' (rad, R = Rz(yaw) Ry(pitch)),
    optional 'w_pos', 'w_rot'}}. -> {ok, measured: {side: {pos, quat, err_pos_m, err_rot_rad, arm}, calls, seconds,
    loads, torso}, why}"""
    tg = {}
    for side, t in targets.items():
        q = t.get('quat')
        if q is None and 'yaw' in t:
            q = O.quat_yaw_pitch(float(t['yaw']), float(t.get('pitch', -0.785)))
        tg[side] = {'pos': [float(v) for v in t['pos']], 'quat': q,
                    'w_pos': float(t.get('w_pos', 10.0)), 'w_rot': float(t.get('w_rot', 5.0))}
    r = O.move_hands(ctx, tg, torso=torso, posture=posture, w_posture=w_posture, step_m=step_m, step_rad=step_rad,
                     tol_m=tol_m, log=log)
    return {'ok': r['ok'], 'why': r['why'],
            'measured': {**r['measured'], 'calls': r['calls'], 'seconds': r['seconds'], 'loads': r['loads'],
                         'torso': r['torso']}}
