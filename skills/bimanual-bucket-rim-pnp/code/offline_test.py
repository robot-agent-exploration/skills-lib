"""Offline checks (no robot, no Gemini): module imports, pinch IK at the tested stance, knee clearance, route
planning on the C1 scene, and the silhouette fit on a saved frame with a fixed target box.

    MPLCONFIGDIR=$TMPDIR python3 code/basket_pnp/offline_test.py [frame_dir_with_bucket]"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grasp_plan as gp  # noqa: E402
import nav  # noqa: E402
import perception as pc  # noqa: E402
import steps as S  # noqa: E402
from lib import frames  # noqa: E402


def check_ik(c=(0.595, 0.0), rim=0.37):
    pins = gp.pinch_points(c, rim, depth=S.PINCH_DEPTH)
    out, q = {}, {}
    for stage, dz in (('pre', S.PRE_DZ), ('grasp', 0.0)):
        for side in ('left', 'right'):
            R = gp.grip_rotation(side, S.TILT, flip=True)
            seeds = [q[('pre', side)]] if stage == 'grasp' else None          # as steps.pregrasp
            r = gp.solve_arm(S.TORSO_GRASP, side, gp.ee_for_pinch(pins[side] + [0, 0, dz], R), R, seeds=seeds)
            q[(stage, side)] = r['q']
            out[f'{stage}_{side}'] = (round(r['pos_err'] * 1000, 2), round(r['margin'], 3))
    ok = all(v[0] < 2.0 and v[1] > 0.1 for v in out.values())
    kc = gp.knee_clearance(S.TORSO_GRASP, c, rim)
    print(f'IK (pos err mm, joint margin rad): {out}  knee clearance {kc:.3f} m -> {"OK" if ok and kc > 0.03 else "FAIL"}')
    return ok and kc > 0.03


def check_route():
    # C1-like scene in a local odometry frame: robot at the origin facing +x, bucket 0.79 ahead, cabinet behind it
    pose = (0.0, 0.0, 0.0)
    obs = {'bucket': nav.circle([0.79, -0.02], 0.16),
           'cabinet': nav.Polygon([[1.16, 0.34], [2.01, 0.37], [2.02, -0.23], [1.18, -0.26]])}
    goal = np.array([0.79 - S.STANCE_D, -0.02])
    r = nav.plan_route(pose, goal, 0.0, obs, min_clear=0.12, final_straight=0.15, target='bucket', target_clear=0.05)
    print('route to a stance 0.19 m ahead:', None if r is None else [{k: round(v, 3) for k, v in m.items()} for m in r['moves']])
    return r is not None


def check_silhouette(fdir):
    _, _, rgb, _ = frames.load(fdir, stride=16)
    om = pc.orange_mask(rgb)
    ys, xs = np.nonzero(om)
    if len(xs) < 400:
        print('no orange pixels in', fdir)
        return False
    box = [int(np.percentile(xs, 1)), int(np.percentile(ys, 1)), int(np.percentile(xs, 99)), int(np.percentile(ys, 99))]
    tm = pc.target_mask(rgb, box)
    P, _ = pc.mask_points(fdir, tm & om)
    f = pc.fit_silhouette(fdir, tm, np.median(P[:, :2], axis=0), z0=pc.floor_z_near(fdir, np.median(P[:, :2], axis=0)))
    print('silhouette fit:', {k: round(v, 3) if isinstance(v, float) else np.round(v, 3).tolist() for k, v in f.items()})
    return f['iou'] > 0.8


if __name__ == '__main__':
    fdir = sys.argv[1] if len(sys.argv) > 1 else None
    res = {'ik': all(check_ik(c, 0.37) for c in ((0.58, -0.04), (0.595, 0.0), (0.60, 0.045), (0.61, 0.0))),
           'route': check_route()}
    if fdir:
        res['silhouette'] = check_silhouette(fdir)
    print('offline checks:', res)
    sys.exit(0 if all(res.values()) else 1)
