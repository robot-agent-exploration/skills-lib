"""Banana pick at its measured middle, closing across the local tangent, fingers tilted along the banana (trial 3,
2026-10-10: the agent measures the centreline by pixel depth, the arc-length middle and the local tangent; yaw =
tangent + 90 deg, the tilt about the closing axis points the fingers along the banana and keeps every joint off its
limit - fingers straight down the right arm was at full stretch). Descends in 1 cm steps with a 6 N stop.
  python3 skill-library/skills/stack-cups/code/banana_pick.py pre <tip_x> <tip_y> <tip_z> <yaw_deg> <tilt_rad> <side>
  python3 skill-library/skills/stack-cups/code/banana_pick.py down        # 1 cm steps with a 6 N stop to the grasp, look
  python3 skill-library/skills/stack-cups/code/banana_pick.py close       # close, lift 5 cm, report width + wrist force"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import _deps  # noqa: F401,E402
import numpy as np  # noqa: E402
import pp_cups  # noqa: F401,E402
import chair_ctl as C  # noqa: E402
import pp_motion as M  # noqa: E402
from scipy.spatial.transform import Rotation as Rr  # noqa: E402
from api.runtime import Context  # noqa: E402

ST = ROOT / 'output' / 'banana_pick.json'
TIP = 0.192


def main(ctx, stage, a):
    if stage == 'pre':
        tip = np.array([float(v) for v in a[:3]]); yaw, tilt, side = float(a[3]), float(a[4]), a[5]
        R = Rr.from_euler('z', math.radians(yaw)) * Rr.from_euler('x', tilt)
        q = R.as_quat().tolist(); f = R.apply([0, 0, -1.0])
        grasp = (tip - TIP * f).round(4)
        ctx.value('rby1_gripper', side=side, action='open')
        r = M.move_arm(ctx, side, (grasp + [0, 0, 0.08]).tolist(), q)
        ST.write_text(json.dumps({'side': side, 'q': q, 'grasp': grasp.tolist()}))
        return {'ok': r['ok'], 'why': r.get('why'), 'grasp': grasp.tolist(), 'frame': C.observe(ctx)}
    st = json.loads(ST.read_text()); side = st['side']; g = np.array(st['grasp'])
    if stage == 'down':
        for up in (0.05, 0.03, 0.02, 0.01, 0.0):
            r = M.cart(ctx, side, (g + [0, 0, up]).tolist(), speed=0.015, stop_n=6.0)
            if not r['ok']:
                return {'ok': False, 'why': f'stopped at +{up}: {r["why"]}', 'frame': C.observe(ctx)}
        return {'ok': True, 'ee': C.status(ctx)[f'{side}_ee']['position'], 'frame': C.observe(ctx)}
    if stage == 'close':
        w0 = np.array(C.status(ctx)['wrist_force_n'][side])
        gr = ctx.value('rby1_gripper', side=side, action='close', current_a=0.5)
        e = C.status(ctx)[f'{side}_ee']
        r = M.cart(ctx, side, (np.array(e['position']) + [0, 0, 0.05]).tolist(), speed=0.02, stop_n=30)
        s = C.status(ctx)
        return {'ok': gr['measured_width_m'] > 0.012, 'width_mm': round(gr['measured_width_m'] * 1000, 1),
                'blocked': gr.get('blocked_by_object'), 'dF': (np.array(s['wrist_force_n'][side]) - w0).round(1).tolist(),
                'lift': r.get('ok'), 'frame': C.observe(ctx)}


if __name__ == '__main__':
    with Context() as ctx:
        print(json.dumps(main(ctx, sys.argv[1], sys.argv[2:]), default=str))
