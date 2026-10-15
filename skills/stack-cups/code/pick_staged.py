"""Staged rim pick of one cup with agent looks between (pp_cups.pick_cup split into stages). v4 (trial 3, 2026-10-10):
the one-shot pick slipped (2.4 cm pinch on the rim lip) and moved the cup 4 cm; this one keeps the hand out of the
camera's view while locating (a hand hovering over the cup merged into its mask: rim 0.97 m, height 0.23 m), pinches
4 cm deep and lets the agent look before descending and before closing. CUP_CLEAR=0 skips the move-clear (its IK
flipped a shoulder once when started from ready: start from ready and skip it).
  python3 skill-library/skills/stack-cups/code/pick_staged.py pre "<cup name>" left [depth]   # locate, plan, open, move to the pre-grasp, look
  python3 skill-library/skills/stack-cups/code/pick_staged.py down [dx dy]                    # optional xy correction, descend, look
  python3 skill-library/skills/stack-cups/code/pick_staged.py close                           # close, lift 6 cm, held check, raise
State: output/cup_pick_staged.json"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
from stack_cups import pp_cups as K  # noqa: E402
from office_chair_tuck import chair_ctl as C  # noqa: E402
from office_chair_tuck import kin  # noqa: E402
from stack_cups import pp_grasp as G  # noqa: E402
from stack_cups import pp_motion as M  # noqa: E402
from api.runtime import Context  # noqa: E402

ST = ROOT / 'output' / 'cup_pick_staged.json'
from stack_cups.params import DESK_LOOK  # hyper-parameters: stack_cups/params.py


def main(ctx, stage, args):
    if stage == 'pre':
        name, side = args[0], args[1]
        depth = float(args[2]) if len(args) > 2 else 0.04
        e = C.status(ctx)[f'{side}_ee']             # hand out of the camera's view first (it hid the cup, 19:53)
        clear = [0.30, 0.38 if side == 'left' else -0.38, max(1.20, e['position'][2])]
        if __import__('os').environ.get('CUP_CLEAR', '1') == '1' and np.linalg.norm(np.array(e['position'][:2]) - clear[:2]) > 0.05:
            M.move_arm(ctx, side, clear, e['quaternion'])
        loc = K.locate(ctx, [name], look=DESK_LOOK)
        cup = loc['measured']['cups'][name]
        if not cup.get('ok'):
            return {'ok': False, 'why': cup.get('why'), 'frames': loc['frames']}
        s = C.status(ctx)
        obj = dict(cup, rim_in=K.CUP_RIM_IN, depth=depth)
        plan = G.plan_rim(obj, side, s['torso'], {'left_arm': s['left_arm'], 'right_arm': s['right_arm']},
                          pitches=(0,), cur_quat=s[f'{side}_ee']['quaternion'])
        if not plan['ok']:
            return {'ok': False, 'why': plan['why']}
        ctx.value('rby1_gripper', side=side, action='open')
        r = M.move_arm(ctx, side, plan['pre']['pos'], plan['pre']['quat'])
        if not r['ok']:
            return {'ok': False, 'why': f"pre-grasp: {r['why']}"}
        M.look_at(ctx, [cup['centre'][0], cup['centre'][1], cup['rim_z']])
        fid = C.observe(ctx)
        ST.write_text(json.dumps({'name': name, 'side': side, 'depth': depth, 'cup': cup, 'plan': plan}, default=str))
        return {'ok': True, 'cup': {k: cup[k] for k in ('centre', 'r', 'rim_z', 'height')}, 'rim_angle': plan['rim_angle'],
                'yaw_deg': round(math.degrees(plan['yaw']), 1), 'pinch': plan['pinch'], 'fingers': plan['fingers'],
                'pre': plan['pre']['pos'], 'grasp': plan['grasp']['pos'], 'frame': fid}
    st = json.loads(ST.read_text())
    side, plan = st['side'], st['plan']
    if stage == 'down':
        d = np.array([float(args[0]), float(args[1]), 0.0]) if len(args) >= 2 else np.zeros(3)
        pts = [list(np.array(p) + d) for p in plan['approach'] + [plan['grasp']['pos']]]
        if np.any(d):
            M.cart(ctx, side, list(np.array(plan['pre']['pos']) + d), speed=0.015, stop_n=K.CONTACT_N)
        for p in pts:
            r = M.cart(ctx, side, p, speed=0.015, stop_n=K.CONTACT_N)
            if not r['ok']:
                M.cart(ctx, side, list(np.array(plan['pre']['pos']) + d), speed=0.02, stop_n=30)
                return {'ok': False, 'why': f"approach: {r['why']}"}
        st['offset'] = d.tolist(); ST.write_text(json.dumps(st, default=str))
        fid = C.observe(ctx)
        return {'ok': True, 'ee': C.status(ctx)[f'{side}_ee']['position'], 'frame': fid}
    if stage == 'close':
        cup, depth = st['cup'], st['depth']
        g = ctx.value('rby1_gripper', side=side, action='close', current_a=M.CLOSE_A)
        ee = C.status(ctx)[f'{side}_ee']['position']
        M.move_arm(ctx, side, [ee[0], ee[1], ee[2] + K.LIFT_CUP], plan['grasp']['quat'])
        fid = C.observe(ctx)
        verdict, low = K.held_check(ctx, fid, cup.get('support_z') or 0.74)
        out = {'width_mm': round(g['measured_width_m'] * 1000, 1), 'held': verdict, 'lowest': low, 'frame': fid}
        if verdict != 'held':
            out['ok'] = None
            return out
        h = cup['height'] if cup.get('height') and 0.05 < cup['height'] < 0.25 else K.CUP_H
        pinch = {'rim_angle': plan['rim_angle'], 'r': cup['r'], 'rim_in': K.CUP_RIM_IN, 'depth': depth, 'h': h,
                 'yaw': plan['yaw'], 'pitch': plan['pitch']}
        bottom = cup['rim_z'] + K.CARRY_CLEAR + K.NEST_RISE
        e = C.status(ctx)[f'{side}_ee']
        tip_z = e['position'][2] + (np.asarray(kin.R_from_quat(e['quaternion'])) @ [0, 0, -G.TIP_LOW])[2]
        lift = bottom - (tip_z + depth - h)
        if lift > 0:
            M.move_arm(ctx, side, [e['position'][0], e['position'][1], e['position'][2] + lift], e['quaternion'])
        # hand the held cup to the stack-cups state so `run_cups.py align/insert` continue from here
        cs = K.load_state() if hasattr(K, 'load_state') else None
        state_p = ROOT / 'output' / 'cup_state.json'
        cs = json.loads(state_p.read_text()) if state_p.exists() else {'held': {}}
        cs.setdefault('held', {})[side] = {'name': st['name'], 'pinch': pinch, 'rise': round(low - 0.743, 3)}
        state_p.write_text(json.dumps(cs, default=str))
        out.update(ok=True, pinch=pinch)
        return out
    return {'ok': False, 'why': f'unknown stage {stage}'}


if __name__ == '__main__':
    with Context() as ctx:
        print(json.dumps(main(ctx, sys.argv[1], sys.argv[2:]), default=str))
