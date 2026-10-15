"""Drop a held object into an open container (bucket / basket) from above, one hand: a CLOSED-FORM hand pose and
the controller's whole-body optimal control (oc-hand-reach) - no IK or yaw search.

Pose: the fingers pitched `pitch` (default -0.7 rad: forward-down, the elbow stays low) and yawed from the shoulder
towards the target (oc_reach.yaw_toward); the ee sits `hold` back along the finger axis from the point where the
object is held (fingertips' middle 0.20 m; a cup pinched 4 cm deep hangs ~0.12 m lower), so that point is over
the target. The caller sets the torso posture first with a tested joint move (a floor bucket's rim at 0.37 m
needed torso `lowest` + lean 0.15 for the left hand with the cups; `ready` sufficed for the banana with the right
hand) and checks the image before releasing.

Replaces code/compose natural_ik.py + drop_plan.py (searched yaws for the smallest joint cost, 2026-10-09)."""
import math

import numpy as np

from oc_hand_reach import oc_reach as O


def drop_pose(ctx, side, target, hold=0.20, pitch=-0.7):
    """target: [x, y, z] base frame of the held point (fingertips' middle) at release. -> (ee pos, quat)"""
    yaw = O.yaw_toward(ctx, side, target)
    R = O.Rz(yaw) @ O.Ry(pitch)
    f = R @ np.array([0.0, 0.0, -1.0])
    pos = np.array(target, float) - hold * f
    return pos.round(4).tolist(), O.quat_from_R(R)


def drop(ctx, side, target, hold=0.20, pitch=-0.7, open_width=0.10, lift=0.10, release=True, log=print):
    """Move the hand so the held point is at `target`, open the gripper (if release), lift the hand by `lift`.
    -> {ok, why, measured: {pose, oc, opened}}"""
    pos, quat = drop_pose(ctx, side, target, hold, pitch)
    log(f'  drop pose {side}: ee {pos} yaw/pitch from the shoulder')
    r = O.move_hands(ctx, {side: {'pos': pos, 'quat': quat}}, log=log)
    if not r['ok']:
        return {'ok': False, 'why': f"drop pose: {r['why']}", 'measured': {'pose': pos, 'oc': r}}
    opened = None
    if release:
        opened = ctx.value('rby1_gripper', side=side, action='width', width_m=open_width)
        up = (np.array(r['measured'][side]['pos']) + [0, 0, lift]).tolist()
        O.move_hands(ctx, {side: {'pos': up, 'quat': None}}, log=log)
    return {'ok': True, 'why': '', 'measured': {'pose': pos, 'oc': {k: r[k] for k in ('calls', 'seconds', 'loads')},
                                                'opened': opened}}
