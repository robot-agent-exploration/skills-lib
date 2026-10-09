"""Start-of-task checks and the neutral posture."""
import numpy as np

from ..core import robot
from ..core.runner import StepFailed


def prepare(rt, grippers_closed=True):
    """Armed, no EMO / fault, grippers homed (and closed into balls), torso and arms at ready, hands empty."""
    s = rt.status()
    if not s['motion']['armed'] or s['emo_pressed'] or s['safety']['fault']:
        raise StepFailed(f"robot not ready: armed {s['motion']['armed']} emo {s['emo_pressed']} fault {s['safety']['fault']}")
    if rt.world.held:
        raise StepFailed(f'world says {rt.world.held} is held; resume from a carry step instead')
    ops = s['grippers']['openings_m']
    if grippers_closed and (not s['grippers']['homed'] or any(o is None or o > 0.005 for o in ops.values())):
        for side in ('right', 'left'):
            rt.v('rby1_gripper', side=side, action='close', current_a=0.5)
    near_ready = np.allclose(s['torso'], robot.TORSO_READY, atol=0.02) and all(
        np.allclose(s[f'{k}_arm'], robot.ARM_READY[k], atol=0.03) for k in ('right', 'left'))
    if not near_ready:
        robot.ready_posture(rt.ctx, rt.log)
    F = robot.wrist_force(rt.ctx)
    return {'odometry': s['odometry'], 'wrist_force': {k: np.round(v, 1).tolist() for k, v in F.items()}}
