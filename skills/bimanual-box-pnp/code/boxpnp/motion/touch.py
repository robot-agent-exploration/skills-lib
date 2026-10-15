"""Wrist-force contact search with the controller's Cartesian steps (one arm)."""
import numpy as np

from ..core import robot

from ..params import CONTACT_DF_N, STEP_M  # hyper-parameters: boxpnp/params.py


def step_until_contact(ctx, side, direction, max_travel=0.08, step=STEP_M, threshold=CONTACT_DF_N, log=print,
                       knee_y=None, speed=0.02):
    """Step one hand along `direction` until its wrist force changes by > threshold.

    knee_y: refuse a step that brings this ee closer than knee_y to the base x axis (|y|) - squatting, the
    wrist meets the torso knee link in the model there. -> {ok, travel, dF, F0, F, why}"""
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    F0 = robot.wrist_force(ctx)[side]
    travel, dF = 0.0, 0.0
    while travel < max_travel - 1e-9:
        d = min(step, max_travel - travel)
        if knee_y is not None:
            y_now = robot.status(ctx)[f'{side}_ee']['position'][1]
            if abs(y_now + direction[1] * d) < knee_y:
                return {'ok': False, 'travel': travel, 'dF': dF, 'F0': F0.tolist(),
                        'why': f'knee guard: ee y {y_now:+.3f} would pass |y| {knee_y}'}
        try:
            robot.hand_step(ctx, side, direction * d, speed=speed)
        except RuntimeError as e:          # handled stop (force guard): report, do not push further
            F = robot.wrist_force(ctx)[side]
            log(f'{side}: step stopped: {e}')
            return {'ok': False, 'stopped': True, 'travel': travel, 'dF': float(np.linalg.norm(F - F0)),
                    'F0': F0.tolist(), 'F': F.tolist(), 'why': str(e)[:300]}
        travel += d
        F = robot.wrist_force(ctx)[side]
        dF = float(np.linalg.norm(F - F0))
        log(f'{side}: travel {travel:.3f} dF {dF:.1f} N')
        if dF > threshold:
            return {'ok': True, 'travel': travel, 'dF': dF, 'F0': F0.tolist(), 'F': F.tolist()}
    return {'ok': False, 'travel': travel, 'dF': dF, 'F0': F0.tolist(), 'why': 'no contact'}
