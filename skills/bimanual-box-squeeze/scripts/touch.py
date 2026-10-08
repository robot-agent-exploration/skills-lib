"""Contact by wrist force: step both arms along joint waypoints until each one touches.

The wrist F/T readings carry large posture-dependent offsets (e.g. right z ~41 N at rest) and their
frame is not documented, so contact = norm of the change from a no-contact baseline taken in the
same gripper orientation. A 1 cm inward step into the box raised it by 7-15 N on 2026-10-08.
"""
import json

import numpy as np

CONTACT_N = 4.5


def wrist_forces(ctx):
    s = ctx.value('rby1_status')
    return {k: np.array(v, float) for k, v in s['wrist_force_n'].items()}, s


def squeeze_delta(ctx, baseline):
    f, s = wrist_forces(ctx)
    return {k: float(np.linalg.norm(f[k] - np.asarray(baseline[k]))) for k in baseline}, s


def step_until_contact(ctx, waypoints, baseline, threshold=CONTACT_N, log=print):
    """waypoints: {'right': [q, ...], 'left': [q, ...]}. Each arm stops at the first waypoint where its
    force change exceeds `threshold`. A refused / stopped motion raises RuntimeError (caller backs off)."""
    idx = {s: -1 for s in waypoints}
    contact = {s: False for s in waypoints}
    delta = {s: 0.0 for s in waypoints}
    while True:
        targets = {f'{s}_arm': {'positions': list(w[idx[s] + 1])}
                   for s, w in waypoints.items() if not contact[s] and idx[s] + 1 < len(w)}
        if not targets:
            break
        reply = ctx.value('rby1_move_body', targets=targets)
        if not reply.get('settled', False):
            raise RuntimeError('move not settled: ' + json.dumps(reply)[:300])
        for key in targets:
            idx[key[:-4]] += 1
        delta, status = squeeze_delta(ctx, baseline)
        for s in waypoints:
            contact[s] = contact[s] or delta[s] > threshold
        log(f'touch step {idx} dF {{{", ".join(f"{k}: {v:.1f}" for k, v in delta.items())}}} contact {contact} '
            f'loads {status["safety"]["highest_torque_load"]}')
    return {'index': idx, 'contact': contact, 'delta': delta}
