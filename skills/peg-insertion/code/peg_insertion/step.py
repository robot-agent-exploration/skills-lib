"""Run one proven step from the shell (from the workspace root, for api.runtime; <skill>/code on PYTHONPATH):
    python3 -u -m peg_insertion.step centre '{"centre":[x,y],"z":z,"yaw_deg":psi}'
    python3 -u -m peg_insertion.step centre_tri '{"centre":[x,y],"z":z,"yaw_deg":psi,"hole_yaw_deg":h}'
    python3 -u -m peg_insertion.step hop '{"xy":[x,y],"z_surf":z,"yaw_deg":psi}'
(In the session: code/peg_insertion/step.py with PYTHONPATH=.:code/peg_insertion.)
"""
import json
import sys

import numpy as np

from . import fixture
from .plan import quat


def centre(ctx, p):
    yaw = np.radians(p['yaw_deg'])
    return fixture.centre_in_hole(ctx, p['centre'], p['z'], yaw, quat(yaw))


def centre_tri(ctx, p):
    yaw = np.radians(p['yaw_deg'])
    return fixture.centre_triangle(ctx, p['centre'], p['z'], np.radians(p['hole_yaw_deg']), quat(yaw))


def hop(ctx, p):
    """Guarded hop search around xy: hover 1.5 mm above the surface contact z_surf, guarded down;
    a contact more than 3 mm below z_surf means the peg entered. Rings of 0.5 and 1.0 mm."""
    from .motion import move
    yaw = np.radians(p['yaw_deg']); q = quat(yaw)
    xy = np.array(p['xy']); zs = p['z_surf']
    offs = [(0.0, 0.0)] + [(r * np.cos(t), r * np.sin(t)) for r, n in ((0.0005, 8), (0.001, 12), (0.0015, 16))
                           for t in np.arange(n) * 2 * np.pi / n]
    for k, (dx, dy) in enumerate(offs):
        tgt = xy + [dx, dy]
        move(ctx, [tgt[0], tgt[1], zs + 0.0015], q, tol_mm=0.25, timeout_s=4)
        g = ctx.value('franka_guarded_move', direction=[0, 0, -1], max_distance_m=0.03, speed_mps=0.004,
                      force_threshold_n=4, retract_m=0)
        z = g['contact_position'][2] if g['contact'] else g['final_position'][2]
        print(f'  probe {k} off ({dx*1000:+.2f},{dy*1000:+.2f}) mm: z {z:.4f}', flush=True)
        if z < zs - 0.003:
            return {'ok': True, 'xy': tgt.tolist(), 'z': z, 'depth': zs - z, 'probes': k + 1}
    return {'ok': False, 'probes': len(offs)}


STEPS = {'centre': centre, 'centre_tri': centre_tri, 'hop': hop}

if __name__ == '__main__':
    from api.runtime import Context
    name, params = sys.argv[1], json.loads(sys.argv[2])
    with Context() as ctx:
        print(json.dumps(STEPS[name](ctx, params)))
