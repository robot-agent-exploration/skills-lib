"""Put a held cube back down where the gripper is and return to the watch pose.

    python <skill>/code/reset.py        (from the workspace root)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))                       # workspace root (api/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_the_cube import controller as C  # noqa: E402


def place_and_watch(ctx):
    st = ctx.value('franka_status')
    x, y, z = st['tcp_position']
    q = st['tcp_quaternion']
    held = st['gripper']['width']
    if 0.012 < held < 0.06:                       # a cube is in the fingers: put it down here
        C.move(ctx, [x, y, max(z, 0.16)], q)
        C.move(ctx, [x, y, C.TABLE_Z + 0.023], q, tol=6, accept_mm=10)
        ctx.value('franka_gripper', action='open')
        C.wait_gripper(ctx)
        C.move(ctx, [x, y, 0.28], q)
    else:
        ctx.value('franka_gripper', action='open')
    C.goto_watch(ctx)
    return {'ok': True, 'placed_at': [x, y] if 0.012 < held < 0.06 else None, 'tcp': C.tcp(ctx)}


if __name__ == '__main__':
    from api.runtime import Context
    with Context() as session:
        print(json.dumps(place_and_watch(session)))
