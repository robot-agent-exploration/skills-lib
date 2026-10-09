"""Offline checks (no robot motion): imports, depth detectors and the stance plan on frames saved in session
practice_pnp_box_61befa; with --gemini also the Gemini locators (a few cents). Frames missing in another workspace
are skipped.   MPLCONFIGDIR=$TMPDIR python3 skill-library/skills/bimanual-box-pnp/code/offline_test.py [--gemini]"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from boxpnp.core.frames import frame_dir          # noqa: E402
from boxpnp.core.runner import workspace_root     # noqa: E402
from boxpnp.core.world import World               # noqa: E402
from boxpnp.motion import carry, navigation, place, squeeze   # noqa: E402,F401
from boxpnp.perception import box as pbox, locate, sensing, table as ptable   # noqa: E402,F401
from boxpnp.tasks import floor_to_table, table_to_floor       # noqa: E402

P1 = '20261008231539327b6a'             # box on the island, robot 0.55 m from it (odometry below)
P1_ODOM = {'x': 1.4046, 'y': -1.357, 'yaw': -1.9436}
EDGE = '20261009000618718682'           # held box over the island, near edge 0.39 m ahead (left view)


def have(f):
    return (frame_dir(f) / 'meta.json').exists()


class FakeRT:
    def __init__(self, f, odom):
        self.world, self.frame_odom, self._o = World(), {f: odom}, odom
        self.out = workspace_root() / 'output' / 'offline_test'
        self.out.mkdir(parents=True, exist_ok=True)

    def pose(self):
        return (self._o['x'], self._o['y'], self._o['yaw'])

    def odom(self):
        return self._o

    def held_box_region(self):
        return None

    def log(self, *a):
        print('   ', *a)


def test_compositions():
    assert [s.name for s in table_to_floor.steps()][-1] == 'verify_floor'
    assert [s.name for s in floor_to_table.steps()][-1] == 'retreat'


def test_stance_plan():
    if not have(P1):
        return
    t = ptable.find_table(P1)
    t['support_z'] = t['top_z']
    b = pbox.find_box(P1, support_z=t['top_z'] - 0.01)
    assert t['ok'] and b['ok'] and 0.36 < b['size'][0] < 0.42, (t.get('why'), b.get('size'))
    rt = FakeRT(P1, P1_ODOM)
    rt.world.put('table', 'table', t, P1_ODOM)
    rt.world.put('box', 'box', b, P1_ODOM)
    obj = rt.world.in_base('box', P1_ODOM)
    h = obj['yaw_short']                                    # face_blend 0
    hv = np.array([np.cos(h), np.sin(h)])
    target = np.array(obj['center']) - hv * table_to_floor.STANCE_D
    moves = navigation.plan_turn_drive_turn(target - hv * table_to_floor.STRAIGHT, h) + \
        [{'forward': table_to_floor.STRAIGHT}]
    clear = navigation.known_clearance(rt, moves, ['table'])
    print('stance plan', [{k: round(v, 3) for k, v in m.items()} for m in moves], 'clearance', clear)
    assert clear['table'] >= 0.03
    rel = 0.0
    box_r = {'center': [table_to_floor.STANCE_D, 0.0], 'long_axis': [-np.sin(rel), np.cos(rel)],
             'short_axis': [np.cos(rel), np.sin(rel)], 'size': b['size'], 'top_z': b['top_z']}
    from boxpnp.core import robot
    for z in (b['top_z'] + 0.11, t['top_z'] + 0.20, t['top_z'] + 0.18):
        p = squeeze.plan_balls(box_r, robot.TORSO_ZERO, z, squeeze.Q_CARRY, squeeze.SEED_TABLE)
        assert p['ok'], p


def test_gemini():
    sys.path.insert(0, str(workspace_root()))
    from api.runtime import Context
    with Context() as ctx:
        if have(P1):
            d = locate.locate_box(ctx, P1, support='black table top', support_z=0.685)
            print('gemini box', d.get('center'), d.get('vlm_corner_err_m'), d.get('why'))
            assert d['ok'] and np.hypot(d['center'][0] - 0.90, d['center'][1] + 0.26) < 0.04
        if have(EDGE):
            e = locate.table_near_edge(ctx, EDGE)
            print('gemini mask edge', e.get('edge_x'), e.get('bands'), e.get('why'))
            assert e['ok'] and 0.36 < e['edge_x'] < 0.42


if __name__ == '__main__':
    test_compositions()
    test_stance_plan()
    if '--gemini' in sys.argv:
        test_gemini()
    print('offline tests passed')
