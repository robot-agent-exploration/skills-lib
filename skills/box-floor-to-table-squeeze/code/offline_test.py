"""Offline checks of perception and motion checking on frames saved in session pnp_box_f75f61 (no robot calls).

    MPLCONFIGDIR=$TMPDIR python3 code/rby1_tasks/tests/offline_test.py
Frames missing in another workspace are skipped."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parents[1]):          # works from code/rby1_tasks/tests and the skill's code/
    if (p / 'rby1_tasks').is_dir():
        sys.path.insert(0, str(p))
WORKSPACE = next(p for p in HERE.parents if (p / 'api' / 'runtime.py').exists())

from rby1_tasks.core.world import World                 # noqa: E402
from rby1_tasks.perception import box as pbox           # noqa: E402
from rby1_tasks.perception import frames as pframes     # noqa: E402
from rby1_tasks.primitives import navigation, squeeze   # noqa: E402

OUT = WORKSPACE / 'output' / 'offline_test'
OUT.mkdir(parents=True, exist_ok=True)


def have(frame):
    return (pframes.frame_dir(frame) / 'meta.json').exists()


class FakeRT:
    def __init__(self, frame, odom):
        self.world, self.out, self.frame_odom, self._o = World(), OUT, {frame: odom}, odom

    def pose(self):
        return (self._o['x'], self._o['y'], self._o['yaw'])

    def odom(self):
        return self._o

    def held_box_region(self):
        return None

    def log(self, *a):
        print('   ', *a)


def test_box():
    for f in ['20261008204144e43330', '2026100820421207c84e', '202610082043390cb24c', '20261008211704e76157',
              '20261008221401c29c92', '20261008220958a2c917']:
        if have(f):
            b = pbox.find_box(f)
            assert b['ok'] and 0.36 < b['size'][0] < 0.42, (f, b.get('size'))
            near = pbox.find_box(f, near=b['center'], radius=0.6)
            assert near['ok'], f
            print('box', f, np.round(b['center'], 3), np.round(b['size'], 3))
    for f in ['2026100820542769dfda', '20261008210421a7d43b']:         # held / on the table: not on the floor
        if have(f):
            assert not pbox.find_box(f)['ok'], f


def test_path_check():
    f = '202610082206436164e6'                  # facing the island, ~1.6 m away
    if not have(f):
        return
    rt = FakeRT(f, {'x': 2.6183, 'y': -1.5959, 'yaw': -2.5664})
    short = navigation.check_path(rt, [{'forward': 0.3}], tag='test_short', fids=[f])
    long_ = navigation.check_path(rt, [{'forward': 1.6}], tag='test_long', fids=[f])
    print('path check: 0.3 m', short['obstacle_points'], '1.6 m', long_['obstacle_points'], long_['picture'])
    assert short['ok'] and not long_['ok']


def test_ik():
    if have('202610082043390cb24c'):
        b = pbox.find_box('202610082043390cb24c')
        p = squeeze.plan_balls(b, squeeze.TORSO_LOW, squeeze.BALL_Z_PRE, squeeze.Q_EE, squeeze.SEED)
        assert p['ok'], p


if __name__ == '__main__':
    test_box()
    test_path_check()
    test_ik()
    print('offline tests passed')
