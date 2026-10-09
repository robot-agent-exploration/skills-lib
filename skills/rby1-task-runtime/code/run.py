"""Skill entry (rby1-task-runtime): prepare the robot for a task skill.

    main(ctx, grippers_closed=True) -> {ok, measured, why}
Checks armed / no EMO / no fault, homes and closes both grippers (closed round fingers = the 'ball' tool of the
squeeze skills) or leaves them, brings torso and arms to `ready`, reads the wrist-force baseline."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rby1rt import posture                              # noqa: E402
from rby1rt.runner import Runtime, StepFailed, workspace_root   # noqa: E402


def main(ctx, grippers_closed=True, outdir=None):
    import time
    rt = Runtime(ctx, outdir or workspace_root() / 'output' / 'runs' / f'{time.strftime("%Y%m%d_%H%M%S")}_prepare')
    try:
        m = posture.prepare(rt, grippers_closed=grippers_closed)
    except (StepFailed, RuntimeError) as e:
        return {'ok': False, 'measured': {}, 'why': str(e)[:800]}
    return {'ok': True, 'measured': m, 'why': 'armed, grippers ready, torso and arms at ready'}


if __name__ == '__main__':
    import json
    sys.path.insert(0, str(workspace_root()))
    from api.runtime import Context
    with Context() as session:
        print(json.dumps(main(session), indent=1, default=str))
