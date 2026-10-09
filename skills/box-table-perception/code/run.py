"""Skill entry (box-table-perception): localize the box and / or the table top from the head camera, now.

    main(ctx, target='box', support='floor', describe=None, near=None) -> {ok, measured, why, checkpoint}
target 'box': Gemini finds the box and points at its top face, depth fits the rectangle (centre, size, yaw,
top height, fully_visible); 'table': Gemini finds the table, depth fits its top (centre, size, corners, height);
'both'. Nothing but the head moves. measured holds the base-frame results and the frame ids."""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from boxperc import locate as ploc   # noqa: E402
from boxperc import sensing          # noqa: E402
from rby1rt.runner import Runtime, StepFailed, workspace_root   # noqa: E402


def main(ctx, target='box', support='floor', describe=None, near=None, outdir=None):
    rt = Runtime(ctx, outdir or workspace_root() / 'output' / 'runs' / f'{time.strftime("%Y%m%d_%H%M%S")}_perceive')
    measured = {}
    try:
        if target in ('table', 'both'):
            measured['table'] = sensing.find_surface(rt, 'table', describe=describe if target == 'table' else None)
        if target in ('box', 'both'):
            sz = rt.world.objects.get('table', {}).get('top_z', 0.0) if support != 'floor' else 0.0
            d = sensing.locate(rt, 'box', near=near, support_z=sz - 0.01 if sz else 0.0,
                               support=support if support == 'floor' else 'black table top',
                               describe=describe if target == 'box' else None,
                               looks=None if near is not None else ([[1.0, 0.0, sz + 0.1]] if sz else None))
            if d is None:
                return {'ok': False, 'measured': measured, 'why': 'no box found', 'frames': rt.frames}
            measured['box'] = {k: d[k] for k in ('center', 'size', 'top_z', 'support_z', 'long_axis', 'short_axis',
                                                 'yaw_short', 'fully_visible', 'vlm_corner_err_m', 'frame') if k in d}
    except (StepFailed, RuntimeError) as e:
        return {'ok': False, 'measured': measured, 'why': str(e)[:600], 'frames': rt.frames}
    return {'ok': True, 'measured': measured, 'why': 'localized', 'frames': rt.frames,
            'checkpoint': {'subagent': 'haiku', 'frames': [f'runtime/frames/{f}/head_camera.jpg' for f in rt.frames[-2:]],
                           'question': 'Is the object the robot is about to act on the closed cardboard box (or the '
                                       'black-top cabinet), fully visible and alone?', 'answer': 'yes / no + why',
                           'pass_if': 'yes', 'on_fail': 'move the head / robot for a clear view and run again'}}


if __name__ == '__main__':
    import json
    sys.path.insert(0, str(workspace_root()))
    from api.runtime import Context
    with Context() as session:
        print(json.dumps(main(session, *(sys.argv[1:2] or ['box'])), indent=1, default=str))
