"""Skill entry (box-table-to-floor-squeeze): take the closed box off a black-top table and set it down on the
floor in front of the robot, checkpoint by checkpoint (composition of the library primitives).

    main(ctx, start=None, resume=None, checkpoints=True, params=None, stop_after=None) -> {ok, measured, why,
        checkpoint: {step, next, brief}, next, outdir}
Resume after reviewing a checkpoint brief with main(ctx, resume=r['outdir'], start=r['next']).

Steps: prepare -> find_table -> find_box (cp) -> plan_stance (cp) -> drive_stance -> check_stance (cp) ->
grasp (cp) -> retract -> back_off -> set_down (cp) -> verify_floor (cp)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_RT = HERE.parents[1] / 'rby1-task-runtime' / 'code'
sys.path.insert(0, str(_RT))
from rby1rt import use       # noqa: E402

use('box-table-perception', 'perception-checked-base-motion', 'bimanual-squeeze-grasp', 'squeezed-box-transfer')
from basemove import navigation          # noqa: E402
from boxperc import sensing              # noqa: E402
from boxtransfer import carry            # noqa: E402
from rby1rt import posture               # noqa: E402
from rby1rt.runner import Step, run_task  # noqa: E402
from squeezegrasp import squeeze         # noqa: E402

STANCE_D = 0.60        # box centre ahead of the base origin at the table grasp (tested 0.60; clearance ~8 cm)
STRAIGHT = 0.35        # straight final leg = the reverse allowed after the grasp (tested 0.35)
FACE_BLEND = 0.5       # heading between the box axis (0) and the table-edge normal (1)


def steps(box='box', table='table', stance_d=STANCE_D, straight=STRAIGHT, face_blend=FACE_BLEND):
    return [
        Step('prepare', posture.prepare, torso='zero'),
        Step('find_table', sensing.find_surface, name=table),
        Step('find_box', sensing.find_object, name=box, kind='box', support='black table top', on=table,
             looks=[[1.0, 0.0, 0.85]],
             checkpoint={'subagent': 'haiku', 'question': 'Is the cardboard box standing alone on the black table '
                         'top, fully visible, with nothing on it or leaning on it?', 'answer': 'yes / no + why',
                         'pass_if': 'yes', 'on_fail': 'stop; ask the operator'}),
        Step('plan_stance', navigation.plan_stance_straight, name=box, stance_d=stance_d, straight=straight,
             avoid=(table,), face=table, face_blend=face_blend,
             checkpoint={'subagent': 'haiku', 'question': 'The green / orange outlines are the robot base footprint '
                         'along its planned path on the floor. Is that floor area free of people and objects '
                         '(the black-top cabinet ahead is the target and may be close to the last outline)?',
                         'answer': 'yes / no + what is in the way', 'pass_if': 'yes',
                         'on_fail': 'do not drive; replan or ask the operator'}),
        Step('drive_stance', navigation.execute_plan),
        Step('check_stance', navigation.check_surface_stance, name=box, stance_d=stance_d,
             checkpoint={'subagent': 'haiku', 'question': 'Is the cardboard box on the black table top straight '
                         'ahead of the robot, roughly centred between the two robot arms?',
                         'answer': 'yes / no', 'pass_if': 'yes', 'on_fail': 'stop and look'}),
        Step('grasp', squeeze.grasp_from_surface, name=box, surface=table,
             checkpoint={'subagent': 'haiku', 'question': 'Is the cardboard box held between the two robot grippers '
                         '(one at each end), lifted a few cm above the black table top and level?',
                         'answer': 'yes / no + why', 'pass_if': 'yes',
                         'on_fail': 'lower it back onto the table (lower_until_contact), release, stop'}),
        Step('retract', carry.recentre_and_retract),
        Step('back_off', navigation.retreat, dist=straight),
        Step('set_down', carry.set_down_on_floor,
             checkpoint={'subagent': 'haiku', 'question': 'Is the cardboard box standing upright on the floor in '
                         'front of the robot, with both grippers clear of it?', 'answer': 'yes / no',
                         'pass_if': 'yes', 'on_fail': 'stop and look'}),
        Step('verify_floor', sensing.verify_on_floor, name=box),
    ]


def main(ctx, start=None, resume=None, checkpoints=True, params=None, stop_after=None, **task_kwargs):
    return run_task(ctx, steps(**task_kwargs), task='box_table_to_floor', start=start, stop_after=stop_after,
                    resume=resume, checkpoints=checkpoints, params=params)


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('--start')
    ap.add_argument('--resume')
    ap.add_argument('--stop-after')
    ap.add_argument('--no-checkpoints', action='store_true')
    a = ap.parse_args()
    from rby1rt.runner import workspace_root
    sys.path.insert(0, str(workspace_root()))
    from api.runtime import Context
    with Context() as session:
        r = main(session, start=a.start, resume=a.resume, checkpoints=not a.no_checkpoints, stop_after=a.stop_after)
    print(json.dumps(r, indent=1, default=str)[:6000])
