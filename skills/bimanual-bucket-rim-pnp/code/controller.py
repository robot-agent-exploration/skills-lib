"""Controller: pick the orange bucket off the floor with BOTH grippers (rim pinch, open fingers) and place it on the
black-top cabinet next to the cardboard box. Steps run through the library step runner (bimanual-box-pnp
core/runner.py): each checkpoint stops the run and returns a perception brief for the agent / a subagent / Gemini;
resume with --resume <outdir> --start <next>.

    python3 code/basket_pnp/controller.py                       # to the first checkpoint
    python3 code/basket_pnp/controller.py --resume output/runs/<run> --start <next> [--params '{"heading_offset": 1.57}']
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib import WORKSPACE  # noqa: E402
from boxpnp.core.runner import Step, run_task  # noqa: E402
import steps as S  # noqa: E402


def cp(question, answer, pass_if, on_fail, subagent='haiku'):
    return {'subagent': subagent, 'question': question, 'answer': answer, 'pass_if': pass_if, 'on_fail': on_fail}


def task_steps():
    return [
        Step('prepare', S.prepare),
        Step('find_bucket', S.find_bucket, checkpoint=cp(
            'Does the green ellipse lie on the rim of the ORANGE bucket that stands on the floor beside the white '
            'drawer cabinet (not a red bucket far away)? Is the bucket upright?', 'yes / no + why', 'yes',
            'stop; point the head at the right bucket and rerun find_bucket')),
        Step('find_cabinet', S.find_cabinet, checkpoint=cp(
            'Is there a white drawer cabinet with a black top and a cardboard box on that top in the image?',
            'yes / no', 'yes', 'look around (other looks) and rerun find_cabinet')),
        Step('plan_grasp_approach', S.plan_grasp_approach, checkpoint=cp(
            'In the plan picture (robot frame, robot at the black triangle), does the swept area avoid the red '
            'outlines except the bucket at the very end? In the camera frame, is the floor along the way free of '
            'people and objects?', 'yes / no + what is in the way', 'yes', 'replan with another heading_offset')),
        Step('drive_grasp_approach', S.drive_plan),
        Step('final_approach', S.final_approach, checkpoint=cp(
            'Look at the bucket in front of the robot. Where are the two places where the thin metal wire handle is '
            'attached to the bucket (just below the rim): at the FRONT/BACK (towards / away from the camera) or at '
            'the LEFT/RIGHT sides of the bucket in the image?', 'front_back | left_right', 'front_back',
            'retreat 0.5 m and rerun plan_grasp_approach with heading_offset +-1.57 (go 90 deg around the bucket)',
            subagent='agent (zoom on the bucket)')),
        Step('pregrasp', S.pregrasp, checkpoint=cp(
            'Are both open grippers above the bucket rim, one at its left side and one at its right side, with one '
            'black finger inside the rim and one outside for each gripper?', 'yes / no + which gripper is off',
            'yes', 'raise the hands, re-measure the rim, rerun pregrasp')),
        Step('grasp_lift', S.grasp_lift, checkpoint=cp(
            'Is the orange bucket hanging level between the two grippers, clear of the floor?', 'yes / no',
            'yes and held_dFz >= 6 N', 'if it slipped: set it down (lower_to_touch on the floor) and redo')),
        Step('plan_place', S.plan_place, checkpoint=cp(
            'In the plan picture, does the route reach the cabinet edge straight on without the swept base crossing '
            'the cabinet outline, with the target far from the box?', 'yes / no', 'yes',
            'choose another edge (params edge_choice) or move the robot')),
        Step('drive_place', S.drive_plan),
        Step('approach_edge', S.approach_edge, checkpoint=cp(
            'Is the held bucket above the black cabinet top (its bottom over the top, not over the floor), clear of '
            'the cardboard box?', 'yes / no', 'yes', 'adjust: drive a few cm or rerun plan_place')),
        Step('lower_to_touch', S.lower_to_touch, checkpoint=cp(
            'Is the bucket standing on the black cabinet top with its whole bottom on the top (not overhanging the '
            'edge), upright?', 'yes / no', 'yes', 'lift 3 cm, adjust, lower again')),
        Step('release', S.release),
        Step('verify', S.verify, checkpoint=cp(
            'Is the orange bucket standing upright on the black top of the white cabinet next to the cardboard box, '
            'with both grippers away from it?', 'yes / no', 'yes', 'report')),
    ]


def main(ctx, start=None, resume=None, checkpoints=True, params=None, stop_after=None):
    return run_task(ctx, task_steps(), task='basket_pnp', start=start, resume=resume, checkpoints=checkpoints,
                    params=params, stop_after=stop_after)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--start')
    ap.add_argument('--resume')
    ap.add_argument('--stop-after')
    ap.add_argument('--params')
    ap.add_argument('--no-checkpoints', action='store_true')
    a = ap.parse_args()
    from api.runtime import Context
    with Context() as session:
        r = main(session, start=a.start, resume=a.resume, checkpoints=not a.no_checkpoints,
                 params=json.loads(a.params) if a.params else None, stop_after=a.stop_after)
    print(json.dumps(r, indent=1, default=str)[:5000])
