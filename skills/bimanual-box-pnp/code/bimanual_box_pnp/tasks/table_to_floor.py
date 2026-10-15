"""Task: take the box off a black-top table and set it down on the floor in front of the robot.

prepare -> find_table -> find_box (cp) -> plan_stance (cp) -> look_around -> drive_stance -> check_stance (cp) ->
grasp (cp) -> retract -> back_off -> set_down (cp) -> verify_floor
The approach ends with a straight leg of `straight` m: after the grasp the robot reverses only along it."""
from ..core import posture
from ..core.runner import Step
from ..motion import carry, navigation, squeeze
from ..perception import sensing

from ..params import STANCE_D, STRAIGHT, FACE_BLEND  # hyper-parameters: bimanual_box_pnp/params.py


def steps(box='box', table='table', stance_d=STANCE_D, straight=STRAIGHT, face_blend=FACE_BLEND):
    return [
        Step('prepare', posture.prepare, torso='zero'),
        Step('find_table', sensing.find_surface, name=table),
        Step('find_box', sensing.find_object, name=box, kind='box', support='black table top', on=table,
             looks=[[1.0, 0.0, 0.85]],
             checkpoint={'subagent': 'haiku', 'question': 'Is the cardboard box standing alone on the black table '
                         'top, fully visible, with nothing on it or leaning on it?', 'answer': 'yes / no + why',
                         'pass_if': 'yes', 'on_fail': 'stop and look'}),
        Step('plan_stance', navigation.plan_stance_straight, name=box, stance_d=stance_d, straight=straight,
             avoid=(table,), face=table, face_blend=face_blend,
             checkpoint={'subagent': 'haiku', 'question': 'The coloured outlines are the robot base footprint along '
                         'its planned path on the floor. Is that floor area free of people and objects (the black-top '
                         'cabinet ahead is the target and may be close to the last outline)?',
                         'answer': 'yes / no + what is in the way', 'pass_if': 'yes',
                         'on_fail': 'do not drive; replan'}),
        Step('look_around', navigation.look_around_torso),
        Step('drive_stance', navigation.execute_plan),
        Step('check_stance', navigation.check_surface_stance, name=box, stance_d=stance_d,
             checkpoint={'subagent': 'haiku', 'question': 'Is the cardboard box on the black table top straight '
                         'ahead of the robot, centred between the two robot arms?', 'answer': 'yes / no',
                         'pass_if': 'yes', 'on_fail': 'stop and look'}),
        Step('grasp', squeeze.grasp_from_surface, name=box, surface=table,
             checkpoint={'subagent': 'haiku', 'question': 'Is the cardboard box held between the two robot grippers '
                         '(one at each end, at the middle of each end face) and level?', 'answer': 'yes / no + why',
                         'pass_if': 'yes (lift-off itself is measured by depth, not asked)',
                         'on_fail': 'lower it back onto the table (place.lower_until_contact), release, stop'}),
        Step('retract', carry.recentre_and_retract),
        Step('back_off', navigation.retreat, dist=straight),
        Step('set_down', carry.set_down_on_floor,
             checkpoint={'subagent': 'haiku', 'question': 'Is the cardboard box standing upright on the floor in '
                         'front of the robot, with both grippers clear of it?', 'answer': 'yes / no',
                         'pass_if': 'yes', 'on_fail': 'stop and look'}),
        Step('verify_floor', sensing.verify_on_floor, name=box),
    ]
