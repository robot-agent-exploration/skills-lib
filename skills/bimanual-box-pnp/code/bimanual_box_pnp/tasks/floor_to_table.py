"""Task: pick the box off the floor and put it on the black-top table ahead (squeeze, lift, approach, place).

prepare -> find_box (cp) -> plan_approach -> drive_approach -> check_stance (loops) -> grasp (cp) -> lift (cp) ->
find_table -> approach_table (cp) -> place (cp) -> verify (cp) -> retreat
approach_table drives straight at the table ahead (start facing it, its near edge 0.5-1.6 m away)."""
from ..core import posture
from ..core.runner import Step
from ..motion import carry, navigation, place, squeeze
from ..perception import sensing

from ..params import CARRY_Z  # hyper-parameters: bimanual_box_pnp/params.py


def steps(box='box', table='table', carry_z=CARRY_Z):
    return [
        Step('prepare', posture.prepare),
        Step('find_box', sensing.find_object, name=box, kind='box', support='floor', looks=[[1.0, 0.0, 0.1]],
             checkpoint={'subagent': 'haiku', 'question': 'Is the closed cardboard box standing upright and alone on '
                         'the floor, fully visible, with clear floor between it and the robot?',
                         'answer': 'yes / no + why', 'pass_if': 'yes', 'on_fail': 'stop and look'}),
        Step('plan_approach', navigation.plan_approach, name=box),
        Step('drive_approach', navigation.execute_plan),
        Step('check_stance', navigation.check_stance, name=box, replan_step='plan_approach'),
        Step('grasp', squeeze.grasp_from_floor, name=box,
             checkpoint={'subagent': 'haiku', 'question': 'Are both robot grippers pressed against the two ends of the '
                         'cardboard box, at the middle of each end face?', 'answer': 'yes / no',
                         'pass_if': 'yes', 'on_fail': 'recover_from_squat, re-localize, retry once'}),
        Step('lift', carry.lift_from_floor, carry_z=carry_z,
             checkpoint={'subagent': 'haiku', 'question': 'Is the cardboard box held level between the two grippers, '
                         'clear of the floor?', 'answer': 'yes / no', 'pass_if': 'yes',
                         'on_fail': 'set it down (set_down_on_floor) and stop'}),
        Step('find_table', sensing.find_surface, name=table),
        Step('approach_table', navigation.final_table_approach, name=table,
             checkpoint={'subagent': 'haiku', 'question': 'Is the black table top directly in front of the held box, '
                         'its near edge just below / in front of the box, with nothing where the box will go?',
                         'answer': 'yes / no', 'pass_if': 'yes', 'on_fail': 'stop and look'}),
        Step('place', place.place_on_surface, name=box, surface=table,
             checkpoint={'subagent': 'haiku', 'question': 'Is the cardboard box standing on the black table top, '
                         'fully on it, with both grippers clear of it?', 'answer': 'yes / no', 'pass_if': 'yes',
                         'on_fail': 'stop and look'}),
        Step('verify', sensing.verify_on_surface, name=box, surface=table),
        Step('retreat', navigation.retreat, dist=0.3),
    ]
