"""Task pnp_box: pick the cardboard box up from the floor and put it on the black-top table (island).

Composition of primitives; each step checks its own preconditions and result. With checkpoints on, the run
pauses after the steps marked (cp) so the agent can look at the pictures and numbers before the next motion:
  prepare -> find_box (cp) -> plan_approach_box (cp) -> execute_approach -> check_stance
  (loops to plan_approach_box) -> grasp (cp) -> lift (cp) -> find_table (cp) -> plan_table (cp) ->
  execute_table -> final_table (cp) -> place (cp) -> verify (cp) -> retreat
"""
from ..core.runner import Step
from ..primitives import carry, navigation, place, posture, sensing, squeeze

TABLE_PRIOR_ODOM = [1.30, -2.40]     # island centre in this session's odometry (fits 1.25-1.36, -2.32..-2.44)
TABLE_SIZE = (1.0, 0.62)
CARRY_Z = 1.17                       # ee z while carrying: box bottom ~0.81 m, table top 0.67-0.74


def steps(box='box', table='table', table_prior=TABLE_PRIOR_ODOM, carry_z=CARRY_Z):
    return [
        Step('prepare', posture.prepare),
        Step('find_box', sensing.find_object, checkpoint=True, name=box, kind='box'),
        Step('plan_approach_box', navigation.plan_approach, checkpoint=True, name=box),
        Step('execute_approach', navigation.execute_plan),
        Step('check_stance', navigation.check_stance, name=box, replan_step='plan_approach_box'),
        Step('grasp', squeeze.grasp_from_floor, checkpoint=True, name=box),
        Step('lift', carry.lift_from_floor, checkpoint=True, carry_z=carry_z),
        Step('find_table', sensing.find_surface, checkpoint=True, name=table, prior_odom=table_prior, size=TABLE_SIZE),
        Step('plan_table', navigation.plan_table_approach, checkpoint=True, name=table),
        Step('execute_table', navigation.execute_plan),
        Step('final_table', navigation.final_table_approach, checkpoint=True, name=table),
        Step('place', place.place_on_surface, checkpoint=True, name=box, surface=table),
        Step('verify', sensing.verify_on_surface, checkpoint=True, name=box, surface=table),
        Step('retreat', navigation.retreat, dist=0.4),
    ]
