"""Reverse task / scene reset: take the box off the table top and set it down on the floor in front of the robot.

Precondition: the robot faces the table with the box on it ~0.55 m ahead (e.g. right after pnp_box place).
    prepare -> find_table -> find_box_on_table -> grasp_on_table -> recentre -> back_off -> set_down -> retreat
"""
from ..core.runner import Step
from ..primitives import carry, navigation, posture, sensing, squeeze
from .pnp_box import TABLE_PRIOR_ODOM, TABLE_SIZE

TABLE_TOP_Z = 0.68


def steps(box='box', table='table', table_prior=TABLE_PRIOR_ODOM, back_off=0.5):
    return [
        Step('prepare', posture.prepare),
        Step('find_table', sensing.find_surface, name=table, prior_odom=table_prior, size=TABLE_SIZE),
        Step('find_box_on_table', sensing.find_object, name=box, kind='box', support_z=TABLE_TOP_Z - 0.01,
             max_turns=0, looks=[[0.6, 0.0, 0.9], [0.8, 0.2, 0.9], [0.8, -0.2, 0.9]]),
        Step('grasp_on_table', squeeze.grasp_from_surface, name=box, surface=table),
        Step('recentre', carry.recentre_and_retract),
        Step('back_off', navigation.retreat, dist=back_off),
        Step('set_down', carry.set_down_on_floor),
        Step('retreat', navigation.retreat, dist=0.3),
    ]
