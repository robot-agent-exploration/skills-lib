"""Hyper-parameters of the bimanual-bucket-rim-pnp skill, in one list; the modules import their values from here.

The values are as the sessions that wrote the skill set them, with their comments. Values marked
CALIBRATE depend on the setup (cameras, table, layout, postures): in a setup the skill was not
tested in, measure them again as their comment says, or pass the measured value in.
"""

# Calibrate in a new setup ---------------------------------------------------------------

# CALIBRATE (grasp_plan.py): the custom fingers' geometry (robot-interface, 'Custom fingers');
#   measure it again if this robot's fingers differ
PINCH_D = 0.20          # m along -z_ee from ee to the middle of the open round fingertips (closed ball 0.195)

# CALIBRATE (steps.py): the ready posture copied from the session; check it against the ready
#   posture in robot-interface
TORSO_READY = [0, 0.524, -1.047, 0.524, 0, 0]

# CALIBRATE (steps.py): the ready posture copied from the session; check it against the ready
#   posture in robot-interface
ARM_READY = {'right': [-0.7854, -0.5236, 0, -1.5708, 0, 0.7854, 0], 'left': [-0.7854, 0.5236, 0, -1.5708, 0, 0.7854, 0]}

# CALIBRATE (steps.py): below the backend's joint torque stop (150 % in the session); check robot-
#   interface
LOAD_STOP = 140.0

# grasp_plan.py -------------------------------------------------------------------------

PINCH_DEPTH = 0.03      # m below the rim top
WALL_R = 0.146          # m: wall radius at the rim (outer lip ~0.152, inner ~0.140)

# perception.py -------------------------------------------------------------------------

ORANGE_HSV = ((3, 120, 70), (22, 255, 255))     # bucket hue 5-7 measured; the red distractor bucket is ~179
RIM_BAND = 0.035          # m below the rim top used for the circle fit
R_RANGE = (0.11, 0.19)    # plausible rim radius (5-gallon bucket: outer rim ~0.155)
R_PRIOR = 0.152

# steps.py ------------------------------------------------------------------------------

TORSO_MID = [0, 0.785, -1.571, 0.785, 0, 0]
TORSO_ZERO = [0, 0, 0, 0, 0, 0]
TORSO_GRASP = [0, 1.047, -2.094, 1.247, 0, 0]          # low + lean 0.2 (lean steps 0.1)
TORSO_LIFT1 = [0, 0.85, -1.7, 0.95, 0, 0]
STANCE_D = 0.595         # bucket centre ahead of the base origin at the grasp: pinch IK ok for 0.58-0.60
                         # (y -0.04..+0.045, rim 0.37), fails at 0.62; knee clearance >= 3 cm needs >= 0.58
TILT = 0.6               # finger axis tilted forward (rad)
STEPS_PINCH_DEPTH = 0.05       # pinch point below the rim top (fingers reach ~0.2 m along -z_ee)
PRE_DZ = 0.10            # pre-grasp height above the grasp pose
CLOSE_A = 0.35           # gripper current (right first, then left)
PINCH_MAX_W = 0.015      # a pinch of the thin wall reads 3-11 mm
CARRY_STEPS = ([-0.03, 0, 0.10], [0, 0, 0.15], [0, 0, 0.04])   # hand moves at torso zero after the lift
HELD_DF_MIN = 6.0        # summed wrist force change after the lift (bucket ~1.5 kg)
EDGE_IN = 0.21           # bucket centre inside the cabinet edge when placing
TOUCH_DF = 4.0
BUCKET_R = 0.16
