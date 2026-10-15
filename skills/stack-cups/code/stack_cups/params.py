"""Hyper-parameters of the stack-cups skill, in one list; the modules import their values from here.

The values are as the sessions that wrote the skill set them, with their comments. Values marked
CALIBRATE depend on the setup (cameras, table, layout, postures): in a setup the skill was not
tested in, measure them again as their comment says, or pass the measured value in.
"""

# Calibrate in a new setup ---------------------------------------------------------------

# CALIBRATE (banana_pick.py): the custom fingers' geometry (robot-interface, 'Custom fingers');
#   measure it again if this robot's fingers differ
TIP = 0.192

# CALIBRATE (pp_cups.py): torso postures; 'ready' is copied from the session: check it against the
#   ready posture in robot-interface
TORSO_POS = {'mid': [0.0, 0.7854, -1.5708, 0.7854, 0.0, 0.0], 'ready': [0.0, 0.5236, -1.0472, 0.5236, 0.0, 0.0]}

# CALIBRATE (pp_grasp.py): the custom fingers' geometry (robot-interface, 'Custom fingers');
#   measure it again if this robot's fingers differ
TIP_LOW = 0.192

# CALIBRATE (pp_grasp.py): the custom fingers' geometry (robot-interface, 'Custom fingers');
#   measure it again if this robot's fingers differ
GRIP_C = 0.175

# CALIBRATE (pp_grasp.py): the custom fingers' geometry (robot-interface, 'Custom fingers');
#   measure it again if this robot's fingers differ
OPEN_HALF = 0.05          # open fingers' inner faces at ±0.05 m along ee x (opening 0.10)

# CALIBRATE (pp_grasp.py): the backend's reach rule (0.69 m from link_<side>_arm_0 in the
#   session); check robot-interface
REACH_M = 0.69                # backend reach rule: an ee target within 0.69 m of link_<side>_arm_0

# nav.py --------------------------------------------------------------------------------

REAR_R = 0.62        # rear sweep radius about the axle

# pick_staged.py ------------------------------------------------------------------------

DESK_LOOK = (0.50, 0.10, 0.80)

# pp_cups.py ----------------------------------------------------------------------------

CUP_RIM_IN = 0.004        # cup wall inward of the rim 2.4 cm down (straight-ish aluminium cups)
NEST_RISE = 0.024         # each nested cup's rim stands this much above the one below (measured 0.024-0.026)
CARRY_CLEAR = 0.06        # held cup's bottom above the target rim while moving over it (0.018 knocked the target over)
CONTACT_N = 5.0           # wrist force change = the outer finger met the target rim -> release
ALIGN_TOL = 0.012         # held cup bottom centre within this of the target axis before inserting
ALIGN_STEPS = 3           # closed-loop xy corrections at most
CUP_H = 0.112             # cup height if the depth support estimate fails
LIFT_CUP = 0.06           # lift before the held check (a 4 cm lift was misjudged as "not risen")
HELD_CLEAR = 0.025        # held cup's lowest point at least this above the desk = lifted
LEFT_ON_DESK = 0.015      # lowest point within this of the desk = clearly not lifted -> release and back off

# pp_grasp.py ---------------------------------------------------------------------------

DESK_CLEAR = 0.004        # tips above the support when closing on a lying object (tested 0.003-0.004)
TOUCH_MARGIN = 0.012      # the force-guarded descent may go this far below the planned height (support error)
PRE_UP = 0.09             # pre-grasp above the grasp (top grasps)
BOX_PITCHES = (0, -30, -20, -45)  # fingers down when reachable (near boxes), else tilted forward (deg); tested -30
BOX_GRIP_DOWN = 0.035     # grip centre below the box top (tested: the tips pinched the top flaps)
RIM_DEPTH = 0.024         # tips' lowest point below the rim for a rim pinch (tested 0.024 on 4.4-5 cm bowls)
RIM_IN = 0.010            # wall at that depth, inward of the rim (tapered bowls)
RIM_ANGLES = (180, 157.5, -157.5, 135, -135, 112.5, -112.5, 90, -90)   # plan_rim_search: rim points tried, deg from +x
RIM_PITCHES = (0, -20, -35)   # fingers tilted forward about a sideways (y) closing axis, for far rims (tested -35)
RIM_TILT_X = 0.45             # plan_rim: tilt by pitches[-1] only when the fingers-down rim point is farther ahead (m)
YAW_TIE = 0.2                 # plan_rim: both yaws within this (rad) of equally far from the current yaw -> reach bearing

# pp_motion.py --------------------------------------------------------------------------

CLOSE_A = 0.5
TOUCH_N = 5.0
STOP_N = 8.0
LIFT_CHECK = 0.04
HELD_RISE = 0.02
CARRY = {'left': (0.40, 0.30, 1.12), 'right': (0.40, -0.30, 1.12)}
CLEAR = {'left': (0.34, 0.33, 1.15), 'right': (0.34, -0.33, 1.15)}     # pitched -45°: arms out of the camera view
TORSO = {'mid': [0.0, 0.7854, -1.5708, 0.7854, 0.0, 0.0], 'lowest': [0.0, 1.2, -2.4, 1.2, 0.0, 0.0]}
BUCKET_DIST = 0.62        # bucket centre ahead of the base origin at the drop stance (bumper 0.304, near rim 0.47)
DESK_EDGE = 0.215         # desk front edge ahead of the base origin for reaching (bumper under the top; torso clear)

# pp_perception.py ----------------------------------------------------------------------

ELONG_STRAIGHT_DEG = 25      # grasp where the ridge turns less than this within ±3.5 cm
BOX_MAX_SHORT = 0.085        # an upright box thicker than this does not fit the 0.10 m opening with margin

# pp_vis.py -----------------------------------------------------------------------------

COLOURS = {  # OpenCV HSV ranges (h 0-180); fallback only, Gemini masks are the default
    'yellow': [((20, 120, 120), (38, 255, 255))],
    'red': [((0, 120, 80), (8, 255, 255)), ((165, 120, 80), (180, 255, 255))],
    'orange': [((8, 140, 120), (20, 255, 255))],
}
