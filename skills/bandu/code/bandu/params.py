"""Hyper-parameters of the bandu skill, in one list; the modules import their values from here.

The values are as the sessions that wrote the skill set them, with their comments. Values marked
CALIBRATE depend on the setup (cameras, table, layout, postures): in a setup the skill was not
tested in, measure them again as their comment says, or pass the measured value in.
"""
import math
import numpy as np

# Calibrate in a new setup ---------------------------------------------------------------

# CALIBRATE (clearance.py): how far from the robot's base axis the table is checked for collisions
#   (m); depends on how the table meets the robot's base column
TABLE_FROM_BASE = 0.25    # the table is checked only this far from the base axis (the base column stands lower)

# CALIBRATE (controller.py): fallback free table spots (base frame, m); pick free table in this
#   layout
ALT_SPOTS = [(0.42, -0.30), (0.62, -0.30), (0.52, 0.30), (0.40, 0.30), (0.66, 0.30), (0.52, -0.24)]

# CALIBRATE (exterior.py): the table surface height in the base frame (m); measure the median
#   height of bare-table pixels with franka_pixel_to_position (depth and contact agreed within ~2 mm
#   in the session)
# (motion.py) table top (depth and contact agree within ~2 mm)
TABLE_Z = 0.093

# CALIBRATE (exterior.py): the base-frame x and y ranges (m) of the table searched for pieces;
#   cover the table in front of the robot in this layout
REGION = dict(x=(0.30, 0.80), y=(-0.40, 0.40))     # table area searched for pieces

# CALIBRATE (motion.py): just above the floor of the harness's workspace box (0.10 m in the
#   session); read it from franka_status
TCP_Z_MIN = 0.1005       # workspace box floor is 0.10

# CALIBRATE (motion.py): an absolute travel height (m) for a table at TABLE_Z; shift it with the
#   table height
TRANSIT_Z = 0.27         # safe travel height (tallest stack top ~0.24)

# CALIBRATE (motion.py): below the harness's largest move per franka_move_pose call (0.4 m in the
#   session); check robot-interface
MAX_STEP = 0.35          # m per move_pose call (tool limit 0.4)

# CALIBRATE (perceive.py): the start posture; it must match the harness's home preset
#   (franka_move_joints preset 'home')
HOME = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]

# CALIBRATE (perceive.py): below the harness's largest joint change per franka_move_joints call
#   (1.5 rad in the session); check robot-interface
MAX_JOINT_STEP = 1.4            # rad per franka_move_joints call (tool limit 1.5)

# CALIBRATE (perception.py): the wrist camera's position in the TCP frame (m); read it from a
#   frame's meta.json (T_base_cam of the wrist camera against the TCP pose)
# Wrist camera in the TCP frame (fixed; from meta.json): t = (-0.0725, 0.0305, -0.0804).
CAM_IN_TCP = np.array([-0.0725, 0.0305, -0.0804])

# CALIBRATE (pieces.py): an absolute travel height (m) for a table at TABLE_Z; shift it with the
#   table height
PIECES_TRANSIT_Z = 0.30          # empty-hand travel height: fingertips 4.6 cm over the full tower (0.2415)

# CALIBRATE (pieces.py): an absolute height (m) for a table at TABLE_Z; shift it with the table
#   height
CARRY_BOTTOM_Z = 0.28     # a carried piece's bottom stays this high (level moves sag up to 2.3 cm)

# CALIBRATE (tower.py): the base block's centre (base frame, m) and top height in the session's
#   layout; in another layout measure them and pass base_center and base_top_z to main
BASE = dict(center=(0.5245, 0.032), axis=(0.0, 1.0), top_z=0.110, half_len=0.079, half_wid=0.028)

# CALIBRATE (tower.py): free table spots (base frame, m) where pieces are put when the tower comes
#   apart; pick free table in this layout
# Where the disassembly puts each piece (picked with grasps at yaw 0 and 90, so the hand's
# cross-shaped footprint around each spot is kept clear of taller pieces; see check_plan()).
SPOTS = {
    'knob':   (0.42, 0.20),
    'dowel':  (0.39, -0.03),
    'urn':    (0.64, 0.22),
    'goblet': (0.62, -0.15),
    'barrel': (0.45, -0.17),
    'ring':   (0.66, 0.06),
    'slant':  (0.53, -0.25),     # bottom centre; it stands leaning towards +y
}

# align.py ------------------------------------------------------------------------------

FOOT_GAP = 0.020                 # foot bottom above the rim while measuring
MAX_OFFSET = 0.0008              # m, aimed foot-vs-lip offset (user: at most 1 mm)
ACCEPT_OFFSET = 0.0010           # m, still accepted when the corrections run out
MAX_TILT = math.radians(2.5)     # tilt corrected above this (the fit jitters ~1 deg; the throat takes ~3)
ACCEPT_TILT = math.radians(3.0)
MAX_ITER = 6
FOOT_ROWS = (0.003, 0.004, 0.005, 0.006, 0.007)   # heights above the held piece's bottom for its centre
SEATED_TOL = 0.006
UPRIGHT_ROWS = {'urn': np.arange(0.016, 0.037, 0.002)}     # above the goblet's rim, below the lip
MAX_SEATED_TILT = 12.0     # deg: a seated urn reads 7-8 (shape/perspective bias; the goblet on the table 2), caught on the rim 22-24

# clearance.py --------------------------------------------------------------------------

LINK_PAD = 0.01           # added to every link capsule radius (models are approximate)

# controller.py -------------------------------------------------------------------------

GRASP_YAW = math.pi / 2      # final grasp closes along base x (the first, centring grasp along y)
PRIOR_MATCH_R = 0.03         # a recorded table position is confirmed by a cluster this close (m)
PICK_ATTEMPTS = 2            # tested: 1 retry after re-localizing
TABLE_PLACE_TOL = 0.010      # placing on the table: contact this close to the table model (m)
SPOT_CLEAR_R = 0.07          # a table spot is free if no cluster centre is this close (m)
MIN_GRIPPER_SPAN = 0.075     # homed max_width (0.080 when right; 0.055 after homing on the knob's ball)
BALL_PICK_YAWS = [math.pi / 2, math.pi / 3, -math.pi / 3, math.pi / 6, -math.pi / 6, 0.0]   # first conflict-free

# holes.py ------------------------------------------------------------------------------

TILT = math.radians(22)        # wrist camera axis leans this much towards TCP +x
VIEW_HEIGHT = 0.15             # camera height above the rim (m)
MIN_SCORE = 12.0               # grey levels rim-minus-opening (seen 30-70 on the urn)
VIEW_YAWS = (math.pi / 2, 0.0, -math.pi / 2)   # visit order; images at the first and last (via 0: a 180 deg turn in one move hits J7)

# motion.py -----------------------------------------------------------------------------

FINGERTIP_BELOW_TCP = 0.0124
APPROACH_DZ = 0.04       # move_pose stops this far above a low target; guarded move does the rest
GUARD_SPEED = 0.01       # m/s for the final descent
GUARD_FORCE = 4.0        # N above baseline = contact
MAX_DESCENT = 0.12       # m per move_pose descent before a guarded approach (overshoot <= ~2.5 cm)

# perceive.py ---------------------------------------------------------------------------

CLASS_SLACK = 0.006             # a Gemini label is accepted if the height is this close to its band
MAX_EXTENT = 0.070              # a standing piece's cluster is smaller than this in x and y
PRESENT_MIN_POINTS = 60

# perception.py -------------------------------------------------------------------------

# Wood: hue 5-30, saturation > 50 (table: saturation < 25).
WOOD_H = (5, 30)
WOOD_S_MIN = 50
WOOD_V_MIN = 40

# pieces.py -----------------------------------------------------------------------------

WIDTH_TOL = 0.006
GRIP_N = 20.0             # grasp force; pieces that get inserted use their 'grip' (40 N: a ball-held knob
                          # turned in a 20 N grip at rim contacts). Never re-grasp while holding: a second
                          # grasp on a held urn reported is_grasped False, stopped squeezing and the urn fell.
PALM_ABOVE_FINGERTIPS = 0.047   # hand body underside (TCP + 0.035) above the fingertip ends
PROBE_SPEED = 0.005      # m/s, guarded probe down
PROBE_FORCE = 1.5        # N
SEATED_TOL_DEFAULT = 0.006   # contact this close to the seat counts as inserted (PIECES seat_tol overrides)
MAX_PROBES = 30

# tower.py ------------------------------------------------------------------------------

MAX_OPENING_SPREAD = 0.006    # the two views must agree this well, else the support's xy is used
