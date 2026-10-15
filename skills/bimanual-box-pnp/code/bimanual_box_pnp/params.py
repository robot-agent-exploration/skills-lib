"""Hyper-parameters of the bimanual-box-pnp skill, in one list; the modules import their values from here.

The values are as the sessions that wrote the skill set them, with their comments. Values marked
CALIBRATE depend on the setup (cameras, table, layout, postures): in a setup the skill was not
tested in, measure them again as their comment says, or pass the measured value in.
"""

# Calibrate in a new setup ---------------------------------------------------------------

# CALIBRATE (bimanual_box_pnp/core/kin.py): the custom fingers' geometry (robot-interface, 'Custom
#   fingers'); measure it again if this robot's fingers differ
BALL_D = 0.195          # closed round fingers: ball centre along -z_ee (library skill, checked in RGB)

# CALIBRATE (bimanual_box_pnp/core/robot.py): the ready posture copied from the session; check it against
#   the ready posture in robot-interface
TORSO_READY = [0, 0.5236, -1.0472, 0.5236, 0, 0]

# CALIBRATE (bimanual_box_pnp/core/robot.py): the ready posture copied from the session; check it against
#   the ready posture in robot-interface
ARM_READY = {'right': [-0.7854, -0.5236, 0, -1.5708, 0, 0.7854, 0], 'left': [-0.7854, 0.5236, 0, -1.5708, 0, 0.7854, 0]}

# CALIBRATE (bimanual_box_pnp/core/robot.py): the longest base move per call used in the session (m); check
#   the base limits in robot-interface
MAX_FORWARD_CALL = 0.8

# CALIBRATE (bimanual_box_pnp/core/robot.py): below the backend's torso change per call (1.2 rad in the
#   session); check robot-interface
TORSO_MAX_STEP = 1.15   # rad per call (backend cap 1.2)

# CALIBRATE (bimanual_box_pnp/motion/carry.py): below the backend's joint torque stop (150 % in the
#   session); check robot-interface
MAX_LOAD = 1.30             # stop before the backend's 150 % torque stop

# CALIBRATE (bimanual_box_pnp/tasks/floor_to_table.py): an absolute hand height (m) that cleared the table
#   tops of the session (0.67-0.74 m); check it against the table heights in this layout
CARRY_Z = 1.17          # ee z while carrying: box bottom ~0.81 m, table tops 0.67-0.74 (tested)

# bimanual_box_pnp/core/footprint.py ----------------------------------------------------

RES = 0.04
FLOOR_Z = 0.06
OBST_Z = (0.15, 1.7)
MIN_PTS = 3                 # points per cell (stride-2 sampling) to call it occupied / free

# bimanual_box_pnp/core/kin.py ----------------------------------------------------------

LIMIT_MARGIN = 0.2      # rad inside the URDF limits (the backend uses 0.1; real stops can be inside)

# bimanual_box_pnp/core/robot.py --------------------------------------------------------

TORSO_ZERO = [0, 0, 0, 0, 0, 0]
TORSO_READY_LEAN = [0, 0.5236, -1.0472, 0.5936, 0, 0]     # ready + 0.07 chest lean (tested)

# bimanual_box_pnp/motion/carry.py ------------------------------------------------------

SQUEEZE_LOW_N = 8.0         # below: grip at risk -> stop
SQUEEZE_TIGHTEN_N = 11.0    # below: 2 mm tighter (~1.4 N/mm measured)
SQUEEZE_HIGH_N = 30.0       # above: 1 mm looser (lowering the arms raised it 16 -> 24 N)
MAX_TIGHTEN_MM = 8
EE_TO_BOX_BOTTOM = 0.355    # ee z - box bottom z for the floor grasp at BALL_Z 0.18 (table touchdowns 0.35-0.36)
SETDOWN_SQUEEZE_N = 20.0    # set-down squeeze cap: P1 lowered the arms with 26 N and left_arm_5 reached 131 %
SETDOWN_MAX_LOAD = 1.25     # stop lowering the arms here and let the squat do the rest
TORSO_LOWEST_FLAT = [0, 1.2, -2.4, 1.2, 0, 0]   # lowest squat, no lean: hands ~2 cm further forward than leaning

# bimanual_box_pnp/motion/navigation.py -------------------------------------------------

MARGIN = 0.06            # m around the robot footprint
SEG_FORWARD = 0.30       # m per closed-loop segment
SEG_TURN = 0.6           # rad per closed-loop segment
MAX_HIT_POINTS = 25      # obstacle points (stride 3) inside the swept footprint that stop a segment
BORDER_PX = 20           # ignore depth this close to the image border (A3: 18 spurious points at row 708)

# bimanual_box_pnp/motion/place.py ------------------------------------------------------

TOUCH_DFZ_N = 4.0           # vertical (base frame) wrist force change per side = the surface took the object
STEP_DOWN = 0.01

# bimanual_box_pnp/motion/squeeze.py ----------------------------------------------------

SQUAT = [[0, 0.785, -1.571, 0.855, 0, 0], [0, 1.047, -2.094, 1.117, 0, 0], [0, 1.2, -2.4, 1.27, 0, 0]]
TORSO_LOW = SQUAT[-1]                       # lowest + 0.07 lean
BALL_Z_PRE = 0.20                           # balls above the support after the squat (box top 0.263)
BALL_Z = 0.18                               # touch / squeeze height above the support (tested 0.18)
STANDOFF = 0.09                             # ball centre starts this far outside the depth end
MAX_TOUCH_TRAVEL = 0.11
KNEE_Y = 0.205                              # |ee y| below this when squatting: wrist meets the knee link (model)
SQUEEZE_MM = 4                              # extra squeeze after both contacts (~1.4 N/mm)
MIN_SQUEEZE_N = 9.0
Q_EE = {'right': [-0.187, -0.1618, 0.7327, 0.634], 'left': [0.107, -0.2229, -0.4199, 0.8732]}     # floor grasp
Q_CARRY = {'right': [-0.2127, -0.1838, 0.726, 0.6276], 'left': [0.1222, -0.2528, -0.4153, 0.8653]}   # torso zero

# bimanual_box_pnp/motion/touch.py ------------------------------------------------------

CONTACT_DF_N = 4.5      # |F - F0| that counts as contact (noise ~1 N)
STEP_M = 0.01

# bimanual_box_pnp/perception/box.py ----------------------------------------------------

FLOOR_Z_MIN = 0.04
OBJECT_Z_MAX = 0.45
GRID = 0.01
BOX_LONG = (0.28, 0.50)
BOX_SHORT = (0.10, 0.30)          # far views shrink the short side (0.13 at 3.2 m)
BOX_TOP_Z = (0.18, 0.34)          # above the support
CARDBOARD_H = (10, 25)            # OpenCV hue; box top 14-18, orange bucket 6, red backpack 177
CARDBOARD_S = (55, 185)           # box 90-125, bucket 250, floor 3
MIN_CARDBOARD_FRAC = 0.45

# bimanual_box_pnp/perception/locate.py -------------------------------------------------

MAX_CORNER_ERR = 0.08       # m: Gemini corners vs depth rectangle beyond this -> not trusted (checkpoint)
MIN_MASK_POINTS = 400       # masked depth points on the top; fewer = the view sees too little of it (rejected)

# bimanual_box_pnp/perception/sensing.py ------------------------------------------------

SEARCH_LOOKS = [[1.6, 0.0, 0.1], [1.2, 0.6, 0.1], [1.2, -0.6, 0.1], [2.5, 0.0, 0.1]]
TABLE_LOOKS = [[1.5, 0.0, 0.7], [1.3, 0.55, 0.7], [1.3, -0.55, 0.7]]
FULL_SIZE = {'box': (0.33, 0.15)}       # depth-only fallback: a closer detection smaller than this is partial
MOVED_TOL = 0.10                        # m: a re-detected object this far from its prediction moved (contact?)

# bimanual_box_pnp/perception/table.py --------------------------------------------------

TABLE_Z = (0.60, 0.85)
DARK_MAX = 90                   # gray level of the black top
TABLE_OBST_Z = (0.15, 0.9)            # obstacles; floor depth noise reached 0.13 m near the image border, cables ~0.1

# bimanual_box_pnp/tasks/table_to_floor.py ----------------------------------------------

STANCE_D = 0.62        # box centre ahead at the table grasp (P1 0.60 with face_blend 0.5; 0.62 box-facing: base 8 cm
                       # from the top edge, IK ok to 0.62)
STRAIGHT = 0.35        # straight final leg = the reverse allowed after the grasp (tested 0.35)
FACE_BLEND = 0.0       # 0: face the box squarely (symmetric squeeze, box hangs level); 1: face the table edge
