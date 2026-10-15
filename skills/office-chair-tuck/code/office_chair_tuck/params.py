"""Hyper-parameters of the office-chair-tuck skill, in one list; the modules import their values from here.

The values are as the sessions that wrote the skill set them, with their comments. Values marked
CALIBRATE depend on the setup (cameras, table, layout, postures): in a setup the skill was not
tested in, measure them again as their comment says, or pass the measured value in.
"""
import math

# Calibrate in a new setup ---------------------------------------------------------------

# CALIBRATE (chair_ctl.py): the custom fingers' geometry (robot-interface, 'Custom fingers');
#   measure it again if this robot's fingers differ
FINGER = 0.20            # open fingertips' middle along ee -z (robot-interface "Custom fingers")

# CALIBRATE (chair_ctl.py): the ready posture copied from the session; check it against the ready
#   posture in robot-interface
# ---------- arm motions ----------
READY_ARM = {'right': [-0.7854, -0.5236, 0.0, -1.5708, 0.0, 0.7854, 0.0],
             'left': [-0.7854, 0.5236, 0.0, -1.5708, 0.0, 0.7854, 0.0]}

# chair_ctl.py --------------------------------------------------------------------------

TORSO_ZERO = [0.0] * 6
PITCH = math.radians(-45)
HAND_HALF = 0.10         # hands at bar centre ± this along the bar (tested 0.10)
GRASP_DEEPER = 0.04      # v3: the ee goes this much further along the fingers than 'tips' middle on the bar centre':
                         # at 0 the fingers bit only the bar's top edge (9-13 mm, operator: "missed", "be deeper");
                         # 0.04 deeper closed around the whole bar (26 / 35 mm), 2026-10-09 compositional session
RELIEVE_PCT = 85         # a wrist joint above this after closing -> relieve and re-close softly
ABORT_PCT = 110          # any joint above this while moving the chair -> stop
WRIST_ABORT_N = 55       # wrist force change (N) while moving the chair -> stop
UP_POSE = {'right': (0.40, -0.30, 1.30), 'left': (0.40, 0.30, 1.30)}   # hands up, fingers clear of the chair
ROLL_STEP = 0.11          # wrist-roll correction while the gripper is open (trial 2: 108-113 % -> 35-45 %)
# ---------- approach the chair from afar (Gemini + depth) ----------
STANDOFF = 0.88          # base origin to chair column when grasping (bar front ≈0.60-0.65 ahead)
PRE_DIST = 1.40          # pre-pose on the chair axis, column distance
CLEAR_R = 0.75           # robot axle path must stay this far from the chair column

# chair_moves.py ------------------------------------------------------------------------

CHAIR_MOVES_RELIEVE_PCT = 90.0       # a wrist / forearm joint above this between turn steps -> relieve that side
CHAIR_MOVES_ABORT_PCT = 125.0        # still above this after relief -> stop
CHAIR_MOVES_WRIST_ABORT_N = 55.0

# run_tuck.py ---------------------------------------------------------------------------

TUCK_GAP = 0.42          # bar (front surface) to desk front edge when tucked: armrests just under the edge (M1, P2)
SQUARE_TOL = 0.06        # rad: chair yaw below this is left to the grasp's tangent yaws
CENTRE_TOL = 0.06        # m: lateral bar offset below this is not corrected
