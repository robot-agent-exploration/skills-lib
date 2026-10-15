"""Hyper-parameters of the swipe-table skill, in one list; the modules import their values from here.

The values are as the sessions that wrote the skill set them, with their comments. Values marked
CALIBRATE depend on the setup (cameras, table, layout, postures): in a setup the skill was not
tested in, measure them again as their comment says, or pass the measured value in.
"""

# Calibrate in a new setup ---------------------------------------------------------------

# CALIBRATE (controller.py): a clean-table frame of the session; in another workspace it is absent
#   and the skill works without a reference, or pass --reference with a clean frame of this setup
# (paper_controller.py) towel-scene clean table; box + bin are masked out
# Clean table seen from home with fixed exterior cameras (2026-10-06 09:35), used when the run
# starts with a stain already on the table; checked against the live frame before use.
STORED_REFERENCE = '20261006093511e0ce16'

# CALIBRATE (motions.py): an absolute height (m) for a table at TABLE_Z; shift it with the table
#   height
Z_FLOOR = 0.11          # never command the TCP below this while pressing (fingertips ~9 mm lower)

# CALIBRATE (paper_controller.py): where stains are searched (base frame x0, x1, y0, y1, m); where
#   the operator pours in this layout
STAIN_ROI = (0.36, 0.80, -0.42, 0.42)   # where stains are searched (operator pours 40-65 cm out)

# CALIBRATE (perception.py): the table surface height in the base frame (m); measure the median
#   height of bare-table pixels with franka_pixel_to_position
TABLE_Z = 0.09

# CALIBRATE (perception.py): the table area mapped for stains (base frame x, m); cover the table
#   in this layout
MAP_X = (0.20, 0.85)

# CALIBRATE (perception.py): the table area mapped for stains (base frame y, m); cover the table
#   in this layout
MAP_Y = (-0.50, 0.50)

# CALIBRATE (steps.py): an absolute height (m) for a table at TABLE_Z; shift it with the table
#   height
# --- tunables (tested on 2026-10-06, wet dark terry towel ~45x70 cm, white table z=0.09) ---
CARRY_Z = 0.45            # towel (wad grasp) hangs almost clear of the table here

# CALIBRATE (steps.py): an absolute height (m) for a table at TABLE_Z; shift it with the table
#   height
LOWER_Z = 0.16            # lowering the carried towel to here piles it up under the gripper

# CALIBRATE (steps.py): an absolute height (m) for a table at TABLE_Z; shift it with the table
#   height
SAFE_Z = 0.30

# CALIBRATE (steps.py): below the harness's largest move per franka_move_pose call (0.40 m in the
#   session); check robot-interface
MAX_SEG = 0.35            # split long moves (move_pose accepts <= 0.40 m)

# CALIBRATE (steps.py): an absolute height (m) for a table at TABLE_Z; shift it with the table
#   height
PINCH_PRESS_Z = 0.10      # flat towel: press fingertips into cloth (stalls ~0.108)

# CALIBRATE (steps.py): gripper path limits found with the session's camera keep-out and IK (base
#   frame, m); check them with franka_status (camera_keepout) and IK in this setup
PATH_X = (0.33, 0.68)     # gripper path limits while scrubbing near the table (beyond x 0.68 the

# CALIBRATE (steps.py): gripper path limits like PATH_X
PATH_Y = (-0.36, 0.36)    # (fit_region then shrinks to IK-safe postures: camera keep-out, margins); pile reaches ~7 cm further

# CALIBRATE (steps.py): reach at CARRY_Z from the session's IK map (base frame, m); check it with
#   IK in this setup
CARRY_MAX_X = 0.60        # at CARRY_Z, x > 0.65 is out of reach for |y| ~0.38 (IK map 2026-10-06)

# CALIBRATE (steps.py): reach at CARRY_Z like CARRY_MAX_X
CARRY_MAX_Y = 0.32

# CALIBRATE (steps.py): the start posture copied from the session; it should be the harness's home
#   preset (franka_move_joints preset 'home')
HOME_JOINTS = [-0.0155, -0.7602, -0.0021, -2.3552, 0.0206, 1.5761, 0.752]   # libfranka ready pose here

# CALIBRATE (steps.py): an absolute height (m) for a table at TABLE_Z; shift it with the table
#   height
CONTACT_PROBE_Z = 0.12   # press target to find the pad's contact height (pad stalls ~0.13-0.15)

# CALIBRATE (steps.py): towel grasp limits found with the session's camera keep-out and IK (base
#   frame, m); check them like PATH_X
GRASP_X = (0.30, 0.68)

# CALIBRATE (steps.py): towel grasp limits like GRASP_X
GRASP_Y = (-0.34, 0.34)

# CALIBRATE (tissues.py): an absolute height (m) for a table at TABLE_Z; shift it with the table
#   height
PULL_Z = 0.47             # lift to here: the sheet comes free (sheet ~20 cm long)

# CALIBRATE (tissues.py): an absolute height (m) for a table at TABLE_Z; shift it with the table
#   height
DROP_Z = 0.115            # release with the pinch nearly at the table: the sheet crumples there
                          # (released at 0.13 the hanging sheet folded over 5-7 cm to one side)

# CALIBRATE (tissues.py): an absolute height (m) for a table at TABLE_Z; shift it with the table
#   height
TRAVEL_Z = 0.45

# controller.py -------------------------------------------------------------------------

MAX_PASSES = 3
STAIN_MARGIN = 0.035        # gripper path beyond the stain bbox on pass 1 (droplets nearby)
RESIDUE_MARGIN = 0.015      # beyond the residue bbox on later passes
VERIFY_ROI_MARGIN = 0.15    # residue search area beyond everything wiped (smears travel)
CLEAN_AREA_M2 = 4e-4        # residue below 4 cm2 (vs clean reference) counts as clean
CLEAN_AREA_NOREF_M2 = 8e-4
CLEAN_WRIST_M2 = 3e-4       # wrist-camera residue (thin streaks) below 3 cm2 counts as clean
SOAP_BEFORE_PASS = 3        # ask the operator for water + soap before this pass and later ones
STAIN_PRESENT_M2 = 3e-4     # a stain above 3 cm2 at the start means the operator already poured
POUR_ASKS = 3               # ask the operator to pour at most this many times
TOWEL_SPOT_MIN_DIST = 0.12  # park towel at least this far (x/y gap) from the wiped region

# motions.py ----------------------------------------------------------------------------

PRESS_DEPTH = 0.012     # command this far below the measured contact height (light press)
PRESS_TOL_MM = 20.0
STROKE_TIMEOUT_S = 15.0
MAX_REFUSED = 3          # refused scrub waypoints tolerated per scrub

# paper_controller.py -------------------------------------------------------------------

SHEETS_PER_PASS = 6
PAPER_CONTROLLER_MAX_PASSES = 2
STAIN_PRESENT_NOREF_M2 = 8e-4   # without a reference: false patches up to ~5 cm2 near the base
BOX_MOVED_M = 0.03              # stop if the tissue box shifts more than this (pulled along with a sheet)
LEFTOVER_RISE_M = 0.018     # real leftover sheets rose 22-50 mm, false patches (shadows) 0-17 mm
LEFTOVER_MIN_M2 = 2e-4
MAX_LEFTOVER_PICKS = 8          # leftover-sheet picks per pass at most

# perception.py -------------------------------------------------------------------------

MAP_RES = 0.0025

# steps.py ------------------------------------------------------------------------------

PINCH_TOL_MM = 10.0
WAD_FORCE = 50.0
PINCH_FORCE = 40.0
MIN_WAD_WIDTH = 0.004     # a grasp narrower than this holds only a thin pinch
WRIST_MARGIN = 0.06        # wrist residue check covers the wiped region + this
CAMERA_SLACK = 0.01        # extra camera clearance beyond the service keep-out (0.20 m, enforced live by the service)

# tissues.py ----------------------------------------------------------------------------

GRASP_FORCE = 20.0
TCP_ABOVE_BOX = 0.030     # pinch with the TCP at least this far above the box top (fingertips ~2 cm)
