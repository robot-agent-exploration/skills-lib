"""Hyper-parameters of the air-fry-pan skill, in one list; the modules import their values from here.

The values are as the sessions that wrote the skill set them, with their comments. Values marked
CALIBRATE depend on the setup (cameras, table, layout, postures): in a setup the skill was not
tested in, measure them again as their comment says, or pass the measured value in.
"""
import numpy as np

# Calibrate in a new setup ---------------------------------------------------------------

# CALIBRATE (controller.py): an absolute TCP height (m) set for this table and fryer; shift it by
#   the difference of their heights from the session's
DROP_Z = 0.36                # TCP height at release (housing clears fryer top)

# CALIBRATE (controller.py): the exterior cameras that see the fryer; in the session the right
#   camera had moved after a collision, so only the left was used
PERC_CAMS = ('left',)        # right exterior camera moved after a collision: unusable

# CALIBRATE (controller.py): the cameras that confirm steps; use the cameras this setup has
VIEW_CAMS = ('wrist', 'left')

# CALIBRATE (controller.py): an absolute TCP height (m) set for this table and fryer; shift it
#   like DROP_Z
BALL_DROP_Z = 0.30           # TCP height at ball release (~3 cm above rim)

# CALIBRATE (controller.py): a wrist-camera viewpoint over free table beside the fryer (base
#   frame, m); pick free table in this layout
TABLE_VIEW = np.array([0.42, 0.26, 0.55])   # wrist view of the table left of the fryer

# CALIBRATE (geom.py): the cameras whose calibration is trusted; check each against known points
USABLE_CAMERAS = ('left', 'wrist')

# CALIBRATE (kin.py): the start posture; it must match the harness's home preset
#   (franka_move_joints preset 'home')
HOME = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])

# CALIBRATE (motion.py): below the harness's largest joint change per franka_move_joints call (1.5
#   rad in the session); check robot-interface
MAX_JOINT_STEP = 1.4      # rad per move_joints call (service limit 1.5)

# CALIBRATE (motion.py): above the harness's joint-limit margin for targets (0.2 rad in the
#   session); check robot-interface
JOINT_TARGET_MARGIN = 0.21  # service rejects targets within 0.2 rad of its limits

# CALIBRATE (perception.py): the table surface height in the base frame (m); measure the median
#   height of bare-table pixels with franka_pixel_to_position
TABLE_Z = 0.09

# CALIBRATE (perception.py): the base-frame x and y ranges (m) of the table where the fryer is
#   searched; cover where it stands in this layout
# Search region for the fryer (generous; robot workspace in front of the base).
FRYER_REGION = ((0.35, 0.95), (-0.45, 0.30))

# controller.py -------------------------------------------------------------------------

# ---- geometry learned in manual run 1 (offsets along the drawer axis d, metres) ----------
FACE22_MODEL = 0.1674        # grip outer face at z=0.22 in the collision-model frame (closed)
GRASP_FROM_FACE = -0.0116    # handle grasp TCP along d, relative to face(z=0.22)
GRASP_BELOW_ARM_TOP = 0.009  # handle grasp TCP z below the arm top
HANDLE_TILT = np.deg2rad(5)  # approach axis tilted toward the fryer
BASKET_FROM_FACE = -0.1474   # basket rim centre (closed drawer) relative to face(z=0.22)
PROTRUSION_CLOSED = 0.065    # grip face - body front below this = drawer closed (closed ~0.056-0.059)
PULL = 0.125                 # drawer pull distance (tested 0.126; front tips ~2-3 cm on release)
PULL_STEP = 0.03
GRASP_FORCE_HANDLE = 50.0    # N
BANANA_FORCE = 20.0          # N
DROP_BETA = np.deg2rad(60)   # hand tilt about the banana axis at the drop
DROP_MID_FROM_BASKET = 0.015  # banana middle beyond basket centre along d
ROT_SPOT = (0.26, -0.05, 0.50)  # (along d, along t, z) where the held banana is re-oriented
FINGER_HALF = 0.00875        # closed-finger half thickness along hand x
PUSH_Z_BELOW_ARM_TOP = 0.0205
VIEW_QUAT = np.array([1.0, 0.0, 0.0, 0.0])   # wrist camera looks down and slightly forward
VIEW_DRAWER = (0.215, 0.0, 0.50)  # (along d, along t, z) wrist view of handle + open basket
TOP_VIEW_OFFSETS = [np.array(v) for v in ((-0.07, 0.06, 0.20), (-0.09, 0.05, 0.18), (-0.11, 0.04, 0.16))]  # TCP rel. lid
BALL_FORCE = 10.0            # N (ball may be hollow)
BALL_DROP_FROM_BASKET = 0.045  # ball released above basket centre + this along d (exposed part)
MIN_MARGIN = 0.15            # joint margin (rad) required for planned configurations
MIN_CLEAR = 0.008            # m, coarse model clearance required
