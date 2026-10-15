"""Hyper-parameters of the find-the-cube skill, in one list; the modules import their values from here.

The values are as the sessions that wrote the skill set them, with their comments. Values marked
CALIBRATE depend on the setup (cameras, table, layout, postures): in a setup the skill was not
tested in, measure them again as their comment says, or pass the measured value in.
"""

# Calibrate in a new setup ---------------------------------------------------------------

# CALIBRATE (arrange.py): where the cups are lined up (base frame x, m); free table in this layout
ROW_X = 0.55

# CALIBRATE (arrange.py): where the cups are lined up (base frame y, m); free table in this layout
ROW_Y = (-0.18, -0.06, 0.06, 0.18)

# CALIBRATE (controller.py): the TCP position (base frame, m) from which the wrist camera sees the
#   whole play area; check the view in this layout
WATCH_POS = [0.47, 0.0, 0.62]     # wrist camera sees x 0.35..0.95, y -0.5..0.5 from here

# CALIBRATE (controller.py): the wrist image region above the gripper fingers, in pixels of a
#   1280-wide image; check it against this wrist camera's resolution and mounting
ROI = (0, 0, 1280, 520)           # wrist image region above the gripper fingers

# CALIBRATE (controller.py): the table surface height in the base frame (m); measure the median
#   height of bare-table pixels with franka_pixel_to_position
TABLE_Z = 0.09

# CALIBRATE (controller.py): absolute heights (m) of a cube top on a table at TABLE_Z; shift them
#   with the table height
CUBE_Z = (0.103, 0.16)            # cube top ~0.12 (3 cm cube)

# CALIBRATE (controller.py): an absolute height (m) for a table at TABLE_Z; shift it with the
#   table height
CUP_TOP_MIN_Z = 0.15              # inverted cup top ~0.183 (9.3 cm cup)

# CALIBRATE (controller.py): an absolute TCP height (m) for a table at TABLE_Z; shift it with the
#   table height
CLOSEUP_Z = 0.50                 # ... at this height (camera ~0.4 m from the cup top)

# CALIBRATE (controller.py): the wrist camera's offset from the TCP in TCP x and y (m); read it
#   from a frame's meta.json (T_base_cam of the wrist camera against the TCP pose)
CAM_OFFSET_TCP = (-0.071, 0.029)  # wrist camera from the TCP in TCP x/y (it rides behind the fingers)

# controller.py -------------------------------------------------------------------------

N_CUPS = 4
CUP_GRASP_BELOW_TOP = 0.012       # fingertips 1.2 cm below the cup top: 3 cm / 15 N pinched a cube
CUP_FORCE = 8                     # sitting high inside the cup through the wall (trial 8)
CUBE_FORCE = 20
SMALL_FORCE = 12                  # objects under 45 mm
GENTLE_FORCE = 6                  # re-grasp force after the grip closed in (red cube slid out at 20 N)
SLIP_MM = 4                       # finger closing during the 2 cm test lift that counts as slipping
HAND_HALF_LENGTH = 0.105         # Franka Hand body ~21 cm along the finger axis ...
HAND_HALF_WIDTH = 0.03           # ... and ~6 cm across
HAND_CLEARANCE = 0.04            # hand-body rectangle to other cups' centres (cup radius 3 cm + 1 cm)
RIDGE_DL = 12                    # brightness step (Lab L) between the halves of a ridged top
DROP_ABOVE_GRASP = 0.02          # release the moved cup with its rim ~2 cm above the table
DROP_CLEARANCE = 0.09            # dropped cup centre to other cups / the object (rims touch at 7.5 cm)
SAVE_RECORDING = True            # copy the shuffle's wrist frames to output/recordings/
CLOSEUP_BACK = 0.10              # close-up TCP this far towards the robot from the cup ...
SIDE_LOOK = 0.10                 # sideways offset of the second look after a blocked descent
CAM_CLEARANCE = 0.08              # camera centre to other cups' centres when the TCP is low

# shuffle.py ----------------------------------------------------------------------------

TPL_HALF = 15            # template half size at half resolution (60x60 px at full resolution)
SEARCH = 10              # local template search radius at half resolution
VISIBLE_SCORE = 0.85     # correlation at P that means the cube is in view (plain table < 0.8)
FOLLOW_SCORE = 0.92      # above this, follow small nudges of the cube
UNCOVER_SCORE = 0.92     # a cover is cancelled only by a clear view of the cube (own: 0.97-0.99)
TPL_DIRT_MAX = 0.05      # template patch may hold at most this much green / skin
COVER_PX = 65            # cup track centre within this of P (full-res px; the tall cup's blob
                         # centre sits up to ~55 px off the cube under it near the image edges) ...
COVER_FRAMES = 12        # ... for this many frames while the cube is occluded = covered
MOVE_MIN_PX = 30         # a depth detection farther than this from P is a move candidate
MOVE_CONFIRM_PX = 20     # consecutive detections within this of each other confirm the move
CUBE_SIZE_MM = (33, 62)  # plausible cube footprint from depth (trials: 40-52 mm)
CUBE_AREA = (350, 1200)  # depth pixels at step 2 (trials: 510-780)
CUBE_MIN_H = 0.02        # height above the local table (trials: 0.023-0.041)
# cube dropped into an upright cup: a white region enclosed by that cup's green blob
HOLE_AREA = (80, 1600)   # half-res px (cube top ~22x22 px half-res, partly hidden by the rim)
HOLE_WHITE = 0.55        # fraction of the hole matching the object's colour
HOLE_COLOUR_DIST = 35    # Lab distance (L weighted 0.3: the object lies in the cup's shadow)
HOLE_DARK_V = 70         # darker pixels inside a cup are its shadowed wall, not the object
HOLE_SKIN_MAX = 0.35     # fraction of the hole that looks like skin (a hand holding the object)
INSIDE_PX = 70           # hole centroid to cup track centre (full-res px)
INSIDE_FRAMES = 8        # consecutive frames with the cube seen inside the same cup

# tracker.py ----------------------------------------------------------------------------

SCALE = 0.5                      # process at half resolution
HSV_LO = (55, 90, 50)            # green cup colour (OpenCV H 0-180)
HSV_HI = (100, 255, 255)
MIN_BLOB_PX = 150                # at half resolution
