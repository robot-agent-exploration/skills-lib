"""Hyper-parameters of the peg-insertion skill, in one list; the modules import their values from here.

The values are as the sessions that wrote the skill set them, with their comments. Values marked
CALIBRATE depend on the setup (cameras, table, layout, postures): in a setup the skill was not
tested in, measure them again as their comment says, or pass the measured value in.
"""

# fixture.py ----------------------------------------------------------------------------

PUSH = 0.0015        # commanded push per call beyond the last measured position (m)
BLOCKED = 0.0007     # moved this much less than commanded = blocked by the wall (m)

# insert.py -----------------------------------------------------------------------------

# Values for the L1 (3 mm clearance) holes; first tested 2026-10-06 on the hexagon.
APPROACH_ABOVE = 0.008     # start probing this far above the expected surface contact (m)
PROBE_STEP = 0.001         # descent step while probing (m)
CONTACT_DFZ = 3.0          # |delta fz| (N) that counts as contact
FREE_BELOW = 0.006         # no contact this far below the expected surface -> peg is in the hole
PRESS = 0.0035             # command this far below the surface contact while searching (~10 N)
DROP = 0.002               # measured z this far below the surface contact = peg entered the hole
SEARCH_TOL_MM = 4.0        # > PRESS so pressed search moves return instead of winding up force
SPIRAL_PITCH = 0.002       # radial spacing of the spiral (m)
SPIRAL_STEP = 0.002        # arc length per move (m)
SPIRAL_RMAX = 0.010        # search radius (m)
DESCEND_STEP = 0.002       # descent step inside the hole (m)
JAM_DFZ = 8.0              # |delta fz| (N) that counts as blocked while descending in the hole
BOTTOM_SLACK = 0.004       # blocked within this of the expected bottom = bottomed out (ok)
HOP_UP = 0.0015            # hover this far above the surface contact between search points (m)
HOP_STEP = 0.002           # candidate spacing (m); diagonal gap 1.41 mm < 1.5 mm per-side clearance
HOP_A_MAX = 0.010          # search half-range along the finger axis (gripper x) (m)
HOP_B_MAX = 0.004          # search half-range along the closing axis (gripper y) (m)
SLIDE_PRESS = 0.0008       # command this far below the surface contact while sliding (~2 N)
SLIDE_DROP = 0.0005        # measured z this far below contact = peg entering the hole
SLIDE_STEP = 0.0005        # arc length per slide step (m)
SLIDE_PITCH = 0.0006       # spiral radial spacing (m)
SLIDE_RMAX = 0.0022        # slide search radius (m)
SLIDE_BIAS = (0.0, 0.0)    # optional base-frame offset of the spiral centre (camera hint)
FINE_STEP = 0.0006         # fine hop grid spacing (m), < 2 x 0.5 mm per-side clearance
FINE_RMAX = 0.0020         # fine hop search radius (m)
FINE_UP = 0.0006           # hover above contact between presses (m)
FINE_PRESS = 0.0015        # press target below contact (m); tol 0.5 mm so it really presses
FINE_DROP = 0.0008         # measured z this far below contact = peg entered the hole
FINE_LIST = None           # optional explicit candidate offsets [(dx, dy), ...] (base frame, m)
FINE_PLACE = False         # correct the hover xy with measured positions before each press
COVER_R = 0.00045          # a press at p rules out hole positions within this radius (m)
PLACE_TOL = 0.00015        # hover placement accuracy target (m), via measured-position correction
ADAPT_RMAX = 0.0022        # adaptive search radius (m)
TESTED = []                # optional seed of already-pressed xy positions (base frame)
RASTER_HALF = 0.0018       # half-size of the raster square (m)
RASTER_LINE = 0.0008       # spacing between raster lines (m) (< 2 x 0.5 mm clearance)
RASTER_SEG = 0.0010        # segment length per call (m): enough spring force to beat friction
RASTER_PRESS = 0.0010      # press target below contact (m) (~3 N)
RASTER_DROP = 0.0005       # measured z this far below contact = entered the hole
