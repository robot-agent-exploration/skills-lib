"""Hyper-parameters of the tidy-chair-cups-basket-box skill, in one list; the modules import their values from here.

The values are as the sessions that wrote the skill set them, with their comments. Values marked
CALIBRATE depend on the setup (cameras, table, layout, postures): in a setup the skill was not
tested in, measure them again as their comment says, or pass the measured value in.
"""

# chair_route.py ------------------------------------------------------------------------

CHAIR_EXTRA = (0.50, -0.36, 1.24, 0.36)   # held chair box (x0, y0, x1, y1) in the base frame for sweep_check
