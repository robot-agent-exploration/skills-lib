"""Hyper-parameters of the oc-hand-reach skill, in one list; the modules import their values from here.

The values are as the sessions that wrote the skill set them, with their comments. Values marked
CALIBRATE depend on the setup (cameras, table, layout, postures): in a setup the skill was not
tested in, measure them again as their comment says, or pass the measured value in.
"""

# Calibrate in a new setup ---------------------------------------------------------------

# CALIBRATE (oc_reach.py): the backend's per-call caps per target (m, rad); check them against
#   robot-interface
CAP_M, CAP_RAD = 0.25, 1.0      # backend per-call caps per target

# oc_reach.py ---------------------------------------------------------------------------

NATURAL = {'right': [-0.6, -0.45, 0.0, -1.9, 0.0, 1.2, 0.0],
           'left': [-0.6, 0.45, 0.0, -1.9, 0.0, 1.2, 0.0]}     # elbow-down reference (compositional session)
# posture prior: joint -> weight. Shoulder roll (joint 1) is the one that runs into its limit when both hands come
# close together in front (right_arm_1 reached -0.012 rad with a 0.2 weight and the backend stopped the move,
# 2026-10-09): 1.5 at -/+0.45 keeps it inside; the elbow (joint 3) only gets a weak pull; the upper-arm twist is
# left free so the elbow can swivel. The wrist pitch (joint 5) ran to 1.892 rad (limit) for a high hand pitched -45
# deg (chair hands_up) with only joints 1 / 3 weighted: 0.6 towards 1.2 makes the swivel keep it lower.
# Budget: <= 10 joint targets per call = torso 4 + 3 per moving arm.
PRIOR = {1: 1.5, 3: 0.2, 5: 0.6}
