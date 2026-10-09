"""rby1rt: shared runtime of the step-wise RB-Y1 task skills in this library.

robot (tool wrappers, body constants), kin (pinocchio FK / IK incl. the closed-ball tool), geom (base <-> odometry),
world (objects in the odometry frame, what is held), frames (saved head-camera frames), footprint (robot footprint
for base-motion checks), vlm (Gemini structured perception), runner (steps, checkpoints with subagent briefs,
STOP-file monitoring), posture (prepare). Other skills import it after `use()` put their code dirs on sys.path."""
import sys
from pathlib import Path

SKILLS = Path(__file__).resolve().parents[3]          # skill-library/skills


def use(*slugs):
    """Make other library skills' code packages importable (adds skills/<slug>/code to sys.path)."""
    for s in slugs:
        p = str(SKILLS / s / 'code')
        if not (SKILLS / s / 'code').is_dir():
            raise ImportError(f'library skill {s} not found under {SKILLS}')
        if p not in sys.path:
            sys.path.insert(0, p)
