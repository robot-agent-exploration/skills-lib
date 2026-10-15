# Rainbow RB-Y1 skill library

Skills that agents learned on a Rainbow Robotics RB-Y1 (model A) and merged into this
library across sessions, 2026-10-08 to 2026-10-10. Each session read the library, reused
and improved the skills that matched its task, and merged back what it learned; the
history of this branch is that record, one `Merge session/<session>: ...` commit per
merge.

The protocol for using and contributing is [CONTRIBUTING.md on
`main`](https://github.com/robot-agent-exploration/skills-lib/blob/main/CONTRIBUTING.md).
The skills are listed in [INDEX.md](INDEX.md).

## The robot

- Wheeled base (differential drive), 6-joint torso, two 7-DoF arms with grippers (custom
  curved fingers: closed ball 0.195 m, open tips ~0.20 m), 2-DoF head with a ZED 2i RGB-D
  camera.
- The code calls the robot through the RAX harness context `ctx` (`api.runtime.Context`)
  and its `rby1_*` tools: status, observe, joint / body / end-effector moves, the
  controller's Cartesian and whole-body optimal-control commands, base moves, grippers,
  dry-run checks and hold. The tools, limits and safety layers are described in
  [`RBY1.md` on `real/rainbow`](https://github.com/robot-agent-exploration/rax/tree/real/rainbow).
- Other dependencies: numpy, opencv, shapely and pinocchio in the agent environment, and
  the harness's Gemini helper (`.agents/skills/gemini/scripts/vlm.py`) for perception.

## The skills

| Skill | What it covers |
| --- | --- |
| `bimanual-box-pnp` | a carton between the floor and a table, squeezed between both hands |
| `bimanual-bucket-rim-pnp` | a 5-gallon bucket off the floor onto a cabinet, pinched by its rim |
| `office-chair-tuck` | an office chair grasped by its backrest and moved with the base |
| `stack-cups` | one cup nested into another with a one-hand rim pinch |
| `oc-hand-reach` | hand poses by the controller's whole-body optimal control, the primitive the others use |
| `tidy-chair-cups-basket-box` | the six-step room tidy that chains the skills above |

All are `draft`: none has two unattended successful runs of its current code yet
(`oc-hand-reach` has five, all of version 1; the others at most one). The sessions themselves (transcripts, robot tool calls, camera
archives) are the RAX RB-Y1 sessions; the frames in `evidence/images/` come from them.

Two earlier box skills (`bimanual-box-squeeze`, `box-floor-to-table-squeeze`) were
replaced by `bimanual-box-pnp` and removed in 6d1abd9; they are in its parent, b4614fd.
