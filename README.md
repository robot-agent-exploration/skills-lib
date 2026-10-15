# Franka FR3 skill library

Skills that agents learned on a Franka FR3, one per task: each is the program the agent
wrote and tested while it explored the task on the real robot, imported from the session
with its records and frames.

The protocol for using and contributing is [CONTRIBUTING.md on
`main`](https://github.com/robot-agent-exploration/skills-lib/blob/main/CONTRIBUTING.md).
The skills are listed in [INDEX.md](INDEX.md).

## The robot

- One 7-DoF Franka FR3 arm with a Franka Hand, on a table (top at z ~0.09 m in the base
  frame `fr3_link0`: x forward over the table, y to the robot's left, z up). The TCP is
  `fr3_hand_tcp`, between the fingertip pads.
- Cameras: two exterior RGB-D cameras on stands beside the table (`left`, `right`) and one
  on the wrist. The stands are obstacles: the robot service keeps every arm link 0.20 m
  from them and refuses motions that come closer.
- The code calls the robot through the RAX harness context `ctx` (`api.runtime.Context`,
  e.g. `ctx.value('franka_move_pose', position=[...])`) and its `franka_*` tools:
  status, observe, straight-line TCP moves, guarded moves that stop on contact force,
  joint moves, gripper, hold, IK / FK. The tools, limits and safety layers are described
  on [`real/franka`](https://github.com/robot-agent-exploration/rax/tree/real/franka).
- Other dependencies: numpy, scipy and opencv in the agent environment; skills that ask a
  vision-language model use the harness's `gemini_generate` tool.

## Where the skills came from

Five sessions of 2026-10-03 to 2026-10-10, one task each, run before the shared library
existed: each agent kept its procedure in its own task memory and code. They were
imported into this format afterwards (CONTRIBUTING.md, section 6): the code is the
session's final code with small changes for the library layout, and the runs in
`evidence/runs.jsonl` are the session's records (`"commit": "pre-library"`).
