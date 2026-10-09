# push_chair controller (RB-Y1)

Grasp a wheeled office chair by the top bar of its mesh backrest with both grippers, move it with the base and tuck
it under the yellow (light-wood) desk. Also packaged as the skill-library skill `office-chair-tuck`.

Run from the workspace root, one step at a time, and look at the returned frames before the next step:

```
python code/push_chair/run_tuck.py find                 # Gemini + depth: chair column / facing (head scan)
python code/push_chair/run_tuck.py approach plan        # print the base moves to stand square behind the chair
python code/push_chair/run_tuck.py approach             # execute them (observes after each drive)
python code/push_chair/run_tuck.py measure              # hands wide, bar profile, chair yaw (+ desk front if seen)
python code/push_chair/run_tuck.py square               # turn by the chair yaw, lane change to centre on the bar
python code/push_chair/run_tuck.py grasp                # two-hand grasp (tangent yaw, pitch -45, fingers 0.20 m)
python code/push_chair/run_tuck.py transport 0.3 plan   # print the moves to bring the chair in front of the desk
python code/push_chair/run_tuck.py transport 0.3        # pull 0.3 (right caged), turn, drive, turn to desk yaw
python code/push_chair/run_tuck.py measure              # robot square to the desk: desk front + bar
python code/push_chair/run_tuck.py tuck                 # push to bar-to-desk-edge 0.42 m (force-stopped)
python code/push_chair/run_tuck.py release
python code/push_chair/run_tuck.py verify
python code/push_chair/run_tuck.py desk                 # once, square in front of the desk: store its pose (odometry)
```

Lower-level steps: `pull <m>`, `drive <m>`, `turn_to <odometry yaw>`. State: `output/push_chair_state.json`.

Modules: `chair_ctl.py` (primitives + guards), `perception.py` (Gemini chair/desk points, desk front from colour +
depth), `geom.py` (depth -> base frame, top-down maps), `kin.py` (pinocchio IK), `plan.py` (footprint sweep checks).

## Perception checkpoints

| Step | Subagent | Frames | Question | Expected | Pass if | On fail |
|---|---|---|---|---|---|---|
| find | haiku | `find` frame | Is there an office chair with a mesh backrest, and is the floor between the robot and the chair free of people and objects? | yes/no + obstacles | chair visible, path free | ask the operator / re-scan |
| approach (each drive) | agent | frames returned by `approach` | Is the robot's path and its rear sweep (r 0.62 m behind the axle) clear? | clear / blocked | clear | stop, replan |
| measure | haiku | `measure` frame | Is the chair directly in front with its backrest towards the camera, nothing in between? | yes/no | yes | reposition, measure |
| grasp | haiku | `grasp` frame | Are both grippers closed around the backrest's top bar? | yes/no each | both (widths 15-30 mm) | open, back out, measure, retry once |
| transport (turns) | agent / operator | stream, look-arounds | Will the held chair (≈1.0 m around the axle) or its casters hit desk legs or furniture? | clear / contact | clear | back off 0.1 m, lane change, retry |
| verify | haiku | `verify` frame | Chair under the desk, square, armrests at/under the edge, centred? | yes/no + angle/offset | yes | pull out 0.3, square, re-centre, tuck |

## Tested (2026-10-09)
M1 manual tuck, P1 park left of the desk against the wall, P2 parked -> tucked (gap 0.419 m, centre -0.025 m).
`find` / `approach` / `transport` are new (v2) and not yet run end to end.
