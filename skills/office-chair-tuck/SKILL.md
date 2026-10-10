---
name: office-chair-tuck
description: Grasp a wheeled office chair by the top bar of its mesh backrest with both grippers and move it with the base (pull out, turn, drive, push in) to tuck it under a desk or park it against a wall; step-wise controller with depth bar profile, tangent-matched gripper yaw, load guards and agent checkpoints.
status: draft
robot: Rainbow RB-Y1
entry: code/run.py
version: 4
uses: [oc-hand-reach]
---

# Office chair: grasp the backrest and tuck / park the chair

## When to use
- A 5-star office chair on casters with a framed mesh backrest whose top bar is at z≈1.0-1.05 m (tested: one
  chair, bar ≈0.40 m wide, curved: x ≈ 1-1.7·y² around its middle, ≈2.5 cm thick), armrests ≈0.70 m high.
- Tuck it under a desk (tested: light-wood sit-stand desk, top 0.745 m, the armrests pass under) or park it against
  a wall; any start pose as long as the robot can stand square behind the backrest with open floor around
  (the base turns sweep r≈0.62 m behind the drive axle; the held chair sweeps ≈1.0 m around the axle).
- Not for: chairs without a graspable top bar, lifting chairs, stairs/ramps, a person sitting on the chair.

## Preconditions
- Custom round fingers (closed ball 0.195 m, open tips ~0.20 m along ee -z; robot-interface "Custom fingers").
- Torso ZERO (standing: shoulders 1.37 m), chest 0, hands up/wide (`measure` puts them there), grippers homed.
- The robot stands square behind the backrest, bar front surface ≈0.55-0.65 m ahead (`measure` reports
  `bar_x`); `square` turns by the measured chair yaw and lane-changes to centre the bar (|y| < 0.06 m).
- Before `tuck`: the robot square to the desk and `measure` has seen the desk front (desk front x stored).

## Procedure
0. From afar: `find` (head scan 0 / ±0.45, Gemini chair points + depth -> column, facing), `approach` (plan:
   turn-drive-turn to a pre-pose 1.4 m behind the column on the chair axis, a side waypoint when the straight
   path passes within 0.75 m of the chair, then 0.52 m straight in to the 0.88 m standoff; `approach plan` only
   prints it). Once, while square in front of the desk: `desk` stores the desk front centre and the desk-facing
   yaw in odometry coordinates. After the grasp, `transport [pull_back]` plans and runs pull back (right caged),
   turn towards the target, drive (or reverse), turn to the stored desk yaw, leaving the bar 0.8 m from the
   edge for `tuck`.
1. `measure`: hands wide (y ±0.42-0.46), head (0, 0.66), observe; depth+colour bar profile (dark points
   z 0.95-1.06 per 2.5 cm y bin: front x, top z), flat-top centre, curvature, chair yaw from the bar ends; desk front
   edge from light-wood colour at z 0.70-0.80.
2. `square`: turn the base by the chair yaw (if > 0.06 rad), lane change (back, turn ±35°, drive, turn back to the
   stored odometry yaw) to centre the bar.
3. `grasp` (torso forced to ZERO first: `ensure_torso_zero`): hands at bar centre ±0.10 m; each gripper yawed to
   the local bar tangent (yaw = -atan(dx/dy)), pitched -45°; ee = bar centre - (0.20 - GRASP_DEEPER 0.04)·f
   (f = finger direction; v3: 0.04 deeper so the fingers close around the whole bar, not its top edge); pre-grasp
   0.10 back along f and 4 cm up. v3: pre-grasp of BOTH hands in one whole-body optimal-control problem
   (oc-hand-reach: torso held, elbow-down posture prior, <= 0.35 rad per call; no IK seed search), descent by two
   Cartesian translations (torso pinned), close left 0.4 A then right 0.4 A; a wrist joint over 85 % -> open
   that side to 4 cm and re-close (0.25 A in the step; while turning with the chair re-close FIRM at 0.4 A:
   `chair_moves.turn_to_relieved`). v4: a hot WRIST ROLL (arm_6) after both closed -> `relieve_wrist_roll`: open
   that side to 4 cm, roll its joint 6 by 0.11 rad in the direction the grip pushed it, re-close 0.4 A, undo and try
   the other direction once if the load did not drop (trial 2: left_arm_6 108-113 % -> 35-45 % twice). The two-hand
   pre-grasp settles 3-7 mm off (accepted up to 12 mm); the descent goes to ABSOLUTE grasp targets.
4. Move the chair with the base only: `pull` (right hand caged at 4.5 cm, left grips), `turn_to` absolute odometry
   yaw (0.15 rad steps land ~5 % short, so it re-measures and corrects), `drive`, lane changes. Every step checks
   loads (stop > 110 %) and wrist force change (stop > 55 N).
5. `tuck [desk_front_x bar_x]`: push by desk_front_x - bar_x - 0.42 m (0.10 m steps, the last 0.10 m in 3 cm
   steps, stop on a wrist force rise of 30 N). v4: while holding the chair, measure the desk front yourself (pixel
   depth of its front edge, robot square to it) and pass both values - the stored ones come from the last `measure`,
   which may have been at another place.
6. `release`: open, back out 0.10 m along f and 4 cm up, arms to ready by split joint moves (v4; the optimal-control
   hands_up from here tripped left_arm_6 at 168 %). 7. `verify`: gap, centring, yaw + checkpoint.

## Code
`code/run.py:main(ctx, step, args)` -> `run_tuck.main` (state in `output/push_chair_state.json`); steps: find,
approach [plan], desk, measure, square, grasp, transport [pull_back] [plan], pull, turn_to, drive, tuck, release,
verify. `find`/`approach`/`desk`/`transport` (v2) are planned with the axle model (turns pivot about the drive
axle at +0.228 m) and still need their first end-to-end test. Modules:
`chair_moves.py` (v3: `turn_to_relieved(ctx, yaw)` = absolute odometry turn holding the chair in 0.15 rad steps,
relieving a wrist joint > 90 % (open to 4 cm, re-close at 0.4 A) and checking `lost_grip` (an opening < 12 mm =
closed on air) after every step), `chair_ctl.py` (primitives: `bar_profile`, `bar_grasp_targets`, `grasp_bar`,
`ensure_torso_zero`, `move_arms_joint` (= oc-hand-reach move of one or both hands; name kept), `release`, `hands_up`,
`base_drive`, `base_turn`, `base_turn_to`, `pull`, `push`, `lane_change`, `measure`, `chair_yaw`), `perception.py`
(`desk_front`, Gemini `chair_pose` / `desk_edge` for coarse localization from afar), `geom.py` (depth -> base frame),
`kin.py` (pinocchio IK). Parameters (tested values): `HAND_HALF` 0.10, `FINGER` 0.20, `PITCH` -45°, `GRIP_A` 0.4,
`GRIP_SOFT_A` 0.25, `RELIEVE_PCT` 85, `ABORT_PCT` 110, `WRIST_ABORT_N` 55, `TUCK_GAP` 0.42 (bar front surface to desk
edge with the armrests just under the edge). Every base command retries once on the transient SDK error
"This stream is expired" (seen on ~every second base call on 2026-10-09).

## Checks
- measure: profile ok, bar span ≥ 0.3 m, curvature > 0 (inverted = contaminated by fingers), z_top 1.0-1.05.
- grasp: both widths 15-30 mm with `blocked_by_object`, wrist loads < 100 %; closing on air (< 8 mm) -> open, stop.
- moves: loads < 110 %, wrist Δ < 55 N per step; pulling with both grips overloads right_arm_4 -> `pull` cages it.
- tuck success: `verify` bar-to-desk-edge 0.42 ± 0.06 m, bar centre |y| < 0.10 m, image shows the chair under the
  desk, square, armrests at/under the edge.

### Perception checkpoints
| Step | Subagent model | Frames | Question | Expected answer | Pass if | On fail |
|---|---|---|---|---|---|---|
| measure | haiku | `measure` frame | Is the office chair directly in front of the robot with its backrest towards the camera, and is anything (person, object) between the robot and the chair? | yes/no + what is in between | chair in front, backrest towards camera, nothing between | reposition (drive/turn) and measure again |
| before every base turn / reverse | agent (you) | head pan ±0.47 / chest look-around frames | Is the floor clear where the robot's rear (r 0.62 m) and the held chair (≈1.0 m) sweep? | clear / obstacle + where | clear | plan another sequence (lane change, smaller turn) |
| grasp | haiku | `grasp` frame (head 0, 0.75) | Are both robot grippers closed around the top bar of the chair backrest? | yes/no per gripper | yes for both (widths 15-30 mm agree) | open, back out, measure, retry once |
| near furniture | agent (you) / operator | stream / observe | Are the chair's casters or armrests touching the desk legs or edge? | yes/no | no | back off 0.1 m, lane change away, retry |
| verify | haiku | `verify` frame | Is the office chair pushed under the light-wood desk, facing it squarely, armrests at or under the desk edge, roughly centred? | yes/no + angle / offset | yes | pull out 0.3 m, square, re-centre, tuck again |

## Failure modes
- Fingers modelled with the URDF length (0.073-0.09 m) hit the bar (left wrist +45 N): use 0.20 m.
- Two rigid grips on the curved bar with yaw 0: closing the second gripper twisted right_arm_4 to 108-153 %
  (watchdog re-target): fixed by tangent-matched yaw (both grips then ≤78 %).
- Changing gripper yaw with the Cartesian controller tripped left_arm_6 (8 N·m) at 154-173 %, even in free air:
  change yaw only with joint moves; lateral arm shifts of the held chair also tripped it -> use base lane changes.
- Left hand reaching across to y < -0.05 at torso zero: left_arm_1 at its limit, command timed out.
- Tucked fingertips (0.20 m) hang at z≈0.76: rotating the chest near a desk swept them into the desk top (+50 N).
- Pulling with both grips: right_arm_4 98-113 % -> `pull` cages the right hand.
- Casters / armrests caught on the desk's leg while parking beside the desk (operator stop): keep ≥0.15 m
  lateral clearance from desk legs; lane change sideways.
- Transient SDK "This stream is expired" on base moves: one retry each (handled).
- v3 (compositional session, 2026-10-09): the chair slipped out of two 0.25 A grips during a 90° turn and the turn
  loop kept turning (operator: "you lost the chair") -> firm 0.4 A re-closes and `lost_grip` checks every step.
- Fingers bit only the bar's top edge (widths 9-13 mm, operator: "missed again", "be deeper") -> GRASP_DEEPER 0.04
  (then 26-35 mm, whole bar). A bar centre taken from the flat top was biased to one end: centre on the full
  visible span.
- Grasp started at torso `ready`: pre-grasp 9 cm low, the pinned-torso descent drove the fingers under the bar ->
  `ensure_torso_zero` at the start of `grasp` and `measure`.
- v3's local IK with 3 seeds (current, mirrored other arm, ready) once returned a flipped-shoulder solution (joint1
  -3.1); the Cartesian descent from there timed out 1 cm short (closed there by hand: 28.5 mm, ok) -> pre-grasp and
  hands-up now by the controller's optimal control (oc-hand-reach), no seed search. hands_up's exact -45° pitch at
  (0.40, ±0.30, 1.30) needs the wrist at its stop: hands_up / measure poses use a loose orientation (w_rot 0.3).
- Base turns near furniture: an unrelated 151° turn swung the base's rear into a cabinet; check the swept footprint
  (rear r 0.63 m about the axle, plus the held chair) against the furniture before every turn.
- v4 (trial 2, 2026-10-10): the two-hand optimal-control pre-grasp settled 5.5 / 6.5 mm off (tolerance 5 mm ->
  RuntimeError) -> 12 mm accepted, absolute descent. Closing the second gripper put left_arm_6 at 108-113 % (twice);
  the old relief re-closed the side just closed (the right) -> `relieve_wrist_roll` on the hot side. The release's
  optimal-control hands_up stopped at left_arm_6 168 % -> joint moves. `find` returned another chair (bar width
  0.049 m) -> locate the chair yourself by pixel depth of its backrest bar and casters, plan the move behind it,
  then `measure`. Wheel odometry drifted ~0.45 m over the session: plan every route from fresh frames (polygons
  of desk / cabinet / chair from pixel depth at the current pose), not from old odometry outlines. The desk's
  front-edge angle from a far left-end view was 0.3 rad off: re-measure it from 1.3-1.9 m before the final tuck.

## Evidence
| date | session | run | outcome | unattended | conditions |
|---|---|---|---|---|---|
| 2026-10-09 | push_chair_4ab505 | M1 manual | tucked | no | chair at the desk's left end facing away; many operator hints |
| 2026-10-09 | push_chair_4ab505 | P1 controller steps | parked left of the desk against the wall | no | 1 operator stop (casters vs desk leg) |
| 2026-10-09 | push_chair_4ab505 | P2 controller steps | tucked, square, centred, gap 0.419 m | no | from the parked, 20° yawed chair; steps chained by the agent |
| 2026-10-09 | compositional_manipulation | C1 controller steps | parked left of the desk facing the wall | no | chair lost once in a turn (soft grips), recovered; v3 fixes from it |
| 2026-10-09 | compositional_manipulation | C2 controller steps | tucked, centred, square | no | after a box/cup task; grasp #1 fingers under the bar (torso ready), #2 closed at the stopped pose |
| 2026-10-09 | compositional_manipulation | O1 free-air test of the v3 arm moves | pre-grasp 0.3 mm, hands_up 1.7 mm | yes | synthetic bar 0.62 m ahead, no chair; see oc-hand-reach evidence |
| 2026-10-10 | compositional_manipulation | C3 trial 2 steps (park) | parked left of the desk near the wall | no | v4 fixes made during this run (pre-grasp tolerance, wrist-roll relief by hand, joint release) |
| 2026-10-10 | compositional_manipulation | C4 trial 2 steps (tuck) | tucked, square, centred within 2.5 cm | no | chair located by the agent (find picked another chair); routes from fresh frames; tuck with agent desk / bar values |

![start](evidence/images/2026-10-09-M1-start.jpg)
![grasp](evidence/images/2026-10-09-M1-two-hand-grasp.jpg)
![parked](evidence/images/2026-10-09-P1-end-parked-left.jpg)
![tucked](evidence/images/2026-10-09-P2-end-tucked.jpg)
![parked (compositional C1)](evidence/images/2026-10-09-compose-parked-left.jpg)
![tucked (compositional C2)](evidence/images/2026-10-09-compose-tucked.jpg)
![parked (trial 2)](evidence/images/2026-10-10-compose2-parked-left.jpg)
![tucked (trial 2)](evidence/images/2026-10-10-compose2-tucked.jpg)
