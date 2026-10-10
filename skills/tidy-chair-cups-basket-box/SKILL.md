---
name: tidy-chair-cups-basket-box
description: Task composition on the RB-Y1 - move the desk chair out and park it, stack two cups, carry the stack and a banana into a floor basket, put the basket on a dark-top cabinet, tuck the chair back, put a floor box on the cabinet - chaining office-chair-tuck, stack-cups, bimanual-bucket-rim-pnp and bimanual-box-pnp with the agent at the checkpoints, plus the task-level pieces (swept-footprint check of every base move, guarded chair route, closed-form optimal-control drop into the basket).
status: draft
robot: Rainbow RB-Y1
entry: code/run.py
version: 1
uses: [office-chair-tuck, stack-cups, bimanual-bucket-rim-pnp, bimanual-box-pnp, oc-hand-reach]
---

# Tidy: chair out, cups + banana into the basket, basket and box onto the cabinet, chair back

Session `20261010T011251Z_compositional_manipulation_aa1adf` (2026-10-09), task: "move the chair in front of you
out and park it to the left of the yellow table ... stack the two silver cups together and carry the stacked cups
and banana over to the orange basket to your right and place them in, and pick up the orange basket on to the
cabinet with dark top. Then ... move the chair back and tuck it to the table again. Finally ... pick up the box on
your right and put it also on the cabinet." All six conditions were met once, step by step with the agent at every
checkpoint (operator hints listed under Evidence). Operator rules for this composition: no brute-force searches
(the controller's optimal control instead), look at the images yourself instead of writing detection code, the
agent stays in the loop (no need for full automation).

## When to use

- A lab scene like the one in the evidence images: light-wood desk with an office chair tucked under it, two
  silver cups and a banana on the desk, an orange 5-gal bucket ("basket") and a ~0.4 m carton on the floor to the
  robot's right, a dark-top drawer cabinet (one or two units, top ~0.66-0.70 m) within a few metres, open floor.
- Start: robot ~2.5 m in front of the desk facing it, torso / arms `ready`.
- Not for: unattended runs (draft), cluttered floors (every base move needs a clear swept footprint), people in
  the work area.

## Preconditions

Each library skill checks its own (see their SKILL.md). Task level: armed, no fault; the round custom fingers
(open tips ~0.20 m); a fresh look at the whole scene before each subtask; before EVERY base move a swept-footprint
check (`run.py sweep_check`) against fresh depth and the furniture you identified in the image.

## Procedure

1. Chair out (office-chair-tuck): `find` -> `approach` -> `measure` -> `square` -> `grasp` (torso zero, both hands
   by optimal control, 4 cm deeper than the tips' middle) -> route with `run.py chair_route` (pull, relieved
   turns, drive; firm 0.4 A re-closes, grip-loss check) -> push to the wall left of the desk -> `release`.
2. Stack the cups (stack-cups): scan, pick one cup, align over the other, insert, release (the head look_at
   fallback clamps the pan when the target is too far sideways).
3. Cups + banana into the basket: pick the stacked cups (pinch 4 cm deep: both walls), pick the banana
   (elongated grasp), approach the basket (bucket 0.62 m ahead), torso posture by a tested joint move, then
   `run.py drop` per hand (closed-form pose: fingers pitched -0.7 rad, yaw from the shoulder; the cups needed torso
   `lowest` + lean 0.15 with the left hand), check the image, release.
4. Basket onto the cabinet (bimanual-bucket-rim-pnp): rim pinch with both hands, LIFT AT ONCE when a pinch loads
   a wrist (operator: "stand up fast to relieve"), place on the cabinet near an edge, leaving room for the box.
5. Chair back (office-chair-tuck): find, approach, measure, square, grasp, route back, `tuck` (push to bar-to-edge
   0.42 m), release.
6. Box onto the cabinet (bimanual-box-pnp floor_to_table): find_box -> approach x2 -> grasp + lift as one chain
   (no pause while squatted) -> carry route planned and sweep-checked by the agent -> look at the cabinet yourself,
   pick the free top, measure its edge from pixel depth -> `approach_table` with `edge_x` -> `place` -> retreat.

## Code

`code/run.py:main(ctx, step, **args)` -> `{ok, measured, why}`:
- `sweep_check`: `moves` [[turn rad | drive m]], `frames` (depth frames; `frame_odom` {frame: odometry} when taken
  elsewhere), `obstacles` [[name, odometry polygon]], `extra` held-object box (x0, y0, x1, y1) in the base frame
  (chair: `chair_route.CHAIR_EXTRA`), `margin` 0.06. Turns pivot about the wheel axle 0.228 m ahead; the base
  footprint is bimanual-box-pnp's. ok = no polygon hit and <= 25 depth points in the swept area (look at what the
  points are: the robot's own arm / held object shows up as self points).
- `chair_route`: `moves` [[pull | turn | drive, value]] with the chair held.
- `drop`: `side`, `target` (held point at release, base frame), `hold` 0.20, `pitch` -0.7, `release`.
Modules: `sweep_check.py`, `scene_map.py` (depth frames -> odometry obstacle points, top-down render),
`chair_route.py`, `drop.py`. CLI: `python3 skill-library/skills/tidy-chair-cups-basket-box/code/run.py <step>
'<json args>'`.

## Checks

- Chair parked: image shows it against the wall left of the desk, not touching the desk; tucked: armrests at /
  under the desk edge, centred, square (office-chair-tuck verify).
- Cups nested: one inside the other (stack-cups depth + image).
- In the basket: the stream at the release shows each object falling inside the rim.
- Basket / box on the cabinet: upright, fully on a dark top, grippers clear (image + touchdown force).

### Perception checkpoints

| Step | Subagent model | Frames | Question | Expected answer | Pass if | On fail |
|---|---|---|---|---|---|---|
| before every base move | agent (you) | fresh head views along the swept area (torso yaw / head pan for the rear) | Is the floor clear where the base (rear r 0.63 m about the axle) and the held object sweep? Which furniture is near? | clear / what + where | clear and sweep_check ok | replan (smaller turn, reverse <= 0.3 m on the known path) |
| chair grasp | agent / haiku | grasp frame | Are both grippers closed around the whole top bar (widths 15-35 mm)? | yes / no | yes | open, back out, measure, retry |
| drop | agent | image before opening | Is the held object over the basket opening, below the rim height clearance, not touching the rim? | yes / no | yes | adjust the target by a few cm, re-check |
| cabinet top choice | agent (you) | view of the cabinet | Which dark top is free and large enough, from which side can the robot reach it without the arm passing over the basket? | top + side + pixel points | free top found | look from another side |
| box place | agent | frame after release | Box upright fully on the top, grippers clear? | yes / no | yes | re-grasp from the table (bimanual-box-pnp table grasp) |

## Failure modes

1. Chair lost in a turn with soft grips; fingers biting only the bar's edge; grasp at torso `ready`; seed-search
   IK flipping a shoulder: see office-chair-tuck v3 (all fixed there).
2. Base rear hit the cabinet during a hand-planned 151 deg turn (operator): the front-only check missed it ->
   `sweep_check` of the whole footprint before every base move. The cabinet units were pushed into an L; the box
   then went on the free unit, approached from its free side.
3. Interrupted cup alignment re-localized the held cup as the target: keep the pre-pick scan fix (stack-cups).
4. Bucket pinch overload: small corrective moves did not help; lifting at once did (bucket skill).
5. Cabinet perception by depth clustering / extra Gemini masks merged two units and wasted calls (operator: "just
   look at it yourself"): read the image, query a few pixels for depth, decide.
6. Box skill edge from the wrong (farther) cabinet when the held box hides the near top: pass `edge_x`.

## Evidence

| Run | Date | Session | Outcome | Unattended | Conditions |
|---|---|---|---|---|---|
| R1 | 2026-10-09 | compositional_manipulation | success (all 6 conditions) | no | step by step with the agent; operator hints: deeper grasp, firm grip, natural arms, stand up fast, base crash warning, "just grasp it", "look at it yourself" |
| D1 | 2026-10-09 | same | success (drop pose only, free air, no release) | yes | `run.py drop` right hand target (0.62, -0.12, 0.92): 3 calls, 23 s, 2.3 mm |

![start](evidence/images/2026-10-09-start.jpg)
![chair parked, cups on the desk](evidence/images/2026-10-09-chair-parked-cups-on-desk.jpg)
![end: chair tucked](evidence/images/2026-10-09-end-chair-tucked.jpg)
![end: basket and box on the cabinet](evidence/images/2026-10-09-end-bucket-and-box-on-cabinet.jpg)
