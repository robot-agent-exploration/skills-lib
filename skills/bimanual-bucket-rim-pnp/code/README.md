# basket_pnp — bimanual bucket pick-and-place (rim pinch, open fingers)

Picks the orange 5-gallon bucket (Home Depot, thin wall, wire bail handle) off the floor with BOTH grippers, each
gripper OPEN around the rim wall (one finger inside, one outside) and closed on it, carries it and places it on the
black-top drawer cabinet next to the cardboard box. Closed-finger "ball" squeezes are for box-like objects only.

## Run

```
python3 skill-library/skills/bimanual-bucket-rim-pnp/code/run.py           # runs to the first checkpoint
python3 skill-library/skills/bimanual-bucket-rim-pnp/code/run.py --resume output/runs/<run> --start <next>
python3 skill-library/skills/bimanual-bucket-rim-pnp/code/run.py --resume output/runs/<run> --start plan_grasp_approach --params '{"heading_offset": 1.57}'
MPLCONFIGDIR=$TMPDIR python3 skill-library/skills/bimanual-bucket-rim-pnp/code/offline_test.py   # no robot: IK, knee, route, silhouette
```

Every run writes `output/runs/<time>_basket_pnp/` (log.txt, report_*.json, world.json, plan_*.png, overlays).
At a checkpoint the run stops and prints a brief; look at the frames (or hand them to a haiku subagent / Gemini),
then resume with `--start <next>`. `--params` applies to the step you resume at.

## Steps (steps.py)

| Step | Does | Ends with |
|---|---|---|
| prepare | armed / no fault, grippers open, torso mid, arms ready | neutral posture |
| find_bucket | Gemini segment picks the ORANGE bucket; HSV convex region; truncated-cone silhouette fit (+ depth) | world.bucket (odom), overlay |
| find_cabinet | Gemini mask of the black top + depth at table height -> rectangle, top z; Gemini point on the box | world.cabinet, world.box |
| plan_grasp_approach | stance: bucket 0.60 m ahead along the bearing (+ heading_offset); route with footprint clearance | plan picture |
| drive_grasp_approach | each segment: look along, depth points in the swept footprint <= 25, then move | at the pre-stance |
| final_approach | re-localize, straight steps until the bucket is 0.60 m ahead (|y| < 4 cm) | close view for the ears check |
| pregrasp | close-range rim (silhouette + depth rim fit), pinch IK (tilt 0.6, closing axis base y, 5 cm below the rim), arms to pre-grasp at torso mid, torso low -> lean 0.1 -> 0.2 | fingers 10 cm above the rim |
| grasp_lift | Cartesian descent (stops at the wrist-shin self-collision guard), close RIGHT then LEFT 0.35 A, pinch-load relief (3 mm hand moves), torso up with arms fixed, torso zero, hands up to ee z ~1.23 | bucket held, bottom ~0.75 m |
| plan_place | free target 0.21 m inside an edge of the top, >= 0.42 m from the box; stance facing that edge | plan picture |
| drive_place | as drive_grasp_approach, held bucket ignored in depth, obstacles below 0.6 m only | facing the edge |
| approach_edge | depth edge of the top at table height; drive until the bucket centre is 0.21 m past it (bumper >= 6 cm) | bucket over the top |
| lower_to_touch | torso squat with the arms fixed until a wrist force change > 4 N | bucket on the top |
| release | open both, hands +12 cm, arms ready, base back 0.3 m, torso mid | |
| verify | torso ready view over the bucket; Gemini yes/no | |

## Perception checkpoints

| Step | Subagent | Frames | Question | Expected answer | Pass if | On fail |
|---|---|---|---|---|---|---|
| find_bucket | haiku | find frame + `bucket_<frame>.jpg` overlay | Does the green ellipse lie on the rim of the ORANGE bucket on the floor beside the white cabinet (not a red bucket far away)? Upright? | yes / no + why | yes | point the head at the right bucket, rerun |
| find_cabinet | haiku | find frame | Is there a white drawer cabinet with a black top and a cardboard box on that top? | yes / no | yes | other looks, rerun |
| plan_grasp_approach | haiku | `plan_grasp_approach.png` + look frame | Does the swept area avoid the red outlines (except the bucket at the end)? Floor free of people/objects? | yes / no + what | yes | replan with another heading_offset |
| final_approach | agent (zoom on the bucket) | `stance_<frame>.jpg` | Are the two handle attachment points (ears) FRONT/BACK or LEFT/RIGHT of the bucket in the image? | front_back / left_right | front_back | back off and rerun plan_grasp_approach with heading_offset +-1.57 |
| pregrasp | haiku | pregrasp frame | Are both open grippers above the rim, one at its left and one at its right side, one finger in and one out? | yes / no + which | yes | raise hands, re-measure, rerun |
| grasp_lift | haiku | lift frame | Is the bucket hanging level between the grippers, clear of the floor? | yes / no | yes and held_dFz >= 6 N | set it down, redo |
| plan_place | haiku | `plan_place.png` | Does the route reach the cabinet edge straight on without crossing the outline, target far from the box? | yes / no | yes | params edge_choice |
| approach_edge | haiku | edge frame | Is the held bucket above the black top (not over the floor), clear of the box? | yes / no | yes | adjust a few cm |
| lower_to_touch | haiku | touch frame | Is the bucket standing on the top with its whole bottom on it, upright? | yes / no | yes | lift 3 cm, adjust, lower |
| verify | haiku + Gemini check | verify frame (torso ready) | Bucket upright on the black top next to the box, grippers away? | yes / no | yes | report |

## Lessons encoded (memory/TIPS.md, memory/RUNS.md)

- Two pinches on one thin rim over-constrain the bucket: a few mm of mismatch load the wrist joints (arm_5, 10 N m)
  into the 150 % guard. Close the first gripper while the bucket is free (it self-centres), relieve the second
  pinch with 3 mm hand moves, and LIFT AT ONCE: the low stretched stance is the heavy part (loads drop below 65 %
  once standing with the hands at ee z >= 1.03).
- A pinch of the thin wall reads 3-11 mm and often `blocked_by_object: false`.
- Approach so the handle ears are front/back; the hanging wire then lies under the outer finger harmlessly.
- Lower a held bucket with the torso squat (arms fixed), not the arms: arm lowering ran right_arm_5 into the guard.
- Back off >= 0.4 m from furniture before lowering: the hanging handle caught the cabinet once.
