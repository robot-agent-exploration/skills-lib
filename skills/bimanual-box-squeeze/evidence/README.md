# Evidence (RB-Y1, pnp_box session 2026-10-08)

All paths are relative to this skill folder. Each `frames/<id>/` is a complete head-camera frame:
`head_camera.jpg` (1280x720 RGB), `head_camera_depth.png` (uint16 mm, 0 = invalid) and `meta.json`
(K, T_base_cam, all joints at capture). `images/` are crops / overlays made from those frames.
Image x points to the robot's RIGHT (the image-right gripper is the right gripper).

## images/

| File | Shows | From frame |
|---|---|---|
| `round_fingers_at_ready.jpg` | real gripper with the add-on round fingers (not in the URDF), arms at `ready` | 20261008172609eef847 |
| `box_fit_overlay.jpg` | `find_box` top/bottom rectangle projected on the carton | 2026100817565548eacb |
| `open_fingers_on_box_top_edge.jpg` | OPEN fingers resting on the box's top edge ~5 cm below the URDF fingertip height (the failure that led to closing them) | 20261008181120799631 |
| `ball_calib_axis_ticks_right_gripper.jpg` | gripper axis projected with 2 cm ticks from ee (blue dot = ee): the closed ball ends ~0.20-0.22 m out | 202610081814235c15be |
| `ball_model_overlay.jpg` | modelled balls (centre 0.195 m along -z_ee, red circles) on the real grippers + depth box outline (magenta) | 20261008181657903f54 |
| `left_ball_contact_vs_backoff.jpg` | left ball pressed on the box's left end during the descent (left) vs after 3 cm back-off (right): depth had the end 3 cm too far in | 20261008181821102cec, 202610081818386fe148 |
| `right_gripper_blue_connector.jpg` | blue cable connector at the right gripper's tip | 202610081818386fe148 |
| `squeezed_box_lid_bulges.jpg` | side view during the first torso lift: squeezed lid bulges along its seam | 202610081823540dae8e |
| `black_table_target.jpg` | the black-top drawer island (place target, not reached) | 202610081830079d5060 |

## frames/

| Frame | torso_1 / chest yaw | What it shows | Used by |
|---|---|---|---|
| 20261008172609eef847 | 0.524 / 0 | start pose, box ~2.7 m ahead on the floor | images |
| 20261008175351eaa04d | 0.524 / 0 | approach, box ~1.85 m ahead | dry-run (far: stance moves) |
| 2026100817565548eacb | 0.524 / 0 | approach, box ~1.2 m ahead | images |
| 202610081800373d1e54 | 0.524 / 0 | final approach, box just beyond the robot's front | - |
| 20261008180413b03176 | 0.524 / 0 | run 1 stance, box centre ~0.45 m ahead | dry-run (at stance) |
| 20261008180733864f62 | 1.047 / 0 | squatting, box static on the floor | hold-check negative (before) |
| 20261008180751dd1d47 | 1.200 / 0 | at `lowest`, box static | hold-check negative (after) |
| 20261008181120799631 | 1.200 / 0 | open fingers on the box edge | images |
| 202610081814235c15be | 1.200 / 0 | grippers closed into balls (calibration view) | images |
| 20261008181657903f54 | 1.200 / 0 | balls 3 cm outside the box ends | images |
| 20261008181821102cec | 1.200 / 0 | left ball in contact during the descent | images |
| 202610081818386fe148 | 1.200 / 0 | after the 3 cm back-off | images |
| 20261008182114708426 | 1.200 / 0 | balls at grasp height z 0.18 beside the ends | - |
| 2026100818222501b3c5 | 1.200 / 0 | run 1 squeezed at the floor (both in contact) | - |
| 202610081823540dae8e | 1.150 / 0 | first torso lift, side view | images |
| 2026100818251493d682 | 0.950 / 0 | run 1 box held | hold-check positive (before) |
| 2026100818255517d483 | 0.750 / 0 | run 1 box held, 10.5 cm higher | hold-check positive (after) |
| 20261008182619ce6e00 | 0.750 / 0 | run 1 side view, box off the floor | - |
| 202610081830079d5060 | 0.524 / 0 | facing the black table with the box held | images |
| 202610081832293e419a | 0.524 / 0 | box upright on the floor after the drop | dry-run (close: no detour) |
| 2026100818332193d48b | 0.524 / 0 | run 2 stance, box 5 cm right | dry-run (chest yaw -0.10) |
| 20261008183550a2d2d2 | 1.200 / -0.10 | run 2 after backing off the 40 N descent contact | - |
| 2026100818383778ab5d | 0.524 / -0.10 | run 2 box held after the torso lift | - |
