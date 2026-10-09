# rby1_tasks — modular RB-Y1 task code (pnp_box and compositions)

Layers (each primitive is `fn(rt, **kwargs) -> dict`, raising `StepFailed` on a failed check):

| Layer | Modules | What |
|---|---|---|
| core | `robot.py`, `kin.py`, `geom.py`, `world.py`, `runner.py` | tool wrappers + body constants; pinocchio IK/FK (ball tool 0.195 m along -z_ee); base<->odom; World (objects in the odometry frame, held object, grip data); Step runner (log, resume, report) |
| perception | `frames.py`, `box.py`, `table.py` | frame loading/projection; cardboard box on a support (depth + colour); dark table top, its sides, near edge, side clearance |
| primitives | `posture.py`, `sensing.py`, `navigation.py`, `squeeze.py`, `touch.py`, `carry.py`, `place.py` | prepare; find/relocate/find_surface/verify_on_surface; turn-drive-turn about the axle, corridor + carry-path checks, approach_for_grasp, approach_surface, retreat; bimanual ball squeeze from floor / table, releases; force touch; lift_from_floor, arm_move_z (squeeze regulated), recentre, set_down_on_floor; lower_until_contact, place_on_surface |
| tasks | `pnp_box.py`, `box_table_to_floor.py` | compositions: `steps(...) -> [Step]` |

Run from the workspace root:

```
python3 code/rby1_tasks/run_task.py pnp_box                    # full task
python3 code/rby1_tasks/run_task.py pnp_box --start find_table --world output/runs/<run>/world.json
python3 code/rby1_tasks/run_task.py box_table_to_floor          # reset: table -> floor
python3 code/rby1_tasks/tests/offline_test.py                   # perception / planning checks, no robot
```

New task = a new `tasks/<task>.py` with `steps()`; new object kind = a detector in `perception/` registered in
`primitives/sensing.DETECTORS`. Outputs: `output/runs/<time>_<task>/{log.txt, report.json, world.json}`.
