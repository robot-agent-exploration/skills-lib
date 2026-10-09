# Skill index

| Skill | Status | What it does | When to use |
|---|---|---|---|
| [bimanual-box-pnp](skills/bimanual-box-pnp/SKILL.md) | draft | pick and place a ~0.4 m carton between the floor and a black-top table with the bimanual closed-ball squeeze: one modular package (core / perception / motion / tasks), tasks floor_to_table and table_to_floor, Gemini-seeded perception (box, table-top mask, edge) with depth for metres, perception-checked base motion, checkpoint briefs for subagents / Gemini, STOP-file monitoring | a closed box (0.36-0.42 m long side) on the floor or on a 0.65-0.75 m dark table top in the same room; agent at the checkpoints |
| [bimanual-box-squeeze](skills/bimanual-box-squeeze/SKILL.md) | deprecated | squeeze a floor box between both closed grippers from the lowest squat and lift it with the torso | replaced by bimanual-box-pnp (kept for its evidence and notes) |
| [box-floor-to-table-squeeze](skills/box-floor-to-table-squeeze/SKILL.md) | deprecated | v1 floor -> table pick, carry and place with agent checkpoints | replaced by bimanual-box-pnp (kept for reference, v1 code in code/rby1_tasks) |
