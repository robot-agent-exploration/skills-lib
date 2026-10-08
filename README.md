# rby1 skill library

Skills agents learned on this robot, kept across sessions. Agents read and use them, and
add or improve them on their session branch; the harness tool `skill_library_merge`
checks the format and evidence and merges into `main`. The protocol is the agent skill
`continual-learning`; the rules the merge enforces:

- `INDEX.md`: one row per skill, `| [slug](skills/slug/SKILL.md) | status | what it does | when to use |`.
- `skills/<slug>/SKILL.md` begins with frontmatter (`name` = the folder name, `description`,
  `status`: draft | verified | deprecated, `robot`, `entry`: the code file, e.g.
  `code/run.py`) and has the sections `## When to use`, `## Preconditions`, `## Procedure`, `## Code`, `## Checks`, `## Failure modes`, `## Evidence`.
- The entry is Python defining `main(ctx, ...)` (ctx: `api.runtime.Context`) that returns
  `{"ok": bool, "measured": {...}, "why": str}`.
- `evidence/runs.jsonl`: one JSON object per run of the code with `session`, `date`, `commit`, `outcome`, `unattended`, `conditions`, `checks`, `images`; `images` are
  paths under `evidence/images/` (jpg / png, at most 400 kB each).
- At least one image per skill. `verified` needs at least 2 unattended successful
  runs. A branch may only change `INDEX.md` and files under `skills/`.
- Never delete another session's skill: set its status to `deprecated` and say why.
