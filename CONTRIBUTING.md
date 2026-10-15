# Contributing skills

This is the protocol for reading, adding and improving skills in this repository. It is
the same protocol the agents follow on the robot (the
[`continual-learning`](https://github.com/robot-agent-exploration/rax/tree/main/skills/continual-learning)
skill of the [RAX](https://github.com/robot-agent-exploration/rax) harness), written for
any contributor: an agent session, a harness that merges sessions, or a person.

## How the repository is organized

- `main` holds this protocol, the [README](README.md) and the checker
  [`scripts/check_skills.py`](scripts/check_skills.py). It holds no skills.
- **One branch per robot embodiment** holds that robot's skill library, for example
  `franka` (Franka FR3, one arm) and `rby1` (Rainbow RB-Y1, mobile, two arms). A skill is
  written for one embodiment: its code calls that robot's tools and its evidence comes from
  that robot. A procedure that also works on another robot is contributed again on that
  robot's branch, with that robot's evidence. A robot without a branch gets one through
  [NEW_ROBOT.md](NEW_ROBOT.md).
- Every embodiment branch has the same layout, which is also the layout of the harness's
  local library, so a branch can be cloned straight into the harness:

```
README.md                         the robot, its interface, where its skills came from
INDEX.md                          one row per skill
skills/<slug>/
  SKILL.md                        frontmatter + the seven sections
  code/<entry>.py                 the entry: main(ctx, ...) -> {ok, measured, why}
  code/*.py                       helpers it imports (relative to code/)
  evidence/runs.jsonl             one JSON object per run of the code
  evidence/images/*.jpg           visual references (at most 400 kB each, e.g. 640 px JPEG)
.github/workflows/check-skills.yml  runs the checker from main on pushes and pull requests
```

Keep three kinds of knowledge apart:

| Where | Written by | Holds | Lifetime |
|---|---|---|---|
| the harness's skills (`.agents/skills/`) | people | how to use the robot, its tools, the workflow; read-only | every session |
| the session's `memory/` (tips, progress, runs) | the agent | this task's notes, attempts, frame links | this task |
| this repository | agents and people, through merges | verified, executable procedures for tasks and sub-tasks | every later session |

The harness skills and the live tool schemas take precedence over a library skill: a
library skill never changes a limit, a safety rule or what a tool does.

## 1. Using a skill

1. Read the branch's `INDEX.md`. **Load only the skills relevant to the task**: for each
   part of the task, open the skills whose **When to use** matches it, plus the skills they
   list in `uses:`. Unrelated skills cost context and can mislead. Prefer `verified` over
   `draft`; read `deprecated` only to learn why it was retired.
2. Before running one, check its **Preconditions** against a fresh observation, and its
   tested ranges against your scene. A skill tested on 0.3 m boxes is not evidence for a
   0.6 m box.
3. Run its entry and read the returned `{ok, measured, why}`; confirm the result with your
   own observation. A skill that fails is information: record the run as a failure
   (section 4) and say what differed.
4. A skill may be older than the tools you have. Where a newer tool does a step better
   (for example the controller's whole-body IK in place of a hand-written solver), upgrade
   the skill, bump its `version`, and re-verify it.

With the RAX harness, the session's workspace holds a clone of the library in
`skill-library/`; run a skill's entry from the workspace root:

```python
import importlib.util, json
from api.runtime import Context

spec = importlib.util.spec_from_file_location("skill", "skill-library/skills/<slug>/code/run.py")
skill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skill)
with Context() as ctx:
    print(json.dumps(skill.main(ctx, **{"param": "value"}), indent=1))
```

## 2. What to contribute

Contribute a skill when a procedure **worked on the robot** and is useful beyond that exact
scene: a task or a sub-task with a clear start and end ("pick a box from the floor with
both hands", "place a held object on a table", "localize a cardboard box"). Also
contribute:

- an **improvement** to an existing skill: wider tested conditions, better parameters, a
  new failure mode and how the code now detects it;
- a **draft** of a procedure that is automated but not yet verified, so the next session can
  finish it instead of starting over;
- a **deprecation** of a skill that proved wrong or was replaced (status `deprecated`, why,
  and what to use instead). Never delete another contributor's skill or evidence.

One folder is one skill. Name it by what it does, or by the task it solves, in lower-case
words with hyphens (`pick-box-floor-two-hands`, `peg-insertion`). A task-level skill may
call sub-task skills and declare them in `uses:`.

## 3. Make skills modular, so new tasks can compose them

A skill is reused on the **same robot** in setups its author never saw: other camera
placements and calibrations, other rooms, surfaces and layouts, other objects of the same
kind, and other tasks that call it as one step. Write every skill so that a later session
can reuse its pieces for a task nobody has seen yet: "pick a box from the floor" should
serve "put the box on a shelf", "hand the box to a person" and "stack two boxes" without
being rewritten.

**Keep the manipulation logic; take the setup from the harness.** The skill holds what
carries over: the steps, the motion and contact strategy in the object's or target's
frame, and the checks between steps. The environment setup (camera parameters, the
robot's postures, limits and workspace) is read from the harness at run time, never copied
from the session that wrote the skill; what the harness does not provide is measured or
passed in.

**Keep hyper-parameters in one list**, separate from the logic (for example
`code/params.py`), each with what it controls, its unit, its tested range and how it was
set, so a new task or setup knows where to adjust it. Say in `## When to use` and in each
run's `conditions` which setups were tested.

**Cut the task into primitives**: perception (frame -> pose and size), positioning (reach
pose), grasping, transport, placing, release and retreat. Split where the robot is in a
stable, describable state; never in the middle of a contact.

**Give each primitive a contract**, in its SKILL.md (`## Code`) and its docstring:

- inputs as parameters in base-frame metres / radians / named sides, never constants tied
  to one scene: the object's pose and size, which arm, the target height, speeds and
  forces with their tested ranges;
- the state it needs at the start (posture, gripper, what is held, what must be visible)
  and the state it leaves at the end: the end state of one primitive is the start state the
  next one checks;
- its return value `{ok, measured, why}`, with what the next step needs in `measured`;
- on failure: stop in a safe state, say which precondition or check failed, and leave
  enough in `measured` for a caller to retry or recover.

**Keep the code composable**:

- one function per primitive, `def primitive(ctx, ..., log=print) -> dict`, with no
  globals, no hidden state and no dependence on the working directory; perception helpers
  take a frame and return data without moving the robot; motion helpers do one thing;
- the entry's `main(ctx, **params)` only chains the skill's own primitives; anything a
  different task could use lives in its own module or its own skill;
- reuse instead of copying: import another skill's functions and declare it in the
  frontmatter (`uses: [localize-box]`; the checker verifies those skills exist):

  ```python
  import importlib.util
  from pathlib import Path
  SKILLS = Path(__file__).resolve().parents[2]          # skills/

  def load(slug, module="run"):
      path = SKILLS / slug / "code" / f"{module}.py"
      spec = importlib.util.spec_from_file_location(f"{slug.replace('-', '_')}_{module}", path)
      mod = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(mod)
      return mod
  ```

- changing a primitive others use: keep its signature and contract, or add a new function
  and leave the old one working; bump `version` and say what changed. Check `uses:` in the
  other skills first.

**Prefer the robot's own tools inside primitives** (whole-body IK, Cartesian moves,
guarded moves, the planning tools) over hand-written solvers: they are measured, and they
improve without the skill changing.

**Leave room for the agent's perception.** A skill runs in scenes its author never saw, so
it must not be a blind script:

- each primitive re-observes what it acts on and returns the frame ids it used;
- keep geometry in code (depth, pixel to position, kinematics) and leave semantically rich
  judgments (which object is the target, whether a grasp holds, why it failed) to the agent
  or a perception subagent at marked **checkpoints**;
- list every checkpoint in a `### Perception checkpoints` table under `## Checks` (write
  "None" if there are none): step, subagent model (a fast model for structured checks, a
  stronger one for diagnosis), frames, the exact question, expected answer, pass criterion,
  action on fail. The code returns the same brief with its result:

  ```python
  "checkpoint": {
      "subagent": "haiku",
      "frames": ["runtime/frames/<id>/head_camera.jpg"],
      "question": "Are both fingertips below the box's top edge, one at each end face?",
      "answer": "yes | no, plus the pixel of each fingertip",
      "pass_if": "yes for both fingertips",
      "on_fail": "back off 3 cm outward, re-localize the end faces, retry once; then stop",
  }
  ```

- inputs that come from perception are parameters the agent can supply from its own
  verified observation, with the skill's own localization as the default;
- when a check is ambiguous, stop safely and return the frames and `why` instead of
  guessing.

## 4. The format (checked by `scripts/check_skills.py`)

`INDEX.md` row: `| [<slug>](skills/<slug>/SKILL.md) | <status> | what it does | when to use |`.
The status in the row matches the one in `SKILL.md`.

`SKILL.md`:

```markdown
---
name: <slug>                       # equals the folder name
description: <one line: what it does and when to use it>
status: draft                      # draft | verified | deprecated
robot: <the embodiment, e.g. Franka FR3>
entry: code/run.py                 # the .py file that defines main(ctx, ...)
version: 1                         # bump when behaviour changes
uses: [localize-box]               # optional: skills of this branch it imports
---

# <Title>

## When to use
Observable conditions: the object (kind, size range, weight if known), where it is, the
scene (clutter, lighting), the robot's start posture. Also when NOT to use it.

## Preconditions
What the code checks before moving, and how (tools, thresholds). The start posture,
gripper state, what must be visible.

## Procedure
The steps in order, each with its tool calls and the reason for its parameters.

## Code
The entry and how to call it; every parameter with its tested range and default; what it
returns in `measured`; where it logs.

## Checks
The completion check of each step (image, depth, gripper width, force, joint state) and
the final success check, with thresholds.

### Perception checkpoints
| Step | Subagent model | Frames | Question | Expected answer | Pass if | On fail |
|---|---|---|---|---|---|---|

## Failure modes
Each failure seen: how it shows, how the code detects it, what it does, and what the
operator or a later session should do.

## Evidence
A table of runs (date, session, outcome, unattended, conditions) summarizing
`evidence/runs.jsonl`, and the visual references: `![start](evidence/images/<file>.jpg)`.
```

The code:

- re-localizes everything it acts on from fresh observations;
- checks its preconditions first and stops with `ok: False` and a clear `why` when one
  fails; checks each step before the next; ends in a safe state on failure;
- uses the robot only through `ctx` (the session's tools) and reads the setup from it;
  hyper-parameters in their one list (section 3); no absolute paths, no credentials; only
  the agent environment's packages;
- returns `{"ok": bool, "measured": {...}, "why": str}`.

Each line of `evidence/runs.jsonl` (append; never rewrite another contributor's lines):

```json
{"session": "<session name>", "date": "<YYYY-MM-DD>", "commit": "<commit of the code that ran>",
 "outcome": "success", "unattended": true,
 "conditions": "cardboard box 0.30 x 0.20 x 0.15 m on the floor 0.9 m ahead, robot at ready",
 "checks": {"box_lifted_m": 0.12, "opening_m": 0.21},
 "images": ["evidence/images/2026-10-08-start.jpg", "evidence/images/2026-10-08-end.jpg"],
 "notes": "optional"}
```

- `outcome` is `success` or `failure`. Record failures too: they define the skill's
  limits.
- `unattended: true` only when the code ran from start to end with no help from the agent
  or the operator (a scene reset before the run is fine).
- `commit` is the commit of the code that ran. For runs recorded before the code was in
  the library (a skill imported from a session), write `"pre-library"` and say in `notes`
  how the code that ran differs from the committed code.
- `images`: at least the start and end frame of the run, downscaled, under
  `evidence/images/`.

**Status** follows the evidence:

| Status | Needs |
|---|---|
| `draft` | the seven sections, a compiling entry with `main(ctx, ...)`, at least one image |
| `verified` | additionally at least 2 unattended successful runs **of the committed code**, with images, ideally on different layouts or sessions |
| `deprecated` | why, and what replaces it |

When you change a verified skill's behaviour, bump `version`; it stays verified only if
new unattended runs of the new code meet the bar, otherwise set it back to `draft`.

Before committing, run the checker on your checkout of the embodiment branch:

```bash
curl -sO https://raw.githubusercontent.com/robot-agent-exploration/skills-lib/main/scripts/check_skills.py
python3 check_skills.py . --base upstream/<embodiment>  # only what you changed, plus the contribution rules
python3 check_skills.py .                               # the whole library
```

## 5. Commit and merge

A contribution changes only `INDEX.md` and files under `skills/`, and deletes no skill.

**From a session or by hand (GitHub).** Fork the repository on GitHub, then work on a
branch of your fork that starts from the embodiment branch (a fork copies only `main` by
default, so take the embodiment branch from `upstream`):

```bash
git clone https://github.com/<you>/skills-lib.git && cd skills-lib
git remote add upstream https://github.com/robot-agent-exploration/skills-lib.git
git fetch upstream
git checkout -b contrib/<embodiment>/<session-or-topic> upstream/<embodiment>
# add or change skills/<slug>/ and its INDEX.md row
git add -A && git commit -m "<slug>: what changed"
git fetch upstream && git merge upstream/<embodiment>     # take in other merges; resolve conflicts
python3 check_skills.py . --base upstream/<embodiment>
git push -u origin contrib/<embodiment>/<session-or-topic>
```

Then open a pull request from that branch into `<embodiment>` of
`robot-agent-exploration/skills-lib` (never into `main`). The workflow runs the checker on
it; a first-time contributor's run waits for a maintainer's approval. Pull requests are
merged with a **merge commit**, not squashed or rebased, so each session's commits and
evidence stay in the history. Maintainers may push `contrib/` branches to this repository
directly instead of a fork.

On a conflict, merge `upstream/<embodiment>` into your branch, keep **both** sides'
`runs.jsonl` lines, reconcile the text (the higher `version` wins only when its evidence
supports it), and commit.

**From the RAX harness.** The harness keeps a local library per robot
(`<artifact root>/skill-library/<robot>/`, or `$RAX_SKILL_LIBRARY/<robot>`; branch
`main`) and merges each session's `session/<session>` branch into it with the tool
`skill_library_merge`, which applies the same checks. To start it from this repository and
to publish what the sessions learned:

```bash
git clone -b <embodiment> -o upstream https://github.com/robot-agent-exploration/skills-lib.git \
    runtime/skill-library/<robot>
git -C runtime/skill-library/<robot> checkout -b main
# ... sessions run and merge ...
git -C runtime/skill-library/<robot> remote add fork https://github.com/<you>/skills-lib.git
git -C runtime/skill-library/<robot> push fork main:contrib/<embodiment>/<lab-or-date>
```

and open a pull request into `<embodiment>` as above. The harness's merge commits
(`Merge session/<session>: ...`) carry over.

Merge at least when a skill is first automated (as `draft`), when it becomes `verified` or
changes, and always before a session ends, so nothing learned is lost.

## 6. Importing skills from earlier sessions

Sessions that ran before this library existed kept their procedures in the session's
`memory/` and `code/`. To import one:

- refactor it into the format above: one folder per task or sub-task, the code under
  `code/` with an entry that defines `main(ctx, ...)`, the session's notes distilled into
  the seven sections;
- keep the code as it ran on the robot; limit refactoring to what the format needs (paths,
  an entry wrapper, removing one-off scripts) and say in `## Code` what changed since the
  runs;
- note in `## Code` what still depends on the session's setup and where the
  hyper-parameters are (section 3);
- take the runs from the session's records into `evidence/runs.jsonl` with
  `"commit": "pre-library"` and frames from the session as images;
- set the status from that evidence: runs of code that has since changed do not count
  toward `verified`.

## 7. Adding a robot

A robot without a branch starts with an issue: a maintainer creates its branch from
[`templates/robot-branch/`](templates/robot-branch/) with
[`scripts/new_robot_branch.sh`](scripts/new_robot_branch.sh), and the first skills come in
by pull request as in section 5. The steps, and when a robot needs its own branch, are in
[NEW_ROBOT.md](NEW_ROBOT.md).
