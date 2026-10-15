# Robot agent skill library

Task skills that robot agents learned on real robots, one library per robot embodiment.
Each skill is an executable procedure the agent wrote while it explored the task, kept
with its preconditions, checks, failure modes and the evidence of its runs: run records
and camera frames. A later session, on a new task or in a new scene, reads the library,
runs or composes the skills that match, and contributes back what it improves.

The skills come from sessions of [RAX: Scaling Online Exploration for Robot
Agents](https://robot-agent-exploration.com/) ([code](https://github.com/robot-agent-exploration/rax)),
where a coding agent (Claude Code or Codex CLI) drives a robot through perception and
control tools, with no demonstrations, and compiles what worked into code.

## Branches

`main` holds the protocol. **Each robot embodiment has its own branch**, with one folder
per skill:

| Branch | Robot | Skills |
| --- | --- | --- |
| [`franka`](https://github.com/robot-agent-exploration/skills-lib/tree/franka) | Franka FR3: one 7-DoF arm, Franka Hand, two exterior cameras and a wrist camera | find the cube, air fryer, wipe the table, peg-in-hole, Bandu tower |
| [`rby1`](https://github.com/robot-agent-exploration/skills-lib/tree/rby1) | Rainbow RB-Y1: wheeled base, 6-joint torso, two 7-DoF arms, grippers, head ZED 2i | box, bucket and chair moves, cup stacking, whole-body hand reach, a six-step room tidy |

The robots' harness branches in the RAX repository describe their tools:
[`real/franka`](https://github.com/robot-agent-exploration/rax/tree/real/franka) and
[`real/rainbow`](https://github.com/robot-agent-exploration/rax/tree/real/rainbow).

## Layout of an embodiment branch

```
README.md                    the robot and where its skills came from
INDEX.md                     one row per skill: status, what it does, when to use it
skills/<slug>/
  SKILL.md                   when to use, preconditions, procedure, code, checks,
                             failure modes, evidence
  code/                      the entry main(ctx, ...) -> {ok, measured, why} and its helpers
  evidence/runs.jsonl        one record per run of the code: outcome, unattended, conditions, checks
  evidence/images/           frames of those runs
```

A skill's status follows its evidence: `draft` (automated, documented), `verified` (at
least two unattended successful runs of the committed code) or `deprecated` (with what
replaces it).

## Using a skill

```bash
git clone -b rby1 https://github.com/robot-agent-exploration/skills-lib.git skill-library
```

Read `INDEX.md`, open only the skills that match your task, check their preconditions
against your scene, and run the entry through the robot's harness context (`ctx`). The
RAX harness clones the library into each session's workspace and gives the agent the
[`continual-learning`](https://github.com/robot-agent-exploration/rax/tree/main/skills/continual-learning)
skill to read, run and grow it.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md): what to contribute, how to keep skills modular,
the format and evidence rules, and how to commit and merge into an embodiment branch.
Check a contribution with [`scripts/check_skills.py`](scripts/check_skills.py) before
opening a pull request; the embodiment branches run it on every push and pull request.
