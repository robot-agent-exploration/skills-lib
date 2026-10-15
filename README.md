<div align="center">

<img src="https://raw.githubusercontent.com/robot-agent-exploration/rax/main/docs/assets/rax-mark.svg" alt="RAX logo" width="72">

# RAX Skill Library

**Real-robot task skills written by robot agents, one library per embodiment.**

[![RAX](https://img.shields.io/badge/Harness-RAX-00a3a6)](https://github.com/robot-agent-exploration/rax) [![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](https://robot-agent-exploration.com/rax.pdf) [![Project WebPage](https://img.shields.io/badge/Project-webpage-00a3a6)](https://robot-agent-exploration.com/) [![Format](https://img.shields.io/badge/format-check__skills.py-lightgrey)](scripts/check_skills.py)

</div>

---

[RAX](https://github.com/robot-agent-exploration/rax) is the general harness: a coding
agent (Claude Code or Codex CLI) drives a robot through perception and control tools, with
no demonstrations, and compiles what worked into code. **This repository keeps what those
agents learned on real robots.**

Each skill is an executable procedure the agent wrote while it explored a task, stored
with its preconditions, checks, failure modes and the evidence of its runs (run records
and camera frames). A later session, on a new task or in a new scene, reads the library,
runs or composes the skills that match, and merges back what it improves.

```
 explore a task ──► write a skill ──► merge into the library
        ▲                                     │
        └──── next session reads, reuses ◄────┘
              and composes the skills
```

## 🌿 Libraries

`main` holds the protocol. **Each robot embodiment has its own branch**, with one folder
per skill.

| Branch | Robot | Skills | Status |
| --- | --- | --- | --- |
| [`rby1`](https://github.com/robot-agent-exploration/skills-lib/tree/rby1) | Rainbow RB-Y1: wheeled base, 6-joint torso, two 7-DoF arms, grippers, head ZED 2i | bimanual box and bucket pick-and-place, office-chair tuck, cup stacking, whole-body hand reach, and a six-step room tidy that composes them | ✅ 6 skills |
| [`franka`](https://github.com/robot-agent-exploration/skills-lib/tree/franka) | Franka FR3: one 7-DoF arm, Franka Hand, two exterior cameras and a wrist camera | find the cube, air fryer, wipe the table, peg-in-hole, Bandu tower | 🚧 refactoring |

The Franka sessions ran before this library existed; their procedures are being
refactored into skills by the
[import path](CONTRIBUTING.md#6-importing-skills-from-earlier-sessions).

Each library's harness lives in the RAX repository, which also documents the robot's
tools: [`real/rainbow`](https://github.com/robot-agent-exploration/rax/tree/real/rainbow)
and [`real/franka`](https://github.com/robot-agent-exploration/rax/tree/real/franka).

## 🚀 Using a skill

```bash
git clone -b rby1 https://github.com/robot-agent-exploration/skills-lib.git skill-library
```

1. Read `INDEX.md` and open only the skills that match your task.
2. Check each skill's preconditions against your scene.
3. Run its entry through the robot's harness context (`ctx`).

Inside RAX this happens automatically: the harness clones the robot's library into each
session's workspace and gives the agent the
[`continual-learning`](https://github.com/robot-agent-exploration/rax/tree/main/skills/continual-learning)
skill to read, run and grow it. RAX's
[real-robot adaptation guide](https://github.com/robot-agent-exploration/rax/blob/main/docs/REAL_ROBOT_ADAPTATION.md#a-skill-library-that-persists-across-sessions)
sets this up for every new robot; `real/rainbow` is the reference.

## 📁 Layout of a library

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

A skill's status follows its evidence:

| Status | Meaning |
| --- | --- |
| `draft` | Automated and documented. |
| `verified` | At least two unattended, successful runs of the committed code. |
| `deprecated` | Superseded; names what replaces it. |

## 🤝 Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers what to contribute, how to keep skills modular
so new tasks can compose them, the format and evidence rules, and how to commit and merge
into an embodiment branch. Check a contribution with
[`scripts/check_skills.py`](scripts/check_skills.py) before opening a pull request; the
embodiment branches run it on every push and pull request.

## 📝 Citation

These skills come from sessions of
[RAX: Scaling Online Exploration for Robot Agents](https://robot-agent-exploration.com/).
BibTeX is coming soon; until then, please cite the paper as listed on the project website.
