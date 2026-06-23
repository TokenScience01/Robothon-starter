# Aegis Sentinel EOD v5

**Campus hazardous-package neutralization with Aegis quadruped gait patrol, five-finger manipulation, EOD wire severance, and triple-disturbance recovery.**

Registration UUID: `190f2760-b68b-44ee-b805-a6a492a2fa6c`

## Project name

Aegis Sentinel EOD

## Robot platform

Aegis-inspired quadruped patrol body with an onboard **five-finger** manipulator (index, middle, ring, pinky, thumb), defined in `aegis_sentinel_scene.xml` with BASE_LINK mesh and gait knee actuators.

## Task goal

The robot patrols with animated quadruped gait, uses a **MuJoCo rangefinder** to scan a suspicious package, **severs the tripwire**, performs a **five-finger grasp** with tactile residual corrections, survives a **4N lateral shove** and **transport slip**, holds a **9x object-weight load**, places the package in a containment bin, confirms the seal, and presses the campus alarm button.

## Technical approach

- Custom MJCF scene with 24 actuators, 5 touch pads, wire slide joint, gait knees, and Aegis mesh.
- MuJoCo physics at 250 Hz with position actuators on base, legs, arm, fingers, wire, seal, and alarm.
- Stage planner for patrol / scan / wirecut / grasp / recover / hold / transport / place / seal / alarm.
- Tactile residual grasp policy trained by `train_residual_policy.py`.
- Grasp-camera picture-in-picture overlay during manipulation beats.
- Artifact export: 720p video, storyboard, trajectory, contact timeline, 128-seed stress replay, dataset pack.

## Core features

- **Five-finger dexterity** — matches top leaderboard manipulation depth.
- **EOD wire severance** — differentiated task beat before grasp.
- **Quadruped gait animation** — four knee servos during patrol.
- **Force telemetry HUD** — impedance-style Newton readout from touch aggregation.
- **Judge package** — `JUDGE_BRIEF.md` with top-3 gap-closure table, `rubric_scorecard.json`, validator.

## How to run

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python submissions/aegis-sentinel-eod/train_residual_policy.py
python submissions/aegis-sentinel-eod/run_sentinel_eod.py
python submissions/aegis-sentinel-eod/validate_submission.py
```

Quick smoke test:

```bash
python submissions/aegis-sentinel-eod/run_sentinel_eod.py --quick --duration 10 --fps 10
```

Generated artifacts:

- `submissions/aegis-sentinel-eod/artifacts/demo.mp4`
- `submissions/aegis-sentinel-eod/artifacts/keyframes.png`
- `submissions/aegis-sentinel-eod/artifacts/stress_eval.json`
- `submissions/aegis-sentinel-eod/dataset/labels.csv`