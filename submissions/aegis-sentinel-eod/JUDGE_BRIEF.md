# Aegis Sentinel EOD - Judge Brief

Registration UUID: `190f2760-b68b-44ee-b805-a6a492a2fa6c`

## Why This Entry Targets Top 10

Aegis Sentinel EOD is a **campus hazardous-package neutralization** benchmark built around a differentiated platform: an **Aegis-inspired quadruped patrol body with an onboard tri-finger manipulator**. Unlike the medicine-triage leaderboard cluster, this project combines **mobile patrol + MuJoCo rangefinder scan + tactile tri-finger grasp + containment placement + alarm confirmation** in one reproducible run.

The controller is not a replay-only animation. It steps MuJoCo physics, reads **rangefinder / frame-position / touch** sensors, and applies a **tactile residual grasp policy** during reach, grasp, lift, transport, and place.

## Inspect First

1. `artifacts/keyframes.png` - six-panel storyboard: patrol, scan, grasp, transport, bin placement, alarm.
2. `artifacts/demo.mp4` - HUD video with stage, range, active fingers, grip, residual norm, and confidence.
3. `aegis_sentinel_scene.xml` - MJCF scene with patrol platform, tri-finger hand, free package body, containment bin, alarm button, actuators, and sensors.
4. `learned_policy_weights.json` and `dataset/training_report.json` - residual policy evidence.
5. `artifacts/contact_timeline.json` - tri-finger contact samples and stable-contact count.
6. `artifacts/stress_eval.json` - fixed-seed improvement summary.
7. `artifacts/mission_report.json` - final success criteria and rubric alignment.

## Quantitative Evidence

- Platform: Aegis-inspired quadruped patrol body + onboard 3-finger manipulator
- Actuated channels: 14 position servos
- MuJoCo sensors: rangefinder, 3 touch pads, 4 frame-position sensors, alarm joint position
- Residual corrections: logged per frame during manipulation stages
- Task phases: patrol, scan, align, reach, grasp, lift, transport, place, alarm, complete
- Final success checks: package grasped, bin placement error, alarm pressed

## Rubric Mapping

- **Runnability**: one command regenerates video, trajectory, report, policy card, contact timeline, and stress replay.
- **MuJoCo depth**: custom MJCF, free package body, slide alarm joint, touch sensors, rangefinder, contacts, position actuators.
- **Task design**: campus EOD patrol with scan, grasp, containment, and alarm confirmation.
- **Control**: stage planner plus tactile residual policy driven by MuJoCo sensor streams.
- **Dexterity**: thumb-opposed tri-finger grasp, contact balancing, transport, and release.
- **Engineering quality**: validator, judge brief, scorecard, manifest, training report.
- **Presentation**: HUD video plus generated keyframe storyboard.
- **Innovation**: quadruped-plus-manipulator campus EOD benchmark, distinct from gantry-only dexterous triage entries.

## Run

```bash
python3 -m pip install -r requirements.txt
python submissions/aegis-sentinel-eod/train_residual_policy.py
python submissions/aegis-sentinel-eod/run_sentinel_eod.py
python submissions/aegis-sentinel-eod/validate_submission.py
```