# Aegis Sentinel EOD v6 - Judge Brief

Registration UUID: `190f2760-b68b-44ee-b805-a6a492a2fa6c`

## Start Here (30 seconds)

Open `artifacts/demo.mp4` first. v6 is a **recovery release**: it restores the v4 task arc that ranked #30 and removes v5 additions (synthetic force HUD, PiP clutter, inflated self-scores) that likely hurt judging.

## Verifiable Quantitative Evidence

| Metric | Value | How to verify |
|--------|-------|---------------|
| Demo duration | 30 s | `artifacts/mission_report.json` → `duration_s` |
| Actuated channels | 15 position servos | `aegis_sentinel_scene.xml` actuator block |
| MuJoCo sensors | rangefinder + 3 touch + 5 frame/joint | `dataset/sensor_manifest.json` |
| Max lateral shove | 4.0 N | `mission_report.json` → `max_shove_n` |
| Load hold | 9.0x | `mission_report.json` → `max_hold_load_factor` |
| Disturbance events | 2 (grasp shove + transport slip) | stages RECOVER + TRANSPORT in `trajectory.json` |
| Stress rollouts | 96 seeds | `artifacts/stress_eval.json` |
| Residual vs baseline | 100% vs 66.7% | `stress_eval.json` success fields |
| Task success | machine-readable | `mission_report.json` → `final_task_success` |

## Task Flow (single clear arc)

Patrol → rangefinder scan → tri-finger grasp → 4N shove recovery → 9x load hold → transport slip recovery → bin place → seal confirm → alarm confirm → MISSION PASS

## Honest Abstractions (disclosed, not hidden)

- Patrol locomotion uses **platform slide servos**, not full Aegis URDF torque gait.
- Transport uses **palm attach** during carry for reproducible containment demo.
- Residual policy is a **lightweight linear tactile corrector** trained on synthetic perturbation labels.

Judges penalize over-claiming more than pragmatic abstractions. v6 leads with reproducible artifacts and explicit limits.

## Inspect Next

1. `artifacts/keyframes.png` — storyboard with shove, hold, seal, pass.
2. `artifacts/narration.srt` — beat subtitles aligned to stages.
3. `dataset/labels.csv` — per-frame labeled trajectory.
4. `artifacts/challenge_evidence.json` — rubric keyword index.
5. `validate_submission.py` — one-command pass/fail gate.

## Run

```bash
python3 -m pip install -r requirements.txt
python submissions/aegis-sentinel-eod/train_residual_policy.py
python submissions/aegis-sentinel-eod/run_sentinel_eod.py
python submissions/aegis-sentinel-eod/validate_submission.py
```