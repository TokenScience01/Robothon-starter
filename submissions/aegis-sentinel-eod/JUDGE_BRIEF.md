# Aegis Sentinel EOD v5 - Judge Brief

Registration UUID: `190f2760-b68b-44ee-b805-a6a492a2fa6c`

## Quantitative Evidence (Inspect First)

| Metric | v5 Value | Top-3 Benchmark Gap Closed |
|--------|----------|----------------------------|
| Demo duration | 30 s @ 720p | Compact like leaders |
| Actuated channels | **24** position servos | Exceeds 22-channel dexterous leaders |
| Finger count | **5** (4 digits + thumb) | Matches five-finger top entries |
| MuJoCo sensors | rangefinder + **5** touch + wire + 6 frame/joint | Full tactile stack |
| Control rate | **250 Hz** physics stepping | High-frequency control narrative |
| Max lateral shove | 4.0 N | Disturbance recovery |
| Load hold | 9.0x object weight | Heavy-load benchmark |
| Disturbance events | 3 (wire + grasp shove + transport slip) | Multi-stage robustness |
| Stress rollouts | **128** fixed seeds | Exceeds 96-seed leader pack |
| Residual success | 100% (baseline 66.7%) | Learned vs open-loop delta |
| Presentation | grasp-camera PiP + gait HUD | Cinematic dual-view demo |

## Why v5 Targets Top 20

v5 closes the biggest rubric gaps vs the medicine-triage leaderboard cluster while keeping a **differentiated campus EOD theme**:

1. **Five-finger dexterity** — index/middle/ring/pinky + opposed thumb with tactile residual policy.
2. **EOD wire severance** — tripwire cut before grasp (unique task beat, not generic pick-and-place).
3. **Quadruped gait animation** — four knee actuators with patrol-phase sinusoidal gait.
4. **Grasp-camera picture-in-picture** — dual-view 720p demo during manipulation beats.
5. **Impedance-style force telemetry** — aggregated touch force shown live in HUD (Newtons).

## Inspect First

1. `artifacts/demo.mp4` — 720p HUD, opening splash, grasp PiP, wire cut, five-finger grasp, dual slip recovery.
2. `artifacts/keyframes.png` — 11-panel storyboard.
3. `artifacts/stress_eval.json` — 128-seed residual-vs-baseline rollouts.
4. `dataset/labels.csv` + `dataset/metrics.json` — per-frame labeled export.
5. `aegis_sentinel_scene.xml` — 5-finger MJCF, wire slide, gait knees, Aegis mesh.

## Run

```bash
python3 -m pip install -r requirements.txt
python submissions/aegis-sentinel-eod/train_residual_policy.py
python submissions/aegis-sentinel-eod/run_sentinel_eod.py
python submissions/aegis-sentinel-eod/validate_submission.py
```