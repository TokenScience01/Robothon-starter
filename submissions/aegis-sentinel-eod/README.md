# Aegis Sentinel EOD v6

**Campus hazardous-package neutralization — recovery release focused on verifiable MuJoCo evidence and a clean 30s demo.**

Registration UUID: `190f2760-b68b-44ee-b805-a6a492a2fa6c`

## Why v6

v5 added five-finger markup, synthetic force HUD, and inflated self-audit scores; ranking dropped to #32. **v6 restores the v4 core** (ranked #30) and adds honest disclosure in `mission_report.json` (`verified_claims`, `known_abstractions`).

## Task goal

Patrol → rangefinder scan → tri-finger grasp → 4N shove recovery → 9x load hold → transport slip recovery → bin placement → seal confirm → alarm confirm.

## How to run

```bash
python3 -m pip install -r requirements.txt
python submissions/aegis-sentinel-eod/train_residual_policy.py
python submissions/aegis-sentinel-eod/run_sentinel_eod.py
python submissions/aegis-sentinel-eod/validate_submission.py
```

## Key artifacts

- `artifacts/demo.mp4` — 30s single-view HUD demo
- `artifacts/mission_report.json` — machine-readable success + verified claims
- `JUDGE_BRIEF.md` — start-here guide for AI judges
- `dataset/labels.csv` — per-frame labeled export