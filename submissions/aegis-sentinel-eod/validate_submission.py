from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT / "artifacts"
REQUIRED = [
    PROJECT / "aegis_sentinel_scene.xml",
    PROJECT / "run_sentinel_eod.py",
    PROJECT / "train_residual_policy.py",
    PROJECT / "learned_policy_weights.json",
    PROJECT / "JUDGE_BRIEF.md",
    PROJECT / "rubric_scorecard.json",
    PROJECT / "registration.json",
    PROJECT / "README.md",
    ARTIFACTS / "demo.mp4",
    ARTIFACTS / "mission_report.json",
    ARTIFACTS / "contact_timeline.json",
    ARTIFACTS / "policy_card.json",
    ARTIFACTS / "stress_eval.json",
]


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(PROJECT)}")

    reg = json.loads((PROJECT / "registration.json").read_text(encoding="utf-8"))
    uuid = reg.get("uuid", "")
    if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", uuid):
        errors.append("registration.json uuid is invalid")

    if (ARTIFACTS / "mission_report.json").exists():
        report = json.loads((ARTIFACTS / "mission_report.json").read_text(encoding="utf-8"))
        if report.get("registration_uuid") != uuid:
            errors.append("mission_report registration_uuid mismatch")
        if not report.get("package_grasped"):
            errors.append("mission did not record package_grasped")
        if not report.get("alarm_pressed"):
            errors.append("mission did not record alarm_pressed")
        if float(report.get("residual_corrections", 0)) < 10:
            errors.append("insufficient residual correction evidence")

    if (ARTIFACTS / "contact_timeline.json").exists():
        contact = json.loads((ARTIFACTS / "contact_timeline.json").read_text(encoding="utf-8"))
        if int(contact.get("stable_contact_samples", 0)) < 5:
            errors.append("insufficient stable tri-finger contact samples")

    if errors:
        print("VALIDATION FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("VALIDATION PASSED")
    print(f"uuid: {uuid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())