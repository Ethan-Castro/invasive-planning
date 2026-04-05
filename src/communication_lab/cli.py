from __future__ import annotations

import json

from communication_lab.environment import build_environment_report


def main() -> None:
    report = build_environment_report()
    print(json.dumps(report.model_dump(mode="json"), indent=2))
