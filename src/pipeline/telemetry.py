from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class PipelineRunRecorder:
    run_ts: str
    mode: str
    replay: bool
    replay_ts: str | None = None
    started_at_utc: str = field(default_factory=utc_now_iso)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def start_step(self, name: str) -> None:
        self.steps.append(
            {
                "name": name,
                "status": "running",
                "started_at_utc": utc_now_iso(),
                "finished_at_utc": None,
                "duration_seconds": None,
                "error": None,
            }
        )

    def finish_step(self, name: str, *, status: str, error: str | None = None) -> None:
        for step in reversed(self.steps):
            if step["name"] != name or step["status"] != "running":
                continue
            finished_at = utc_now_iso()
            step["status"] = status
            step["finished_at_utc"] = finished_at
            started = datetime.fromisoformat(str(step["started_at_utc"]).replace("Z", "+00:00"))
            ended = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            step["duration_seconds"] = max((ended - started).total_seconds(), 0.0)
            step["error"] = error
            return

    def build_summary(
        self,
        *,
        status: str,
        artifacts: dict[str, str | None],
        error: str | None = None,
    ) -> dict[str, Any]:
        finished_at = utc_now_iso()
        started = datetime.fromisoformat(self.started_at_utc.replace("Z", "+00:00"))
        ended = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        return {
            "run_ts": self.run_ts,
            "mode": self.mode,
            "replay": self.replay,
            "replay_ts": self.replay_ts,
            "status": status,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": finished_at,
            "duration_seconds": max((ended - started).total_seconds(), 0.0),
            "steps": self.steps,
            "artifacts": {key: value for key, value in sorted(artifacts.items())},
            "error": error,
        }


def pipeline_run_summary_path(project_root: Path, run_ts: str, *, replay: bool) -> Path:
    stem = f"pipeline_run_replay_{run_ts}.json" if replay else f"pipeline_run_{run_ts}.json"
    return project_root / "artifacts" / "pipeline_runs" / stem


def write_pipeline_run_summary(path: Path, summary: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path
