from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_future_metrics_roadmap(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "rows": [], "status_counts": {}}

    rows: list[dict[str, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 4:
            continue
        if cells[0].lower() in {"capability", "---"}:
            continue
        rows.append(
            {
                "capability": cells[0],
                "status": cells[1],
                "unlocked_stats_now": cells[2],
                "remaining_work": cells[3],
            }
        )

    status_counts: dict[str, int] = {}
    for row in rows:
        status = row["status"].strip().lower()
        status_counts[status] = status_counts.get(status, 0) + 1

    remaining = [
        row["capability"]
        for row in rows
        if row["status"].strip().lower() not in {"implemented", "partial"}
    ]

    return {
        "path": str(path),
        "exists": True,
        "row_count": len(rows),
        "rows": rows,
        "status_counts": status_counts,
        "implemented_count": status_counts.get("implemented", 0),
        "partial_count": status_counts.get("partial", 0),
        "planned_count": status_counts.get("planned", 0),
        "remaining_capabilities": remaining,
    }
