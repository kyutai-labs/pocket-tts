#!/usr/bin/env python3
import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_ts(s: str):
    try:
        # ISO 8601 with Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Generate a Markdown status dashboard from Ralph JSONL logs."
    )
    ap.add_argument(
        "--log-dir",
        default=".ralph/logs",
        help="Directory containing <run_id>.jsonl files",
    )
    ap.add_argument(
        "--out", default="docs/status-dashboard.md", help="Output Markdown file"
    )
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = {}
    issues = defaultdict(lambda: {"events": [], "last": None})

    if not log_dir.exists():
        out_path.write_text(
            "# Ralph Status Dashboard\n\nNo logs found.\n", encoding="utf-8"
        )
        return 0

    for p in sorted(log_dir.glob("*.jsonl")):
        run_id = p.stem
        runs.setdefault(
            run_id,
            {
                "events": [],
                "first": None,
                "last": None,
                "agents": set(),
                "issues": set(),
            },
        )
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                logger.debug("Failed to parse JSON line in %s: %s", p, exc)
                continue
            runs[run_id]["events"].append(rec)
            ts = parse_ts(rec.get("ts", ""))
            if ts:
                if runs[run_id]["first"] is None or ts < runs[run_id]["first"]:
                    runs[run_id]["first"] = ts
                if runs[run_id]["last"] is None or ts > runs[run_id]["last"]:
                    runs[run_id]["last"] = ts
            if "agent" in rec:
                runs[run_id]["agents"].add(rec["agent"])
            if "issue" in rec:
                runs[run_id]["issues"].add(rec["issue"])
                issues[rec["issue"]]["events"].append(rec)
                issues[rec["issue"]]["last"] = rec

    # Build markdown
    lines = []
    lines.append("# Ralph Status Dashboard")
    lines.append("")
    lines.append(
        f"_Generated: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}_"
    )
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("| Run ID | Start (UTC) | End (UTC) | Agents | Issues | Last event |")
    lines.append("|---|---|---|---|---:|---|")
    for run_id, meta in sorted(
        runs.items(),
        key=lambda kv: (kv[1]["first"] or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    ):
        start = (
            meta["first"].isoformat().replace("+00:00", "Z") if meta["first"] else ""
        )
        end = meta["last"].isoformat().replace("+00:00", "Z") if meta["last"] else ""
        agents = ", ".join(sorted(meta["agents"])) if meta["agents"] else ""
        issue_count = len(meta["issues"])
        last_event = meta["events"][-1]["event"] if meta["events"] else ""
        lines.append(
            f"| `{run_id}` | {start} | {end} | {agents} | {issue_count} | {last_event} |"
        )

    lines.append("")
    lines.append("## Issues")
    lines.append("")
    lines.append(
        "| Issue | Last event | Last status | Last run | Last timestamp (UTC) |"
    )
    lines.append("|---:|---|---|---|---|")
    for issue, meta in sorted(issues.items(), key=lambda kv: kv[0]):
        last = meta["last"] or {}
        ts = last.get("ts", "")
        lines.append(
            f"| {issue} | {last.get('event', '')} | {last.get('status', '')} | `{last.get('run_id', '')}` | {ts} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- This dashboard is log-derived. If you want it updated automatically, run:"
    )
    lines.append("  ```bash")
    lines.append("  ./scripts/ralph-dashboard.py")
    lines.append("  ```")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
