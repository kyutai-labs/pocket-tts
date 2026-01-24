#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main():
    ap = argparse.ArgumentParser(
        description="Append a structured Ralph event to a JSONL log."
    )
    ap.add_argument(
        "--run-id",
        required=True,
        help="Run identifier (e.g., 20260121T120102Z-windsurf)",
    )
    ap.add_argument(
        "--event",
        required=True,
        help="Event name (e.g., run_start, issue_selected, lock_acquired)",
    )
    ap.add_argument(
        "--issue", type=int, default=None, help="Issue number, if applicable"
    )
    ap.add_argument(
        "--agent",
        default=None,
        help="Agent identifier (e.g., windsurf, claude, orchestrator)",
    )
    ap.add_argument("--status", default=None, help="Optional status string")
    ap.add_argument("--data", default=None, help="Optional JSON string payload")
    ap.add_argument("--log-dir", default=".ralph/logs", help="Base log directory")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.run_id}.jsonl"

    payload = {}
    if args.data:
        try:
            payload = json.loads(args.data)
        except Exception as e:
            print(f"error: --data must be valid JSON: {e}", file=sys.stderr)
            return 2

    rec = {"ts": utc_now_iso(), "run_id": args.run_id, "event": args.event}
    if args.issue is not None:
        rec["issue"] = args.issue
    if args.agent:
        rec["agent"] = args.agent
    if args.status:
        rec["status"] = args.status
    if payload:
        rec["data"] = payload

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
