# Ralph logging

Structured logs:
- `.ralph/logs/<run_id>.jsonl` (JSON Lines)

Generate dashboard:
```bash
./scripts/ralph-dashboard.py
```

Recommended `.gitignore` entries:
```
.ralph/logs/
.ralph/backup/
```


## Real-time dashboard (recommended)
Generate and update the dashboard continuously as logs arrive:

```bash
./scripts/ralph-dashboard-watch.py
```

To view it in a browser with live refresh:
```bash
./scripts/ralph-dashboard-watch.py --serve --port 8735
# then open: http://localhost:8735/status-dashboard.html
```

Notes:
- The watcher regenerates `docs/status-dashboard.md` and `docs/status-dashboard.html` whenever any `.ralph/logs/*.jsonl` changes.
- Stop with Ctrl+C.
