#!/usr/bin/env python3
import argparse
import http.server
import socketserver
import threading
import time
from pathlib import Path


def generate(dashboard_py: Path, log_dir: Path, out_md: Path):
    # Call the generator script as a module-less subprocess replacement by importing it is messy;
    # simplest is exec it in-process via runpy.
    import runpy
    import sys

    argv0 = sys.argv
    try:
        sys.argv = [str(dashboard_py), "--log-dir", str(log_dir), "--out", str(out_md)]
        runpy.run_path(str(dashboard_py), run_name="__main__")
    finally:
        sys.argv = argv0


def md_to_html(md_text: str, refresh_seconds: int) -> str:
    # Minimal Markdown-to-HTML conversion for tables/headings/code fences.
    # This intentionally avoids external deps.
    import html

    lines = md_text.splitlines()
    out = []
    in_code = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            if in_code:
                out.append("<pre><code>")
            else:
                out.append("</code></pre>")
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("|") and line.endswith("|"):
            # Table handling: accumulate in a simple state machine outside; easiest: render as <pre> if complex.
            out.append(html.escape(line))
        elif line.strip() == "":
            out.append("<br/>")
        else:
            out.append(f"<p>{html.escape(line)}</p>")

    body = "\n".join(out)
    # If markdown tables exist, show them in <pre> to preserve alignment.
    if "|" in md_text and "\n|" in md_text:
        body = "<pre>\n" + html.escape(md_text) + "\n</pre>"

    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="{refresh_seconds}">
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Ralph Status Dashboard</title>
    <style>
      body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; padding: 16px; }}
      pre {{ white-space: pre-wrap; word-break: break-word; }}
      code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
    </style>
  </head>
  <body>
    {body}
  </body>
</html>"""


def serve_dir(directory: Path, port: int):
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        # serve from directory
        import os

        cwd = os.getcwd()
        try:
            os.chdir(directory)
            httpd.serve_forever()
        finally:
            os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser(
        description="Watch Ralph logs and regenerate dashboard in near real-time."
    )
    ap.add_argument(
        "--log-dir",
        default=".ralph/logs",
        help="Directory containing <run_id>.jsonl files",
    )
    ap.add_argument(
        "--out-md", default="docs/status-dashboard.md", help="Output markdown path"
    )
    ap.add_argument(
        "--out-html", default="docs/status-dashboard.html", help="Output HTML path"
    )
    ap.add_argument(
        "--interval", type=float, default=1.0, help="Polling interval seconds"
    )
    ap.add_argument("--refresh", type=int, default=2, help="HTML meta-refresh seconds")
    ap.add_argument(
        "--serve", action="store_true", help="Serve docs/ over HTTP for live viewing"
    )
    ap.add_argument(
        "--port", type=int, default=8735, help="HTTP port when --serve is set"
    )
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    dashboard_py = repo_root / "scripts" / "ralph-dashboard.py"
    log_dir = repo_root / args.log_dir
    out_md = repo_root / args.out_md
    out_html = repo_root / args.out_html

    if args.serve:
        docs_dir = out_html.parent
        t = threading.Thread(target=serve_dir, args=(docs_dir, args.port), daemon=True)
        t.start()
        print(
            f"Serving {docs_dir} at http://localhost:{args.port}/status-dashboard.html"
        )

    last_sig = None
    while True:
        # Compute signature from mtimes + sizes
        sig_parts = []
        if log_dir.exists():
            for p in sorted(log_dir.glob("*.jsonl")):
                st = p.stat()
                sig_parts.append((p.name, int(st.st_mtime), st.st_size))
        sig = tuple(sig_parts)
        if sig != last_sig:
            generate(dashboard_py, log_dir, out_md)
            md = (
                out_md.read_text(encoding="utf-8")
                if out_md.exists()
                else "# Ralph Status Dashboard\n\nNo logs.\n"
            )
            html = md_to_html(md, args.refresh)
            out_html.parent.mkdir(parents=True, exist_ok=True)
            out_html.write_text(html, encoding="utf-8")
            print(f"updated: {out_md} and {out_html} (logs changed)")
            last_sig = sig
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
