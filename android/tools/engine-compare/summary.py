"""Summarise judge.json across cells. usage: summary.py CELLDIR... [--detail]"""

import json
import pathlib
import sys
from typing import Any

import numpy as np

detail = "--detail" in sys.argv
cells = [a for a in sys.argv[1:] if not a.startswith("--")]


def agg(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = sum(r["n_exp"] for r in rows)
    errs = sum(r["sub"] + r["del"] + r["ins"] for r in rows)
    wps = sum(r["n_exp"] for r in rows) / sum(r["speech_s"] for r in rows)
    gaps = [g for r in rows for g in r["gaps"]]
    nb = sum(r["n_bound"] for r in rows)
    short = sum(1 for r in rows if r["wps_exp"] and r["wps_exp"] > 4.0)
    return dict(
        wer=errs / n,
        wps=wps,
        gap_med=float(np.median(gaps)) if gaps else float("nan"),
        gap_lt100=sum(g < 0.10 for g in gaps),
        gaps_n=len(gaps),
        nb=nb,
        short=short,
        n=len(rows),
        p=float(np.mean([r["p_mean"] for r in rows])),
    )


print(
    "| cell | corpus | WER | wps (expected words / speech s) | boundary pause median s | boundaries <100 ms quiet / measured / total | chunks >4 wps | whisper p |"
)
print("|---|---|---|---|---|---|---|---|")
for c in cells:
    rows = json.loads((pathlib.Path(c) / "judge.json").read_text())
    for corpus, sel in (
        ("TalkBack", lambda r: r["item"].startswith("tb")),
        ("ReadAloud", lambda r: r["item"].startswith("ra")),
    ):
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        a = agg(rs)
        print(
            f"| {pathlib.Path(c).name} | {corpus} | {a['wer']:.1%} | {a['wps']:.2f} | {a['gap_med']:.2f} | {a['gap_lt100']}/{a['gaps_n']}/{a['nb']} | {a['short']}/{a['n']} | {a['p']:.2f} |"
        )
    if detail:
        for r in rows:
            print(
                f"    {r['item']}.{r['chunk']} wer={r['wer']} wps={r['wps_exp']} gaps={r['gaps']} | {r['heard']}"
            )
