# savant_park_factors_to_csv.py
"""Fetch Baseball Savant Statcast park factor table and write a CSV.

This project uses Savant park factors as a **prior** for MLBC park factors.
MLBC-derived home/away performance should quickly dominate the blend, but
Savant provides stability for low-sample stadium-years (e.g., first year
of a new lease).

Example:
  python3 savant_park_factors_to_csv.py --year 2002 --batside L --rolling 3 \
    --stat index_wOBA --out data/savant_park_factors_prior.csv

Notes:
- Savant can change page structure. Keeping a "last known good" CSV in git
  makes projections reproducible.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests


def _build_url(*, year: int, batside: str, stat: str, rolling: int) -> str:
    return (
        "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
        f"?type=year&year={int(year)}&batSide={batside}&stat={stat}"
        f"&condition=All&rolling={int(rolling)}&parks=mlb"
    )


def _extract_data_json(html: str):
    # Pull the JSON blob: data = [...];  (works on Savant pages where data is embedded in <script>)
    m = re.search(r"\bdata\s*=\s*(\[.*?\])\s*;", html, flags=re.DOTALL)
    if not m:
        # Fallback: sometimes it's not bracketed exactly as expected
        m = re.search(r"\bdata\s*=\s*(.*?);\s*\n", html, flags=re.DOTALL)
    if not m:
        raise RuntimeError("Could not locate embedded 'data = ...;' block. Page format may have changed.")
    return json.loads(m.group(1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2002)
    ap.add_argument("--batside", type=str, default="L", choices=["L", "R"])
    ap.add_argument("--stat", type=str, default="index_wOBA")
    ap.add_argument("--rolling", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("data/savant_park_factors_prior.csv"))
    ap.add_argument(
        "--user-agent",
        type=str,
        default="mlbcsimleague scraper (contact: you@example.com)",
        help="Set a descriptive User-Agent.",
    )
    args = ap.parse_args(argv)

    url = _build_url(year=args.year, batside=args.batside, stat=args.stat, rolling=args.rolling)

    headers = {"User-Agent": args.user_agent}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    data = _extract_data_json(r.text)
    df = pd.DataFrame(data)

    # Optional: keep the most useful columns first if they exist
    preferred = [
        "venue_name",
        "year",
        "index_woba",
        "woba",
        "wobacon",
        "xwobacon",
        "bacon",
        "xbacon",
        "obp",
        "hits",
        "single",
        "double",
        "triple",
        "hr",
        "bb",
        "so",
        "runs",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df):,} rows, {len(df.columns):,} cols)")


if __name__ == "__main__":
    main()

