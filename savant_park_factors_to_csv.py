# savant_park_factors_to_csv.py
import json
import re
from pathlib import Path

import pandas as pd
import requests

URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=year&year=2002&batSide=L&stat=index_wOBA&condition=All&rolling=3&parks=mlb"
OUT = Path("savant_park_factors_2002_L_index_wOBA_rolling3.csv")

def main():
    headers = {
        # Good etiquette: identify yourself
        "User-Agent": "mlbcimleague scraper (contact: you@example.com)",
    }
    r = requests.get(URL, headers=headers, timeout=30)
    r.raise_for_status()
    html = r.text

    # Pull the JSON blob: data = [...];  (works on Savant pages where data is embedded in <script>)
    m = re.search(r"\bdata\s*=\s*(\[.*?\])\s*;", html, flags=re.DOTALL)
    if not m:
        # Fallback: sometimes it's not bracketed exactly as expected
        m = re.search(r"\bdata\s*=\s*(.*?);\s*\n", html, flags=re.DOTALL)
    if not m:
        raise RuntimeError("Could not locate embedded 'data = ...;' block. Page format may have changed.")

    data = json.loads(m.group(1))
    df = pd.DataFrame(data)

    # Optional: keep the most useful columns first if they exist
    preferred = [
        "venue_name", "year",
        "index_woba", "woba", "wobacon", "xwobacon", "bacon", "xbacon",
        "obp", "hits", "single", "double", "triple", "hr", "bb", "so",
        "runs",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(df):,} rows, {len(df.columns):,} cols)")

if __name__ == "__main__":
    main()
