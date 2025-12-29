#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _is_home_key(k: str) -> bool:
    k = _norm(k)
    return k in {"home", "at home", "home games"}


def _is_away_key(k: str) -> bool:
    k = _norm(k)
    return k in {"away", "road", "on road", "away games", "road games"}


def _safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b


def _ops_from_row(h: int, bb: int, ab: int, doubles: int, triples: int, hr: int) -> Optional[float]:
    """
    Crude OPS from counting stats (if OPS/OBP/SLG missing).
    OBP ~= (H+BB)/ (AB+BB)
    SLG computed from total bases / AB
    """
    if ab is None or ab <= 0:
        return None
    if h is None or bb is None:
        return None
    # singles:
    if None in (doubles, triples, hr):
        return None
    singles = h - doubles - triples - hr
    if singles < 0:
        return None
    tb = singles + 2 * doubles + 3 * triples + 4 * hr
    obp = (h + bb) / (ab + bb) if (ab + bb) > 0 else None
    slg = tb / ab if ab > 0 else None
    if obp is None or slg is None:
        return None
    return obp + slg


@dataclass
class PFRow:
    team: str
    stadium: Optional[str]
    season_year: int
    home_ops: float
    away_ops: float
    home_ab: int
    away_ab: int
    pf_ops_index: float  # 100 = neutral


# ----------------------------
# DB schema for park factors
# ----------------------------

# ----------------------------
# Savant priors + stadium/venue mapping
# ----------------------------
SAVANT_PRIOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS park_factors_savant_venue (
  venue_name TEXT PRIMARY KEY,
  pf_woba_index REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS stadium_savant_map (
  stadium TEXT PRIMARY KEY,
  venue_name TEXT,
  method TEXT,
  score REAL
);
"""

# A small set of known renames/aliases. Keep this list short and additive.
# The auto-mapper will cover most names; this handles "branding" changes.
STADIUM_ALIASES: Dict[str, str] = {
    # SF
    "pacific bell park": "oracle park",
    "sbc park": "oracle park",
    "att park": "oracle park",
    # OAK
    "network coliseum": "oakland coliseum",
    "mcafee coliseum": "oakland coliseum",
    # HOU
    "enron field": "minute maid park",
    # ARI
    "bank one ballpark": "chase field",
    # MIL
    "miller park": "american family field",
}


def _norm_stadium_name(s: Optional[str]) -> str:
    """Normalize stadium/venue strings for matching."""
    s = (s or "").strip().lower()
    # common punctuation
    for ch in [".", ",", "'", "\"", "(", ")", "-", "_", "/"]:
        s = s.replace(ch, " ")
    s = s.replace("&", " and ")
    # strip common generic words
    stop = {
        "the",
        "stadium",
        "park",
        "field",
        "ballpark",
        "baseball",
        "grounds",
    }
    parts = [p for p in s.split() if p and p not in stop]
    return " ".join(parts)


def load_savant_priors_from_csv(conn: sqlite3.Connection, csv_path: str) -> int:
    """Load Savant prior CSV into SQLite.

    Expects a column `venue_name` and either `index_woba` or `pf_woba_index`.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return 0

    cols = {c.lower(): c for c in df.columns}
    if "venue_name" not in cols:
        raise ValueError("Savant CSV missing 'venue_name' column")

    idx_col = None
    for c in ("pf_woba_index", "index_woba"):
        if c in cols:
            idx_col = cols[c]
            break
    if idx_col is None:
        raise ValueError("Savant CSV must include 'index_woba' (or 'pf_woba_index')")

    df = df[[cols["venue_name"], idx_col]].copy()
    df.columns = ["venue_name", "pf_woba_index"]
    df = df.dropna(subset=["venue_name", "pf_woba_index"])

    # Some Savant exports may be 1xx.xx as string
    df["pf_woba_index"] = pd.to_numeric(df["pf_woba_index"], errors="coerce")
    df = df.dropna(subset=["pf_woba_index"])

    conn.executescript(SAVANT_PRIOR_SCHEMA)
    conn.executemany(
        """
        INSERT INTO park_factors_savant_venue (venue_name, pf_woba_index)
        VALUES (?, ?)
        ON CONFLICT(venue_name) DO UPDATE SET pf_woba_index=excluded.pf_woba_index
        """,
        [(str(r.venue_name), float(r.pf_woba_index)) for r in df.itertuples(index=False)],
    )
    conn.commit()
    return int(len(df))


def _get_savant_venues(conn: sqlite3.Connection) -> List[str]:
    try:
        rows = conn.execute("SELECT venue_name FROM park_factors_savant_venue").fetchall()
    except sqlite3.OperationalError:
        return []
    return [str(r[0]) for r in rows if r and r[0]]


def _best_fuzzy_match(name: str, choices: List[str]) -> Tuple[Optional[str], float]:
    """Return (best_choice, score) using difflib over normalized strings."""
    import difflib

    n = _norm_stadium_name(name)
    if not n or not choices:
        return None, 0.0

    best = None
    best_score = 0.0
    for c in choices:
        score = difflib.SequenceMatcher(a=n, b=_norm_stadium_name(c)).ratio()
        if score > best_score:
            best_score = score
            best = c
    return best, float(best_score)


def ensure_stadium_savant_map(conn: sqlite3.Connection, *, min_score: float = 0.86) -> int:
    """Auto-build stadium->Savant venue mapping.

    Uses:
    - normalization
    - alias dictionary
    - fuzzy matching fallback

    Writes results to stadium_savant_map for reproducibility/debugging.
    """
    conn.executescript(SAVANT_PRIOR_SCHEMA)

    venues = _get_savant_venues(conn)
    if not venues:
        return 0

    # stadiums in use (from team_stadiums)
    try:
        stadium_rows = conn.execute("SELECT DISTINCT stadium FROM team_stadiums WHERE stadium IS NOT NULL").fetchall()
    except sqlite3.OperationalError:
        return 0

    stadiums = [str(r[0]) for r in stadium_rows if r and r[0]]

    inserted = 0
    for stadium in stadiums:
        # already mapped?
        r = conn.execute("SELECT venue_name FROM stadium_savant_map WHERE stadium=?", (stadium,)).fetchone()
        if r and r[0]:
            continue

        norm = _norm_stadium_name(stadium)
        norm = STADIUM_ALIASES.get(norm, norm)

        # exact normalized match first
        exact = None
        for v in venues:
            if _norm_stadium_name(v) == norm:
                exact = v
                break

        if exact:
            conn.execute(
                "INSERT OR REPLACE INTO stadium_savant_map (stadium, venue_name, method, score) VALUES (?,?,?,?)",
                (stadium, exact, "exact_norm", 1.0),
            )
            inserted += 1
            continue

        best, score = _best_fuzzy_match(norm, venues)
        if best and score >= float(min_score):
            conn.execute(
                "INSERT OR REPLACE INTO stadium_savant_map (stadium, venue_name, method, score) VALUES (?,?,?,?)",
                (stadium, best, "fuzzy", float(score)),
            )
            inserted += 1
        else:
            # keep row for visibility; venue_name stays NULL
            conn.execute(
                "INSERT OR REPLACE INTO stadium_savant_map (stadium, venue_name, method, score) VALUES (?,?,?,?)",
                (stadium, None, "unmatched", float(score)),
            )
            inserted += 1

    conn.commit()
    return inserted


def savant_prior_index_for_stadium(conn: sqlite3.Connection, stadium: Optional[str]) -> Optional[float]:
    if not stadium:
        return None
    try:
        r = conn.execute("SELECT venue_name FROM stadium_savant_map WHERE stadium=?", (stadium,)).fetchone()
    except sqlite3.OperationalError:
        return None
    if not r or not r[0]:
        return None
    venue = str(r[0])
    try:
        r2 = conn.execute("SELECT pf_woba_index FROM park_factors_savant_venue WHERE venue_name=?", (venue,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return float(r2[0]) if r2 and r2[0] is not None else None
PF_SCHEMA = """
CREATE TABLE IF NOT EXISTS park_factors_team_year (
  team TEXT NOT NULL,
  season_year INTEGER NOT NULL,
  stadium TEXT,
  home_ops REAL,
  away_ops REAL,
  home_ab INTEGER,
  away_ab INTEGER,
  pf_ops_index REAL NOT NULL,
  pf_ops_index_mlbc REAL,
  pf_ops_index_prior REAL,
  pf_ops_blend_w REAL,
  PRIMARY KEY (team, season_year)
);
"""


def load_team_stadium_for_year(conn: sqlite3.Connection, team: str, season_year: int) -> Optional[str]:
    team = (team or "").strip()

    # 1) exact match
    row = conn.execute(
        """
        SELECT stadium
        FROM team_stadiums
        WHERE team = ?
          AND (? BETWEEN COALESCE(since_year, 0) AND COALESCE(through_year, 9999))
        LIMIT 1
        """,
        (team, season_year),
    ).fetchone()
    if row:
        return row[0]

    # 2) fuzzy: stadiums table team contains abbreviation OR abbreviation contains team
    row = conn.execute(
        """
        SELECT stadium
        FROM team_stadiums
        WHERE (? LIKE '%' || team || '%' OR team LIKE '%' || ? || '%')
          AND (? BETWEEN COALESCE(since_year, 0) AND COALESCE(through_year, 9999))
        LIMIT 1
        """,
        (team, team, season_year),
    ).fetchone()
    return row[0] if row else None



def compute_park_factors(
    conn: sqlite3.Connection,
    min_ab_per_side: int = 150,
    shrink_to_100_ab: int = 6000,
    savant_prior_csv: Optional[str] = None,
    savant_min_score: float = 0.86,
    savant_k_ab: int = 2500,
) -> List[PFRow]:

    """Compute TEAM-YEAR park factor from Home vs Away OPS using splits.

    Baseline (existing behavior):
      - PF is derived from MLBC home vs away OPS and shrunk toward 100 by AB.

    If `savant_prior_csv` is provided:
      - loads Savant venue indices (index_wOBA) as a *prior*
      - auto-maps MLBC stadium names -> Savant venue_name
      - blends MLBC-derived PF with prior using AB-based weighting


    if savant_prior_csv:
        # These will create tables if missing.
        load_savant_priors_from_csv(conn, savant_prior_csv)
        ensure_stadium_savant_map(conn, min_score=float(savant_min_score))
    Args:
      shrink_to_100_ab: AB used to shrink MLBC raw PF toward 100.
      savant_k_ab: AB used to weight MLBC vs Savant prior (smaller => MLBC dominates faster).
    """

    # Join splits -> player season for team/year
    rows = conn.execute(
        """
        SELECT
          s.player_id,
          ps.season_year,
          ps.team,
          lower(trim(s.split_key)) as split_key,
          s.ab,
          s.ops,
          s.h, s.bb, s.doubles, s.triples, s.hr
        FROM player_splits_batting s
        JOIN player_season_batting ps
          ON ps.player_id = s.player_id
         AND (s.season_year IS NULL OR s.season_year = ps.season_year)
        WHERE lower(trim(s.split_key)) IN ('home','away')
          AND ps.team IS NOT NULL
          AND trim(ps.team) <> ''
        """
    ).fetchall()

    acc: Dict[Tuple[str, int, str], Tuple[float, int]] = {}

    for player_id, season_year, team, split_key, ab, ops, h, bb, d2, d3, hr in rows:
        side = "home" if split_key == "home" else "away"

        ab_i = int(ab) if ab is not None else 0
        if ab_i <= 0:
            continue

        ops_f = float(ops) if ops is not None else None
        if ops_f is None:
            ops_f = _ops_from_row(
                h=int(h) if h is not None else None,
                bb=int(bb) if bb is not None else None,
                ab=ab_i,
                doubles=int(d2) if d2 is not None else None,
                triples=int(d3) if d3 is not None else None,
                hr=int(hr) if hr is not None else None,
            )
        if ops_f is None:
            continue

        key = (str(team).strip(), int(season_year), side)
        cur_sum, cur_ab = acc.get(key, (0.0, 0))
        acc[key] = (cur_sum + ops_f * ab_i, cur_ab + ab_i)

    teams_years = {(t, y) for (t, y, s) in acc.keys()}

    out: List[PFRow] = []
    for team, year in sorted(teams_years, key=lambda x: (x[0], x[1])):
        h_sum, h_ab = acc.get((team, year, "home"), (0.0, 0))
        a_sum, a_ab = acc.get((team, year, "away"), (0.0, 0))

        if h_ab < min_ab_per_side or a_ab < min_ab_per_side:
            continue

        home_ops = h_sum / h_ab
        away_ops = a_sum / a_ab
        raw_pf = _safe_div(home_ops, away_ops)
        if raw_pf is None:
            continue

        total_ab = h_ab + a_ab
        w_shrink = total_ab / (total_ab + shrink_to_100_ab)
        shrunk_pf = 1.0 + (raw_pf - 1.0) * w_shrink
        idx_mlbc = 100.0 * shrunk_pf

        stadium = load_team_stadium_for_year(conn, team, year)

        # Optional: blend in Savant prior (index_wOBA) as a stabilizing prior.
        idx_prior = None
        w_blend = None
        idx_final = float(idx_mlbc)
        if savant_prior_csv and stadium:
            idx_prior = savant_prior_index_for_stadium(conn, stadium)
            if idx_prior is not None:
                w_blend = total_ab / (total_ab + max(1, int(savant_k_ab)))
                idx_final = float(w_blend * idx_mlbc + (1.0 - w_blend) * float(idx_prior))

        r = PFRow(
            team=team,
            stadium=stadium,
            season_year=year,
            home_ops=home_ops,
            away_ops=away_ops,
            home_ab=h_ab,
            away_ab=a_ab,
            pf_ops_index=float(idx_final),
        )
        # Store optional diagnostics for DB write.
        r.__dict__["pf_ops_index_mlbc"] = float(idx_mlbc)
        r.__dict__["pf_ops_index_prior"] = (float(idx_prior) if idx_prior is not None else None)
        r.__dict__["pf_ops_blend_w"] = (float(w_blend) if w_blend is not None else None)

        out.append(r)

    return out


def write_to_db(conn: sqlite3.Connection, rows: List[PFRow]) -> None:
    conn.executescript(PF_SCHEMA)
    conn.executemany(
        """
        INSERT INTO park_factors_team_year (
          team, season_year, stadium,
          home_ops, away_ops, home_ab, away_ab,
          pf_ops_index,
          pf_ops_index_mlbc, pf_ops_index_prior, pf_ops_blend_w
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(team, season_year) DO UPDATE SET
          stadium=excluded.stadium,
          home_ops=excluded.home_ops,
          away_ops=excluded.away_ops,
          home_ab=excluded.home_ab,
          away_ab=excluded.away_ab,
          pf_ops_index=excluded.pf_ops_index,
          pf_ops_index_mlbc=excluded.pf_ops_index_mlbc,
          pf_ops_index_prior=excluded.pf_ops_index_prior,
          pf_ops_blend_w=excluded.pf_ops_blend_w
        """,
        [
            (
                r.team,
                r.season_year,
                r.stadium,
                r.home_ops,
                r.away_ops,
                r.home_ab,
                r.away_ab,
                r.pf_ops_index,
                r.__dict__.get("pf_ops_index_mlbc"),
                r.__dict__.get("pf_ops_index_prior"),
                r.__dict__.get("pf_ops_blend_w"),
            )
            for r in rows
        ],
    )
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mlbc.sqlite")
    ap.add_argument("--min-ab-per-side", type=int, default=150)
    ap.add_argument("--shrink-to-100-ab", type=int, default=6000)
    ap.add_argument(
        "--savant-prior-csv",
        default="",
        help="Optional Savant prior CSV (expects venue_name + index_woba).",
    )
    ap.add_argument(
        "--savant-min-score",
        type=float,
        default=0.86,
        help="Min fuzzy-match score to map stadium->venue.",
    )
    ap.add_argument(
        "--savant-k-ab",
        type=int,
        default=2500,
        help="AB weight for MLBC vs Savant blend (smaller => MLBC dominates faster).",
    )
    ap.add_argument("--out-csv", default="", help="Optional CSV output path")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys=ON;")

    rows = compute_park_factors(
        conn,
        min_ab_per_side=args.min_ab_per_side,
        shrink_to_100_ab=args.shrink_to_100_ab,
        savant_prior_csv=(args.savant_prior_csv or None),
        savant_min_score=float(args.savant_min_score),
        savant_k_ab=int(args.savant_k_ab),
    )
    write_to_db(conn, rows)

    df = pd.DataFrame([r.__dict__ for r in rows])

    if not df.empty and "season_year" in df.columns:
        df = df.sort_values(
            ["season_year", "pf_ops_index"], ascending=[False, False]
        )

    print(df.head(25).to_string(index=False))

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote {len(df)} rows to {args.out_csv}")

    conn.close()


if __name__ == "__main__":
    main()
