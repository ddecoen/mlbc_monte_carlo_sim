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
) -> List[PFRow]:
    """
    Compute TEAM-YEAR park factor from Home vs Away OPS using splits,
    but source team/year from player_season_batting (more reliable).
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
        w = total_ab / (total_ab + shrink_to_100_ab)
        shrunk_pf = 1.0 + (raw_pf - 1.0) * w
        idx = 100.0 * shrunk_pf

        stadium = load_team_stadium_for_year(conn, team, year)

        out.append(
            PFRow(
                team=team,
                stadium=stadium,
                season_year=year,
                home_ops=home_ops,
                away_ops=away_ops,
                home_ab=h_ab,
                away_ab=a_ab,
                pf_ops_index=idx,
            )
        )

    return out

    """
    Method:
      - For each (team, year), aggregate player splits:
          weighted_home_ops = sum(ops * AB) / sum(AB)
          weighted_away_ops = sum(ops * AB) / sum(AB)
      - Raw PF = home_ops / away_ops
      - Convert to index: 100 * PF
      - Apply Bayesian shrinkage to 1.0 (100) based on total AB:
          PF_shrunk = 1 + (PF_raw - 1) * w
          w = total_ab / (total_ab + shrink_to_100_ab)

    Notes:
      - Uses ops from splits table if present; if ops NULL, tries to reconstruct from counts.
      - Requires team to exist on split rows (team column) and season_year to exist.
    """
    hk = tuple(_norm(x) for x in home_keys)
    ak = tuple(_norm(x) for x in away_keys)

    # Pull relevant split rows
    rows = conn.execute(
        """
        SELECT
          season_year,
          team,
          split_key,
          ab,
          ops,
          h, bb, doubles, triples, hr
        FROM player_splits_batting
        WHERE season_year IS NOT NULL
          AND team IS NOT NULL
          AND ab IS NOT NULL
        """
    ).fetchall()

    # Accumulators: (team, year, side) -> (sum_ops_ab, sum_ab)
    acc: Dict[Tuple[str, int, str], Tuple[float, int]] = {}

    for season_year, team, split_key, ab, ops, h, bb, d2, d3, hr in rows:
        team = str(team).strip()
        if not team:
            continue
        k = _norm(split_key)
        side = None
        if k in hk:
            side = "home"
        elif k in ak:
            side = "away"
        else:
            continue

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

        key = (team, int(season_year), side)
        cur_sum, cur_ab = acc.get(key, (0.0, 0))
        acc[key] = (cur_sum + ops_f * ab_i, cur_ab + ab_i)

    # Build PF rows
    out: List[PFRow] = []
    teams_years = {(t, y) for (t, y, s) in acc.keys()}

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
        w = total_ab / (total_ab + shrink_to_100_ab)
        shrunk_pf = 1.0 + (raw_pf - 1.0) * w
        idx = 100.0 * shrunk_pf

        stadium = load_team_stadium_for_year(conn, team, year)

        out.append(
            PFRow(
                team=team,
                stadium=stadium,
                season_year=year,
                home_ops=home_ops,
                away_ops=away_ops,
                home_ab=h_ab,
                away_ab=a_ab,
                pf_ops_index=idx,
            )
        )

    return out


def write_to_db(conn: sqlite3.Connection, rows: List[PFRow]) -> None:
    conn.executescript(PF_SCHEMA)
    conn.executemany(
        """
        INSERT INTO park_factors_team_year (
          team, season_year, stadium, home_ops, away_ops, home_ab, away_ab, pf_ops_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(team, season_year) DO UPDATE SET
          stadium=excluded.stadium,
          home_ops=excluded.home_ops,
          away_ops=excluded.away_ops,
          home_ab=excluded.home_ab,
          away_ab=excluded.away_ab,
          pf_ops_index=excluded.pf_ops_index
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
    ap.add_argument("--out-csv", default="", help="Optional CSV output path")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys=ON;")

    rows = compute_park_factors(
        conn,
        min_ab_per_side=args.min_ab_per_side,
        shrink_to_100_ab=args.shrink_to_100_ab,
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
