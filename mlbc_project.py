
#!/usr/bin/env python3
"""
mlbc_project.py

Monte Carlo projection engine for MLBC Sim League (High Heat 2003) hitters.

Exports for Streamlit:
- connect(db_path)
- project_players(conn, ...)  # backward compatible with older mlbc_app.py kwargs

CLI:
  python mlbc_project.py --db mlbc.sqlite --players 2030 --target-year 2141 --sims 5000 --k-ab 150 --hr-dispersion 150
"""

from __future__ import annotations

def forward_years(target_year: int, last_season_year: int) -> int:
    """Number of seasons projected forward from the last observed season."""
    try:
        return max(0, int(target_year) - int(last_season_year))
    except Exception:
        return 0



import argparse
import math
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# DB helpers (Streamlit imports)
# -----------------------------
def connect(db_path: str) -> sqlite3.Connection:
    """Streamlit-friendly connection helper."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
        return [r["name"] for r in rows] if rows else []
    except sqlite3.OperationalError:
        return []


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("none", "nan"):
            return None
        return float(s)
    except Exception:
        return None


def _parse_age_text(birth_or_age_text: Any) -> Optional[int]:
    """
    Parse players.birth_or_age_text, e.g. "Age: 26 Height: ..."
    """
    if birth_or_age_text is None:
        return None
    s = str(birth_or_age_text)
    m = re.search(r"\bAge[:\s]*([0-9]{1,2})\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([0-9]{1,2})\b", s)
    if m:
        age = int(m.group(1))
        if 10 <= age <= 60:
            return age
    return None


# -----------------------------
# Team + stadium resolution
# -----------------------------
def resolve_team_full(conn: sqlite3.Connection, team_raw: Optional[str]) -> Optional[str]:
    if not team_raw:
        return None
    t = str(team_raw).strip()
    if not t:
        return None

    cols = _table_columns(conn, "team_aliases")
    if not cols:
        return t

    # schema-adaptive alias/full columns
    alias_candidates = ["abbrev", "team", "team_abbrev", "alias", "short", "code"]
    full_candidates = ["team_full", "full_name", "name_full", "team_name", "long", "name"]

    alias_col = next((c for c in alias_candidates if c in cols), None)
    full_col = next((c for c in full_candidates if c in cols), None)

    if not alias_col or not full_col:
        return t

    q = f"SELECT {full_col} AS team_full FROM team_aliases WHERE {alias_col}=? COLLATE NOCASE LIMIT 1"
    try:
        r = conn.execute(q, (t,)).fetchone()
        return (r["team_full"] if r and r["team_full"] else t)
    except sqlite3.OperationalError:
        return t


def resolve_stadium_for_team_year(conn: sqlite3.Connection, team_any: Optional[str], season_year: int) -> Optional[str]:
    if not team_any:
        return None
    try:
        r = conn.execute(
            """
            SELECT stadium
            FROM team_stadiums
            WHERE team = ?
              AND (? BETWEEN COALESCE(since_year, 0) AND COALESCE(through_year, 9999))
            LIMIT 1
            """,
            (team_any, season_year),
        ).fetchone()
        if r and r["stadium"]:
            return str(r["stadium"])
    except sqlite3.OperationalError:
        return None

    try:
        r = conn.execute(
            """
            SELECT stadium
            FROM team_stadiums
            WHERE team = ? COLLATE NOCASE
              AND (? BETWEEN COALESCE(since_year, 0) AND COALESCE(through_year, 9999))
            LIMIT 1
            """,
            (team_any, season_year),
        ).fetchone()
        if r and r["stadium"]:
            return str(r["stadium"])
    except sqlite3.OperationalError:
        return None

    return None


# -----------------------------
# League context + player stats
# -----------------------------
@dataclass
class LeagueContext:
    obp: float
    slg: float
    ops: float
    hr_per_ab: float
    k_per_ab: float


def league_context_recent(conn: sqlite3.Connection, recent_years: int = 3) -> LeagueContext:
    try:
        max_year = conn.execute("SELECT MAX(season_year) AS y FROM player_season_batting").fetchone()
    except sqlite3.OperationalError:
        return LeagueContext(obp=0.330, slg=0.420, ops=0.750, hr_per_ab=0.030, k_per_ab=0.160)

    if not max_year or max_year["y"] is None:
        return LeagueContext(obp=0.330, slg=0.420, ops=0.750, hr_per_ab=0.030, k_per_ab=0.160)

    y0 = int(max_year["y"])
    years = [y0 - i for i in range(max(1, int(recent_years)))]

    q = f"""
    SELECT
      SUM(obp * ab) / NULLIF(SUM(ab), 0) AS obp,
      SUM(slg * ab) / NULLIF(SUM(ab), 0) AS slg,
      SUM(COALESCE(ops, obp + slg) * ab) / NULLIF(SUM(ab), 0) AS ops,
      SUM(COALESCE(hr,0)) / NULLIF(SUM(ab), 0) AS hr_per_ab,
      SUM(COALESCE(k,0)) / NULLIF(SUM(ab), 0) AS k_per_ab
    FROM player_season_batting
    WHERE season_year IN ({",".join(["?"] * len(years))})
      AND ab IS NOT NULL AND ab > 0
      AND obp IS NOT NULL AND slg IS NOT NULL
    """
    r = conn.execute(q, years).fetchone()

    obp = float(r["obp"]) if r and r["obp"] is not None else 0.330
    slg = float(r["slg"]) if r and r["slg"] is not None else 0.420
    ops = float(r["ops"]) if r and r["ops"] is not None else (obp + slg)
    hr_per_ab = float(r["hr_per_ab"]) if r and r["hr_per_ab"] is not None else 0.030
    k_per_ab = float(r["k_per_ab"]) if r and r["k_per_ab"] is not None else 0.160

    return LeagueContext(obp=obp, slg=slg, ops=ops, hr_per_ab=hr_per_ab, k_per_ab=k_per_ab)


def player_recent_seasons(conn: sqlite3.Connection, player_id: int, n: int = 3) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT
          player_id, season_year,
          ab, COALESCE(hr,0) AS hr, COALESCE(bb,0) AS bb, COALESCE(k,0) AS k,
          obp, slg, ops
        FROM player_season_batting
        WHERE player_id = ?
        ORDER BY season_year DESC
        LIMIT ?
        """,
        conn,
        params=(player_id, n),
    )
    if df.empty:
        return df
    df["ops"] = df["ops"].fillna(df["obp"] + df["slg"])
    return df


def weighted_recent_ops(df_recent: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if df_recent is None or df_recent.empty:
        return None, None
    df = df_recent[df_recent["ab"].fillna(0) > 0].copy()
    if df.empty:
        return None, None
    w = df["ab"].astype(float).to_numpy()
    ops = df["ops"].astype(float).to_numpy()
    return float(np.average(ops, weights=w)), float(w.sum())

def weighted_recent_hr_per_ab(df_recent: pd.DataFrame) -> tuple[float | None, float | None]:
    """Recency-weighted HR/AB from last up to 3 seasons.

    Correct units:
      - each season contributes a rate (HR/AB)
      - we weight by recency only (not AB), and separately return total AB
        for regression strength.
    """
    if df_recent is None or df_recent.empty:
        return None, None

    df = df_recent.copy()
    df = df[df["ab"].fillna(0) > 0]
    if df.empty:
        return None, None

    # Most recent first
    df = df.sort_values("season_year", ascending=False).reset_index(drop=True)

    rec_w = [0.60, 0.30, 0.10]
    rates = []
    weights = []
    total_ab = 0.0

    for i in range(min(3, len(df))):
        ab = float(df.loc[i, "ab"])
        hr = float(df.loc[i, "hr"])
        if ab <= 0:
            continue
        rates.append(hr / ab)
        weights.append(rec_w[i])
        total_ab += ab

    if not weights:
        return None, None

    wsum = sum(weights)
    hr_rate = sum(r * w for r, w in zip(rates, weights)) / wsum
    return float(hr_rate), float(total_ab)

def estimate_true_talent_ops(ops_recent: float, ab_recent: float, lg_ops: float, k_ab: float) -> float:
    ab_recent = max(0.0, float(ab_recent))
    k_ab = max(1.0, float(k_ab))
    return float((ops_recent * ab_recent + lg_ops * k_ab) / (ab_recent + k_ab))


def gb_effect_multiplier(gb_pct: Optional[float]) -> float:
    if gb_pct is None:
        return 1.0
    gb = float(gb_pct)
    return float(np.clip(1.0 + (43.0 - gb) / 10.0 * 0.04, 0.85, 1.18))


def estimate_hr_rate(df_recent: pd.DataFrame, lg_hr_per_ab: float, k_ab: float, gb_pct: Optional[float]) -> Optional[float]:
    if df_recent is None or df_recent.empty:
        return None
    df = df_recent[df_recent["ab"].fillna(0) > 0].copy()
    if df.empty:
        return None
    ab = float(df["ab"].sum())
    hr = float(df["hr"].fillna(0).sum())
    hr_per_ab_recent = hr / ab if ab > 0 else lg_hr_per_ab
    base = (hr_per_ab_recent * ab + lg_hr_per_ab * k_ab) / (ab + k_ab)
    base *= gb_effect_multiplier(gb_pct)
    return float(np.clip(base, 0.001, 0.200))


# -----------------------------
# Park factors
# -----------------------------
def get_park_pf_ops_for_team_year(conn: sqlite3.Connection, team_any: Optional[str], season_year: int) -> Tuple[Optional[int], float, float]:
    """Return (pf_year_used, home_idx, away_idx) from park_factors_team_year.

    This repo's current DB schema stores:
      park_factors_team_year(team, season_year, home_idx, away_idx, ...)

    Fallbacks:
      - If exact (team, season_year) missing, use latest available season_year for that team.
      - If table/cols missing, return 100/100.
    """
    if not team_any:
        return None, 100.0, 100.0

    # If we were passed a full team name but PFs are keyed by abbrev, try reverse alias lookup.
    team_keys = [str(team_any)]
    try:
        cols_alias = _table_columns(conn, "team_aliases")
        if cols_alias and "abbrev" in cols_alias and "team_full" in cols_alias:
            r = conn.execute(
                "SELECT abbrev FROM team_aliases WHERE team_full=? COLLATE NOCASE LIMIT 1",
                (str(team_any),),
            ).fetchone()
            if r and r["abbrev"]:
                team_keys.insert(0, str(r["abbrev"]))  # prefer abbrev first
    except Exception:
        pass


    # Ensure table/cols exist
    cols = _table_columns(conn, "park_factors_team_year")
    if not cols or "home_idx" not in cols or "away_idx" not in cols:
        return None, 100.0, 100.0

    def _q(where_team: str, year: Optional[int]) -> Optional[sqlite3.Row]:
        if year is None:
            return conn.execute(
                """
                SELECT season_year, home_idx, away_idx
                FROM park_factors_team_year
                WHERE team = ?
                ORDER BY season_year DESC
                LIMIT 1
                """,
                (where_team,),
            ).fetchone()
        return conn.execute(
            """
            SELECT season_year, home_idx, away_idx
            FROM park_factors_team_year
            WHERE team = ? AND season_year = ?
            LIMIT 1
            """,
            (where_team, int(year)),
        ).fetchone()

    try:
        r = _q(str(team_any), int(season_year))
        if not r:
            r = conn.execute(
                """
                SELECT season_year, home_idx, away_idx
                FROM park_factors_team_year
                WHERE team = ? COLLATE NOCASE AND season_year = ?
                LIMIT 1
                """,
                (str(team_any), int(season_year)),
            ).fetchone()

        if r and r["home_idx"] is not None and r["away_idx"] is not None:
            return int(r["season_year"]), float(r["home_idx"]), float(r["away_idx"])

        # fallback to latest for this team
        r = _q(str(team_any), None)
        if r and r["home_idx"] is not None and r["away_idx"] is not None:
            return int(r["season_year"]), float(r["home_idx"]), float(r["away_idx"])
    except sqlite3.OperationalError:
        return None, 100.0, 100.0

    return None, 100.0, 100.0


def park_multiplier(home_idx: float, away_idx: float) -> float:
    return float(((home_idx / 100.0) * 0.5) + ((away_idx / 100.0) * 0.5))


# -----------------------------
# Monte Carlo
# -----------------------------
def simulate_ops_hr(
    rng: np.random.Generator,
    sims: int,
    ab_proj: int,
    ops_mu_prepark: float,
    park_mult: float,
    hr_rate_adj: float,
    hr_dispersion: float,
    ops_sigma: float,
    corr_ops_hr: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray]:
    sims = int(max(1, sims))
    ab_proj = int(max(1, ab_proj))

    corr = float(np.clip(corr_ops_hr, -0.99, 0.99))
    z1 = rng.standard_normal(size=sims)
    z2 = rng.standard_normal(size=sims)
    hr_shock = z1
    ops_shock = corr * z1 + math.sqrt(max(1e-9, 1 - corr * corr)) * z2

    ops_mu = float(ops_mu_prepark * park_mult)
    sigma = float(max(0.005, ops_sigma))
    ops_sim = np.clip(ops_mu + ops_shock * sigma, 0.400, 1.600)

    # Beta-binomial HR
    disp = float(max(20.0, hr_dispersion))
    p0 = float(np.clip(hr_rate_adj, 0.001, 0.200))
    logit = math.log(p0 / (1.0 - p0))
    logit_sim = logit + hr_shock * 0.25
    p_mean_sim = np.clip(1.0 / (1.0 + np.exp(-logit_sim)), 0.001, 0.250)

    alpha = np.maximum(p_mean_sim * disp, 0.5)
    beta = np.maximum((1.0 - p_mean_sim) * disp, 0.5)

    p_sim = np.clip(rng.beta(alpha, beta), 0.001, 0.300)
    p_sim = np.clip(p_sim * (park_mult ** 1.2), 0.001, 0.350)

    hr_sim = rng.binomial(n=ab_proj, p=p_sim)
    return ops_sim, hr_sim


# -----------------------------
# Backward-compatible arg mapping
# -----------------------------
def _normalize_kwargs(
    *,
    sims: Optional[int],
    k_ab: Optional[int],
    lg_years: Optional[int],
    seed: Optional[int],
    hr_dispersion: Optional[float],
    legacy_kwargs: Dict[str, Any],
) -> Tuple[int, int, int, int, float]:
    """
    Map legacy Streamlit kwargs to current params.
    Accepts things like:
      - n_sims, nsims
      - k_ab_prior
      - recent_years, league_years, lg_recent_years
      - hr_disp
    """
    if sims is None:
        for k in ("n_sims", "nsims", "num_sims", "sims"):
            if k in legacy_kwargs and legacy_kwargs[k] is not None:
                sims = int(legacy_kwargs[k])
                break
    if k_ab is None:
        for k in ("k_ab_prior", "k_ab", "k_ab_reg", "k_ab_regression"):
            if k in legacy_kwargs and legacy_kwargs[k] is not None:
                k_ab = int(legacy_kwargs[k])
                break
    if lg_years is None:
        for k in ("recent_years", "league_years", "lg_recent_years", "n_years", "years"):
            if k in legacy_kwargs and legacy_kwargs[k] is not None:
                lg_years = int(legacy_kwargs[k])
                break
    if seed is None:
        for k in ("rng_seed", "seed"):
            if k in legacy_kwargs and legacy_kwargs[k] is not None:
                seed = int(legacy_kwargs[k])
                break
    if hr_dispersion is None:
        for k in ("hr_disp", "hr_dispersion"):
            if k in legacy_kwargs and legacy_kwargs[k] is not None:
                hr_dispersion = float(legacy_kwargs[k])
                break

    # defaults
    sims = int(sims) if sims is not None else 5000
    k_ab = int(k_ab) if k_ab is not None else 150
    lg_years = int(lg_years) if lg_years is not None else 3
    seed = int(seed) if seed is not None else 7
    hr_dispersion = float(hr_dispersion) if hr_dispersion is not None else 150.0
    return sims, k_ab, lg_years, seed, hr_dispersion


# -----------------------------
# Main projection (Streamlit + CLI)
# -----------------------------
def project_players(
    conn: sqlite3.Connection,
    players: Optional[Sequence[int]] = None,
    *,
    player_ids: Optional[Sequence[int]] = None,
    target_year: int,
    sims: Optional[int] = None,
    k_ab: Optional[int] = None,
    lg_years: Optional[int] = None,
    seed: Optional[int] = None,
    hr_dispersion: Optional[float] = None,
    **legacy_kwargs: Any,
) -> Tuple[pd.DataFrame, LeagueContext]:
    """
    Backward-compatible:
      - players=[...] OR player_ids=[...]
      - accepts legacy kwargs (k_ab_prior, recent_years, n_sims, hr_disp, etc.)
    """
    if players is None and player_ids is None:
        raise ValueError("Provide players= or player_ids=")
    if players is None:
        players = player_ids
    assert players is not None

    sims_v, k_ab_v, lg_years_v, seed_v, hr_disp_v = _normalize_kwargs(
        sims=sims, k_ab=k_ab, lg_years=lg_years, seed=seed, hr_dispersion=hr_dispersion, legacy_kwargs=legacy_kwargs
    )

    players_list = [int(x) for x in players]
    rng = np.random.default_rng(int(seed_v))
    lg = league_context_recent(conn, recent_years=lg_years_v)

    out_rows: List[Dict[str, Any]] = []

    for pid in players_list:
        prow = conn.execute(
            "SELECT player_id, name, birth_or_age_text, gb_pct FROM players WHERE player_id=?",
            (int(pid),),
        ).fetchone()

        name = prow["name"] if prow and prow["name"] else f"player_{pid}"
        # -----------------------------
        # Age handling (advance to target_year)
        # -----------------------------
        # Prefer season-level age from the most recent season year we have for the player.
        # Then advance to the requested target_year.
        age = None
        last_season_year = None
        age_last_season = None
        
        r_last = conn.execute(
            """
            SELECT season_year, age
            FROM player_season_batting
            WHERE player_id=?
            ORDER BY season_year DESC
            LIMIT 1
            """,
            (int(pid),),
        ).fetchone()
        
        if r_last and r_last["season_year"] is not None:
            last_season_year = int(r_last["season_year"])
            age_last_season = r_last["age"]
        
        if age_last_season is not None:
            age_base = int(age_last_season)
        else:
        
            age_base = _parse_age_text(prow["birth_or_age_text"]) if prow else None
        
        if age_base is not None and last_season_year is not None:
            age = int(age_base + (int(target_year) - int(last_season_year)))
        else:
            age = age_base
        
        gb_pct = _safe_float(prow["gb_pct"]) if prow else None

        recent = player_recent_seasons(conn, int(pid), n=3)
        ops_recent3, ab_recent = weighted_recent_ops(recent)

        if ops_recent3 is None or ab_recent is None:
            out_rows.append(
                dict(
                    player_id=int(pid),
                    name=name,
                    target_year=int(target_year),
                    team=None,
                    stadium=None,
                    age=age,
                    ab_proj=None,
                    gb_pct=gb_pct,
                    park_pf_year=None,
                    park_pf_home_idx=100.0,
                    park_pf_away_idx=100.0,
                    ops_recent3=None,
                    ops_true_talent_prepark=None,
                    OPS_p10=None,
                    OPS_p50=None,
                    OPS_p90=None,
                    HR_p10=None,
                    HR_p50=None,
                    HR_p90=None,
                    park_mult=1.0,
                )
            )
            continue

        team_raw = None
        cols_ps = _table_columns(conn, "player_season_batting")
        if "team" in cols_ps:
            r = conn.execute(
                """
                SELECT team
                FROM player_season_batting
                WHERE player_id=?
                ORDER BY season_year DESC
                LIMIT 1
                """,
                (int(pid),),
            ).fetchone()
            team_raw = (r["team"] if r else None)

        team_full = resolve_team_full(conn, str(team_raw) if team_raw is not None else None)
        stadium = resolve_stadium_for_team_year(conn, team_full or team_raw, int(target_year))

        pf_year = int(target_year) - 1
        pf_year_used, home_idx, away_idx = get_park_pf_ops_for_team_year(conn, team_full or team_raw, pf_year)
        pmult = park_multiplier(home_idx, away_idx)

        ops_tt_prepark = estimate_true_talent_ops(
            ops_recent=float(ops_recent3),
            ab_recent=float(ab_recent),
            lg_ops=float(lg.ops),
            k_ab=float(k_ab_v),
        )
        ops_tt_prepark *= (gb_effect_multiplier(gb_pct) ** 0.6)

        ab_last = float(recent.iloc[0]["ab"]) if not recent.empty else float(ab_recent / 3.0)
        ab_proj = int(np.clip(ab_last, 250, 700))

        hr_rate_recent, ab_hr_recent = weighted_recent_hr_per_ab(recent)
        if hr_rate_recent is None or ab_hr_recent is None:
            hr_rate_adj = float(lg.hr_per_ab)
        else:
            # Regress HR/AB toward league with an AB prior
            k_ab_hr = float(k_ab_v)
            hr_rate_adj = float((hr_rate_recent * ab_hr_recent + float(lg.hr_per_ab) * k_ab_hr) / (ab_hr_recent + k_ab_hr))

        # GB% effect (kept modest)
        hr_rate_adj *= (gb_effect_multiplier(gb_pct) ** 0.6)

        # Drift HR rate toward league the farther we project out (stabilizes long-range forecasts)
        if last_season_year is not None:
            fwd = forward_years(int(target_year), int(last_season_year))
            # 6% of the gap closes per forward year (tunable)
            shrink = min(0.40, 0.06 * fwd)
            hr_rate_adj = float((1.0 - shrink) * hr_rate_adj + shrink * float(lg.hr_per_ab))

        if hr_rate_adj is None:
            hr_rate_adj = float(lg.hr_per_ab)

        sample_ab = float(ab_recent)
        ops_sigma = float(np.clip(0.10 - 0.04 * (sample_ab / (sample_ab + 400.0)), 0.05, 0.11))

        # Wider HR uncertainty the farther out we project
        hr_disp_eff = float(hr_disp_v)
        if last_season_year is not None:
            fwd = forward_years(int(target_year), int(last_season_year))
            hr_disp_eff = float(max(40.0, hr_disp_eff / (1.0 + 0.35 * fwd)))

        ops_sim, hr_sim = simulate_ops_hr(
            rng=rng,
            sims=int(sims_v),
            ab_proj=int(ab_proj),
            ops_mu_prepark=float(ops_tt_prepark),
            park_mult=float(pmult),
            hr_rate_adj=float(hr_rate_adj),
            hr_dispersion=float(hr_disp_eff),
            ops_sigma=float(ops_sigma),
        )

        OPS_p10, OPS_p50, OPS_p90 = np.percentile(ops_sim, [10, 50, 90]).tolist()
        HR_p10, HR_p50, HR_p90 = np.percentile(hr_sim, [10, 50, 90]).tolist()

        out_rows.append(
            dict(
                player_id=int(pid),
                name=name,
                target_year=int(target_year),
                team=team_full or team_raw,
                stadium=stadium,
                age=age,
                ab_proj=int(ab_proj),
                gb_pct=gb_pct,
                park_pf_year=pf_year_used if pf_year_used is not None else pf_year,
                park_pf_home_idx=float(home_idx),
                park_pf_away_idx=float(away_idx),
                ops_recent3=float(ops_recent3),
                ops_true_talent_prepark=float(ops_tt_prepark),
                OPS_p10=float(OPS_p10),
                OPS_p50=float(OPS_p50),
                OPS_p90=float(OPS_p90),
                HR_p10=int(round(HR_p10)),
                HR_p50=int(round(HR_p50)),
                HR_p90=int(round(HR_p90)),
                park_mult=float(pmult),
                hr_rate_adj=float(hr_rate_adj),
                ops_sigma=float(ops_sigma),
                sims=int(sims_v),
                k_ab=int(k_ab_v),
                hr_dispersion=float(hr_disp_v),
            )
        )

    df = pd.DataFrame(out_rows)

    preferred = [
        "player_id", "name", "target_year", "team", "stadium", "age",
        "ab_proj", "gb_pct",
        "park_pf_year", "park_pf_home_idx", "park_pf_away_idx", "park_mult",
        "ops_recent3", "ops_true_talent_prepark",
        "OPS_p10", "OPS_p50", "OPS_p90",
        "HR_p10", "HR_p50", "HR_p90",
        "hr_rate_adj", "ops_sigma",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    for c in ["ops_recent3", "ops_true_talent_prepark", "OPS_p10", "OPS_p50", "OPS_p90", "park_mult", "hr_rate_adj", "ops_sigma"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(3)
    for c in ["park_pf_home_idx", "park_pf_away_idx"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    return df, lg


# -----------------------------
# CLI
# -----------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLBC Monte Carlo hitter projections (OPS + HR).")
    p.add_argument("--db", required=True, help="SQLite DB path (mlbc.sqlite)")
    p.add_argument("--players", nargs="+", type=int, required=True, help="Player IDs (e.g., 2030 2126)")
    p.add_argument("--target-year", type=int, required=True, help="Target season year (e.g., 2141)")
    p.add_argument("--sims", type=int, default=5000, help="Monte Carlo sims")
    p.add_argument("--k-ab", type=int, default=150, help="Regression AB")
    p.add_argument("--lg-years", type=int, default=3, help="League context window (years)")
    p.add_argument("--seed", type=int, default=7, help="RNG seed")
    p.add_argument("--hr-dispersion", type=float, default=150.0,
                   help="Beta-binomial HR dispersion (smaller => wider HR tails). Try 100–200.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    conn = connect(args.db)
    try:
        df, lg = project_players(
            conn,
            player_ids=args.players,
            target_year=args.target_year,
            sims=args.sims,
            k_ab=args.k_ab,
            lg_years=args.lg_years,
            seed=args.seed,
            hr_dispersion=args.hr_dispersion,
        )
        print()
        print(df.to_string(index=False))
        print()
        print("League context (recent):")
        print(f"  OBP={lg.obp:.3f} SLG={lg.slg:.3f} OPS={lg.ops:.3f} HR/AB={lg.hr_per_ab:.4f} K/AB={lg.k_per_ab:.4f}")
        print()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
