
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

def batter_platoon_splits(conn, player_id: int) -> dict:
    """
    Returns platoon split stats from player_splits_batting (single-season snapshot).
    Keys: 'vs RH Pitchers', 'vs LH Pitchers'
    Values: {'ab': float, 'ops': float}
    """
    try:
        df = pd.read_sql_query(
            """
            select split_key, ab, ops
            from player_splits_batting
            where player_id = ?
              and split_key in ('vs RH Pitchers', 'vs LH Pitchers')
            """,
            conn,
            params=(int(player_id),),
        )
    except Exception:
        return {}

    out: dict = {}
    for _, r in df.iterrows():
        k = str(r["split_key"])
        out[k] = {
            "ab": float(r["ab"] if r["ab"] is not None else 0.0),
            "ops": float(r["ops"] if r["ops"] is not None else 0.0),
        }
    return out



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




# -----------------------------
# Pitching: league context + player stats
# -----------------------------
@dataclass
class PitchingLeagueContext:
    era: float
    whip: float
    ops_allowed: float
    k_per_ip: float
    bb_per_ip: float
    hr_per_ip: float
    fip_const: float



def pitching_league_context_recent(conn: sqlite3.Connection, recent_years: int = 3) -> PitchingLeagueContext:
    """League averages from player_season_pitching over the most recent N years."""
    try:
        max_year = conn.execute("SELECT MAX(season_year) AS y FROM player_season_pitching").fetchone()
    except sqlite3.OperationalError:
        # reasonable defaults
        return PitchingLeagueContext(era=4.50, whip=1.35, ops_allowed=0.720, k_per_ip=0.90, bb_per_ip=0.33)

    if not max_year or max_year["y"] is None:
        return PitchingLeagueContext(era=4.50, whip=1.35, ops_allowed=0.720, k_per_ip=0.90, bb_per_ip=0.33)

    y0 = int(max_year["y"])
    years = [y0 - i for i in range(max(1, int(recent_years)))]

    q = f"""
    SELECT
      SUM(era * ip) / NULLIF(SUM(ip), 0) AS era,
      SUM(whip * ip) / NULLIF(SUM(ip), 0) AS whip,
      SUM(ops_allowed * ip) / NULLIF(SUM(ip), 0) AS ops_allowed,
      SUM(COALESCE(k,0)) / NULLIF(SUM(ip), 0) AS k_per_ip,
      SUM(COALESCE(bb,0)) / NULLIF(SUM(ip), 0) AS bb_per_ip
    FROM player_season_pitching
    WHERE season_year IN ({",".join(["?"] * len(years))})
      AND ip IS NOT NULL AND ip > 0
      AND era IS NOT NULL
    """

    r = conn.execute(q, years).fetchone()

    era = float(r["era"]) if r and r["era"] is not None else 4.50
    whip = float(r["whip"]) if r and r["whip"] is not None else 1.35
    ops_allowed = float(r["ops_allowed"]) if r and r["ops_allowed"] is not None else 0.720
    k_per_ip = float(r["k_per_ip"]) if r and r["k_per_ip"] is not None else 0.90
    bb_per_ip = float(r["bb_per_ip"]) if r and r["bb_per_ip"] is not None else 0.33

    hr_rows = conn.execute(
    """
    select sum(hr) as hr, sum(ip) as ip
    from player_splits_pitching
    where split_group = 'platoon'
    """
    ).fetchone()
    lg_hr = float(hr_rows["hr"] or 0.0)
    lg_ip_hr = float(hr_rows["ip"] or 0.0)
    hr_per_ip = lg_hr / max(1e-9, lg_ip_hr)

    # You likely already have lg_ip, lg_k, lg_bb from seasons; if not, compute them:
    tot = conn.execute(
        """
        select sum(ip) as ip, sum(k) as k, sum(bb) as bb
        from player_season_pitching
        """
    ).fetchone()
    lg_ip = float(tot["ip"] or 0.0)
    lg_k  = float(tot["k"] or 0.0)
    lg_bb = float(tot["bb"] or 0.0)

    # Estimate lg_hr using hr_per_ip * lg_ip
    lg_hr_est = hr_per_ip * lg_ip

    fip_const = float(era) - (13.0 * lg_hr_est + 3.0 * lg_bb - 2.0 * lg_k) / max(1e-9, lg_ip)



    return PitchingLeagueContext(era=era, whip=whip, ops_allowed=ops_allowed, k_per_ip=k_per_ip, bb_per_ip=bb_per_ip, hr_per_ip=hr_per_ip, fip_const=fip_const)


def pitcher_recent_seasons(conn: sqlite3.Connection, player_id: int, n: int = 3) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT
          player_id, season_year,
          team, g, gs, ip,
          era, ops_allowed, whip,
          COALESCE(k,0) AS k, COALESCE(bb,0) AS bb
        FROM player_season_pitching
        WHERE player_id = ?
        ORDER BY season_year DESC
        LIMIT ?
        """,
        conn,
        params=(player_id, n),
    )
    return df


def _weighted_recent_rate(df_recent: pd.DataFrame, value_col: str, weight_col: str) -> Tuple[Optional[float], Optional[float]]:
    if df_recent is None or df_recent.empty:
        return None, None
    df = df_recent[df_recent[weight_col].fillna(0) > 0].copy()
    if df.empty:
        return None, None
    w = df[weight_col].astype(float).to_numpy()
    v = df[value_col].astype(float).to_numpy()
    return float(np.average(v, weights=w)), float(w.sum())


def estimate_true_talent_rate(rate_recent: float, w_recent: float, lg_rate: float, k_w: float) -> float:
    w_recent = max(0.0, float(w_recent))
    k_w = max(1.0, float(k_w))
    return float((rate_recent * w_recent + lg_rate * k_w) / (w_recent + k_w))


def simulate_pitching(
    rng: np.random.Generator,
    sims: int,
    *,
    era_mu: float,
    ops_mu: float,
    whip_mu: float,
    k_per_ip_mu: float,
    bb_per_ip_mu: float,
    ip_proj: float,
    # dispersions
    era_sigma: float,
    ops_sigma: float,
    whip_sigma: float,
    # correlation between "good" outcomes
    corr: float = 0.45,
) -> Dict[str, np.ndarray]:
    sims = int(max(1, sims))
    ip_proj = float(max(1.0, ip_proj))

    # Shared skill shock: negative => better pitcher (lower ERA/OPS/WHIP, higher K, lower BB)
    z1 = rng.standard_normal(size=sims)
    z2 = rng.standard_normal(size=sims)
    z_skill = z1
    z2 = corr * z1 + math.sqrt(max(1e-9, 1 - corr * corr)) * z2

    # Rate stats: clamp to plausible ranges.
    era = np.clip(era_mu + (-z_skill) * float(era_sigma), 0.50, 12.00)
    ops = np.clip(ops_mu + (-z_skill) * float(ops_sigma), 0.350, 1.200)
    whip = np.clip(whip_mu + (-z_skill) * float(whip_sigma), 0.60, 2.50)

    # K/BB: model K and BB counts from Poisson on rates, using same z_skill.
    k_per_ip = np.clip(k_per_ip_mu * (1.0 + 0.12 * (-z_skill)), 0.10, 2.50)
    bb_per_ip = np.clip(bb_per_ip_mu * (1.0 + 0.18 * (z_skill)), 0.01, 1.50)

    k = rng.poisson(lam=np.clip(k_per_ip * ip_proj, 0.1, 1e9))
    bb = rng.poisson(lam=np.clip(bb_per_ip * ip_proj, 0.1, 1e9))

    return {
        "ERA": era,
        "OPS_allowed": ops,
        "WHIP": whip,
        "K": k.astype(int),
        "BB": bb.astype(int),
    }


def project_pitchers(
    conn: sqlite3.Connection,
    pitchers: Optional[Sequence[int]] = None,
    *,
    player_ids: Optional[Sequence[int]] = None,
    target_year: int,
    sims: int = 5000,
    k_ip: float = 100.0,
    lg_years: int = 3,
    seed: int = 7,
) -> Tuple[pd.DataFrame, PitchingLeagueContext]:
    """Pitcher projections (ERA, OPS allowed, WHIP, K/BB, plus IP context).

    - Regresses recent rates to league rates using k_ip (innings-equivalent prior)
    - Monte Carlo simulates correlated outcomes

    NOTE: Requires player_season_pitching to exist.
    """
    if pitchers is None and player_ids is None:
        raise ValueError("Provide pitchers= or player_ids=")
    if pitchers is None:
        pitchers = player_ids
    assert pitchers is not None

    rng = np.random.default_rng(int(seed))
    lg = pitching_league_context_recent(conn, recent_years=int(lg_years))

    out_rows: List[Dict[str, Any]] = []

    for pid in [int(x) for x in pitchers]:
        prow = conn.execute(
            "SELECT player_id, name, birth_or_age_text FROM players WHERE player_id=?",
            (int(pid),),
        ).fetchone()
        name = prow["name"] if prow and prow["name"] else f"player_{pid}"

        recent = pitcher_recent_seasons(conn, int(pid), n=3)
        if recent is None or recent.empty:
            out_rows.append(
                {
                    "player_id": int(pid),
                    "name": name,
                    "target_year": int(target_year),
                    "team": None,
                    "gs_recent": None,
                    "ip_proj": None,
                    "ERA_p10": None,
                    "ERA_p50": None,
                    "ERA_p90": None,
                    "OPS_allowed_p10": None,
                    "OPS_allowed_p50": None,
                    "OPS_allowed_p90": None,
                    "WHIP_p10": None,
                    "WHIP_p50": None,
                    "WHIP_p90": None,
                    "KBB_p10": None,
                    "KBB_p50": None,
                    "KBB_p90": None,
                }
            )
            continue

        team = str(recent.iloc[0]["team"]) if "team" in recent.columns else None
        team = team if team and team.lower() not in ("nan", "none") else None

        # Recent weighted by IP
        era_recent, ip_recent = _weighted_recent_rate(recent, "era", "ip")
        ops_recent, _ = _weighted_recent_rate(recent, "ops_allowed", "ip")
        whip_recent, _ = _weighted_recent_rate(recent, "whip", "ip")

        # K/BB rates from totals over recent IP
        ip_sum = float(recent["ip"].fillna(0).sum())
        k_sum = float(recent["k"].fillna(0).sum())
        bb_sum = float(recent["bb"].fillna(0).sum())
        k_per_ip_recent = (k_sum / ip_sum) if ip_sum > 0 else None
        bb_per_ip_recent = (bb_sum / ip_sum) if ip_sum > 0 else None

        if era_recent is None or ip_recent is None or ip_recent <= 0:
            # can't project without any IP
            out_rows.append(
                {
                    "player_id": int(pid),
                    "name": name,
                    "target_year": int(target_year),
                    "team": team,
                    "gs_recent": int(recent["gs"].fillna(0).sum()) if "gs" in recent.columns else None,
                    "ip_proj": None,
                    "ERA_p10": None,
                    "ERA_p50": None,
                    "ERA_p90": None,
                    "OPS_allowed_p10": None,
                    "OPS_allowed_p50": None,
                    "OPS_allowed_p90": None,
                    "WHIP_p10": None,
                    "WHIP_p50": None,
                    "WHIP_p90": None,
                    "KBB_p10": None,
                    "KBB_p50": None,
                    "KBB_p90": None,
                }
            )
            continue

        # Regress each rate to league mean.
        k_ip_v = float(max(1.0, k_ip))
        era_tt = estimate_true_talent_rate(float(era_recent), float(ip_recent), float(lg.era), k_ip_v)
        ops_tt = estimate_true_talent_rate(float(ops_recent) if ops_recent is not None else float(lg.ops_allowed), float(ip_recent), float(lg.ops_allowed), k_ip_v)
        whip_tt = estimate_true_talent_rate(float(whip_recent) if whip_recent is not None else float(lg.whip), float(ip_recent), float(lg.whip), k_ip_v)

        k_per_ip_tt = estimate_true_talent_rate(
            float(k_per_ip_recent) if k_per_ip_recent is not None else float(lg.k_per_ip),
            float(ip_recent),
            float(lg.k_per_ip),
            k_ip_v,
        )
        bb_per_ip_tt = estimate_true_talent_rate(
            float(bb_per_ip_recent) if bb_per_ip_recent is not None else float(lg.bb_per_ip),
            float(ip_recent),
            float(lg.bb_per_ip),
            k_ip_v,
        )

                # --- HR/IP estimate for FIP (from platoon splits) ---
        hr_per_ip_recent = None
        try:
            split_df = pitcher_splits_pitching(conn, int(pid), split_group="platoon")
            if split_df is not None and len(split_df) > 0:
                ip_split = float(split_df["ip"].fillna(0.0).sum())
                hr_split = float(split_df["hr"].fillna(0.0).sum())
                if ip_split > 1e-9:
                    hr_per_ip_recent = hr_split / ip_split
        except Exception:
            hr_per_ip_recent = None

        hr_per_ip_tt = estimate_true_talent_rate(
            float(hr_per_ip_recent) if hr_per_ip_recent is not None else float(lg.hr_per_ip),
            float(ip_recent),
            float(lg.hr_per_ip),
            k_ip_v,
        )


        # IP projection: use last season IP as baseline, with starter boost.
        ip_last = float(recent.iloc[0]["ip"]) if recent.iloc[0]["ip"] is not None else float(ip_recent / 3.0)
        gs_last = int(recent.iloc[0]["gs"]) if "gs" in recent.columns and recent.iloc[0]["gs"] is not None else 0
        is_sp = gs_last >= 10

        # Basic: SP get a higher cap.
        if is_sp:
            ip_proj = float(np.clip(ip_last, 60.0, 230.0))
        else:
            ip_proj = float(np.clip(ip_last, 20.0, 120.0))

        # Uncertainty: more IP reduces noise.
        ip_eff = float(max(10.0, ip_recent))
        era_sigma = float(np.clip(0.85 * math.sqrt(80.0 / (ip_eff + 40.0)), 0.25, 1.20))
        ops_sigma = float(np.clip(0.080 * math.sqrt(80.0 / (ip_eff + 40.0)), 0.020, 0.110))
        whip_sigma = float(np.clip(0.18 * math.sqrt(80.0 / (ip_eff + 40.0)), 0.05, 0.30))

        

        sims_out = simulate_pitching(
            rng=rng,
            sims=int(sims),
            era_mu=float(era_tt),
            ops_mu=float(ops_tt),
            whip_mu=float(whip_tt),
            k_per_ip_mu=float(k_per_ip_tt),
            bb_per_ip_mu=float(bb_per_ip_tt),
            ip_proj=float(ip_proj),
            era_sigma=float(era_sigma),
            ops_sigma=float(ops_sigma),
            whip_sigma=float(whip_sigma),
        )

        ERA_p10, ERA_p50, ERA_p90 = np.percentile(sims_out["ERA"], [10, 50, 90]).tolist()
        OPS_p10, OPS_p50, OPS_p90 = np.percentile(sims_out["OPS_allowed"], [10, 50, 90]).tolist()

        # FIP(no HBP): derive a HR total from HR/IP and IP proj
        # FIP = (13*HR + 3*BB - 2*K) / IP + const
        hr_sim = np.clip(hr_per_ip_tt * float(ip_proj) * (1.0 + 0.18 * rng.standard_normal(size=int(sims))), 0.0, 9999.0)
        bb_sim = sims_out["BB"].astype(float)
        k_sim = sims_out["K"].astype(float)
        ip_sim = float(ip_proj)
        fip = (13.0 * hr_sim + 3.0 * bb_sim - 2.0 * k_sim) / max(1e-9, ip_sim) + float(lg.fip_const)
        fip = np.clip(fip, 0.50, 12.00)

        FIP_p10, FIP_p50, FIP_p90 = np.percentile(fip, [10, 50, 90]).tolist()

        WHIP_p10, WHIP_p50, WHIP_p90 = np.percentile(sims_out["WHIP"], [10, 50, 90]).tolist()

        # K/BB ratio distribution
        k = sims_out["K"].astype(float)
        bb = sims_out["BB"].astype(float)
        kbb = np.where(bb <= 0, 99.0, np.clip(k / bb, 0.0, 99.0))
        KBB_p10, KBB_p50, KBB_p90 = np.percentile(kbb, [10, 50, 90]).tolist()

        out_rows.append(
            {
                "player_id": int(pid),
                "name": name,
                "target_year": int(target_year),
                "team": team,
                "is_sp": bool(is_sp),
                "gs_last": int(gs_last),
                "ip_recent": float(ip_recent),
                "ip_proj": float(ip_proj),
                "era_recent3": float(era_recent),
                "ops_allowed_recent3": float(ops_recent) if ops_recent is not None else None,
                "whip_recent3": float(whip_recent) if whip_recent is not None else None,
                "ERA_p10": float(ERA_p10),
                "ERA_p50": float(ERA_p50),
                "ERA_p90": float(ERA_p90),
                "OPS_allowed_p10": float(OPS_p10),
                "OPS_allowed_p50": float(OPS_p50),
                "OPS_allowed_p90": float(OPS_p90),
                "WHIP_p10": float(WHIP_p10),
                "WHIP_p50": float(WHIP_p50),
                "WHIP_p90": float(WHIP_p90),
                "KBB_p10": float(KBB_p10),
                "KBB_p50": float(KBB_p50),
                "KBB_p90": float(KBB_p90),
                "sims": int(sims),
                "k_ip": float(k_ip_v),
            }
        )

    df = pd.DataFrame(out_rows)

    preferred = [
        "player_id",
        "name",
        "target_year",
        "team",
        "is_sp",
        "gs_last",
        "ip_recent",
        "ip_proj",
        "era_recent3",
        "ops_allowed_recent3",
        "whip_recent3",
        "ERA_p10",
        "ERA_p50",
        "ERA_p90",
        "OPS_allowed_p10",
        "OPS_allowed_p50",
        "OPS_allowed_p90",
        "WHIP_p10",
        "WHIP_p50",
        "WHIP_p90",
        "KBB_p10",
        "KBB_p50",
        "KBB_p90",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    for c in [
        "ip_recent",
        "ip_proj",
        "era_recent3",
        "ops_allowed_recent3",
        "whip_recent3",
        "ERA_p10",
        "ERA_p50",
        "ERA_p90",
        "OPS_allowed_p10",
        "OPS_allowed_p50",
        "OPS_allowed_p90",
        "WHIP_p10",
        "WHIP_p50",
        "WHIP_p90",
        "KBB_p10",
        "KBB_p50",
        "KBB_p90",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(3)

    return df, lg
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
          avg, obp, slg, ops
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


def age_curve_multiplier(
    age: Optional[int],
    *,
    # Typical hitter aging: rise into late 20s, plateau, then decline.
    peak_start: int = 27,
    peak_end: int = 29,
    decline_start: int = 31,
    # Slope controls (per-year) outside plateau.
    pre_peak_gain: float = 0.010,
    post_decline_loss: float = 0.007,
    # Clamp overall effect so we don't do anything crazy.
    min_mult: float = 0.85,
    max_mult: float = 1.15,
    # "Star" override knobs. If a hitter is far above league, we assume a longer peak
    # and gentler decline.
    star: bool = False,
) -> float:
    """Return a multiplicative adjustment to apply to rate stats (OPS-ish / HR rate).

    This is intentionally simple and stable:
    - Ages [peak_start..peak_end] => multiplier ~ 1.0
    - Before peak_start => linear gain into peak (pre_peak_gain per year)
    - After decline_start => linear decline (post_decline_loss per year)

    If star=True, we extend peak_end and soften decline.
    """
    if age is None:
        return 1.0

    a = int(age)

    if star:
        peak_end = max(peak_end, 31)
        decline_start = max(decline_start, 33)
        pre_peak_gain *= 0.80
        post_decline_loss *= 0.60

    if a < peak_start:
        mult = 1.0 - (peak_start - a) * float(pre_peak_gain)
    elif peak_start <= a <= peak_end:
        mult = 1.0
    elif peak_end < a < decline_start:
        # extended plateau / gentle fade until decline_start
        mult = 1.0 - (a - peak_end) * float(post_decline_loss) * 0.25
    else:
        mult = 1.0 - (a - decline_start) * float(post_decline_loss)

    return float(np.clip(mult, float(min_mult), float(max_mult)))


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


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    try:
        r = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
        return bool(r)
    except Exception:
        return False

def _zscore_fit(df: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    mu = df[cols].mean(axis=0)
    sd = df[cols].std(axis=0).replace(0.0, 1.0)
    return mu, sd

def _zscore_apply(df: pd.DataFrame, cols: list[str], mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    return (df[cols] - mu) / sd

def load_edits_training_set(conn: sqlite3.Connection, *, min_ab: int = 150) -> pd.DataFrame:
    """
    Training rows: players with edits + at least one MLB season with AB >= min_ab.
    Labels:
      - ops_recent3 (AB-weighted over last 3 seasons)
      - hr_per_ab_recent3 (recency-weighted over last 3 seasons)
    """
    if not _table_exists(conn, "player_edits_hitting"):
        return pd.DataFrame()

    train = pd.read_sql_query(
        """
        select
          e.player_id,
          e.perf_ab, e.perf_h, e.perf_2b, e.perf_3b, e.perf_hr, e.perf_bb, e.perf_so,
          e.avg_vs_l, e.obp_vs_l, e.slg_vs_l,
          e.avg_vs_r, e.obp_vs_r, e.slg_vs_r,
          e.pull_pct, e.middle_pct, e.oppo_pct, e.gb_pct,
          e.arm_strength, e.run_speed
        from player_edits_hitting e
        where exists (
          select 1
          from player_season_batting b
          where b.player_id = e.player_id
            and b.ab >= ?
        )
        """,
        conn,
        params=(int(min_ab),),
    )
    if train.empty:
        return train

    labels = []
    for pid in train["player_id"].astype(int).tolist():
        recent = player_recent_seasons(conn, int(pid), n=3)
        ops_recent3, ab_recent = weighted_recent_ops(recent)
        hr_rate_recent, ab_hr_recent = weighted_recent_hr_per_ab(recent)

        labels.append(
            {
                "player_id": int(pid),
                "ops_recent3": ops_recent3,
                "ab_recent3": ab_recent,
                "hr_per_ab_recent3": hr_rate_recent,
                "ab_hr_recent3": ab_hr_recent,
            }
        )

    lab = pd.DataFrame(labels)
    out = train.merge(lab, on="player_id", how="left")
    out = out[(out["ops_recent3"].notna()) & (out["hr_per_ab_recent3"].notna())].copy()
    return out

def predict_from_edits_knn(
    conn: sqlite3.Connection,
    *,
    player_id: int,
    age: Optional[int],
    k: int = 30,
    min_ab: int = 150,
) -> tuple[Optional[float], Optional[float], int]:
    """
    Returns (ops_est, hr_per_ab_est, n_used).
    """
    if not _table_exists(conn, "player_edits_hitting"):
        return None, None, 0

    tgt = pd.read_sql_query(
        """
        select
          player_id,
          perf_ab, perf_h, perf_2b, perf_3b, perf_hr, perf_bb, perf_so,
          avg_vs_l, obp_vs_l, slg_vs_l,
          avg_vs_r, obp_vs_r, slg_vs_r,
          pull_pct, middle_pct, oppo_pct, gb_pct,
          arm_strength, run_speed
        from player_edits_hitting
        where player_id = ?
        """,
        conn,
        params=(int(player_id),),
    )
    if tgt.empty:
        return None, None, 0

    train = load_edits_training_set(conn, min_ab=int(min_ab))
    if train.empty:
        return None, None, 0

    feat_cols = [
        "perf_ab", "perf_h", "perf_2b", "perf_3b", "perf_hr", "perf_bb", "perf_so",
        "avg_vs_l", "obp_vs_l", "slg_vs_l",
        "avg_vs_r", "obp_vs_r", "slg_vs_r",
        "pull_pct", "middle_pct", "oppo_pct", "gb_pct",
        "arm_strength", "run_speed",
    ]

    # Include age as a feature (rookies benefit; training ages not yet available -> median-filled)
    train = train.copy()
    tgt = tgt.copy()
    train["age_feat"] = np.nan
    tgt["age_feat"] = float(age) if age is not None else np.nan
    feat_cols2 = feat_cols + ["age_feat"]

    # Compute medians; drop feature columns with no signal (median NaN)
    med = train[feat_cols2].median(axis=0, numeric_only=True)
    feat_cols2 = [c for c in feat_cols2 if c in med.index and pd.notna(med[c])]
    if len(feat_cols2) < 5:
        return None, None, 0

    # Fill NaNs with column medians (restricted to usable columns)
    med = train[feat_cols2].median(axis=0, numeric_only=True)
    train[feat_cols2] = train[feat_cols2].fillna(med)
    tgt[feat_cols2] = tgt[feat_cols2].fillna(med)

    # Force numeric dtypes (prevents object-dtype issues)
    train[feat_cols2] = train[feat_cols2].apply(pd.to_numeric, errors="coerce")
    tgt[feat_cols2] = tgt[feat_cols2].apply(pd.to_numeric, errors="coerce")

    # Re-fill any NaNs introduced by coercion
    med = train[feat_cols2].median(axis=0, numeric_only=True)
    train[feat_cols2] = train[feat_cols2].fillna(med)
    tgt[feat_cols2] = tgt[feat_cols2].fillna(med)

    # Final guard: if still NaNs, bail out
    if train[feat_cols2].isna().any().any() or tgt[feat_cols2].isna().any().any():
        return None, None, 0

    # Force numeric dtypes (prevents object-dtype fillna/NaN issues)
    train[feat_cols2] = train[feat_cols2].apply(pd.to_numeric, errors="coerce")
    tgt[feat_cols2] = tgt[feat_cols2].apply(pd.to_numeric, errors="coerce")

    # Re-fill any NaNs introduced by coercion
    med = train[feat_cols2].median(axis=0, numeric_only=True)
    train[feat_cols2] = train[feat_cols2].fillna(med)
    tgt[feat_cols2] = tgt[feat_cols2].fillna(med)



    # Standardize
    mu, sd = _zscore_fit(train, feat_cols2)
    X = _zscore_apply(train, feat_cols2, mu, sd).to_numpy(dtype=float)
    x0 = _zscore_apply(tgt, feat_cols2, mu, sd).to_numpy(dtype=float)[0]

    d = np.sqrt(((X - x0) ** 2).sum(axis=1))
    k_eff = int(min(max(5, k), len(train)))
    idx = np.argsort(d)[:k_eff]
    d_sel = d[idx]
    w = 1.0 / (d_sel + 1e-6)

    ops_vals = train.iloc[idx]["ops_recent3"].to_numpy(dtype=float)
    hr_vals = train.iloc[idx]["hr_per_ab_recent3"].to_numpy(dtype=float)

    ops_est = float(np.average(ops_vals, weights=w))
    hr_est = float(np.average(hr_vals, weights=w))

    if not np.isfinite(ops_est):
        return None, None, 0
    if not np.isfinite(hr_est):
        hr_est = None

    return ops_est, hr_est, int(k_eff)



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

    # Destination overrides (FA signing / relocation scenarios)
    override_team = legacy_kwargs.pop("override_team", None)
    override_stadium = legacy_kwargs.pop("override_stadium", None)
    home_game_share = float(legacy_kwargs.pop("home_game_share", 0.5) or 0.5)
    home_game_share = float(np.clip(home_game_share, 0.0, 1.0))

    pct_vs_rhp = float(legacy_kwargs.pop("pct_vs_rhp", 0.70) or 0.70)
    pct_vs_rhp = float(np.clip(pct_vs_rhp, 0.0, 1.0))


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

        # Rookie / small-sample detection (league rule): max season AB < 150
        max_ab = 0
        try:
            if recent is not None and not recent.empty and "ab" in recent.columns:
                max_ab = int(recent["ab"].fillna(0).max())
        except Exception:
            max_ab = 0

        is_rookie_ab150 = max_ab < 150
        ops_source = "mlb_recent3"
        ops_edits_knn = None
        hr_edits_knn = None
        knn_n_used = 0


        # ISO context (not projected): last-season and recent-3 (AB-weighted)
        iso_last = None
        iso_recent3 = None
        try:
            if recent is not None and not recent.empty and "avg" in recent.columns and "slg" in recent.columns:
                # last season iso
                r0 = recent.iloc[0]
                if r0.get("avg") is not None and r0.get("slg") is not None:
                    iso_last = float(r0["slg"]) - float(r0["avg"])

                # recent 3 weighted by AB
                rr = recent.copy()
                rr = rr[rr["ab"].fillna(0) > 0]
                if not rr.empty:
                    iso_series = (rr["slg"].astype(float) - rr["avg"].astype(float))
                    iso_recent3 = float((iso_series * rr["ab"].astype(float)).sum() / rr["ab"].astype(float).sum())
        except Exception:
            iso_last = None
            iso_recent3 = None

        # -----------------------------
        # Age handling
        # -----------------------------
        # Prefer the season-level age from the most recent season we have for the player,
        # then advance it by (target_year - last_season_year).
        # This makes ages move forward correctly when projecting multiple future seasons.
        age: Optional[int] = None
        last_season_year: Optional[int] = None
        age_last_season: Optional[int] = None
        try:
            cols_ps_age = _table_columns(conn, "player_season_batting")
            if "age" in cols_ps_age:
                r_age = conn.execute(
                    """
                    SELECT season_year, age
                    FROM player_season_batting
                    WHERE player_id=?
                    ORDER BY season_year DESC
                    LIMIT 1
                    """,
                    (int(pid),),
                ).fetchone()
                if r_age and r_age["season_year"] is not None:
                    last_season_year = int(r_age["season_year"])
                    age_last_season = int(r_age["age"]) if r_age["age"] is not None else None
        except sqlite3.OperationalError:
            pass

        if age_last_season is not None and last_season_year is not None:
            age = int(age_last_season + (int(target_year) - int(last_season_year)))
        else:
            # Fallback: parse the player page text. This is less reliable because it may be
            # the "current" age on the site (not season-anchored), but it's better than None.
            age = _parse_age_text(prow["birth_or_age_text"]) if prow else None

        # If we have no MLB seasons, try edits-based kNN (rookies/minors).
        if (ops_recent3 is None or ab_recent is None) or is_rookie_ab150:
            ops_est, hr_est, n_used = predict_from_edits_knn(conn, player_id=int(pid), age=age, k=30, min_ab=150)
            if ops_est is not None and np.isfinite(ops_est):
                ops_recent3 = float(ops_est)
                ab_recent = 150.0
                ops_source = "edits_knn"
                ops_edits_knn = float(ops_est)
                knn_n_used = int(n_used)
                if hr_est is not None and np.isfinite(hr_est):
                    hr_edits_knn = float(hr_est)
            else:
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
                        max_ab_season=int(max_ab),
                        is_rookie_ab150=bool(is_rookie_ab150),
                        ops_source=str(ops_source),
                        ops_edits_knn=ops_edits_knn,
                        hr_per_ab_edits_knn=hr_edits_knn,
                        knn_n_used=int(knn_n_used),
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
        # Destination overrides: allow projecting a FA in a new park/team
        team_for_park = override_team or (team_raw or team_full)
        stadium = override_stadium or resolve_stadium_for_team_year(conn, team_full or team_raw, int(target_year))

        # If we couldn't resolve via abbrev/full, try using the park team key for stadium lookup too
        if stadium is None and team_for_park:
            stadium = resolve_stadium_for_team_year(conn, team_for_park, int(target_year))

        # Destination overrides (FA signing scenario)
        team_for_park = override_team or (team_full or team_raw)

        # Stadium resolution:
        # - team_for_park is often an abbrev like "SF" (from season batting)
        # - team_stadiums.team should also be an abbrev, but in case a DB has full names,
        #   try both forms.
        if override_stadium:
            stadium = override_stadium
        else:
            stadium = resolve_stadium_for_team_year(conn, team_for_park, int(target_year))
            if stadium is None and team_full and team_full != team_for_park:
                stadium = resolve_stadium_for_team_year(conn, team_full, int(target_year))

        pf_year = int(target_year) - 1
        pf_year_used, home_idx, away_idx = get_park_pf_ops_for_team_year(conn, team_for_park, pf_year)
        pmult = park_multiplier(home_idx, away_idx)

        ops_tt_prepark = estimate_true_talent_ops(
            ops_recent=float(ops_recent3),
            ab_recent=float(ab_recent),
            lg_ops=float(lg.ops),
            k_ab=float(k_ab_v),
        )
        # -----------------------------
        # Age curve adjustment (affects projections)
        # -----------------------------
        # "Star" heuristic: if a player's regressed OPS talent is well above league, assume
        # a longer peak and slower decline.
        star = ops_tt_prepark >= float(lg.ops) + 0.100
        age_mult = age_curve_multiplier(age, star=bool(star))
        ops_tt_prepark *= float(age_mult)

        ops_tt_prepark *= (gb_effect_multiplier(gb_pct) ** 0.6)

        # -----------------------------
        # Platoon-aware OPS (vs RHP / vs LHP), combined by pct_vs_rhp
        # Uses player_splits_batting snapshot split keys:
        #   'vs RH Pitchers', 'vs LH Pitchers'
        # -----------------------------
        platoon = batter_platoon_splits(conn, int(pid))
        vs_r = platoon.get("vs RH Pitchers")
        vs_l = platoon.get("vs LH Pitchers")

        ops_tt_vs_rhp = None
        ops_tt_vs_lhp = None
        ops_vs_rhp_split = None
        ops_vs_lhp_split = None

        if vs_r and float(vs_r.get("ab", 0.0)) > 0:
            ops_vs_rhp_split = float(vs_r.get("ops", 0.0))
            ops_tt_vs_rhp = estimate_true_talent_ops(
                ops_recent=float(ops_vs_rhp_split),
                ab_recent=float(vs_r.get("ab", 0.0)),
                lg_ops=float(lg.ops),
                k_ab=float(k_ab_v),
            )

        if vs_l and float(vs_l.get("ab", 0.0)) > 0:
            ops_vs_lhp_split = float(vs_l.get("ops", 0.0))
            ops_tt_vs_lhp = estimate_true_talent_ops(
                ops_recent=float(ops_vs_lhp_split),
                ab_recent=float(vs_l.get("ab", 0.0)),
                lg_ops=float(lg.ops),
                k_ab=float(k_ab_v),
            )

        # If we have both platoon talents, combine them; otherwise keep existing ops_tt_prepark
        if ops_tt_vs_rhp is not None and ops_tt_vs_lhp is not None:
            ops_tt_prepark = float(pct_vs_rhp) * float(ops_tt_vs_rhp) + (1.0 - float(pct_vs_rhp)) * float(ops_tt_vs_lhp)


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
            if hr_edits_knn is not None:
                hr_rate_recent = float(hr_edits_knn)
                ab_hr_recent = 150.0
                hr_rate_adj = float(
                    (hr_rate_recent * ab_hr_recent + float(lg.hr_per_ab) * float(k_ab_v)) / (ab_hr_recent + float(k_ab_v))
                )
                hr_rate_adj *= (gb_effect_multiplier(gb_pct) ** 0.6)
            else:
                hr_rate_adj = estimate_hr_rate(recent, lg_hr_per_ab=float(lg.hr_per_ab), k_ab=float(k_ab_v), gb_pct=gb_pct)

        # Apply the same age curve to HR rate. Clamp to avoid extreme aging artifacts.
        hr_rate_adj = float(np.clip(hr_rate_adj * age_mult, 0.001, 0.200))

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
                ISO_last=iso_last,
                ISO_recent3=iso_recent3,
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
                pct_vs_rhp=float(pct_vs_rhp),
                ops_vs_rhp_split=ops_vs_rhp_split,
                ops_vs_lhp_split=ops_vs_lhp_split,
                ops_tt_vs_rhp_prepark=float(ops_tt_vs_rhp) if ops_tt_vs_rhp is not None else None,
                ops_tt_vs_lhp_prepark=float(ops_tt_vs_lhp) if ops_tt_vs_lhp is not None else None,
                max_ab_season=int(max_ab),
                is_rookie_ab150=bool(is_rookie_ab150),
                ops_source=str(ops_source),
                ops_edits_knn=ops_edits_knn,
                hr_per_ab_edits_knn=hr_edits_knn,
                knn_n_used=int(knn_n_used),


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
