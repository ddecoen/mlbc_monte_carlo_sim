import sqlite3
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def load_team_list(db: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    try:
        teams = pd.read_sql_query(
            """
            SELECT DISTINCT team AS abbrev
            FROM player_season_batting
            WHERE team IS NOT NULL AND trim(team) <> ''
            ORDER BY team
            """,
            conn,
        )

        # Optional: map abbrev -> team_full if team_aliases exists.
        try:
            aliases = pd.read_sql_query("SELECT abbrev, team_full FROM team_aliases", conn)
            teams = teams.merge(aliases, on="abbrev", how="left")
        except Exception:
            teams["team_full"] = None

        def _label(row) -> str:
            ab = str(row["abbrev"])
            tf = row.get("team_full", None)
            if tf is not None and str(tf).strip():
                return f"{ab} — {str(tf).strip()}"
            return ab

        teams["label"] = teams.apply(_label, axis=1)
        return teams
    finally:
        conn.close()

@st.cache_data(show_spinner=False)
def cached_team_list(db: str) -> pd.DataFrame:
    return load_team_list(db)

def load_stadium_list(db: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT stadium
            FROM team_stadiums
            WHERE stadium IS NOT NULL AND trim(stadium) <> ''
            ORDER BY stadium
            """,
            conn,
        )
        return df
    finally:
        conn.close()

@st.cache_data(show_spinner=False)
def cached_stadium_list(db: str) -> pd.DataFrame:
    return load_stadium_list(db)


# Import the engine from mlbc_project.py
from mlbc_project import connect, project_players, project_pitchers


st.set_page_config(page_title="MLBC Projections", layout="wide")

st.title("MLBCSimLeague Projections (Monte Carlo)")
st.caption("OPS + HR distributions using recent-3 seasons, regression to league, GB% batted-ball, age curve, and home/away park factors.")

from get_db import ensure_local_db_with_status, ensure_local_db

# Streamlit raises StreamlitSecretNotFoundError when st.secrets is accessed
# without a configured secrets.toml. Make local runs work without any secrets.
try:
    _secrets = st.secrets
except Exception:
    _secrets = {}

with st.sidebar:
    # If you set secrets DB_URL, the app can auto-download a frequently-updated DB.
    # Otherwise it falls back to a local path.
    db_url = _secrets.get("DB_URL", "")
    max_age_hours = float(_secrets.get("DB_MAX_AGE_HOURS", 6))

    default_db = ensure_local_db_with_status(
        url=db_url,
        local_path="mlbc.sqlite",
        max_age_hours=max_age_hours,
    )

    # Allow forcing a refresh on update days.
    c_refresh, c_status = st.columns([1, 2])
    with c_refresh:
        force_refresh = st.button("Force refresh DB", disabled=(not bool(db_url)))
    with c_status:
        st.caption(f"DB cache: {default_db.reason}")

    if force_refresh and db_url:
        # Clear Streamlit caches so new DB is reflected immediately.
        try:
            ensure_local_db(url=db_url, local_path="mlbc.sqlite", max_age_hours=0)
        finally:
            st.cache_data.clear()
            st.cache_resource.clear()
        st.rerun()

    db_path = st.text_input("SQLite DB path", value=default_db.local_path)
    target_year = st.number_input("Target year", min_value=2000, max_value=9999, value=2141, step=1)
    sims = st.number_input("Simulations", min_value=1000, max_value=200000, value=20000, step=1000)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

    st.divider()
    st.subheader("Mode")
    mode = st.radio("Projection type", options=["Hitters", "Pitchers"], horizontal=True)

    # Pitcher-only knobs
    if mode == "Pitchers":
        k_ip = st.number_input("Regression prior (k-ip)", min_value=0.0, max_value=1000.0, value=100.0, step=10.0)
    else:
        k_ip = None
    k_ab = st.number_input("Regression prior (k-ab)", min_value=0, max_value=1000, value=150, step=10)

    if mode != "Pitchers":
        pct_vs_rhp = st.slider("% PA vs RHP (platoon mix)", 0.0, 1.0, 0.70, 0.01, key="pct_vs_rhp",)
    else:
        pct_vs_rhp = None


    st.divider()
    st.subheader("Free Agent / destination park")
    compare_mode = st.checkbox("Compare vs destination park", value=False)

    destination_mode = "Team"
    destination_team = None
    destination_stadium = None

    if compare_mode:
        destination_mode = st.radio(
            "Destination selection",
            options=["Team", "Stadium"],
            horizontal=True,
            help="Pick a destination team (uses that team's park factors) or pick a stadium directly (useful for relocations).",
        )

        if destination_mode == "Team":
            # Always populate from teams actually present in batting table (abbrevs).
            try:
                teams_df = cached_team_list(db_path)
                team_opts = teams_df["abbrev"].dropna().tolist()
                _label_map = dict(zip(teams_df["abbrev"], teams_df["label"]))
            except Exception:
                team_opts = []
                _label_map = {}

            destination_team = st.selectbox(
                "Destination team",
                options=[""] + team_opts,
                index=0,
                format_func=lambda t: "" if t == "" else _label_map.get(t, t),
            )

        else:
            # Stadium mode: pick a stadium directly. We'll resolve the tenant team (if any)
            # for target_year and use that team's park factors as a proxy.
            try:
                stadiums = cached_stadium_list(db_path)
                stadium_opts = stadiums["stadium"].dropna().tolist()
            except Exception:
                stadium_opts = []

            destination_stadium = st.selectbox(
                "Destination stadium",
                options=[""] + stadium_opts,
                index=0,
            )
    league_years = st.number_input("League mean window (years)", min_value=1, max_value=10, value=3, step=1)


def load_player_list(db: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(
            "SELECT player_id, COALESCE(name, 'player_' || player_id) AS name FROM players ORDER BY name",
            conn,
        )
    finally:
        conn.close()
    return df


def load_team_list(db: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    try:
        # Always start from teams that actually appear in the batting table.
        teams = pd.read_sql_query(
            """
            SELECT DISTINCT team AS abbrev
            FROM player_season_batting
            WHERE team IS NOT NULL AND trim(team) <> ''
            ORDER BY team
            """,
            conn,
        )

        # Optional: map abbrev -> team_full if team_aliases exists.
        try:
            aliases = pd.read_sql_query(
                "SELECT abbrev, team_full FROM team_aliases",
                conn,
            )
            teams = teams.merge(aliases, on="abbrev", how="left")
        except Exception:
            teams["team_full"] = None

        def _label(row) -> str:
            ab = str(row["abbrev"])
            tf = row.get("team_full", None)
            if tf is not None and str(tf).strip():
                return f"{ab} — {str(tf).strip()}"
            return ab

        teams["label"] = teams.apply(_label, axis=1)
        return teams
    finally:
        conn.close()


def load_stadium_list(db: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT stadium
            FROM team_stadiums
            WHERE stadium IS NOT NULL AND trim(stadium) <> ''
            ORDER BY stadium
            """,
            conn,
        )
    finally:
        conn.close()
    return df

@st.cache_data(show_spinner=False)
def cached_stadium_list(db: str) -> pd.DataFrame:
    return load_stadium_list(db)


@st.cache_data(show_spinner=False)
def cached_team_list(db: str) -> pd.DataFrame:
    return load_team_list(db)


@st.cache_data(show_spinner=False)
def cached_player_list(db: str) -> pd.DataFrame:
    return load_player_list(db)


def load_stadium_list(db: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT stadium
            FROM team_stadiums
            WHERE stadium IS NOT NULL AND trim(stadium) <> ''
            ORDER BY stadium
            """,
            conn,
        )
    finally:
        conn.close()
    return df


@st.cache_data(show_spinner=False)
def cached_stadium_list(db: str) -> pd.DataFrame:
    return load_stadium_list(db)


def plot_hist(arr: np.ndarray, title: str):
    fig = plt.figure()
    plt.hist(arr, bins=40)
    plt.title(title)
    st.pyplot(fig)


# Load players
try:
    players_df = cached_player_list(db_path)
except Exception as e:
    st.error(f"Could not read players table from DB: {e}")
    st.stop()

# Search + select
q = st.text_input("Search player name (substring)", value="Sung")
filtered = players_df[players_df["name"].str.contains(q, case=False, na=False)].head(50)

# IMPORTANT: keep selection options stable.
# If options are tied to the filtered list, Streamlit reruns can clear selections,
# making it look like "Run projections" does nothing.
all_player_ids = players_df["player_id"].tolist()

st.caption(f"Search matches (showing up to 50): {len(filtered)}")

selected_ids: List[int] = st.multiselect(
    "Select players (up to 10 recommended)",
    options=all_player_ids,
    format_func=lambda pid: f"{pid} - {players_df.loc[players_df['player_id']==pid, 'name'].values[0]}",
    default=[2030] if 2030 in all_player_ids else [],
)

# Convenience quick-pick from current filtered results
quick_pick = st.selectbox(
    "Quick pick from search results",
    options=[""] + filtered["player_id"].tolist(),
    index=0,
    format_func=lambda pid: "" if pid == "" else f"{pid} - {players_df.loc[players_df['player_id']==pid, 'name'].values[0]}",
)
if quick_pick:
    pid = int(quick_pick)
    if pid not in selected_ids:
        selected_ids = selected_ids + [pid]

run = st.button("Run projections")

if run:
    if not selected_ids:
        st.warning("Pick at least one player.")
        st.stop()

    with st.spinner("Running simulations..."):
        conn = connect(db_path)
        if mode == "Pitchers":
            df_base, lg = project_pitchers(
                conn=conn,
                player_ids=selected_ids,
                target_year=int(target_year),
                sims=int(sims),
                seed=int(seed),
                k_ip=float(k_ip or 100.0),
                lg_years=int(league_years),
            )
        else:
            df_base, lg = project_players(
                conn=conn,
                player_ids=selected_ids,
                target_year=int(target_year),
                sims=int(sims),
                seed=int(seed),
                k_ab_prior=int(k_ab),
                league_years=int(league_years),
                pct_vs_rhp=float(pct_vs_rhp or 0.70),
            )

        if mode != "Pitchers" and compare_mode and (destination_team or destination_stadium):
            # Stadium-mode: resolve the "tenant" team for the chosen stadium in target_year
            # and use that team's park factors as a proxy.
            override_team = None
            override_stadium = None

            if destination_mode == "Team" and destination_team:
                override_team = str(destination_team)
            elif destination_mode == "Stadium" and destination_stadium:
                override_stadium = str(destination_stadium)
                try:
                    r = conn.execute(
                        """
                        SELECT team
                        FROM team_stadiums
                        WHERE stadium = ?
                          AND (? BETWEEN COALESCE(since_year, 0) AND COALESCE(through_year, 9999))
                        LIMIT 1
                        """,
                        (override_stadium, int(target_year)),
                    ).fetchone()
                    if r and r["team"]:
                        # team_stadiums.team is typically full name; map to abbrev if possible
                        rr = conn.execute(
                            "SELECT abbrev FROM team_aliases WHERE team_full=? COLLATE NOCASE LIMIT 1",
                            (str(r["team"]),),
                        ).fetchone()
                        override_team = str(rr["abbrev"]) if rr and rr["abbrev"] else None
                except Exception:
                    override_team = None

        if mode != "Pitchers" and compare_mode and (destination_team or destination_stadium):
            # Stadium-mode: resolve tenant team for the chosen stadium in target_year and
            # use that team's PF indices as a proxy. Also override displayed stadium.
            override_team = None
            override_stadium = None

            if destination_mode == "Team" and destination_team:
                override_team = str(destination_team)

            elif destination_mode == "Stadium" and destination_stadium:
                override_stadium = str(destination_stadium)

                # Find tenant team (full name) for that stadium/year
                try:
                    r = conn.execute(
                        """
                        SELECT team
                        FROM team_stadiums
                        WHERE stadium = ?
                          AND (? BETWEEN COALESCE(since_year, 0) AND COALESCE(through_year, 9999))
                        LIMIT 1
                        """,
                        (override_stadium, int(target_year)),
                    ).fetchone()

                    if r and r["team"]:
                        # team_stadiums.team is usually full name; map to abbrev if possible
                        rr = conn.execute(
                            "SELECT abbrev FROM team_aliases WHERE team_full=? COLLATE NOCASE LIMIT 1",
                            (str(r["team"]),),
                        ).fetchone()
                        if rr and rr["abbrev"]:
                            override_team = str(rr["abbrev"])
                except Exception:
                    pass

            st.sidebar.caption(f"DEBUG override_team={override_team} override_stadium={override_stadium}")

            df_dest, _ = project_players(
                conn=conn,
                player_ids=selected_ids,
                target_year=int(target_year),
                sims=int(sims),
                seed=int(seed),
                k_ab_prior=int(k_ab),
                league_years=int(league_years),
                override_team=override_team,
                override_stadium=override_stadium,
                pct_vs_rhp=float(pct_vs_rhp or 0.70),

            )

            # Merge and compute deltas
            df = df_base.merge(
                df_dest[["player_id", "team", "stadium", "park_mult", "OPS_p10", "OPS_p50", "OPS_p90", "HR_p10", "HR_p50", "HR_p90"]].rename(
                    columns={
                        "team": "dest_team",
                        "stadium": "dest_stadium",
                        "park_mult": "dest_park_mult",
                        "OPS_p50": "dest_OPS_p50",
                        "HR_p50": "dest_HR_p50",
                        "OPS_p10": "dest_OPS_p10",
                        "OPS_p90": "dest_OPS_p90",
                        "HR_p10": "dest_HR_p10",
                        "HR_p90": "dest_HR_p90",
                    }
                ),
                on="player_id",
                how="left",
            )
            df["delta_OPS_p50"] = (df["dest_OPS_p50"] - df["OPS_p50"]).round(3)
            df["delta_HR_p50"] = (df["dest_HR_p50"] - df["HR_p50"]).round(2)
            df["delta_OPS_p10"] = (df["dest_OPS_p10"] - df["OPS_p10"]).round(3)
            df["delta_OPS_p90"] = (df["dest_OPS_p90"] - df["OPS_p90"]).round(3)
            df["delta_HR_p10"] = (df["dest_HR_p10"] - df["HR_p10"]).round(2)
            df["delta_HR_p90"] = (df["dest_HR_p90"] - df["HR_p90"]).round(2)

                # If we didn't build a destination-compare df, fall back to base.
        if mode == "Pitchers":
            if compare_mode:
                st.info("Destination park comparison is currently implemented for hitters only.")
            df = df_base
        else:
            # Hitters: keep df as-is if we built it; otherwise default to base.
            if not (compare_mode and (destination_team or destination_stadium)):
                df = df_base


        conn.close()

    if df.empty:
        st.warning("No results for the selected player(s).")
        st.info(
            "This usually means the player has not been ingested into the database yet (missing rows in `player_season_batting`).\n\n"
            "Fix: update the DB by running the scraper with either:\n"
            "- `--ingest-all` (recommended for shared/hosted DB updates), or\n"
            "- `--players <player_id>` to ingest a specific free agent.\n\n"
            "For commissioners: for hosted DB updates, run `--discover-players` then `--ingest-all`."
        )
        st.stop()

    st.subheader("Results table")
    st.dataframe(df, use_container_width=True)

    st.subheader("League context (recent)")
    if mode == "Pitchers":
        st.write(
            {
                "ERA": round(lg.era, 3),
                "WHIP": round(lg.whip, 3),
                "OPS allowed": round(lg.ops_allowed, 3),
                "K/IP": round(lg.k_per_ip, 3),
                "BB/IP": round(lg.bb_per_ip, 3),
            }
        )
    else:
        st.write(
            {
                "OBP": round(lg.obp, 3),
                "SLG": round(lg.slg, 3),
                "OPS": round(lg.ops, 3),
                "HR/AB": round(lg.hr_per_ab, 4),
                "K/AB": round(lg.k_per_ab, 4),
            }
        )

    # Per-player distribution charts
    st.subheader("Distributions")
    for _, row in df.iterrows():
        if mode == "Pitchers":
            st.markdown(f"### {row['name']} ({int(row['player_id'])}) — {row.get('team','')}")
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            c1.metric("ERA p10 / p50 / p90", f"{row['ERA_p10']:.2f} / {row['ERA_p50']:.2f} / {row['ERA_p90']:.2f}")
            c2.metric(
                "OPS allowed p10 / p50 / p90",
                f"{row['OPS_allowed_p10']:.3f} / {row['OPS_allowed_p50']:.3f} / {row['OPS_allowed_p90']:.3f}",
            )
            c3.metric("WHIP p10 / p50 / p90", f"{row['WHIP_p10']:.2f} / {row['WHIP_p50']:.2f} / {row['WHIP_p90']:.2f}")
            c4.metric("K/BB p10 / p50 / p90", f"{row['KBB_p10']:.2f} / {row['KBB_p50']:.2f} / {row['KBB_p90']:.2f}")

            # Context row
            cx1, cx2, cx3 = st.columns(3)
            cx1.metric("IP proj", f"{row.get('ip_proj', float('nan')):.1f}")
            cx2.metric("GS last", f"{int(row.get('gs_last', 0))}")
            cx3.metric("Starter?", "SP" if bool(row.get('is_sp', False)) else "RP")
        else:
            # Header: include destination context if present
            header = f"### {row['name']} ({int(row['player_id'])}) — {row.get('team','')} @ {row.get('stadium','')}"
            if compare_mode and pd.notna(row.get("dest_team")):
                header += f"  →  {row.get('dest_team','')} @ {row.get('dest_stadium','')}"
            st.markdown(header)

            c1, c2, c3, c4 = st.columns(4)

            # Base metrics
            c1.metric(
                "OPS p10 / p50 / p90",
                f"{row['OPS_p10']:.3f} / {row['OPS_p50']:.3f} / {row['OPS_p90']:.3f}",
            )
            def _fmt_int(x):
                try:
                    if x is None or (isinstance(x, float) and np.isnan(x)):
                        return "NA"
                    return str(int(x))
                except Exception:
                    return "NA"

            c2.metric(
                "HR p10 / p50 / p90",
                f"{_fmt_int(row.get('HR_p10'))} / {_fmt_int(row.get('HR_p50'))} / {_fmt_int(row.get('HR_p90'))}",
            )


            # Destination metrics (only if compare ran)
            if compare_mode and pd.notna(row.get("dest_OPS_p50", np.nan)):
                c3.metric(
                    "Dest OPS p10 / p50 / p90",
                    f"{row['dest_OPS_p10']:.3f} / {row['dest_OPS_p50']:.3f} / {row['dest_OPS_p90']:.3f}",
                    delta=f"{row.get('delta_OPS_p50', 0.0):+.3f}",
                )
                c4.metric(
                    "Dest HR p10 / p50 / p90",
                    f"{int(row['dest_HR_p10'])} / {int(row['dest_HR_p50'])} / {int(row['dest_HR_p90'])}",
                    delta=f"{row.get('delta_HR_p50', 0.0):+.2f}",
                )
            else:
                c3.metric("GB%", f"{row.get('gb_pct', 'NA')}")
                c4.metric("Var mult", f"{row.get('var_mult','NA')}")


    st.caption("Tip: Change sims/seed to see stability; raise sims for smoother percentiles.")
