import sqlite3
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import the engine from mlbc_project.py
from mlbc_project import connect, project_players


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
    k_ab = st.number_input("Regression prior (k-ab)", min_value=0, max_value=1000, value=150, step=10)

    st.divider()
    st.subheader("Free Agent / destination park")
    compare_mode = st.checkbox("Compare vs destination team park", value=False)
    destination_team = None
    if compare_mode:
        try:
            teams_df = cached_team_list(db_path)
            team_opts = teams_df["team"].dropna().tolist()
        except Exception:
            team_opts = []
        destination_team = st.selectbox("Destination team", options=[""] + team_opts, index=0)
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
        # Prefer aliases table if present; fallback to player_season_batting
        try:
            df = pd.read_sql_query(
                "SELECT DISTINCT team_full AS team FROM team_aliases WHERE team_full IS NOT NULL AND trim(team_full)<>'' ORDER BY team_full",
                conn,
            )
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            df = pd.read_sql_query(
                "SELECT DISTINCT team AS team FROM player_season_batting WHERE team IS NOT NULL AND trim(team)<>'' ORDER BY team",
                conn,
            )
    finally:
        conn.close()
    return df


@st.cache_data(show_spinner=False)
def cached_team_list(db: str) -> pd.DataFrame:
    return load_team_list(db)
def cached_player_list(db: str) -> pd.DataFrame:
    return load_player_list(db)


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

selected_ids: List[int] = st.multiselect(
    "Select players (up to 10 recommended)",
    options=filtered["player_id"].tolist(),
    format_func=lambda pid: f"{pid} - {players_df.loc[players_df['player_id']==pid, 'name'].values[0]}",
    default=[2030] if 2030 in filtered["player_id"].tolist() else [],
)

run = st.button("Run projections")

if run:
    if not selected_ids:
        st.warning("Pick at least one player.")
        st.stop()

    with st.spinner("Running simulations..."):
        conn = connect(db_path)
        df_base, lg = project_players(
            conn=conn,
            player_ids=selected_ids,
            target_year=int(target_year),
            sims=int(sims),
            seed=int(seed),
            k_ab_prior=int(k_ab),
            league_years=int(league_years),
        )

        if compare_mode and destination_team:
            df_dest, _ = project_players(
                conn=conn,
                player_ids=selected_ids,
                target_year=int(target_year),
                sims=int(sims),
                seed=int(seed),
                k_ab_prior=int(k_ab),
                league_years=int(league_years),
                override_team=str(destination_team),
            )

            # Merge and compute deltas
            df = df_base.merge(
                df_dest[["player_id", "team", "stadium", "park_mult", "OPS_p50", "HR_p50"]].rename(
                    columns={
                        "team": "dest_team",
                        "stadium": "dest_stadium",
                        "park_mult": "dest_park_mult",
                        "OPS_p50": "dest_OPS_p50",
                        "HR_p50": "dest_HR_p50",
                    }
                ),
                on="player_id",
                how="left",
            )
            df["delta_OPS_p50"] = (df["dest_OPS_p50"] - df["OPS_p50"]).round(3)
            df["delta_HR_p50"] = (df["dest_HR_p50"] - df["HR_p50"]).round(2)
        else:
            df = df_base

        conn.close()

    if df.empty:
    st.warning("No results for the selected player(s).")
    st.info(
        "This usually means the player has not been ingested into the database yet (missing rows in `player_season_batting`).\n\n"
        "Fix: update the DB by running the scraper with either:\n"
        "- `--ingest-all` (recommended for shared/hosted DB updates), or\n"
        "- `--players <player_id>` to ingest a specific free agent."
    )
    st.stop()

    st.subheader("Results table")
    st.dataframe(df, use_container_width=True)

    st.subheader("League context (recent)")
    st.write(
        {
            "OBP": round(lg.obp, 3),
            "SLG": round(lg.slg, 3),
            "OPS": round(lg.ops, 3),
            "HR/AB": round(lg.hr_per_ab, 4),
            "K/AB": round(lg.k_per_ab, 4),

        }
    )

    # Per-player distribution charts (rerun quick per player with same seed offset)
    st.subheader("Distributions")
    for _, row in df.iterrows():
        st.markdown(f"### {row['name']} ({int(row['player_id'])}) — {row.get('team','')} @ {row.get('stadium','')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OPS p10 / p50 / p90", f"{row['OPS_p10']:.3f} / {row['OPS_p50']:.3f} / {row['OPS_p90']:.3f}")
        c2.metric("HR p10 / p50 / p90", f"{int(row['HR_p10'])} / {int(row['HR_p50'])} / {int(row['HR_p90'])}")
        c3.metric("GB%", f"{row.get('gb_pct', 'NA')}")
        c4.metric("Var mult", f"{row.get('var_mult','NA')}")

    st.caption("Tip: Change sims/seed to see stability; raise sims for smoother percentiles.")
