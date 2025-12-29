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

with st.sidebar:
    db_path = st.text_input("SQLite DB path", value="mlbc.sqlite")
    target_year = st.number_input("Target year", min_value=2000, max_value=9999, value=2141, step=1)
    sims = st.number_input("Simulations", min_value=1000, max_value=200000, value=20000, step=1000)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    k_ab = st.number_input("Regression prior (k-ab)", min_value=0, max_value=1000, value=150, step=10)
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


@st.cache_data(show_spinner=False)
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
        df, lg = project_players(
            conn=conn,
            player_ids=selected_ids,
            target_year=int(target_year),
            sims=int(sims),
            seed=int(seed),
            k_ab_prior=int(k_ab),
            league_years=int(league_years),
        )
        conn.close()

    if df.empty:
        st.warning("No results (players may not have season batting rows).")
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
