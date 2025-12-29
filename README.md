# MLBC Monte Carlo Sim (High Heat 2003)

A Fangraphs-style projection engine for the **MLBC Sim League** (High Heat Baseball 2003).  
This project scrapes league/player pages into a **SQLite** database, then generates hitter projections using a **Monte Carlo simulation** approach with league context + park effects.

## What it does

### Data pipeline (SQLite)
- Discovers player IDs from league leader pages (supports paging beyond the top 50)
- Scrapes player cards and season batting lines
- Scrapes player splits (home/away + monthly)
- Scrapes team → stadium history (year-bounded)
- Stores everything in a single SQLite DB for repeatable modeling

### Projection model (Monte Carlo)
- Uses a rolling recent-performance baseline (e.g., last 3 seasons)
- Regresses toward league average based on a configurable prior (`--k-ab`)
- Adjusts outcomes for:
  - **Park factors** (home/away blended)
    - computed from MLBC home/away splits
    - optionally blended with a **Baseball Savant prior** (index_wOBA) for stability in low-sample stadium-years
  - **Batted-ball profile** (GB% impacts HR/SLG/OPS tails)
  - Variance scaling (fat/skinny tails)
- Produces percentile outcomes:
  - OPS p10 / p50 / p90
  - HR p10 / p50 / p90


## Park factors (MLBC + optional Savant prior)

### Build / update Savant priors (CSV)

```bash
python3 savant_park_factors_to_csv.py --year 2002 --batside L --rolling 3 \
  --stat index_wOBA --out data/savant_park_factors_prior.csv
```

### Compute MLBC park factors (with optional Savant blending)

```bash
python3 mlbc_parkfactors.py --db mlbc.sqlite \
  --savant-prior-csv data/savant_park_factors_prior.csv \
  --savant-k-ab 2500
```

Notes:
- Stadium→venue mapping is automatic (normalized + fuzzy match + a small alias list in code).
- The blended PF and diagnostics are stored in `park_factors_team_year`:
  `pf_ops_index` (final), plus `pf_ops_index_mlbc`, `pf_ops_index_prior`, `pf_ops_blend_w`.

## Free agent / destination stadium projections

In Streamlit, enable **Compare vs destination team park** and choose a destination team.
The app will show baseline results and destination deltas (OPS_p50 / HR_p50).

You can also call `project_players(..., override_team="...")` directly.
### Interfaces
- **CLI**: run projections for one or many player IDs
- **Streamlit app**: interactive table and league context

---

## Repo layout (expected)

```text
.
├── mlbc_scrape.py          # Scrape + ingest into SQLite
├── mlbc_project.py         # Projection engine + CLI entrypoint
├── mlbc_app.py             # Streamlit UI
├── mlbc.sqlite             # Generated DB (optional to commit; usually ignored)
├── requirements.txt
└── README.md

