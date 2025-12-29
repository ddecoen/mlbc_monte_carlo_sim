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
  - **Park factors** (home park + away baseline)
  - **Batted-ball profile** (GB% impacts HR/SLG/OPS tails)
  - Variance scaling (fat/skinny tails)
- Produces percentile outcomes:
  - OPS p10 / p50 / p90
  - HR p10 / p50 / p90

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

