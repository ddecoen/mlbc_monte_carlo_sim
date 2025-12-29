# Deployment (Streamlit)

This app is designed to be shared with a small group (e.g. ~20 league owners).

## Requirements

Streamlit Community Cloud installs dependencies from `requirements.txt`.

## Database strategy

The SQLite database (`mlbc.sqlite`) changes frequently (e.g. ~19 updates per season).
Instead of committing the DB into git on every update, prefer **hosting the DB as a downloadable artifact** and pulling it at app startup.

Recommended options:
- GitHub Releases asset (simple; versioned)
- S3 / R2 / GCS (best long-term)

## Suggested workflow

1) Run the scraper locally to produce a fresh DB:

```bash
python3 mlbc_scrape.py --db mlbc.sqlite
```

2) Upload the DB to your hosting location.

3) Configure the Streamlit app to download the latest DB into the local container on startup.

## Auto-download (implemented)

This repo includes `get_db.py` and `mlbc_app.py` will use it automatically when configured.

In Streamlit Community Cloud, set secrets:

- `DB_URL`: public URL to your hosted `mlbc.sqlite`
- `DB_MAX_AGE_HOURS`: (optional) default 6

The app will download `mlbc.sqlite` if missing or stale.
