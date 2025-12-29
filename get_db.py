"""get_db.py

Helpers to make Streamlit deployments work with a frequently-updated SQLite DB.

Recommended: host the DB as a public object (Cloudflare R2 / S3) and set:
  - st.secrets["DB_URL"]

The app will download the DB into the local Streamlit container filesystem
when missing or stale.

Why:
- Streamlit Community Cloud containers are ephemeral.
- The MLBC DB updates often (~19x per season), so committing mlbc.sqlite to git
  is inconvenient.

Usage (in Streamlit):

    from get_db import ensure_local_db

    db_path = ensure_local_db(
        url=st.secrets.get("DB_URL", ""),
        local_path="mlbc.sqlite",
        max_age_hours=6,
    )

"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class DownloadResult:
    downloaded: bool
    reason: str
    local_path: str


def _file_age_seconds(path: str) -> Optional[float]:
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return max(0.0, time.time() - float(st.st_mtime))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: str, *, timeout: int = 60) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    tmp = dst + ".tmp"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "mlbc-monte-carlo-sim/streamlit",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    os.replace(tmp, dst)


def ensure_local_db(*, url: str, local_path: str = "mlbc.sqlite", max_age_hours: float = 6.0) -> str:
    """Ensure a local SQLite file exists.

    If `url` is empty, this is a no-op (expects db already present).
    """
    url = (url or "").strip()
    if not url:
        return local_path

    max_age_seconds = float(max_age_hours) * 3600.0
    age = _file_age_seconds(local_path)

    if age is None:
        _download(url, local_path)
        return local_path

    if age > max_age_seconds:
        _download(url, local_path)
        return local_path

    return local_path


def ensure_local_db_with_status(*, url: str, local_path: str = "mlbc.sqlite", max_age_hours: float = 6.0) -> DownloadResult:
    """Like ensure_local_db(), but returns why it did/didn't download."""
    url = (url or "").strip()
    if not url:
        return DownloadResult(downloaded=False, reason="no_url_configured", local_path=local_path)

    max_age_seconds = float(max_age_hours) * 3600.0
    age = _file_age_seconds(local_path)

    if age is None:
        _download(url, local_path)
        return DownloadResult(downloaded=True, reason="missing", local_path=local_path)

    if age > max_age_seconds:
        _download(url, local_path)
        return DownloadResult(downloaded=True, reason=f"stale>{max_age_hours}h", local_path=local_path)

    return DownloadResult(downloaded=False, reason="fresh", local_path=local_path)
