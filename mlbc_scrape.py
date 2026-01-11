#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

BASE = "https://mlbcsimleague.com"


# ----------------------------
# DB schema
# ----------------------------
SCHEMA = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS player_ids (
  player_id INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS players (
  player_id INTEGER PRIMARY KEY,
  name TEXT,
  bats TEXT,
  throws TEXT,
  primary_pos TEXT,
  birth_or_age_text TEXT,
  gb_pct REAL,
  last_seen_ts INTEGER
);

CREATE TABLE IF NOT EXISTS player_season_batting (
  player_id INTEGER NOT NULL,
  season_year INTEGER NOT NULL,
  team TEXT,
  age INTEGER,
  ab INTEGER,
  h INTEGER,
  doubles INTEGER,
  triples INTEGER,
  hr INTEGER,
  bb INTEGER,
  k INTEGER,
  avg REAL,
  obp REAL,
  slg REAL,
  ops REAL,
  PRIMARY KEY(player_id, season_year)
);

CREATE TABLE IF NOT EXISTS player_splits_batting (
  player_id INTEGER NOT NULL,
  split_key TEXT NOT NULL,
  ab INTEGER,
  h INTEGER,
  doubles INTEGER,
  triples INTEGER,
  hr INTEGER,
  bb INTEGER,
  k INTEGER,
  avg REAL,
  obp REAL,
  slg REAL,
  ops REAL,
  PRIMARY KEY(player_id, split_key)
);

CREATE TABLE IF NOT EXISTS team_stadiums (
  team TEXT NOT NULL,
  stadium TEXT NOT NULL,
  since_year INTEGER,
  through_year INTEGER,
  PRIMARY KEY(team, stadium, since_year, through_year)
);


CREATE TABLE IF NOT EXISTS player_season_pitching (
  player_id INTEGER NOT NULL,
  season_year INTEGER NOT NULL,
  team TEXT,
  w INTEGER,
  l INTEGER,
  era REAL,
  sv INTEGER,
  bs INTEGER,
  g INTEGER,
  gs INTEGER,
  cg INTEGER,
  sho INTEGER,
  ip REAL,
  h INTEGER,
  k INTEGER,
  bb INTEGER,
  ops_allowed REAL,
  whip REAL,
  PRIMARY KEY(player_id, season_year)
);

CREATE TABLE IF NOT EXISTS player_splits_pitching (
  player_id INTEGER NOT NULL,
  split_group TEXT NOT NULL,
  split_key TEXT NOT NULL,
  team TEXT,
  w INTEGER,
  l INTEGER,
  era REAL,
  sv INTEGER,
  bs INTEGER,
  g INTEGER,
  gs INTEGER,
  cg INTEGER,
  sho INTEGER,
  ip REAL,
  h INTEGER,
  doubles INTEGER,
  triples INTEGER,
  hr INTEGER,
  bb INTEGER,
  k INTEGER,
  avg REAL,
  obp REAL,
  slg REAL,
  ops REAL,
  whip REAL,
  PRIMARY KEY(player_id, split_group, split_key)
);

CREATE TABLE IF NOT EXISTS team_roster_players (
  team TEXT NOT NULL,
  player_id INTEGER NOT NULL,
  scraped_ts INTEGER,
  PRIMARY KEY(team, player_id)
);

CREATE TABLE IF NOT EXISTS team_aliases (
  abbrev TEXT PRIMARY KEY,
  team_full TEXT NOT NULL
);
"""


def db_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    _ensure_column(conn, "players", "gb_pct", "REAL")
    conn.commit()
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, coltype: str) -> None:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()]
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype};")


# ----------------------------
# HTTP fetcher (stdlib only)
# ----------------------------
@dataclass
class Fetcher:
    timeout: float = 25.0
    user_agent: str = "Mozilla/5.0 (mlbc-scraper)"
    sleep: float = 0.0

    def get(self, url: str) -> str:
        if self.sleep and self.sleep > 0:
            time.sleep(self.sleep)
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read()
        return raw.decode("utf-8", errors="replace")


# ----------------------------
# Helpers
# ----------------------------
PLAYER_ID_RE = re.compile(r"pcard\.php\?id=(\d+)", re.I)
PCARD_SPLIT_ID_RE = re.compile(r"pcardsplit\.php\?id=(\d+)", re.I)
ROSTER_ID_RE = re.compile(r"pcard\.php\?id=(\d+)", re.I)


def _read_html_tables(page_html: str) -> List[pd.DataFrame]:
    return pd.read_html(StringIO(page_html))


def _norm_col(c) -> str:
    s = str(c) if c is not None else ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _collapse_text(page_html: str) -> str:
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", page_html)
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", page_html)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


    # Existing discovery scrapes leaderboard pages.
    # For team roster pages, we also need to parse JavaScript links like:
    #   javascript:pcards('pcard.php?id=2030%27,%27pcard2030%27,495,705)

def _extract_player_ids_from_html(page_html: str) -> List[int]:
    ids = set(int(m.group(1)) for m in PLAYER_ID_RE.finditer(page_html))
    ids.update(int(m.group(1)) for m in PCARD_SPLIT_ID_RE.finditer(page_html))
    return sorted(ids)


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return None
        return float(s)
    except Exception:
        return None




def _safe_token(x: str) -> Optional[str]:
    s = (x or "").strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    return s


def _tokenize_collapsed_table_text(text: str) -> List[str]:
    # collapsed text already has whitespace normalized; split on spaces
    return [t for t in (text or "").split(" ") if t != ""]


def _parse_pitcher_platoon_splits_from_text(text: str, player_id: int) -> List[Dict]:
    # Header sequence we expect
    header = [
        "split", "era", "ip", "h", "2b", "3b", "hr", "bb", "k",
        "avg", "obp", "slg", "ops", "whip",
    ]

    tokens = _tokenize_collapsed_table_text(text)
    # Find header start
    def find_seq(seq):
        seq = [s.lower() for s in seq]
        for i in range(0, len(tokens) - len(seq)):
            if [t.lower() for t in tokens[i:i+len(seq)]] == seq:
                return i
        return None

    hi = find_seq(header)
    if hi is None:
        return []

    out: List[Dict] = []
    i = hi + len(header)

    # Rows look like: vs RH Hitters <era> <ip> <h> <2b> <3b> <hr> <bb> <k> <avg> <obp> <slg> <ops> <whip>
    # We stop when we reach the situational header 'Split Team W L ...'
    stop = ["split", "team", "w", "l", "era"]

    while i < len(tokens):
        # stop check
        if i + len(stop) <= len(tokens) and [t.lower() for t in tokens[i:i+len(stop)]] == stop:
            break

        # split key is non-numeric; may be multiple tokens starting with 'vs'
        if tokens[i].lower() != 'vs':
            # defensive skip
            i += 1
            continue

        # collect split_key tokens until we have enough remaining numeric tokens
        # platoon keys we expect: vs RH Hitters / vs LH Hitters
        if i + 3 >= len(tokens):
            break
        split_key = " ".join(tokens[i:i+3])
        i += 3

        vals = tokens[i:i+13]
        if len(vals) < 13:
            break
        i += 13

        row = {
            "player_id": player_id,
            "split_group": "platoon",
            "split_key": split_key,
            "era": _safe_float(vals[0]),
            "ip": _safe_float(vals[1]),
            "h": _safe_int(vals[2]) or 0,
            "doubles": _safe_int(vals[3]) or 0,
            "triples": _safe_int(vals[4]) or 0,
            "hr": _safe_int(vals[5]) or 0,
            "bb": _safe_int(vals[6]) or 0,
            "k": _safe_int(vals[7]) or 0,
            "avg": _safe_float(vals[8]),
            "obp": _safe_float(vals[9]),
            "slg": _safe_float(vals[10]),
            "ops": _safe_float(vals[11]),
            "whip": _safe_float(vals[12]),
        }
        out.append(row)

    return out


def _parse_pitcher_situational_splits_from_text(text: str, player_id: int) -> List[Dict]:
    header = [
        "split", "team", "w", "l", "era", "sv", "bs", "g", "gs", "sho", "cg",
        "ip", "h", "k", "bb", "avg", "whip",
    ]

    tokens = _tokenize_collapsed_table_text(text)

    def find_seq(seq):
        seq = [s.lower() for s in seq]
        for i in range(0, len(tokens) - len(seq)):
            if [t.lower() for t in tokens[i:i+len(seq)]] == seq:
                return i
        return None

    hi = find_seq(header)
    if hi is None:
        return []

    i = hi + len(header)
    out: List[Dict] = []

    # split_key is variable-length until we hit a team abbrev token.
    # team tokens look like 2-4 uppercase letters.
    TEAM_RE = re.compile(r"^[A-Z]{2,4}$")

    while i < len(tokens):
        # parse split_key tokens until TEAM token
        start = i
        while i < len(tokens) and not TEAM_RE.match(tokens[i]):
            i += 1
        if i >= len(tokens):
            break
        split_key = " ".join(tokens[start:i]).strip()
        team = tokens[i]
        i += 1

        vals = tokens[i:i+14]
        if len(vals) < 14:
            break
        i += 14

        out.append(
            {
                "player_id": player_id,
                "split_group": "situational",
                "split_key": split_key,
                "team": team,
                "w": _safe_int(vals[0]) or 0,
                "l": _safe_int(vals[1]) or 0,
                "era": _safe_float(vals[2]),
                "sv": _safe_int(vals[3]) or 0,
                "bs": _safe_int(vals[4]) or 0,
                "g": _safe_int(vals[5]) or 0,
                "gs": _safe_int(vals[6]) or 0,
                "sho": _safe_int(vals[7]) or 0,
                "cg": _safe_int(vals[8]) or 0,
                "ip": _safe_float(vals[9]),
                "h": _safe_int(vals[10]) or 0,
                "k": _safe_int(vals[11]) or 0,
                "bb": _safe_int(vals[12]) or 0,
                "avg": _safe_float(vals[13]),
                "whip": _safe_float(vals[14]) if len(vals) > 14 else None,
            }
        )

    return out


def scrape_player_splits_pitching(fetcher: Fetcher, player_id: int) -> List[Dict]:
    url = f"{BASE}/pcardsplit.php?id={player_id}"
    html_page = fetcher.get(url)
    text = _collapse_text(html_page)

    out: List[Dict] = []
    out.extend(_parse_pitcher_platoon_splits_from_text(text, player_id))
    out.extend(_parse_pitcher_situational_splits_from_text(text, player_id))
    return out
def _safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return None
        return int(float(s))
    except Exception:
        return None


def _promote_first_row_to_header_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or df.shape[0] < 2:
        return df

    cols_norm = [_norm_col(c) for c in df.columns]
    numericish = all(re.fullmatch(r"\d+", c) or c in ("0", "1", "2") for c in cols_norm)
    if not numericish:
        return df

    first_row = [str(x).strip() for x in df.iloc[0].tolist()]
    first_norm = [_norm_col(x) for x in first_row]
    header_tokens = {"year", "season", "yr", "ab", "avg", "obp", "slg", "ops", "hr", "bb", "so", "k", "h", "2b", "3b"}
    hits = sum(1 for x in first_norm if any(tok == x or tok in x for tok in header_tokens))
    if hits >= 2:
        new_df = df.iloc[1:].copy()
        new_df.columns = first_row
        return new_df.reset_index(drop=True)

    return df




def _score_season_pitching_table(df: pd.DataFrame) -> int:
    """Heuristic score for identifying the pitching season table on pcard.php."""
    if df is None or df.empty or df.shape[1] < 6:
        return -999

    cols = [_norm_col(c) for c in df.columns]
    colset = set(cols)

    score = 0
    if any(c in ("year", "season", "yr") or "year" in c for c in cols):
        score += 6
    else:
        score += 1

    # pitching-specific signals
    for k in ("era", "ip", "whip"):
        if k in colset:
            score += 5

    if "ops" in colset or "ops allowed" in colset:
        score += 3

    for k in ("g", "gs", "w", "l", "bb", "k", "so", "h"):
        if k in colset:
            score += 1

    if "ab" in colset:
        score -= 5  # batting table probably

    return score


def _find_best_season_pitching_df(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    best_df = None
    best_score = -9999
    for df in dfs:
        df2 = _promote_first_row_to_header_if_needed(df)
        df2.columns = [str(c) for c in df2.columns]
        sc = _score_season_pitching_table(df2)
        if sc > best_score:
            best_score = sc
            best_df = df2

    if best_score < 10:
        return None
    return best_df
def _score_season_table(df: pd.DataFrame) -> int:
    if df is None or df.empty or df.shape[1] < 4:
        return -999

    cols = [_norm_col(c) for c in df.columns]
    colset = set(cols)

    score = 0
    if any(c in ("year", "season", "yr") or "year" in c for c in cols):
        score += 6
    else:
        score += 1

    if "ab" in colset or "at bats" in colset:
        score += 6

    for k in ("h", "2b", "3b", "hr", "bb"):
        if k in colset:
            score += 2

    if "so" in colset or "k" in colset:
        score += 2

    for k in ("avg", "obp", "slg", "ops"):
        if k in colset:
            score += 2

    if "team" in colset:
        score += 1
    if "age" in colset:
        score += 1

    return score


def _find_best_season_batting_df(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    best_df = None
    best_score = -9999
    for df in dfs:
        df2 = _promote_first_row_to_header_if_needed(df)
        df2.columns = [str(c) for c in df2.columns]
        sc = _score_season_table(df2)
        if sc > best_score:
            best_score = sc
            best_df = df2
    if best_score < 8:
        return None
    return best_df


def _extract_gb_pct_from_tables(dfs: List[pd.DataFrame]) -> Optional[float]:
    for df in dfs:
        if df is None or df.empty:
            continue
        df2 = _promote_first_row_to_header_if_needed(df)
        try:
            cols = [_norm_col(c) for c in df2.columns]
        except Exception:
            continue

        # Column case
        for idx, c in enumerate(cols):
            if c in ("gb%", "gb %") or "gb%" in c:
                for v in df2.iloc[:, idx].tolist():
                    f = _safe_float(v)
                    if f is not None:
                        return float(f)

        # Row case
        try:
            vals = df2.values.tolist()
        except Exception:
            continue
        for r in vals:
            r_norm = [_norm_col(x) for x in r]
            for j, cell in enumerate(r_norm):
                if cell in ("gb%", "gb %") or "gb%" in cell:
                    if j + 1 < len(r):
                        f = _safe_float(r[j + 1])
                        if f is not None:
                            return float(f)
                    for k in range(len(r)):
                        if k == j:
                            continue
                        f = _safe_float(r[k])
                        if f is not None:
                            return float(f)

    return None


def _extract_gb_pct_from_text(text: str) -> Optional[float]:
    # Best: Distribution block (Pull/Middle/Opp/GB%)
    m = re.search(
        r"Distribution\s+Pull\s+Middle\s+Opp\s+Field\s+GB%\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s+"
        r"([0-9]+(?:\.[0-9]+)?)",
        text,
        re.I,
    )
    if m:
        return float(m.group(4))

    # Fallback: find GB% and take the last number shortly after
    m2 = re.search(r"\bGB%\b([^A-Za-z]{0,120})", text, re.I)
    if m2:
        nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", m2.group(1))
        if nums:
            return float(nums[-1])
    return None


# ----------------------------
# Scrape: Player card seasons + metadata
# ----------------------------
def scrape_player_card(fetcher: Fetcher, player_id: int) -> Tuple[Dict[str, Optional[str]], List[Dict]]:
    url = f"{BASE}/pcard.php?id={player_id}"
    html_page = fetcher.get(url)
    text = _collapse_text(html_page)

    name = None
    # Common: "Sung Jinwoo SF | #27 | 1B" appears in text
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+[A-Z]{2,4}\s*\|", text)
    if m:
        name = m.group(1).strip()
    else:
        m2 = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text)
        if m2:
            name = m2.group(1).strip()

    bats = None
    m = re.search(r"\bBats:\s*([A-Za-z]+)\b", text, re.I)
    if m:
        bats = m.group(1).strip()

    throws = None
    m = re.search(r"\bThrows:\s*([A-Za-z]+)\b", text, re.I)
    if m:
        throws = m.group(1).strip()

    primary_pos = None
    m = re.search(r"\|\s*#\d+\s*\|\s*([A-Za-z0-9/]+)\b", text)
    if m:
        primary_pos = m.group(1).strip()

    birth_or_age_text = None
    m = re.search(r"\bAge:\s*(\d{1,2})\b", text, re.I)
    if m:
        birth_or_age_text = f"Age: {m.group(1)}"

    dfs = _read_html_tables(html_page)
    df = _find_best_season_batting_df(dfs)

    seasons: List[Dict] = []
    if df is not None:
        df.columns = [_norm_col(c) for c in df.columns]

        ycol = None
        for c in df.columns:
            if c in ("year", "season", "yr") or "year" in c:
                ycol = c
                break
        if ycol is None:
            ycol = df.columns[0]

        colmap = {
            "ab": "ab",
            "at bats": "ab",
            "h": "h",
            "2b": "2b",
            "3b": "3b",
            "hr": "hr",
            "bb": "bb",
            "so": "k",
            "k": "k",
            "avg": "avg",
            "obp": "obp",
            "slg": "slg",
            "ops": "ops",
            "team": "team",
            "age": "age",
        }
        df = df.rename(columns={c: colmap[c] for c in df.columns if c in colmap})

        for _, r in df.iterrows():
            year = _safe_int(r.get(ycol))
            if year is None:
                continue
            ab = _safe_int(r.get("ab"))
            if ab is None or ab <= 0:
                continue

            ops = _safe_float(r.get("ops"))
            obp = _safe_float(r.get("obp"))
            slg = _safe_float(r.get("slg"))
            if ops is None and obp is not None and slg is not None:
                ops = obp + slg

            seasons.append(
                {
                    "player_id": player_id,
                    "season_year": year,
                    "team": (str(r.get("team")).strip() if r.get("team") is not None else None),
                    "age": _safe_int(r.get("age")),
                    "ab": ab,
                    "h": _safe_int(r.get("h")) or 0,
                    "doubles": _safe_int(r.get("2b")) or 0,
                    "triples": _safe_int(r.get("3b")) or 0,
                    "hr": _safe_int(r.get("hr")) or 0,
                    "bb": _safe_int(r.get("bb")) or 0,
                    "k": _safe_int(r.get("k")) or 0,
                    "avg": _safe_float(r.get("avg")),
                    "obp": obp,
                    "slg": slg,
                    "ops": ops,
                }
            )

    player_fields = {
        "name": name,
        "bats": bats,
        "throws": throws,
        "primary_pos": primary_pos,
        "birth_or_age_text": birth_or_age_text,
    }
    return player_fields, seasons


def scrape_player_edits(fetcher: Fetcher, player_id: int) -> Dict[str, Optional[float]]:
    url = f"{BASE}/pcarde.php?id={player_id}"
    html_page = fetcher.get(url)

    gb = None
    try:
        dfs = _read_html_tables(html_page)
        gb = _extract_gb_pct_from_tables(dfs)
    except Exception:
        gb = None

    if gb is None:
        text = _collapse_text(html_page)
        gb = _extract_gb_pct_from_text(text)

    return {"gb_pct": gb}


def scrape_player_card_pitching(fetcher: Fetcher, player_id: int) -> List[Dict]:
    """Scrape pitching season lines from pcard.php.

    Returns list of dicts suitable for player_season_pitching.
    """
    url = f"{BASE}/pcard.php?id={player_id}"
    html_page = fetcher.get(url)

    dfs = _read_html_tables(html_page)
    df = _find_best_season_pitching_df(dfs)
    if df is None:
        return []

    df.columns = [_norm_col(c) for c in df.columns]

    ycol = None
    for c in df.columns:
        if c in ("year", "season", "yr") or "year" in c:
            ycol = c
            break
    if ycol is None:
        ycol = df.columns[0]

    colmap = {
        "team": "team",
        "w": "w",
        "l": "l",
        "era": "era",
        "sv": "sv",
        "bs": "bs",
        "g": "g",
        "gs": "gs",
        "cg": "cg",
        "sho": "sho",
        "ip": "ip",
        "h": "h",
        "so": "k",
        "k": "k",
        "bb": "bb",
        "ops": "ops_allowed",
        "whip": "whip",
    }
    df = df.rename(columns={c: colmap[c] for c in df.columns if c in colmap})

    seasons: List[Dict] = []
    for _, r in df.iterrows():
        year = _safe_int(r.get(ycol))
        if year is None:
            continue

        ip = _safe_float(r.get("ip"))
        # skip empty rows
        if ip is None or ip <= 0:
            continue

        seasons.append(
            {
                "player_id": player_id,
                "season_year": year,
                "team": (str(r.get("team")).strip() if r.get("team") is not None else None),
                "w": _safe_int(r.get("w")) or 0,
                "l": _safe_int(r.get("l")) or 0,
                "era": _safe_float(r.get("era")),
                "sv": _safe_int(r.get("sv")) or 0,
                "bs": _safe_int(r.get("bs")) or 0,
                "g": _safe_int(r.get("g")) or 0,
                "gs": _safe_int(r.get("gs")) or 0,
                "cg": _safe_int(r.get("cg")) or 0,
                "sho": _safe_int(r.get("sho")) or 0,
                "ip": ip,
                "h": _safe_int(r.get("h")) or 0,
                "k": _safe_int(r.get("k")) or 0,
                "bb": _safe_int(r.get("bb")) or 0,
                "ops_allowed": _safe_float(r.get("ops_allowed")),
                "whip": _safe_float(r.get("whip")),
            }
        )

    return seasons
    url = f"{BASE}/pcarde.php?id={player_id}"
    html_page = fetcher.get(url)

    gb = None
    try:
        dfs = _read_html_tables(html_page)
        gb = _extract_gb_pct_from_tables(dfs)
    except Exception:
        gb = None

    if gb is None:
        text = _collapse_text(html_page)
        gb = _extract_gb_pct_from_text(text)

    return {"gb_pct": gb}


def scrape_player_splits(fetcher: Fetcher, player_id: int) -> List[Dict]:
    url = f"{BASE}/pcardsplit.php?id={player_id}"
    html_page = fetcher.get(url)

    dfs = _read_html_tables(html_page)
    if not dfs:
        return []

    best = None
    best_score = -999
    for df in dfs:
        df2 = _promote_first_row_to_header_if_needed(df)
        df2.columns = [_norm_col(c) for c in df2.columns]
        cset = set(df2.columns)
        sc = 0
        if "ab" in cset or "at bats" in cset:
            sc += 4
        if "ops" in cset:
            sc += 3
        if "obp" in cset and "slg" in cset:
            sc += 2
        if sc > best_score:
            best_score = sc
            best = df2

    if best is None:
        return []

    split_col = best.columns[0]

    colmap = {
        "ab": "ab",
        "at bats": "ab",
        "h": "h",
        "2b": "2b",
        "3b": "3b",
        "hr": "hr",
        "bb": "bb",
        "so": "k",
        "k": "k",
        "avg": "avg",
        "obp": "obp",
        "slg": "slg",
        "ops": "ops",
    }
    best = best.rename(columns={c: colmap[c] for c in best.columns if c in colmap})

    out: List[Dict] = []
    for _, r in best.iterrows():
        split_key = r.get(split_col)
        if split_key is None:
            continue
        split_key = str(split_key).strip()
        if split_key == "" or split_key.lower() in ("nan", "none"):
            continue

        ab = _safe_int(r.get("ab"))
        if ab is None or ab <= 0:
            continue

        ops = _safe_float(r.get("ops"))
        obp = _safe_float(r.get("obp"))
        slg = _safe_float(r.get("slg"))
        if ops is None and obp is not None and slg is not None:
            ops = obp + slg

        out.append(
            {
                "player_id": player_id,
                "split_key": split_key,
                "ab": ab,
                "h": _safe_int(r.get("h")) or 0,
                "doubles": _safe_int(r.get("2b")) or 0,
                "triples": _safe_int(r.get("3b")) or 0,
                "hr": _safe_int(r.get("hr")) or 0,
                "bb": _safe_int(r.get("bb")) or 0,
                "k": _safe_int(r.get("k")) or 0,
                "avg": _safe_float(r.get("avg")),
                "obp": obp,
                "slg": slg,
                "ops": ops,
            }
        )

    return out


# ----------------------------
# Stadiums + aliases
# ----------------------------
def scrape_team_stadiums(fetcher: Fetcher) -> List[Dict]:
    url = f"{BASE}/stadiums.php"
    html_page = fetcher.get(url)

    txt = _collapse_text(html_page)
    m = re.search(r"MLBC STADIUMS(.*?)(View All League News|View All Team News|POTM)", txt, flags=re.I | re.S)
    block = m.group(1) if m else txt
    block = re.sub(r"\s+", " ", block).strip()

    # Remove the column header if present
    block = re.sub(r"(?i)^stadium\\s+team\\s+since\\s+leased\\s+through\\s+", "", block)

    tokens = block.split(" ")

    def is_year(tok: str) -> bool:
        return bool(re.fullmatch(r"\d{4}", tok))

    out: List[Dict] = []

    # We'll scan for patterns:
    #   <stadium words...> <team words...> <since_year> <through_year|-> ...
    #
    # Since stadium/team are both variable-length, we use the year as anchor:
    # when we see a year, everything immediately before it (since last "stadium boundary")
    # is treated as team, and the stadium is whatever was accumulated earlier.
    #
    # Practical heuristic for this page:
    # - Stadium names always appear before their lease line
    # - Team names come immediately before the year
    #
    # We'll maintain a rolling window of "unassigned words" and carve off the last N tokens
    # as team once we hit the year. N is determined by walking backwards until we hit a token
    # that looks like the end of a stadium name boundary (we approximate by using a known year anchor).
    unassigned: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # Sometimes "-" comes through as "-" already; good.
        # Sometimes it might be "—" etc; normalize.
        if tok in ("\\-", "–", "—"):
            tok = "-"

        if is_year(tok):
            # tok is since_year; next token should be through_year or "-"
            since_year = int(tok)
            through_tok = tokens[i + 1] if i + 1 < len(tokens) else "-"
            if through_tok in ("\\-", "–", "—"):
                through_tok = "-"
            through_year = None if through_tok == "-" else (int(through_tok) if is_year(through_tok) else None)

            # Now we need to split `unassigned` into [stadium words..., team words...].
            #
            # We can do this by looking for the LAST stadium name we saw that ends with common suffixes.
            # But easiest: on this page, the stadium name is the LAST contiguous phrase ending with
            # Stadium/Park/Field/Dome/Grounds/Bowl/Coliseum/etc that appears before the team.
            #
            # We'll scan from the end backwards to find a plausible stadium end token, then split.
            stadium_end_words = {
                "Stadium", "Park", "Field", "Dome", "Grounds", "Bowl", "Coliseum",
                "Ballpark", "Arena", "Center", "Centre"
            }

            # Find the LAST stadium-ending token (end of the actual stadium we want)
            end_j = None
            for j in range(len(unassigned) - 1, -1, -1):
                if unassigned[j] in stadium_end_words:
                    end_j = j
                    break
            
            if end_j is None:
                unassigned = []
                i += 2
                continue
            
            # Find the previous stadium-ending token before end_j; the stadium starts right after it.
            prev_end_j = None
            for j in range(end_j - 1, -1, -1):
                if unassigned[j] in stadium_end_words:
                    prev_end_j = j
                    break
            
            start_idx = (prev_end_j + 1) if prev_end_j is not None else 0
            
            stadium = " ".join(unassigned[start_idx : end_j + 1]).strip()
            team = " ".join(unassigned[end_j + 1 :]).strip()


            if stadium and team:
                out.append(
                    dict(
                        team=team,
                        stadium=stadium,
                        since_year=since_year,
                        through_year=through_year,
                    )
                )

            # reset for next record
            unassigned = []
            i += 2
            continue

        # Normal token: accumulate
        unassigned.append(tok)
        i += 1

    if not out:
        raise ValueError("Failed to parse any stadium leases from stadiums.php")

    # Deduplicate
    seen = set()
    deduped: List[Dict] = []
    for r in out:
        key = (r["team"], r["stadium"], r["since_year"], r["through_year"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    return deduped





def rebuild_team_aliases_from_stadiums(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT DISTINCT team FROM team_stadiums").fetchall()
    aliases = []
    for (team_full,) in rows:
        if not team_full:
            continue
        parts = str(team_full).split()
        if len(parts) >= 2:
            abbrev = (parts[0][0] + parts[1][0]).upper()
            aliases.append((abbrev, str(team_full)))
    conn.execute("DELETE FROM team_aliases;")
    conn.executemany("INSERT OR REPLACE INTO team_aliases(abbrev, team_full) VALUES(?,?)", aliases)
    conn.commit()


# ----------------------------
# Discover players
# ----------------------------


def _extract_player_ids_from_roster_html(page_html: str) -> List[int]:
    ids = [int(m.group(1)) for m in ROSTER_ID_RE.finditer(page_html or "")]
    # de-dupe preserving order
    out: List[int] = []
    seen: set[int] = set()
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def discover_roster_players(
    conn: sqlite3.Connection,
    fetcher: Fetcher,
    *,
    teams: Sequence[str],
    verbose: bool = True,
) -> int:
    """Discover player IDs by scraping each team's ML roster page.

    Defaults should be a list of team abbrevs (e.g. SF, LOUI) from the batting table.
    """

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    discovered = 0
    seen: set[int] = set(r[0] for r in conn.execute("SELECT player_id FROM player_ids").fetchall())

    for t in teams:
        team = str(t).strip()
        if not team:
            continue

        url = f"{BASE}/teamrosters_team.php?team={urllib.parse.quote(team)}"
        try:
            page = fetcher.get(url)
        except Exception as e:
            log(f"[rosters] fetch failed team={team}: {e}")
            continue

        ids = _extract_player_ids_from_roster_html(page)
        if not ids:
            log(f"[rosters] team={team} ids=0")
            continue

        new_ids = [i for i in ids if i not in seen]
        if new_ids:
            conn.executemany("INSERT OR IGNORE INTO player_ids(player_id) VALUES(?)", [(i,) for i in new_ids])
            discovered += len(new_ids)
            for i in new_ids:
                seen.add(i)

        # Store roster membership for debugging/coverage.
        now = int(time.time())
        conn.executemany(
            "INSERT OR REPLACE INTO team_roster_players(team, player_id, scraped_ts) VALUES(?,?,?)",
            [(team, i, now) for i in ids],
        )
        conn.commit()

        log(f"[rosters] team={team} ids={len(ids)} new={len(new_ids)}")

    return discovered
def discover_players(
    conn: sqlite3.Connection,
    fetcher: Fetcher,
    seeds: Sequence[str],
    max_pages: int,
    verbose: bool = True,
) -> int:
    seen: set[int] = set(r[0] for r in conn.execute("SELECT player_id FROM player_ids").fetchall())
    discovered = 0

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    for seed in seeds:
        parsed = urllib.parse.urlparse(seed)
        qs = dict(urllib.parse.parse_qsl(parsed.query))
        for pn in range(1, max_pages + 1):
            qs2 = dict(qs)
            qs2["pn"] = str(pn)
            url = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(qs2)))
            try:
                page = fetcher.get(url)
            except Exception as e:
                log(f"[discover] fetch failed pn={pn}: {e}")
                break

            ids = _extract_player_ids_from_html(page)
            if not ids:
                if pn == 1:
                    log(f"[discover] no player ids on first page: {url}")
                break

            new_ids = [i for i in ids if i not in seen]
            if new_ids:
                conn.executemany("INSERT OR IGNORE INTO player_ids(player_id) VALUES(?)", [(i,) for i in new_ids])
                conn.commit()
                for i in new_ids:
                    seen.add(i)
                discovered += len(new_ids)

            log(f"[discover] seed={seed} pn={pn} ids={len(ids)} new={len(new_ids)} total={len(seen)}")

    return discovered


# ----------------------------
# DB writes
# ----------------------------
def upsert_player(conn: sqlite3.Connection, player_id: int, fields: Dict[str, Optional[str]], gb_pct: Optional[float]) -> None:
    now = int(time.time())
    conn.execute(
        """
        INSERT INTO players(player_id, name, bats, throws, primary_pos, birth_or_age_text, gb_pct, last_seen_ts)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(player_id) DO UPDATE SET
          name=COALESCE(excluded.name, players.name),
          bats=COALESCE(excluded.bats, players.bats),
          throws=COALESCE(excluded.throws, players.throws),
          primary_pos=COALESCE(excluded.primary_pos, players.primary_pos),
          birth_or_age_text=COALESCE(excluded.birth_or_age_text, players.birth_or_age_text),
          gb_pct=COALESCE(excluded.gb_pct, players.gb_pct),
          last_seen_ts=excluded.last_seen_ts
        """,
        (
            player_id,
            fields.get("name"),
            fields.get("bats"),
            fields.get("throws"),
            fields.get("primary_pos"),
            fields.get("birth_or_age_text"),
            gb_pct,
            now,
        ),
    )




def replace_player_seasons_pitching(conn: sqlite3.Connection, seasons: List[Dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO player_season_pitching(
          player_id, season_year, team,
          w, l, era, sv, bs, g, gs, cg, sho,
          ip, h, k, bb,
          ops_allowed, whip
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                s["player_id"],
                s["season_year"],
                s.get("team"),
                s.get("w"),
                s.get("l"),
                s.get("era"),
                s.get("sv"),
                s.get("bs"),
                s.get("g"),
                s.get("gs"),
                s.get("cg"),
                s.get("sho"),
                s.get("ip"),
                s.get("h"),
                s.get("k"),
                s.get("bb"),
                s.get("ops_allowed"),
                s.get("whip"),
            )
            for s in seasons
        ],
    )


def replace_player_splits_pitching(conn: sqlite3.Connection, splits: List[Dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO player_splits_pitching(
          player_id, split_group, split_key,
          team, w, l, era, sv, bs, g, gs, cg, sho,
          ip, h, doubles, triples, hr, bb, k,
          avg, obp, slg, ops, whip
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                r["player_id"],
                r["split_group"],
                r["split_key"],
                r.get("team"),
                r.get("w"),
                r.get("l"),
                r.get("era"),
                r.get("sv"),
                r.get("bs"),
                r.get("g"),
                r.get("gs"),
                r.get("cg"),
                r.get("sho"),
                r.get("ip"),
                r.get("h"),
                r.get("doubles"),
                r.get("triples"),
                r.get("hr"),
                r.get("bb"),
                r.get("k"),
                r.get("avg"),
                r.get("obp"),
                r.get("slg"),
                r.get("ops"),
                r.get("whip"),
            )
            for r in splits
        ],
    )
def replace_player_seasons(conn: sqlite3.Connection, seasons: List[Dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO player_season_batting(
          player_id, season_year, team, age, ab, h, doubles, triples, hr, bb, k, avg, obp, slg, ops
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                s["player_id"],
                s["season_year"],
                s.get("team"),
                s.get("age"),
                s.get("ab"),
                s.get("h"),
                s.get("doubles"),
                s.get("triples"),
                s.get("hr"),
                s.get("bb"),
                s.get("k"),
                s.get("avg"),
                s.get("obp"),
                s.get("slg"),
                s.get("ops"),
            )
            for s in seasons
        ],
    )


def _table_has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()]
    return col in cols


def _infer_split_type(split_key: str) -> str:
    s = (split_key or "").strip().lower()

    if s in ("home", "away"):
        return "home_away"
    if re.fullmatch(r"(apr|may|jun|jul|aug|sep|oct|sep/oct)", s):
        return "month"
    if "vs" in s or s in ("vs l", "vs r", "vsl", "vsr"):
        return "platoon"
    if "1st" in s or "2nd" in s or "3rd" in s or "4th" in s:
        return "inning"
    return "other"


def replace_player_splits(conn: sqlite3.Connection, splits: List[Dict]) -> None:
    has_split_type = _table_has_column(conn, "player_splits_batting", "split_type")

    if has_split_type:
        conn.executemany(
            """
            INSERT OR REPLACE INTO player_splits_batting(
              player_id, split_key, split_type,
              ab, h, doubles, triples, hr, bb, k, avg, obp, slg, ops
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    r["player_id"],
                    r["split_key"],
                    _infer_split_type(r["split_key"]),
                    r.get("ab"),
                    r.get("h"),
                    r.get("doubles"),
                    r.get("triples"),
                    r.get("hr"),
                    r.get("bb"),
                    r.get("k"),
                    r.get("avg"),
                    r.get("obp"),
                    r.get("slg"),
                    r.get("ops"),
                )
                for r in splits
            ],
        )
    else:
        conn.executemany(
            """
            INSERT OR REPLACE INTO player_splits_batting(
              player_id, split_key, ab, h, doubles, triples, hr, bb, k, avg, obp, slg, ops
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    r["player_id"],
                    r["split_key"],
                    r.get("ab"),
                    r.get("h"),
                    r.get("doubles"),
                    r.get("triples"),
                    r.get("hr"),
                    r.get("bb"),
                    r.get("k"),
                    r.get("avg"),
                    r.get("obp"),
                    r.get("slg"),
                    r.get("ops"),
                )
                for r in splits
            ],
        )



def replace_team_stadiums(conn: sqlite3.Connection, rows: List[Dict]) -> None:
    conn.execute("DELETE FROM team_stadiums;")
    conn.executemany(
        "INSERT INTO team_stadiums(team, stadium, since_year, through_year) VALUES(?,?,?,?)",
        [(r["team"], r["stadium"], r.get("since_year"), r.get("through_year")) for r in rows],
    )


# ----------------------------
# Ingest
# ----------------------------
def ingest_players(
    conn: sqlite3.Connection,
    fetcher: Fetcher,
    player_ids: Sequence[int],
    *,
    do_splits_batting: bool,
    do_pitching: bool,
    do_splits_pitching: bool,
) -> None:
    for pid in player_ids:
        try:
            fields, seasons_bat = scrape_player_card(fetcher, pid)
            edits = scrape_player_edits(fetcher, pid)
            upsert_player(conn, pid, fields, edits.get("gb_pct"))

            if seasons_bat:
                replace_player_seasons(conn, seasons_bat)

            if do_splits_batting:
                sp = scrape_player_splits(fetcher, pid)
                if sp:
                    replace_player_splits(conn, sp)

            if do_pitching:
                seasons_p = scrape_player_card_pitching(fetcher, pid)
                if seasons_p:
                    replace_player_seasons_pitching(conn, seasons_p)

            if do_splits_pitching:
                ps = scrape_player_splits_pitching(fetcher, pid)
                if ps:
                    replace_player_splits_pitching(conn, ps)

            conn.commit()
            print(
                f"Saved player {pid}: bat_seasons={len(seasons_bat)} pitch_seasons={len(locals().get('seasons_p', []) or [])} gb_pct={edits.get('gb_pct')}",
                flush=True,
            )
        except Exception as e:
            conn.rollback()
            print(f"[WARN] player {pid} failed: {e}", file=sys.stderr, flush=True)


def ingest_stadiums(conn: sqlite3.Connection, fetcher: Fetcher) -> None:
    rows = scrape_team_stadiums(fetcher)
    replace_team_stadiums(conn, rows)
    rebuild_team_aliases_from_stadiums(conn)
    conn.commit()
    print(f"Saved stadiums: rows={len(rows)}", flush=True)


def ingest_all(
    conn: sqlite3.Connection,
    fetcher: Fetcher,
    *,
    do_splits_batting: bool,
    do_pitching: bool,
    do_splits_pitching: bool,
    limit: Optional[int],
    start_at: int,
) -> None:
    ids = [r[0] for r in conn.execute("SELECT player_id FROM player_ids ORDER BY player_id").fetchall()]
    if start_at:
        ids = ids[start_at:]
    if limit is not None:
        ids = ids[:limit]
    print(f"Starting ingest-all: {len(ids)} player IDs", flush=True)
    ingest_players(
        conn,
        fetcher,
        ids,
        do_splits_batting=do_splits_batting,
        do_pitching=do_pitching,
        do_splits_pitching=do_splits_pitching,
    )


# ----------------------------
# CLI
# ----------------------------
def default_discover_seeds() -> List[str]:
    return [
        f"{BASE}/leagueleaders_batting.php?bc=avg",
        f"{BASE}/leagueleaders_batting.php?bc=ops",
        f"{BASE}/leagueleaders_batting.php?bc=hr",
        f"{BASE}/leagueleaders_batting.php?bc=rbi",
        f"{BASE}/leagueleaders_batting.php?bc=obp",
        f"{BASE}/leagueleaders_batting.php?bc=slg",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=25.0)

    ap.add_argument("--players", nargs="*", type=int, default=[])
    ap.add_argument("--player-id-range", nargs=2, type=int, default=None,
                metavar=("START", "END"),
                help="Ingest a contiguous range of player IDs inclusive, e.g. --player-id-range 0 3500")

    ap.add_argument("--splits", action="store_true", help="Scrape batting splits")
    ap.add_argument("--pitching", action="store_true", help="Scrape pitching season stats")
    ap.add_argument("--pitching-splits", action="store_true", help="Scrape pitching splits (platoon + situational)")

    ap.add_argument("--stadiums", action="store_true")

    ap.add_argument("--discover-players", action="store_true")
    ap.add_argument("--discover-max-pages", type=int, default=50)
    ap.add_argument("--discover-rosters", action="store_true", help="Discover player IDs from each team's ML roster page")
    ap.add_argument("--roster-teams", nargs="*", default=None, help="Optional subset of team abbrevs for roster discovery")
    ap.add_argument("--discover-seeds", nargs="*", default=None)
    ap.add_argument(
        "--discover-id-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Probe player IDs in [START..END] and add 'real' ones to player_ids.",
    )
    ap.add_argument(
        "--discover-id-sleep",
        type=float,
        default=0.05,
        help="Sleep between discover-id-range requests (seconds).",
    )



    ap.add_argument("--ingest-all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-at", type=int, default=0)

    args = ap.parse_args()
    
    if args.player_id_range:
        lo, hi = args.player_id_range
        lo = int(lo)
        hi = int(hi)
        if hi < lo:
            ap.error("player-id-range END must be >= START")
        args.players = list(range(lo, hi + 1))


    conn = db_connect(args.db)
    fetcher = Fetcher(timeout=args.timeout, sleep=args.sleep)

    if args.discover_id_range:
        lo, hi = [int(x) for x in args.discover_id_range]
        if hi < lo:
            ap.error("discover-id-range END must be >= START")

        found = 0
        checked = 0

        for pid in range(lo, hi + 1):
            checked += 1
            try:
                fields, seasons_bat = scrape_player_card(fetcher, pid)

                name = str(fields.get("name") or "").strip()
                team = str(fields.get("team") or "").strip()
                bats = str(fields.get("bats") or "").strip()
                throws = str(fields.get("throws") or "").strip()

                # Filter out known bogus parse artifacts from blank placeholder pages
                bogus_names = {"minor league baseball club"}
                bogus_bats = {"throws"}   # header text mis-read as value
                bogus_throws = {"ml"}     # mis-read fragment

                is_real = bool(name) and (name.lower() not in bogus_names)

                # Require at least one meaningful identity field besides the bogus defaults
                if team:
                    pass
                elif bats and bats.lower() not in bogus_bats:
                    pass
                elif throws and throws.lower() not in bogus_throws:
                    pass
                else:
                    is_real = False

                if is_real:
                    conn.execute(
                        "INSERT OR IGNORE INTO player_ids(player_id) VALUES(?)",
                        (int(pid),),
                    )
                    conn.commit()
                    found += 1
                    if found % 50 == 0:
                        print(f"[discover-id-range] found={found} (latest pid={pid} name={name})", flush=True)


            except Exception:
                pass

            if args.discover_id_sleep:
                time.sleep(float(args.discover_id_sleep))

    print(f"[discover-id-range] checked={checked} found={found}", flush=True)



    if args.discover_rosters:
        if args.roster_teams:
            teams = [t.strip() for t in args.roster_teams if t and t.strip()]
        else:
            teams = [r[0] for r in conn.execute(
                "SELECT DISTINCT team FROM player_season_batting WHERE team IS NOT NULL AND trim(team)<>'' ORDER BY team"
            ).fetchall()]
        n = discover_roster_players(conn, fetcher, teams=teams, verbose=True)
        print(f"Discovered {n} new roster player IDs", flush=True)
    if args.discover_players:
        seeds = args.discover_seeds if args.discover_seeds else default_discover_seeds()
        n = discover_players(conn, fetcher, seeds=seeds, max_pages=args.discover_max_pages, verbose=True)
        print(f"Discovered {n} new player IDs", flush=True)

    if args.stadiums:
        ingest_stadiums(conn, fetcher)

    if args.players:
        ingest_players(
            conn,
            fetcher,
            args.players,
            do_splits_batting=args.splits,
            do_pitching=args.pitching,
            do_splits_pitching=args.pitching_splits,
        )

    if args.ingest_all:
        ingest_all(
            conn,
            fetcher,
            do_splits_batting=args.splits,
            do_pitching=args.pitching,
            do_splits_pitching=args.pitching_splits,
            limit=args.limit,
            start_at=args.start_at,
        )

    conn.close()


if __name__ == "__main__":
    main()
