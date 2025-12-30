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

    # Convert HTML -> plain text (collapses tags, removes script/style)
    text = _collapse_text(html_page)

    # We only want the portion after the "MLBC STADIUMS" header if present
    # (keeps regex from matching random footer numbers).
    m = re.search(r"\bMLBC\s+STADIUMS\b(.+)$", text, flags=re.I)
    if m:
        text = m.group(1)

    # Example patterns in the collapsed page text:
    #   "Kauffman Stadium San Francisco Seals 2140 2142"
    #   "Louisville Bats 2108 -"
    #
    # We'll extract as: stadium, team, since_year, through_year/-
    # Stadium + team are greedy-but-bounded by the year tokens.
    lease_re = re.compile(
        r"""
        (?P<stadium>.+?)\s+
        (?P<team>.+?)\s+
        (?P<since>\d{4})\s+
        (?P<through>\d{4}|-)
        (?=\s|$)
        """,
        flags=re.VERBOSE,
    )

    # The page is essentially a long list; to avoid runaway matches,
    # split into chunks around known year tokens.
    # A simple way: replace multiple spaces, then scan with regex globally.
    text = re.sub(r"\s+", " ", text).strip()

    out: List[Dict] = []
    for mm in lease_re.finditer(text):
        stadium_s = mm.group("stadium").strip()
        team_s = mm.group("team").strip()
        since = _safe_int(mm.group("since"))
        through_raw = mm.group("through").strip()
        through = None if through_raw == "-" else _safe_int(through_raw)

        # Light cleanup: strip any leftover header fragments
        if stadium_s.lower().startswith("stadium team"):
            continue
        if team_s.lower() in ("team", "stadiums"):
            continue

        # Sanity: avoid absurdly long captures if something goes wrong
        if len(stadium_s) > 80 or len(team_s) > 80:
            continue

        out.append(
            {
                "team": team_s,
                "stadium": stadium_s,
                "since_year": since,
                "through_year": through,
            }
        )

    if not out:
        raise ValueError("Failed to parse any stadium leases from stadiums.php")

    return out



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
def ingest_players(conn: sqlite3.Connection, fetcher: Fetcher, player_ids: Sequence[int], do_splits: bool) -> None:
    for pid in player_ids:
        try:
            fields, seasons = scrape_player_card(fetcher, pid)
            edits = scrape_player_edits(fetcher, pid)
            upsert_player(conn, pid, fields, edits.get("gb_pct"))
            if seasons:
                replace_player_seasons(conn, seasons)
            if do_splits:
                sp = scrape_player_splits(fetcher, pid)
                if sp:
                    replace_player_splits(conn, sp)
            conn.commit()
            print(f"Saved player {pid}: seasons={len(seasons)} gb_pct={edits.get('gb_pct')}", flush=True)
        except Exception as e:
            conn.rollback()
            print(f"[WARN] player {pid} failed: {e}", file=sys.stderr, flush=True)


def ingest_stadiums(conn: sqlite3.Connection, fetcher: Fetcher) -> None:
    rows = scrape_team_stadiums(fetcher)
    replace_team_stadiums(conn, rows)
    rebuild_team_aliases_from_stadiums(conn)
    conn.commit()
    print(f"Saved stadiums: rows={len(rows)}", flush=True)


def ingest_all(conn: sqlite3.Connection, fetcher: Fetcher, do_splits: bool, limit: Optional[int], start_at: int) -> None:
    ids = [r[0] for r in conn.execute("SELECT player_id FROM player_ids ORDER BY player_id").fetchall()]
    if start_at:
        ids = ids[start_at:]
    if limit is not None:
        ids = ids[:limit]
    print(f"Starting ingest-all: {len(ids)} player IDs", flush=True)
    ingest_players(conn, fetcher, ids, do_splits=do_splits)


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
    ap.add_argument("--splits", action="store_true")

    ap.add_argument("--stadiums", action="store_true")

    ap.add_argument("--discover-players", action="store_true")
    ap.add_argument("--discover-max-pages", type=int, default=50)
    ap.add_argument("--discover-rosters", action="store_true", help="Discover player IDs from each team's ML roster page")
    ap.add_argument("--roster-teams", nargs="*", default=None, help="Optional subset of team abbrevs for roster discovery")
    ap.add_argument("--discover-seeds", nargs="*", default=None)

    ap.add_argument("--ingest-all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-at", type=int, default=0)

    args = ap.parse_args()

    conn = db_connect(args.db)
    fetcher = Fetcher(timeout=args.timeout, sleep=args.sleep)


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
        ingest_players(conn, fetcher, args.players, do_splits=args.splits)

    if args.ingest_all:
        ingest_all(conn, fetcher, do_splits=args.splits, limit=args.limit, start_at=args.start_at)

    conn.close()


if __name__ == "__main__":
    main()
