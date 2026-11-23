"""Data layer for KickCast.

Downloads, validates and caches the data the project depends on:
season CSV files for the EPL and La Liga (football-data.co.uk) and the
live Fantasy Premier League (FPL) feed. Every download is saved under
data/raw/ so the program can fall back to the cache when offline.

Contents:
    DataFetchError      - custom exception for network failures, bad status
                          codes, malformed data or a missing cache.
    DataSource          - abstract base class with load() and cache helpers.
    FootballCsvSource   - loads season CSVs into a tidy DataFrame.
    FplApiSource        - loads the FPL bootstrap-static JSON and exposes
                          injury news and chance of playing.
"""

import json
import time
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# Project root = the folder that contains src/ (so paths work from anywhere).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

HEADERS = {"User-Agent": "KickCast-student-project/1.0 (FAST-NUCES coursework)"}
TIMEOUT_SECONDS = 15

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"   # small snapshot kept in the repo

LEAGUE_CODES = {"EPL": "E0", "LaLiga": "SP1"}

# Columns every match row must have, and the extra columns we keep if present.
REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
OPTIONAL_COLUMNS = ["HS", "AS", "HST", "AST",
                    "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA"]


class DataFetchError(Exception):
    """Raised on network failure, bad status code, bad data or missing cache."""


def _decode(raw):
    """Decode downloaded bytes to text (some CSVs are not valid UTF-8)."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


class DataSource(ABC):
    """Abstract base class for every data source.

    Subclasses must implement load(). The download, caching and
    fall-back-to-cache logic is shared here so both sources behave the same
    way and can be used polymorphically through load().
    """

    def __init__(self, name, cache_path):
        self._name = name
        self._cache_path = Path(cache_path)

    @property
    def name(self):
        """Human-readable name of this source."""
        return self._name

    @abstractmethod
    def load(self):
        """Return the loaded data (implemented by each subclass)."""

    # ---- cache helpers -------------------------------------------------
    def _resolve(self, path):
        return Path(path) if path else self._cache_path

    def _write_cache(self, raw, path=None):
        """Save raw bytes to the cache file, creating folders if needed."""
        path = self._resolve(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def _read_cache(self, path=None):
        """Return cached bytes, or raise DataFetchError if there is none."""
        path = self._resolve(path)
        if not path.exists():
            raise DataFetchError(f"No cached copy found at {path}")
        return path.read_bytes()

    def is_available(self, path=None):
        """True if a cached copy exists."""
        return self._resolve(path).exists()

    def cache_age_hours(self, path=None):
        """Age of the cached file in hours, or None if there is no cache."""
        path = self._resolve(path)
        if not path.exists():
            return None
        return (time.time() - path.stat().st_mtime) / 3600

    # ---- network -------------------------------------------------------
    def _download(self, url):
        """GET a URL and return the response bytes, or raise DataFetchError."""
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise DataFetchError(f"Network error for {url}: {exc}") from exc
        if response.status_code != 200:
            raise DataFetchError(f"{url} returned status {response.status_code}")
        return response.content

    def _get(self, url, parse, path=None):
        """Download + parse + cache; fall back to the cache on any failure.

        Returns (parsed_data, origin) where origin is "web" or "cache".
        The cache is only overwritten when the new download parsed correctly.
        """
        path = self._resolve(path)
        try:
            raw = self._download(url)
            parsed = parse(raw)
        except DataFetchError as web_error:
            if not path.exists():
                raise DataFetchError(
                    f"{web_error}. No cached copy at {path}.") from web_error
            return parse(path.read_bytes()), "cache"
        self._write_cache(raw, path)
        return parsed, "web"


class FootballCsvSource(DataSource):
    """Season CSVs from football-data.co.uk for one league."""

    BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    # 2021/22 ... 2025/26 (2025/26 is the latest completed season).
    DEFAULT_SEASONS = ["2122", "2223", "2324", "2425", "2526"]

    def __init__(self, league, seasons=None):
        if league not in LEAGUE_CODES:
            raise ValueError(f"League must be one of {list(LEAGUE_CODES)}")
        super().__init__(f"football-data {league}", RAW_DIR)
        self.league = league
        self.seasons = list(seasons) if seasons else list(self.DEFAULT_SEASONS)
        self.status = {}  # season -> "web" or "cache" after load()

    def _season_path(self, season):
        return self._cache_path / f"{self.league}_{season}.csv"

    def season_cache_age_hours(self, season):
        """Age in hours of one season's cached CSV, or None if it is not cached."""
        return self.cache_age_hours(self._season_path(season))

    @classmethod
    def load_sample(cls, league, season="2526", sample_dir=None):
        """Load the small offline snapshot kept in data/sample/ (no network).

        Used as the last fallback when the web and the data/raw cache are both
        unavailable, so a fresh clone still runs offline.

        Raises:
            ValueError: for an unknown league.
            DataFetchError: if the snapshot file is missing or unreadable.
        """
        if league not in LEAGUE_CODES:
            raise ValueError(f"League must be one of {list(LEAGUE_CODES)}")
        path = (Path(sample_dir) if sample_dir else SAMPLE_DIR) / f"{league}_{season}.csv"
        if not path.exists():
            raise DataFetchError(f"No sample snapshot found at {path}")
        df = cls._parse_csv(path.read_bytes())
        df["League"] = league
        df["Season"] = season
        return df.reset_index(drop=True)

    @staticmethod
    def _parse_csv(raw):
        """Turn raw CSV bytes into a tidy DataFrame (or raise DataFetchError)."""
        try:
            df = pd.read_csv(StringIO(_decode(raw)), on_bad_lines="skip")
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                raise DataFetchError(f"CSV is missing columns: {missing}")
            df = df.dropna(subset=REQUIRED_COLUMNS)
            keep = [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
                    if c in df.columns]
            df = df[keep].copy()
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            raise DataFetchError(f"Could not parse CSV: {exc}") from exc
        return df

    def load(self):
        """Return one DataFrame with all seasons, using the cache if offline."""
        frames = []
        self.status = {}
        code = LEAGUE_CODES[self.league]
        for season in self.seasons:
            url = self.BASE_URL.format(season=season, code=code)
            df, origin = self._get(url, self._parse_csv, self._season_path(season))
            df["League"] = self.league
            df["Season"] = season
            frames.append(df)
            self.status[season] = origin
        return pd.concat(frames, ignore_index=True)


class FplApiSource(DataSource):
    """Live Fantasy Premier League feed (bootstrap-static)."""

    URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

    def __init__(self):
        super().__init__("FPL bootstrap-static", RAW_DIR / "fpl_bootstrap.json")
        self.origin = None  # "web" or "cache" after load()

    @staticmethod
    def _parse_json(raw):
        """Parse bytes to a dict and check the keys we need are present."""
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise DataFetchError("FPL response was not valid JSON") from exc
        if not isinstance(data, dict) or "elements" not in data or "teams" not in data:
            raise DataFetchError("FPL JSON is missing 'elements' or 'teams'")
        return data

    @classmethod
    def load_sample(cls, sample_dir=None):
        """Load the FPL snapshot kept in data/sample/ (no network).

        Raises:
            DataFetchError: if the snapshot is missing or not valid FPL JSON.
        """
        path = (Path(sample_dir) if sample_dir else SAMPLE_DIR) / "fpl_bootstrap.json"
        if not path.exists():
            raise DataFetchError(f"No sample snapshot found at {path}")
        return cls._parse_json(path.read_bytes())

    def load(self):
        """Return the full FPL JSON as a dict, using the cache if offline."""
        data, self.origin = self._get(self.URL, self._parse_json)
        return data

    def get_news(self, data=None, only_with_news=True):
        """Return a DataFrame of each player's injury news and chance of playing."""
        if data is None:
            data = self.load()
        team_names = {t["id"]: t["name"] for t in data["teams"]}
        rows = [
            {
                "player": p.get("web_name", ""),
                "team": team_names.get(p.get("team"), "Unknown"),
                "status": p.get("status", ""),
                "news": p.get("news") or "",
                "chance_of_playing": p.get("chance_of_playing_next_round"),
            }
            for p in data["elements"]
        ]
        df = pd.DataFrame(rows, columns=["player", "team", "status", "news",
                                         "chance_of_playing"])
        if only_with_news:
            df = df[df["news"].str.strip() != ""]
        return df.reset_index(drop=True)