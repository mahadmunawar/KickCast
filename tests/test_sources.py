"""Tests for the data layer. No real network access is used."""

import pytest

from src import sources
from src.sources import DataFetchError, FootballCsvSource, FplApiSource

SAMPLE_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H\n"
    "E0,15/08/2025,Liverpool,Bournemouth,4,2,H,1.3\n"
    "E0,16/08/2025,Aston Villa,Newcastle,0,0,D,2.6\n"
    ",,,,,,,\n"
).encode()

SAMPLE_FPL = {
    "teams": [{"id": 1, "name": "Arsenal"}],
    "elements": [
        {"id": 10, "web_name": "Saka", "team": 1, "status": "d",
         "news": "Hamstring injury - 75% chance of playing",
         "chance_of_playing_next_round": 75},
        {"id": 11, "web_name": "Raya", "team": 1, "status": "a",
         "news": "", "chance_of_playing_next_round": None},
    ],
}


def _offline(url):
    raise DataFetchError("simulated network failure")


class FakeResponse:
    status_code = 503
    content = b""


def test_csv_downloads_cleans_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "RAW_DIR", tmp_path)
    src = FootballCsvSource("EPL", seasons=["2526"])
    monkeypatch.setattr(src, "_download", lambda url: SAMPLE_CSV)
    df = src.load()
    assert len(df) == 2                      # blank row dropped
    assert src.status["2526"] == "web"
    assert (tmp_path / "EPL_2526.csv").exists()
    assert df["Date"].dt.year.iloc[0] == 2025
    assert set(df["League"]) == {"EPL"}


def test_csv_falls_back_to_cache_when_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "RAW_DIR", tmp_path)
    src = FootballCsvSource("EPL", seasons=["2526"])
    monkeypatch.setattr(src, "_download", lambda url: SAMPLE_CSV)
    src.load()
    monkeypatch.setattr(src, "_download", _offline)
    df = src.load()
    assert len(df) == 2
    assert src.status["2526"] == "cache"


def test_missing_cache_raises_datafetcherror(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "RAW_DIR", tmp_path)
    src = FootballCsvSource("LaLiga", seasons=["2526"])
    monkeypatch.setattr(src, "_download", _offline)
    with pytest.raises(DataFetchError):
        src.load()


def test_bad_status_code_raises_datafetcherror(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "RAW_DIR", tmp_path)
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResponse())
    with pytest.raises(DataFetchError):
        FplApiSource().load()


def test_bad_csv_does_not_overwrite_good_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "RAW_DIR", tmp_path)
    src = FootballCsvSource("EPL", seasons=["2526"])
    monkeypatch.setattr(src, "_download", lambda url: SAMPLE_CSV)
    src.load()
    monkeypatch.setattr(src, "_download", lambda url: b"<html>error page</html>")
    df = src.load()
    assert len(df) == 2
    assert src.status["2526"] == "cache"


def test_unknown_league_rejected():
    with pytest.raises(ValueError):
        FootballCsvSource("Bundesliga")


def test_fpl_news_parsing():
    news = FplApiSource().get_news(SAMPLE_FPL)
    assert len(news) == 1
    assert news.loc[0, "team"] == "Arsenal"
    assert news.loc[0, "chance_of_playing"] == 75