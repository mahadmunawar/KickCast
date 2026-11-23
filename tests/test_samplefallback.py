"""Tests for the offline sample snapshot and the load_data() fallback chain."""

import json

import pandas as pd
import pytest

from src import engine
from src.engine import LoadedData, describe_age, load_data, season_label
from src.sources import SAMPLE_DIR, DataFetchError, FootballCsvSource, FplApiSource
from tests.synthetic import make_fpl, raw_frame

CSV = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,B365H,B365D,B365A\n"
       "E0,15/08/2025,Arsenal,Chelsea,2,1,H,12,8,5,3,1.8,3.6,4.5\n"
       "E0,16/08/2025,Everton,Fulham,0,0,D,9,10,2,4,2.6,3.2,2.9\n")


# ---- sample loaders ----------------------------------------------------------------------
def test_load_sample_csv_adds_league_and_season(tmp_path):
    (tmp_path / "EPL_2526.csv").write_text(CSV)
    df = FootballCsvSource.load_sample("EPL", sample_dir=tmp_path)
    assert len(df) == 2
    assert set(df["League"]) == {"EPL"} and set(df["Season"]) == {"2526"}
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert df["Date"].iloc[0] == pd.Timestamp("2025-08-15")


def test_load_sample_csv_errors(tmp_path):
    with pytest.raises(DataFetchError):
        FootballCsvSource.load_sample("EPL", sample_dir=tmp_path)      # file missing
    with pytest.raises(ValueError):
        FootballCsvSource.load_sample("Serie A", sample_dir=tmp_path)  # unknown league
    (tmp_path / "LaLiga_2526.csv").write_text("not,a,match,file\n1,2,3,4\n")
    with pytest.raises(DataFetchError):
        FootballCsvSource.load_sample("LaLiga", sample_dir=tmp_path)   # wrong columns


def test_load_sample_fpl(tmp_path):
    with pytest.raises(DataFetchError):
        FplApiSource.load_sample(sample_dir=tmp_path)
    (tmp_path / "fpl_bootstrap.json").write_text("{not json")
    with pytest.raises(DataFetchError):
        FplApiSource.load_sample(sample_dir=tmp_path)
    (tmp_path / "fpl_bootstrap.json").write_text(json.dumps(make_fpl()))
    data = FplApiSource.load_sample(sample_dir=tmp_path)
    assert {"teams", "elements"} <= set(data)


@pytest.mark.skipif(not (SAMPLE_DIR / "EPL_2526.csv").exists(),
                    reason="repository sample snapshot not present")
def test_the_committed_sample_snapshot_loads():
    for league in ("EPL", "LaLiga"):
        df = FootballCsvSource.load_sample(league)
        assert len(df) > 300
        assert set(df["Season"]) == {"2526"}
    assert len(FplApiSource.load_sample()["elements"]) > 100


def test_cache_age_helper(tmp_path, monkeypatch):
    source = FootballCsvSource("EPL")
    monkeypatch.setattr(source, "_cache_path", tmp_path)
    assert source.season_cache_age_hours("2526") is None
    (tmp_path / "EPL_2526.csv").write_text(CSV)
    assert source.season_cache_age_hours("2526") < 1


# ---- helpers ----------------------------------------------------------------------------------
def test_season_label_and_age_text():
    assert season_label("2526") == "2025/26"
    assert season_label("weird") == "weird"
    assert describe_age(None) == "of unknown age"
    assert describe_age(0.2) == "less than an hour old"
    assert describe_age(5) == "5 hours old"
    assert describe_age(72) == "3 days old"


# ---- load_data fallback chain --------------------------------------------------------------------
def _fake_web_load(origin):
    def load(self):
        self.status = {season: origin for season in ("2425", "2526")}
        return raw_frame(self.league)
    return load


def test_load_data_from_the_web(monkeypatch):
    monkeypatch.setattr(FootballCsvSource, "load", _fake_web_load("web"))
    monkeypatch.setattr(FplApiSource, "load", lambda self: setattr(self, "origin", "web") or make_fpl())
    data = load_data()
    assert isinstance(data, LoadedData)
    assert data.origin == "web"
    assert set(data.matches["League"]) == {"EPL", "LaLiga"}
    assert data.fpl is not None
    assert any("downloaded just now" in note for note in data.notes)


def test_load_data_from_the_cache_reports_its_age(monkeypatch):
    monkeypatch.setattr(FootballCsvSource, "load", _fake_web_load("cache"))
    monkeypatch.setattr(FootballCsvSource, "season_cache_age_hours", lambda self, season: 72.0)

    def cached_fpl(self):
        self.origin = "cache"
        return make_fpl()

    monkeypatch.setattr(FplApiSource, "load", cached_fpl)
    monkeypatch.setattr(FplApiSource, "cache_age_hours", lambda self, path=None: 5.0)
    data = load_data()
    assert data.origin == "cache"
    text = " ".join(data.notes)
    assert "3 days old" in text and "5 hours old" in text


def test_load_data_falls_back_to_the_sample_snapshot(monkeypatch):
    def offline(self):
        raise DataFetchError("no network and no cache")

    monkeypatch.setattr(FootballCsvSource, "load", offline)
    monkeypatch.setattr(FplApiSource, "load", offline)
    monkeypatch.setattr(FootballCsvSource, "load_sample",
                        classmethod(lambda cls, league, **kw: raw_frame(league, ("2526",))))
    monkeypatch.setattr(FplApiSource, "load_sample", classmethod(lambda cls, **kw: make_fpl()))
    data = load_data()
    assert data.origin == "sample"
    assert set(data.matches["Season"]) == {"2526"}
    assert data.fpl is not None
    text = " ".join(data.notes)
    assert "sample snapshot" in text


def test_load_data_without_fpl_still_returns_matches(monkeypatch):
    monkeypatch.setattr(FootballCsvSource, "load", _fake_web_load("web"))

    def offline(self):
        raise DataFetchError("fpl down")

    monkeypatch.setattr(FplApiSource, "load", offline)
    monkeypatch.setattr(FplApiSource, "load_sample",
                        classmethod(lambda cls, **kw: (_ for _ in ()).throw(DataFetchError("none"))))
    data = load_data()
    assert data.fpl is None
    assert any("availability report is disabled" in note for note in data.notes)


def test_load_data_raises_when_nothing_at_all_is_available(monkeypatch):
    def offline(self):
        raise DataFetchError("nothing")

    monkeypatch.setattr(FootballCsvSource, "load", offline)
    monkeypatch.setattr(FootballCsvSource, "load_sample", classmethod(lambda cls, league, **kw: offline(None)))
    with pytest.raises(DataFetchError):
        engine.load_data()