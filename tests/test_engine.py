"""Tests for KickCastEngine on synthetic data (no network, no real files)."""

import numpy as np
import pandas as pd
import pytest

from src.engine import POISSON_NAME, KickCastEngine
from src.models import BOOKMAKER, HOME_BASELINE
from tests.synthetic import make_fpl, make_matches


@pytest.fixture(scope="module")
def matches():
    return make_matches()


@pytest.fixture(scope="module")
def engine(matches):
    return KickCastEngine(matches, make_fpl(), ["a note"])


# ---- information -------------------------------------------------------------------------------
def test_leagues_seasons_and_teams(engine):
    assert engine.leagues == ["EPL", "LaLiga"]
    assert engine.seasons == ["2122", "2223", "2324", "2425"]
    epl = engine.teams("EPL")
    assert len(epl) == 12 and epl == sorted(epl)
    assert "Tottenham" in epl and "Man United" in epl
    assert engine.notes == ["a note"]
    with pytest.raises(ValueError):
        engine.teams("Serie A")


def test_engine_rejects_empty_data(matches):
    with pytest.raises(ValueError):
        KickCastEngine(matches.iloc[0:0])


# ---- Elo ----------------------------------------------------------------------------------------
def test_elo_ranking(engine):
    table = engine.elo_ranking("EPL", top=5)
    assert list(table.columns) == ["Rank", "Team", "Elo"]
    assert list(table["Rank"]) == [1, 2, 3, 4, 5]
    assert list(table["Elo"]) == sorted(table["Elo"], reverse=True)
    assert table["Team"].isin(engine.teams("EPL")).all()


def test_elo_ranking_validation(engine):
    with pytest.raises(ValueError):
        engine.elo_ranking("Serie A")
    with pytest.raises(ValueError):
        engine.elo_ranking("EPL", top=0)
    with pytest.raises(ValueError):
        engine.elo_ranking("EPL", season="1999")
    older = engine.elo_ranking("EPL", top=3, season="2122")
    assert len(older) == 3


# ---- fixture prediction ----------------------------------------------------------------------------
def test_predict_fixture_outputs_are_valid(engine):
    p = engine.predict_fixture("EPL", "Tottenham", "Man United")
    assert (p.home, p.away, p.league) == ("Tottenham", "Man United", "EPL")
    for proba in (p.model_proba, p.poisson_proba):
        assert proba.shape == (3,)
        assert proba.sum() == pytest.approx(1.0)
        assert (proba >= 0).all()
    assert len(p.features) == 16 and not p.features.isna().any()
    assert p.matrix.ndim == 2 and p.matrix.shape[0] == p.matrix.shape[1]
    assert 0.0 <= p.over_2_5 <= 1.0
    assert all(isinstance(g, int) for g in p.likely_score)
    assert all(g > 0 for g in p.expected_goals)


def test_a_much_stronger_home_side_is_favoured(engine):
    ranking = engine.elo_ranking("EPL", top=12)
    strongest, weakest = ranking["Team"].iloc[0], ranking["Team"].iloc[-1]
    strong_home = engine.predict_fixture("EPL", strongest, weakest)
    weak_home = engine.predict_fixture("EPL", weakest, strongest)
    for p in (strong_home,):
        assert p.model_proba[0] > p.model_proba[2]
        assert p.poisson_proba[0] > p.poisson_proba[2]
    assert strong_home.model_proba[0] > weak_home.model_proba[0]
    assert weak_home.model_proba[2] > strong_home.model_proba[2]


def test_predict_fixture_validation(engine):
    with pytest.raises(ValueError):
        engine.predict_fixture("EPL", "Tottenham", "Tottenham")
    with pytest.raises(ValueError):
        engine.predict_fixture("EPL", "Tottenham", "Nowhere FC")
    with pytest.raises(ValueError):
        engine.predict_fixture("EPL", "Tottenham", "LaLiga_T0")      # other league's team
    with pytest.raises(ValueError):
        engine.predict_fixture("Serie A", "Tottenham", "Man United")


def test_team_without_recent_history_is_refused(matches):
    early = matches.iloc[:40]                      # the very first matchdays only
    sparse = KickCastEngine(early)
    in_epl = early[early["League"] == "EPL"]
    played = pd.concat([in_epl["HomeTeam"], in_epl["AwayTeam"]]).value_counts()
    short = played[played < 3].index[0]            # fewer matches than min_history
    other = played.index[0]
    with pytest.raises(ValueError, match="not enough recent matches"):
        sparse.predict_fixture("EPL", short, other)


# ---- injury news --------------------------------------------------------------------------------------
def test_availability_report_maps_fpl_names(engine):
    report = engine.availability_report()
    assert list(report.columns) == ["Team", "Player", "News", "Official", "Predicted", "Agree"]
    teams = set(report["Team"])
    assert {"Tottenham", "Man United"} <= teams
    assert "Spurs" not in teams and "Man Utd" not in teams
    assert set(report["Official"]) <= {"out", "doubtful"}      # fit players have no news
    assert not report["Player"].str.contains("Gone").any()       # departed players excluded
    one = engine.availability_report("Tottenham")
    assert set(one["Team"]) == {"Tottenham"} and len(one) == 10


def test_availability_summary_teams_and_honesty(engine):
    summary = engine.availability_summary()
    assert list(summary.columns) == ["out", "doubtful", "fit"]
    assert summary.loc["Tottenham", "out"] == 6
    assert "Tottenham" in engine.availability_teams()
    honesty = engine.availability_honesty()
    assert honesty["n"] == 40
    assert 0.0 <= honesty["majority_baseline"] <= honesty["accuracy"] <= 1.0
    assert honesty["label_counts"] == {"out": 24, "doubtful": 16}
    with pytest.raises(ValueError):
        engine.availability_report("Nowhere FC")


def test_fixture_availability_only_for_the_epl(engine):
    shown = engine.fixture_availability("EPL", "Tottenham", "Man United")
    assert set(shown) == {"Tottenham", "Man United"}
    assert all(len(frame) == 10 for frame in shown.values())
    empty = engine.fixture_availability("EPL", "EPL_T5", "EPL_T6")      # no news for them
    assert all(frame.empty for frame in empty.values())
    assert engine.fixture_availability("LaLiga", "LaLiga_T0", "LaLiga_T1") is None


def test_unmatched_fpl_teams_lists_only_new_clubs(engine):
    assert engine.unmatched_fpl_teams() == ["Hull City"]


def test_engine_without_fpl_data(matches):
    bare = KickCastEngine(matches)
    assert not bare.has_availability
    assert bare.availability_honesty() is None
    assert bare.fixture_availability("EPL", "Tottenham", "Man United") is None
    assert bare.unmatched_fpl_teams() == []
    with pytest.raises(ValueError):
        bare.availability_report()


# ---- evaluation ------------------------------------------------------------------------------------------
def test_evaluation_table(engine):
    table, season = engine.evaluation()
    assert season == "2425"
    assert {HOME_BASELINE, BOOKMAKER, POISSON_NAME, "Logistic Regression",
            "Random Forest"} <= set(table["Method"])
    poisson = table[table["Method"] == POISSON_NAME]
    assert set(poisson["TestSet"]) == {"EPL", "LaLiga", "All"}
    assert table["Accuracy"].between(0, 1).all() and (table["LogLoss"] > 0).all()
    sizes = table.groupby("TestSet")["N"].nunique()
    assert (sizes == 1).all()                      # every method scored on the same matches
    assert engine.evaluation()[0] is table         # cached


def test_poisson_beats_the_home_baseline_on_log_loss_with_real_signal(engine):
    table, _ = engine.evaluation()
    everything = table[table["TestSet"] == "All"].set_index("Method")["LogLoss"]
    assert everything[POISSON_NAME] < everything[HOME_BASELINE]


def test_evaluation_needs_two_seasons(matches):
    single = KickCastEngine(matches[matches["Season"] == "2425"])
    with pytest.raises(ValueError):
        single.evaluation()


# ---- figures -----------------------------------------------------------------------------------------------
def test_figure_savers_write_files(engine, tmp_path):
    prediction = engine.predict_fixture("EPL", "Tottenham", "Man United")
    heat = engine.save_heatmap(prediction, tmp_path / "heat.png")
    elo = engine.save_elo_figure("EPL", top=3, path=tmp_path / "elo.png")
    cal = engine.save_calibration_figure(tmp_path / "cal.png")
    for path in (heat, elo, cal):
        assert path.exists() and path.stat().st_size > 1000
    with pytest.raises(ValueError):
        engine.save_elo_figure("Serie A", path=tmp_path / "x.png")