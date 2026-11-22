"""Tests for the models, baselines, evaluation and Figure 3. No network access."""

import math

import numpy as np
import pandas as pd
import pytest

from src.dataprep import PROB_COLUMNS
from src.features import FeatureBuilder
from src.models import (BOOKMAKER, HOME_BASELINE, HomeBaseline, MatchPredictor,
                        calibration_inputs, make_model, score, time_split)
from src.plots import plot_calibration


# ---- synthetic data ---------------------------------------------------------
def make_matches(seed=0, n_teams=12, seasons=4, leagues=("L1", "L2"), spread=0.5):
    """Round-robin leagues with hidden strengths and a bookmaker that knows them."""
    rng = np.random.default_rng(seed)
    rows = []
    for league in leagues:
        strengths = rng.normal(0, spread, n_teams)
        for s in range(seasons):
            season = f"{21 + s}{22 + s}"
            day = pd.Timestamp(year=2021 + s, month=8, day=1)  # seasons line up across leagues
            busy = set()
            for i in range(n_teams):
                for j in range(n_teams):
                    if i == j:
                        continue
                    if i in busy or j in busy:
                        day = day + pd.offsets.Day(1)
                        busy = set()
                    busy.update((i, j))
                    d = strengths[i] - strengths[j]
                    hg = rng.poisson(math.exp(0.25 + d))
                    ag = rng.poisson(math.exp(-0.10 - d))
                    target = 0 if hg > ag else 2 if hg < ag else 1
                    weights = np.array([math.exp(0.3 + 1.5 * d), 1.1,
                                        math.exp(-1.5 * d)])
                    probs = weights / weights.sum()
                    rows.append([day, league, season, f"{league}_T{i}",
                                 f"{league}_T{j}", hg, ag, target,
                                 rng.poisson(13), rng.poisson(11),
                                 rng.poisson(4), rng.poisson(3.5), *probs])
    cols = (["Date", "League", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
             "Target", "HS", "AS", "HST", "AST"] + PROB_COLUMNS)
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def data():
    matches = make_matches()
    features = FeatureBuilder().build(matches)
    return matches, features


# ---- time_split -------------------------------------------------------------
def test_time_split_is_chronological_and_complete(data):
    matches, features = data
    split = time_split(matches, features)
    assert split.test_season == "2425"
    assert split.train["Date"].max() < split.test["Date"].min()
    assert set(split.train["Season"]) == {"2122", "2223", "2324"}
    assert not split.train[split.feature_names].isna().any().any()
    assert not split.test[split.feature_names].isna().any().any()
    assert len(split.train) + len(split.test) <= len(matches)


def test_time_split_with_explicit_test_season(data):
    matches, features = data
    split = time_split(matches, features, test_season="2324")
    assert set(split.test["Season"]) == {"2324"}
    assert set(split.train["Season"]) == {"2122", "2223"}


def test_time_split_without_training_data_raises(data):
    matches, features = data
    with pytest.raises(ValueError):
        time_split(matches, features, test_season="2122")  # nothing earlier


# ---- metrics and baseline ---------------------------------------------------
def test_score_perfect_and_uniform():
    y = np.array([0, 1, 2, 0])
    perfect = np.eye(3)[y]
    assert score(y, perfect)["accuracy"] == 1.0
    uniform = np.full((4, 3), 1 / 3)
    assert score(y, uniform)["log_loss"] == pytest.approx(math.log(3))


def test_home_baseline_uses_training_frequencies():
    baseline = HomeBaseline().fit([0, 0, 0, 1, 2])
    proba = baseline.predict_proba(4)
    assert proba.shape == (4, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0] == pytest.approx([0.6, 0.2, 0.2])
    with pytest.raises(RuntimeError):
        HomeBaseline().predict_proba(3)


# ---- MatchPredictor ---------------------------------------------------------
@pytest.mark.parametrize("kind", ["logistic", "forest"])
def test_predictor_outputs_valid_probabilities(data, kind):
    matches, features = data
    split = time_split(matches, features)
    model = MatchPredictor(kind).fit(split.train[split.feature_names],
                                     split.train["Target"])
    proba = model.predict_proba(split.test[split.feature_names])
    assert proba.shape == (len(split.test), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()
    predictions = model.predict(split.test[split.feature_names])
    assert set(predictions) <= {0, 1, 2}
    result = model.evaluate(split.test[split.feature_names], split.test["Target"])
    assert set(result) == {"accuracy", "log_loss"}
    importance = model.feature_importance()
    assert len(importance) == 16
    assert importance.sum() == pytest.approx(1.0)
    assert list(importance) == sorted(importance, reverse=True)


def test_predictor_requires_fit_and_known_kind(data):
    matches, features = data
    with pytest.raises(RuntimeError):
        MatchPredictor("logistic").predict_proba(features)
    with pytest.raises(ValueError):
        make_model("neural-net")


def test_predictor_handles_a_class_missing_from_training(data):
    matches, features = data
    split = time_split(matches, features)
    train = split.train[split.train["Target"] != 1]          # no draws at all
    model = MatchPredictor("logistic").fit(train[split.feature_names], train["Target"])
    proba = model.predict_proba(split.test[split.feature_names])
    assert proba.shape[1] == 3
    assert np.allclose(proba[:, 1], 0.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logistic_beats_home_baseline_on_log_loss_with_real_signal(data):
    matches, features = data
    table = MatchPredictor.compare(matches, features, kinds=("logistic",))
    pick = table[(table["Training"] == "Combined") & (table["TestSet"] == "All")]
    model_loss = pick[pick["Method"] == "Logistic Regression"]["LogLoss"].iloc[0]
    base = table[(table["Method"] == HOME_BASELINE) & (table["TestSet"] == "All")]
    assert model_loss < base["LogLoss"].iloc[0]


# ---- compare ------------------------------------------------------------------
def test_compare_table_structure(data):
    matches, features = data
    table = MatchPredictor.compare(matches, features)
    assert list(table.columns) == ["Method", "Training", "TestSet", "N",
                                   "Accuracy", "AccMargin", "LogLoss"]
    assert len(table) == 6 + 2 * 7       # 6 baseline rows + 7 rows per model
    assert table["Accuracy"].between(0, 1).all()
    assert (table["LogLoss"] > 0).all()
    assert set(table["TestSet"]) == {"L1", "L2", "All"}
    assert {"League-specific", "Combined", "Transfer from L1",
            "Transfer from L2", "-"} == set(table["Training"])
    for method in (HOME_BASELINE, BOOKMAKER):
        assert (table["Method"] == method).sum() == 3


def test_compare_uses_identical_test_rows_for_every_method(data):
    matches, features = data
    table = MatchPredictor.compare(matches, features)
    sizes = table.groupby("TestSet")["N"].nunique()
    assert (sizes == 1).all()


def test_compare_is_reproducible(data):
    matches, features = data
    a = MatchPredictor.compare(matches, features, random_state=7)
    b = MatchPredictor.compare(matches, features, random_state=7)
    pd.testing.assert_frame_equal(a, b)


# ---- calibration and Figure 3 -------------------------------------------------
def test_calibration_inputs_and_plot(data, tmp_path):
    matches, features = data
    y, probabilities = calibration_inputs(matches, features, kind="logistic")
    assert set(probabilities) == {"Logistic Regression", BOOKMAKER}
    for proba in probabilities.values():
        assert proba.shape == (len(y), 3)
    out = plot_calibration(y, probabilities, save_path=tmp_path / "cal.png")
    assert out.exists() and out.stat().st_size > 1000


def test_plot_calibration_rejects_bad_shapes(tmp_path):
    with pytest.raises(ValueError):
        plot_calibration(np.array([0, 1, 2]), {"bad": np.ones((3, 2))},
                         save_path=tmp_path / "x.png")