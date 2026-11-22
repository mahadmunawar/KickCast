"""Tests for leakage-free features. No network access is used."""

import math

import numpy as np
import pandas as pd
import pytest

from src.features import ROLLING_STATS, FeatureBuilder


# ---- helpers --------------------------------------------------------------
def make_matches(seed=0, n_teams=8, seasons=3, leagues=("L1", "L2")):
    """Seeded round-robin matches for several leagues (a team plays <= once/day)."""
    rng = np.random.default_rng(seed)
    rows = []
    for league in leagues:
        strengths = rng.normal(0, 0.3, n_teams)
        day = pd.Timestamp("2022-08-01")
        busy = set()
        for season in range(seasons):
            for i in range(n_teams):
                for j in range(n_teams):
                    if i == j:
                        continue
                    if i in busy or j in busy:
                        day = day + pd.offsets.Day(1)
                        busy = set()
                    busy.update((i, j))
                    hg = rng.poisson(math.exp(0.25 + strengths[i] - strengths[j]))
                    ag = rng.poisson(math.exp(-0.10 + strengths[j] - strengths[i]))
                    target = 0 if hg > ag else 2 if hg < ag else 1
                    rows.append([day, league, f"2{season}", f"{league}_T{i}",
                                 f"{league}_T{j}", hg, ag, target,
                                 rng.poisson(13), rng.poisson(11),
                                 rng.poisson(4), rng.poisson(3.5)])
    df = pd.DataFrame(rows, columns=["Date", "League", "Season", "HomeTeam",
                                     "AwayTeam", "FTHG", "FTAG", "Target",
                                     "HS", "AS", "HST", "AST"])
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def manual_rolling(matches, row, side, window, min_history):
    """Recompute one rolling feature set from scratch using only earlier matches."""
    team = row["HomeTeam"] if side == "Home" else row["AwayTeam"]
    earlier = matches[(matches["Date"] < row["Date"])
                      & (matches["League"] == row["League"])
                      & ((matches["HomeTeam"] == team) | (matches["AwayTeam"] == team))]
    earlier = earlier.sort_values("Date").tail(window)
    if len(earlier) < min_history:
        return {f"{side}_{s}": np.nan for s in ROLLING_STATS}
    records = []
    for _, m in earlier.iterrows():
        at_home = m["HomeTeam"] == team
        points = {0: 3, 1: 1, 2: 0}[m["Target"]] if at_home else {0: 0, 1: 1, 2: 3}[m["Target"]]
        records.append({
            "Points": points,
            "GoalsFor": m["FTHG"] if at_home else m["FTAG"],
            "GoalsAgainst": m["FTAG"] if at_home else m["FTHG"],
            "Shots": m["HS"] if at_home else m["AS"],
            "ShotsAgainst": m["AS"] if at_home else m["HS"],
            "ShotsOnTarget": m["HST"] if at_home else m["AST"],
        })
    means = pd.DataFrame(records).mean()
    return {f"{side}_{s}": means[s] for s in ROLLING_STATS}


# ---- structure ------------------------------------------------------------
def test_features_align_with_matches_and_have_expected_columns():
    matches = make_matches()
    fb = FeatureBuilder()
    X = fb.build(matches)
    assert X.index.equals(matches.index)
    assert list(X.columns) == fb.feature_names
    assert len(fb.feature_names) == 16
    assert set(fb.elos) == {"L1", "L2"}


def test_elo_columns_are_never_missing_and_diff_is_consistent():
    X = FeatureBuilder().build(make_matches())
    assert not X[["HomeElo", "AwayElo", "EloDiff", "EloExpected"]].isna().any().any()
    assert np.allclose(X["EloDiff"], X["HomeElo"] - X["AwayElo"])


# ---- correctness: brute-force recomputation ---------------------------------
@pytest.mark.parametrize("window,min_history", [(5, 3), (3, 1), (1, 1)])
def test_rolling_features_match_manual_recomputation(window, min_history):
    matches = make_matches(seed=2)
    X = FeatureBuilder(window=window, min_history=min_history).build(matches)
    for idx in np.linspace(0, len(matches) - 1, 40).astype(int):
        row = matches.loc[idx]
        expected = {**manual_rolling(matches, row, "Home", window, min_history),
                    **manual_rolling(matches, row, "Away", window, min_history)}
        for name, value in expected.items():
            actual = X.loc[idx, name]
            if np.isnan(value):
                assert np.isnan(actual), (idx, name)
            else:
                assert actual == pytest.approx(value), (idx, name)


# ---- leakage -----------------------------------------------------------------
def test_changing_a_match_never_changes_features_up_to_that_match():
    matches = make_matches(seed=3)
    before = FeatureBuilder().build(matches)

    k = len(matches) // 2
    changed = matches.copy()
    changed.loc[k, ["FTHG", "FTAG", "Target"]] = [9, 0, 0]
    changed.loc[k, ["HS", "AS", "HST", "AST"]] = [40, 1, 20, 0]
    after = FeatureBuilder().build(changed)

    up_to_k = matches["Date"] <= matches.loc[k, "Date"]  # includes match k itself
    pd.testing.assert_frame_equal(before[up_to_k], after[up_to_k])
    assert not before.equals(after)   # ...but later matches DO change


def test_first_match_in_the_data_has_no_rolling_features():
    matches = make_matches(seed=4)
    X = FeatureBuilder(window=5, min_history=1).build(matches)
    rolling_cols = [c for c in X.columns if c.startswith(("Home_", "Away_"))]
    assert len(rolling_cols) == 12
    assert X.iloc[0][rolling_cols].isna().all()   # nobody has played before it


def test_min_history_controls_when_features_appear():
    matches = make_matches(seed=5)
    X = FeatureBuilder(window=5, min_history=3).build(matches)
    team = "L1_T0"
    played = matches[(matches["HomeTeam"] == team) | (matches["AwayTeam"] == team)]
    for position, idx in enumerate(played.index[:6]):
        col = "Home_Points" if matches.loc[idx, "HomeTeam"] == team else "Away_Points"
        assert np.isnan(X.loc[idx, col]) == (position < 3)


def test_window_of_one_equals_previous_match():
    matches = make_matches(seed=6)
    X = FeatureBuilder(window=1, min_history=1).build(matches)
    team = "L2_T3"
    played = matches[(matches["HomeTeam"] == team) | (matches["AwayTeam"] == team)]
    prev, curr = played.index[4], played.index[5]
    prev_row = matches.loc[prev]
    expected_gf = prev_row["FTHG"] if prev_row["HomeTeam"] == team else prev_row["FTAG"]
    col = "Home_GoalsFor" if matches.loc[curr, "HomeTeam"] == team else "Away_GoalsFor"
    assert X.loc[curr, col] == expected_gf


# ---- robustness ---------------------------------------------------------------
def test_leagues_do_not_mix():
    matches = make_matches(seed=7)
    both = FeatureBuilder().build(matches)
    only_l1 = FeatureBuilder().build(matches[matches["League"] == "L1"])
    pd.testing.assert_frame_equal(both.loc[only_l1.index], only_l1)


def test_unsorted_input_gives_same_features():
    matches = make_matches(seed=8)
    shuffled = matches.sample(frac=1, random_state=1)
    a = FeatureBuilder().build(matches)
    b = FeatureBuilder().build(shuffled)
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_missing_shot_values_are_tolerated():
    matches = make_matches(seed=9)
    matches.loc[matches.index[::7], "HS"] = np.nan
    X = FeatureBuilder().build(matches)
    assert len(X) == len(matches)


def test_dropna_removes_incomplete_rows():
    matches = make_matches(seed=10)
    full = FeatureBuilder().build(matches)
    trimmed = FeatureBuilder().build(matches, dropna=True)
    assert len(trimmed) == len(full) - full.isna().any(axis=1).sum()
    assert not trimmed.isna().any().any()


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        FeatureBuilder().build(make_matches().drop(columns=["HS"]))


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"window": 3, "min_history": 4},
                                    {"min_history": 0}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        FeatureBuilder(**kwargs)