"""Tests for FeatureBuilder.fixture_features (features of a not-yet-played fixture)."""

import numpy as np
import pytest

from src.features import FeatureBuilder
from tests.synthetic import make_matches


@pytest.fixture(scope="module")
def matches():
    return make_matches(seed=5, n_teams=8, seasons=3)


@pytest.fixture(scope="module")
def full_features(matches):
    return FeatureBuilder().build(matches)


@pytest.mark.parametrize("position", [3, 40, 150, 250, 335])
def test_fixture_features_equal_the_build_row_of_the_same_match(matches, full_features,
                                                                position):
    """Pretend match `position` is still to be played: features from only the
    matches before it must equal the leakage-free row build() gave that match."""
    target = matches.iloc[position]
    earlier = matches.iloc[:position]
    builder = FeatureBuilder()
    builder.build(earlier)
    row = builder.fixture_features(earlier, target["League"], target["HomeTeam"],
                                   target["AwayTeam"])
    assert list(row.columns) == builder.feature_names
    assert len(row) == 1
    expected = full_features.loc[matches.index[position]].to_numpy(dtype=float)
    assert np.allclose(row.iloc[0].to_numpy(dtype=float), expected, equal_nan=True)


def test_rolling_values_are_nan_without_enough_history(matches):
    early = matches.iloc[:6]
    builder = FeatureBuilder()
    builder.build(early)
    first = early.iloc[0]
    # The away side of the very first match has played only that one match.
    assert (early["HomeTeam"].eq(first["AwayTeam"]) | early["AwayTeam"].eq(first["AwayTeam"])).sum() == 1
    row = builder.fixture_features(early, first["League"], first["HomeTeam"],
                                   first["AwayTeam"])
    assert row.filter(like="Away_").isna().all(axis=None)   # 1 match < min_history 3
    assert row[["HomeElo", "AwayElo", "EloDiff", "EloExpected"]].notna().all(axis=None)


def test_elo_columns_are_consistent(matches):
    builder = FeatureBuilder()
    builder.build(matches)
    league = matches.iloc[0]["League"]
    teams = sorted(set(matches[matches["League"] == league]["HomeTeam"]))
    row = builder.fixture_features(matches, league, teams[0], teams[1]).iloc[0]
    assert row["EloDiff"] == pytest.approx(row["HomeElo"] - row["AwayElo"])
    elo = builder.elos[league]
    assert row["EloExpected"] == pytest.approx(elo.expected_score(teams[0], teams[1]))


def test_fixture_features_validation(matches):
    builder = FeatureBuilder()
    league = matches.iloc[0]["League"]
    teams = sorted(set(matches[matches["League"] == league]["HomeTeam"]))
    with pytest.raises(RuntimeError):
        builder.fixture_features(matches, league, teams[0], teams[1])
    builder.build(matches)
    with pytest.raises(ValueError):
        builder.fixture_features(matches, "Serie A", teams[0], teams[1])
    with pytest.raises(ValueError):
        builder.fixture_features(matches, league, teams[0], teams[0])


def test_min_history_property():
    assert FeatureBuilder(window=5, min_history=2).min_history == 2