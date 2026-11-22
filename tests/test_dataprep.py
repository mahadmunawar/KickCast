"""Tests for cleaning and odds conversion. No network access is used."""

import math

import numpy as np
import pandas as pd
import pytest

from src.dataprep import (PROB_COLUMNS, build_dataset, clean_matches,
                          implied_probabilities, overround, summarize)
from src.sources import DataSource

COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
           "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA", "League", "Season"]
NAN = np.nan


def make_df():
    rows = [
        # 0: good row, both odds sets -> average odds should be used
        ["2025-08-15", "Liverpool", "Bournemouth", 4, 2, "H",
         1.30, 5.5, 10.0, 1.31, 5.4, 9.8, "EPL", "2526"],
        # 1: exact duplicate of row 0
        ["2025-08-15", "Liverpool", "Bournemouth", 4, 2, "H",
         1.30, 5.5, 10.0, 1.31, 5.4, 9.8, "EPL", "2526"],
        # 2: result says away win but goals say home win
        ["2025-08-16", " Team A ", "Team B", 1, 0, "A",
         2.0, 3.0, 4.0, 2.0, 3.0, 4.0, "EPL", "2526"],
        # 3: invalid result code
        ["2025-08-16", "Team C", "Team D", 2, 1, "X",
         2.0, 3.0, 4.0, 2.0, 3.0, 4.0, "EPL", "2526"],
        # 4: only Bet365 odds -> fallback
        ["2025-08-17", "Arsenal", "Chelsea", 1, 1, "D",
         2.5, 3.3, 2.9, NAN, NAN, NAN, "EPL", "2526"],
        # 5: no odds at all
        ["2025-08-18", "Everton", "Fulham", 0, 2, "A",
         NAN, NAN, NAN, NAN, NAN, NAN, "EPL", "2526"],
        # 6: La Liga row with the earliest date (tests sorting)
        ["2025-08-10", "Getafe", "Betis", 1, 0, "H",
         2.4, 3.1, 3.0, 2.45, 3.1, 2.95, "LaLiga", "2526"],
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


class StubSource(DataSource):
    """Minimal DataSource that returns a fixed DataFrame."""

    def __init__(self, df):
        super().__init__("stub", "unused")
        self._df = df

    def load(self):
        return self._df.copy()


# ---- implied_probabilities / overround ----------------------------------
def test_normalized_probabilities_sum_to_one():
    probs = implied_probabilities([2.0, 3.0, 4.0])
    assert math.isclose(sum(probs), 1.0)
    assert math.isclose(probs[0], 0.5 / (0.5 + 1 / 3 + 0.25))


def test_raw_probabilities_exceed_one_by_overround():
    raw = implied_probabilities([2.0, 3.0, 4.0], normalize=False)
    assert sum(raw) > 1.0
    assert math.isclose(sum(raw) - 1.0, overround([2.0, 3.0, 4.0]))


@pytest.mark.parametrize("bad", [[1.0, 3.0, 4.0], [2.0, NAN, 4.0],
                                 [2.0, None, 4.0], [], [0.5, 2.0, 3.0]])
def test_invalid_odds_raise_valueerror(bad):
    with pytest.raises(ValueError):
        implied_probabilities(bad)


# ---- clean_matches -------------------------------------------------------
def test_clean_matches_drops_bad_rows_and_reports():
    clean, report = clean_matches(make_df())
    assert len(clean) == 4
    assert report["duplicates"] == 1
    assert report["bad_result_code"] == 1
    assert report["result_contradicts_goals"] == 1
    assert report["output_rows"] == 4
    assert report["rows_without_odds"] == 1


def test_clean_matches_odds_selection_and_probabilities():
    clean, _ = clean_matches(make_df())
    by_home = clean.set_index("HomeTeam")
    assert by_home.loc["Liverpool", "OddsH"] == 1.31      # average preferred
    assert by_home.loc["Arsenal", "OddsH"] == 2.5         # Bet365 fallback
    assert np.isnan(by_home.loc["Everton", "ProbH"])      # no odds
    with_odds = clean.dropna(subset=PROB_COLUMNS)
    assert np.allclose(with_odds[PROB_COLUMNS].sum(axis=1), 1.0)


def test_target_encoding():
    clean, _ = clean_matches(make_df())
    by_home = clean.set_index("HomeTeam")
    assert by_home.loc["Liverpool", "Target"] == 0   # home win
    assert by_home.loc["Arsenal", "Target"] == 1     # draw
    assert by_home.loc["Everton", "Target"] == 2     # away win


# ---- build_dataset / summarize ------------------------------------------
def test_build_dataset_combines_and_sorts_by_date():
    sources = [StubSource(make_df().iloc[:6]), StubSource(make_df().iloc[6:])]
    dataset, report = build_dataset(sources)
    assert set(dataset["League"]) == {"EPL", "LaLiga"}
    assert dataset["Date"].is_monotonic_increasing
    assert dataset.iloc[0]["League"] == "LaLiga"
    assert report["output_rows"] == len(dataset) == 4


def test_summarize_shape_and_percentages():
    dataset, _ = build_dataset([StubSource(make_df())])
    summary = summarize(dataset)
    assert len(summary) == 2                      # EPL + LaLiga, one season each
    shares = summary[["home_win_pct", "draw_pct", "away_win_pct"]].sum(axis=1)
    assert np.allclose(shares, 100.0, atol=0.2)