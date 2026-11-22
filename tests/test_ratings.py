"""Tests for the Elo rating system and Figure 1. No network access is used."""

import math

import numpy as np
import pandas as pd
import pytest

from src.plots import plot_elo_trajectories
from src.ratings import EloRating, compute_elo, elo_table


# ---- helpers --------------------------------------------------------------
def simulate_league(seed=0, n_teams=12, seasons=3, league="L"):
    """Round-robin seasons with hidden team strengths (seeded, deterministic)."""
    rng = np.random.default_rng(seed)
    strengths = rng.normal(0, 0.3, n_teams)
    rows = []
    day = pd.Timestamp("2022-08-01")
    busy = set()  # teams that already play on the current day
    for season in range(seasons):
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                if i in busy or j in busy:  # a team plays at most once per day
                    day = day + pd.offsets.Day(1)
                    busy = set()
                busy.update((i, j))
                hg = rng.poisson(math.exp(0.25 + strengths[i] - strengths[j]))
                ag = rng.poisson(math.exp(-0.10 + strengths[j] - strengths[i]))
                ftr = "H" if hg > ag else "A" if hg < ag else "D"
                rows.append([day, league, f"2{season}", f"T{i}", f"T{j}", hg, ag, ftr])
    df = pd.DataFrame(rows, columns=["Date", "League", "Season", "HomeTeam",
                                     "AwayTeam", "FTHG", "FTAG", "FTR"])
    return df, strengths


# ---- EloRating basics -----------------------------------------------------
def test_equal_teams_without_home_advantage_expect_half():
    elo = EloRating(home_advantage=0)
    assert elo.expected_score("A", "B") == pytest.approx(0.5)


def test_home_advantage_raises_home_expectation():
    elo = EloRating(home_advantage=50)
    assert elo.expected_score("A", "B") > 0.5


def test_expected_scores_are_complementary_without_home_advantage():
    elo = EloRating(home_advantage=0)
    elo.update("A", "B", 3, 0)
    total = elo.expected_score("A", "B") + elo.expected_score("B", "A")
    assert total == pytest.approx(1.0)


def test_update_is_zero_sum():
    elo = EloRating()
    delta = elo.update("A", "B", 2, 1)
    assert delta > 0
    assert elo.rating("A") + elo.rating("B") == pytest.approx(3000.0)


def test_home_win_raises_home_and_lowers_away():
    elo = EloRating()
    elo.update("A", "B", 1, 0)
    assert elo.rating("A") > 1500 > elo.rating("B")


def test_draw_between_equal_teams_hurts_the_home_side():
    elo = EloRating(home_advantage=50)
    elo.update("A", "B", 1, 1)
    assert elo.rating("A") < 1500 < elo.rating("B")


def test_upset_moves_ratings_more_than_expected_win():
    elo = EloRating(home_advantage=0)
    for _ in range(10):
        elo.update("Strong", "Weak", 2, 0)
    expected_win_gain = elo.update("Strong", "Weak", 2, 0)
    upset_loss = elo.update("Strong", "Weak", 0, 1)
    assert abs(upset_loss) > abs(expected_win_gain)


def test_unseen_team_starts_at_initial_rating():
    assert EloRating().rating("Nobody") == 1500.0


@pytest.mark.parametrize("args", [("A", "A", 1, 0), ("A", "B", -1, 0)])
def test_update_rejects_invalid_matches(args):
    with pytest.raises(ValueError):
        EloRating().update(*args)


def test_invalid_k_factor_rejected():
    with pytest.raises(ValueError):
        EloRating(k_factor=0)


# ---- history and table ----------------------------------------------------
def test_history_and_table():
    elo = EloRating()
    elo.update("A", "B", 2, 0, pd.Timestamp("2025-01-01"))
    elo.update("C", "A", 0, 1, pd.Timestamp("2025-01-08"))
    history = elo.history("A")
    assert list(history.columns) == ["Date", "Team", "Rating"]
    assert len(history) == 2
    assert len(elo.history(["A", "B"])) == 3
    table = elo.table()
    assert [team for team, _ in table][0] == "A"
    ratings = [rating for _, rating in table]
    assert ratings == sorted(ratings, reverse=True)
    assert len(elo.table(top=2)) == 2
    assert [t for t, _ in elo.table(teams=["B", "C"])] == ["C", "B"] or \
           [t for t, _ in elo.table(teams=["B", "C"])] == ["B", "C"]


def test_ratings_property_is_a_copy():
    elo = EloRating()
    elo.update("A", "B", 1, 0)
    snapshot = elo.ratings
    snapshot["A"] = 0
    assert elo.rating("A") > 1500


# ---- compute_elo ----------------------------------------------------------
def test_first_matches_use_initial_rating_and_leagues_are_separate():
    a, _ = simulate_league(seed=1, n_teams=4, seasons=1, league="L1")
    b, _ = simulate_league(seed=2, n_teams=4, seasons=1, league="L2")
    both = pd.concat([a, b], ignore_index=True)
    elos, pre = compute_elo(both)
    assert set(elos) == {"L1", "L2"}
    first = pre.iloc[0]
    assert first["HomeElo"] == 1500 and first["AwayElo"] == 1500
    assert set(elos["L1"].ratings) <= {"T0", "T1", "T2", "T3"}


def test_pre_match_values_use_no_future_information():
    df, _ = simulate_league(seed=3)
    _, pre_before = compute_elo(df)

    changed = df.copy()
    k = 200  # flip the outcome of a match in the middle of the data
    home_won = changed.loc[k, "FTHG"] > changed.loc[k, "FTAG"]
    changed.loc[k, ["FTHG", "FTAG"]] = [0, 1] if home_won else [1, 0]
    _, pre_after = compute_elo(changed)

    cutoff = df.loc[k, "Date"]
    unaffected = df["Date"] <= cutoff   # this match and everything up to its date
    pd.testing.assert_frame_equal(pre_before[unaffected], pre_after[unaffected])
    assert not pre_before.equals(pre_after)  # later matches DO change


def test_compute_elo_aligns_with_unsorted_input():
    df, _ = simulate_league(seed=4, n_teams=6, seasons=1)
    shuffled = df.sample(frac=1, random_state=0)
    _, pre_sorted = compute_elo(df)
    _, pre_shuffled = compute_elo(shuffled)
    pd.testing.assert_frame_equal(pre_sorted.sort_index(), pre_shuffled.sort_index())


def test_elo_recovers_true_team_strength():
    df, strengths = simulate_league(seed=5, n_teams=12, seasons=3)
    elos, _ = compute_elo(df)
    final = elos["L"].ratings
    estimated = [final[f"T{i}"] for i in range(12)]
    assert np.corrcoef(estimated, strengths)[0, 1] > 0.7


def test_elo_table_uses_latest_season_only():
    df, _ = simulate_league(seed=6, n_teams=6, seasons=2)
    df.loc[df["Season"] == "20", ["HomeTeam", "AwayTeam"]] = \
        df.loc[df["Season"] == "20", ["HomeTeam", "AwayTeam"]].replace(
            {"T0": "Relegated"})
    elos, _ = compute_elo(df)
    names = [team for team, _ in elo_table(elos, df, "L")]
    assert "Relegated" not in names
    assert len(names) == 6
    with pytest.raises(ValueError):
        elo_table(elos, df, "Nowhere")


# ---- Figure 1 -------------------------------------------------------------
def test_plot_elo_trajectories_saves_png(tmp_path):
    df, _ = simulate_league(seed=7, n_teams=6, seasons=1)
    elos, _ = compute_elo(df)
    out = plot_elo_trajectories(elos["L"], ["T0", "T1"], save_path=tmp_path / "f.png")
    assert out.exists() and out.stat().st_size > 1000


def test_plot_rejects_unknown_team(tmp_path):
    df, _ = simulate_league(seed=7, n_teams=4, seasons=1)
    elos, _ = compute_elo(df)
    with pytest.raises(ValueError):
        plot_elo_trajectories(elos["L"], ["Ghost FC"], save_path=tmp_path / "f.png")
        