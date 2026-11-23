"""Tests for the Poisson scoreline model and Figure 2. No network access."""

import math

import numpy as np
import pandas as pd
import pytest

from src.models import score
from src.plots import plot_scoreline_heatmap
from src.poisson import ScorelineModel, poisson_pmf_vector


# ---- helpers ----------------------------------------------------------------
def make_matches(seed=0, n_teams=10, seasons=3, leagues=("L1",)):
    """Balanced double round-robins with hidden attack/defence strengths."""
    rng = np.random.default_rng(seed)
    rows, truth = [], {}
    for league in leagues:
        attack = rng.normal(0, 0.3, n_teams)
        leak = rng.normal(0, 0.3, n_teams)     # bigger = concedes more
        for i in range(n_teams):
            truth[f"{league}_T{i}"] = (attack[i], leak[i])
        for s in range(seasons):
            for i in range(n_teams):
                for j in range(n_teams):
                    if i == j:
                        continue
                    hg = rng.poisson(math.exp(0.30 + attack[i] + leak[j]))
                    ag = rng.poisson(math.exp(0.05 + attack[j] + leak[i]))
                    rows.append([league, f"2{s}", f"{league}_T{i}",
                                 f"{league}_T{j}", hg, ag])
    df = pd.DataFrame(rows, columns=["League", "Season", "HomeTeam",
                                     "AwayTeam", "FTHG", "FTAG"])
    return df, truth


def pmf(k, rate):
    return math.exp(-rate) * rate ** k / math.factorial(k)


@pytest.fixture(scope="module")
def fitted():
    matches, truth = make_matches(seed=1)
    return ScorelineModel().fit(matches), matches, truth


# ---- poisson_pmf_vector -------------------------------------------------------
def test_pmf_vector_matches_formula_and_sums_to_one():
    vector = poisson_pmf_vector(1.7, 12)
    assert len(vector) == 13
    for k in range(13):
        assert vector[k] == pytest.approx(pmf(k, 1.7))
    assert vector.sum() == pytest.approx(1.0, abs=1e-6)


def test_pmf_vector_with_zero_rate():
    vector = poisson_pmf_vector(0.0, 4)
    assert vector[0] == 1.0 and vector[1:].sum() == 0.0


# ---- fit and strengths ----------------------------------------------------------
def test_strengths_average_to_one_on_a_balanced_schedule(fitted):
    model, _, _ = fitted
    attack = [model.strengths(t)[0] for t in model.teams()]
    defence = [model.strengths(t)[1] for t in model.teams()]
    assert np.mean(attack) == pytest.approx(1.0, abs=1e-9)
    assert np.mean(defence) == pytest.approx(1.0, abs=1e-9)


def test_strengths_recover_true_ability(fitted):
    model, _, truth = fitted
    teams = model.teams()
    fit_attack = [model.strengths(t)[0] for t in teams]
    fit_defence = [model.strengths(t)[1] for t in teams]
    true_attack = [truth[t][0] for t in teams]
    true_leak = [truth[t][1] for t in teams]
    assert np.corrcoef(fit_attack, true_attack)[0, 1] > 0.6
    assert np.corrcoef(fit_defence, true_leak)[0, 1] > 0.6


def test_shrinkage_pulls_strengths_towards_one():
    matches, _ = make_matches(seed=2)
    raw = ScorelineModel(shrinkage=0).fit(matches)
    shrunk = ScorelineModel(shrinkage=100).fit(matches)
    for team in raw.teams():
        assert abs(shrunk.strengths(team)[0] - 1) < abs(raw.strengths(team)[0] - 1) + 1e-12


def test_fit_handles_several_leagues_separately():
    matches, _ = make_matches(seed=3, leagues=("L1", "L2"))
    model = ScorelineModel().fit(matches)
    assert len(model.teams("L1")) == 10 and len(model.teams("L2")) == 10
    assert len(model.teams()) == 20
    with pytest.raises(ValueError):
        model.expected_goals("L1_T0", "L2_T1")        # different leagues


# ---- expected goals ----------------------------------------------------------------
def test_home_advantage_shows_up_in_expected_goals(fitted):
    model, _, _ = fitted
    home_total = away_total = 0.0
    for a in model.teams():
        for b in model.teams():
            if a != b:
                lam_h, lam_a = model.expected_goals(a, b)
                home_total += lam_h
                away_total += lam_a
    assert home_total > away_total


def test_better_attack_and_weaker_defence_raise_expected_goals(fitted):
    model, _, _ = fitted
    teams = model.teams()
    by_attack = sorted(teams, key=lambda t: model.strengths(t)[0])
    best, worst = by_attack[-1], by_attack[0]
    opponent = next(t for t in teams if t not in (best, worst))
    assert model.expected_goals(best, opponent)[0] > model.expected_goals(worst, opponent)[0]


def test_expected_goals_formula(fitted):
    model, matches, _ = fitted
    a, b = model.teams()[:2]
    att_a, def_a = model.strengths(a)
    att_b, def_b = model.strengths(b)
    mu_h, mu_a = matches["FTHG"].mean(), matches["FTAG"].mean()
    lam_h, lam_a = model.expected_goals(a, b)
    assert lam_h == pytest.approx(mu_h * att_a * def_b)
    assert lam_a == pytest.approx(mu_a * att_b * def_a)


# ---- scoreline matrix -------------------------------------------------------------------
def test_scoreline_matrix_is_the_outer_product(fitted):
    model, _, _ = fitted
    a, b = model.teams()[:2]
    lam_h, lam_a = model.expected_goals(a, b)
    m = model.scoreline_matrix(a, b)
    assert m.shape == (7, 7)
    assert (m >= 0).all() and 0.95 < m.sum() <= 1.0
    for i, j in [(0, 0), (2, 1), (1, 3), (6, 6)]:
        assert m[i, j] == pytest.approx(pmf(i, lam_h) * pmf(j, lam_a))


def test_marginals_recover_expected_goals():
    matches, _ = make_matches(seed=4)
    model = ScorelineModel(max_goals=12).fit(matches)
    a, b = model.teams()[:2]
    lam_h, lam_a = model.expected_goals(a, b)
    home_dist, away_dist = model.marginal_goals(model.scoreline_matrix(a, b))
    goals = np.arange(13)
    assert (home_dist * goals).sum() == pytest.approx(lam_h, abs=1e-3)
    assert (away_dist * goals).sum() == pytest.approx(lam_a, abs=1e-3)
    assert home_dist.sum() == pytest.approx(1.0)


# ---- outcome probabilities (crafted matrices, orientation matters) ----------------------
def test_outcome_probabilities_orientation():
    # rows = home goals, columns = away goals
    m = np.array([[0.10, 0.05, 0.02],     # home 0: draw, away wins
                  [0.30, 0.20, 0.03],     # home 1: home win, draw, away win
                  [0.20, 0.07, 0.03]])    # home 2: home win, home win, draw
    home, draw, away = ScorelineModel.outcome_probabilities(m)
    assert home == pytest.approx(0.30 + 0.20 + 0.07)
    assert draw == pytest.approx(0.10 + 0.20 + 0.03)
    assert away == pytest.approx(0.05 + 0.02 + 0.03)
    assert home + draw + away == pytest.approx(1.0)


def test_outcome_probabilities_match_a_brute_force_loop(fitted):
    model, _, _ = fitted
    a, b = model.teams()[1:3]
    m = model.scoreline_matrix(a, b)
    home = draw = away = 0.0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if i > j:
                home += m[i, j]
            elif i == j:
                draw += m[i, j]
            else:
                away += m[i, j]
    expected = np.array([home, draw, away]) / m.sum()
    assert model.outcome_probabilities(m) == pytest.approx(expected)


def test_identical_teams_favour_the_home_side():
    m = ScorelineModel.outcome_probabilities(
        np.outer(poisson_pmf_vector(1.5, 8), poisson_pmf_vector(1.2, 8)))
    assert m[0] > m[2]


def test_most_likely_score_and_prob_over():
    m = np.zeros((4, 4))
    m[2, 1] = 0.5      # 2-1, three goals
    m[0, 0] = 0.3      # 0-0
    m[1, 1] = 0.2      # 1-1
    assert ScorelineModel.most_likely_score(m) == (2, 1)
    assert ScorelineModel.prob_over(m, 2.5) == pytest.approx(0.5)
    assert ScorelineModel.prob_over(m, 1.5) == pytest.approx(0.7)
    assert ScorelineModel.prob_over(m, 0.5) == pytest.approx(0.7)


# ---- predict_proba ---------------------------------------------------------------------------
def test_predict_proba_shape_order_and_sums(fitted):
    model, matches, _ = fitted
    proba = model.predict_proba(matches.head(25))
    assert proba.shape == (25, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba > 0).all()
    row = matches.iloc[0]
    manual = model.outcome_probabilities(
        model.scoreline_matrix(row["HomeTeam"], row["AwayTeam"]))
    assert proba[0] == pytest.approx(manual)


def test_predict_proba_uses_only_fitted_strengths_not_result_goals():
    matches, _ = make_matches(seed=5, seasons=3)
    train = matches[matches["Season"] != "22"]
    test = matches[matches["Season"] == "22"].copy()
    model = ScorelineModel().fit(train)
    before = model.predict_proba(test)
    test[["FTHG", "FTAG"]] = 9
    after = model.predict_proba(test)
    assert np.array_equal(before, after)


def test_unseen_team_is_treated_as_average_when_not_strict(fitted):
    model, _, _ = fitted
    newcomer = pd.DataFrame({"League": ["L1"], "HomeTeam": ["Newcomer FC"],
                             "AwayTeam": [model.teams()[0]]})
    proba = model.predict_proba(newcomer)
    assert proba.shape == (1, 3) and proba.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        model.expected_goals("Newcomer FC", model.teams()[0], league="L1")


def test_poisson_probabilities_are_scored_like_other_models():
    matches, _ = make_matches(seed=6, seasons=4)
    train = matches[matches["Season"] != "23"]
    test = matches[matches["Season"] == "23"]
    outcome = np.where(test["FTHG"] > test["FTAG"], 0,
                       np.where(test["FTHG"] == test["FTAG"], 1, 2))
    proba = ScorelineModel().fit(train).predict_proba(test)
    result = score(outcome, proba)
    assert 0 < result["accuracy"] < 1
    assert result["log_loss"] < math.log(3) + 0.05    # not worse than guessing


# ---- errors -------------------------------------------------------------------------------------
def test_errors_and_validation(fitted):
    model, matches, _ = fitted
    with pytest.raises(RuntimeError):
        ScorelineModel().scoreline_matrix("A", "B", league="L1")
    with pytest.raises(ValueError):
        model.expected_goals(model.teams()[0], model.teams()[0])
    with pytest.raises(ValueError):
        model.expected_goals("Nobody", "Nobody Else")
    with pytest.raises(ValueError):
        model.expected_goals(model.teams()[0], model.teams()[1], league="Mars")
    with pytest.raises(ValueError):
        ScorelineModel(max_goals=0)
    with pytest.raises(ValueError):
        ScorelineModel(shrinkage=-1)
    with pytest.raises(ValueError):
        ScorelineModel().fit(matches.drop(columns=["FTHG"]))
    with pytest.raises(ValueError):
        ScorelineModel().fit(matches.iloc[0:0])
    with pytest.raises(ValueError):
        ScorelineModel.outcome_probabilities(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        ScorelineModel.outcome_probabilities(np.array([0.2, 0.8]))


# ---- Figure 2 ----------------------------------------------------------------------------------------
def test_heatmap_is_saved(fitted, tmp_path):
    model, _, _ = fitted
    a, b = model.teams()[:2]
    out = plot_scoreline_heatmap(model.scoreline_matrix(a, b), a, b,
                                 save_path=tmp_path / "heat.png")
    assert out.exists() and out.stat().st_size > 1000


def test_heatmap_rejects_bad_matrix(tmp_path):
    with pytest.raises(ValueError):
        plot_scoreline_heatmap(np.array([0.5, 0.5]), "A", "B",
                               save_path=tmp_path / "x.png")