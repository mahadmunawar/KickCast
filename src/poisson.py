"""Poisson scoreline model for KickCast.

Estimates each team's attack and defence strength and turns them into a
2-D matrix of scoreline probabilities using NumPy broadcasting (outer
product of two Poisson probability vectors). Home win, draw and away win
probabilities are obtained by summing the lower triangle, diagonal and
upper triangle of the matrix, with no element loops.

How the model works:
    attack  = goals a team scores per match / league average   (1.0 = average)
    defence = goals a team concedes per match / league average (below 1.0 = tight)
    expected home goals = league home average * home attack * away defence
    expected away goals = league away average * away attack * home defence
Goals of each side are then modelled as independent Poisson variables.

Matrix layout: rows = home goals, columns = away goals, so cell [2, 1] is the
probability of a 2-1 home win.

Contents:
    poisson_pmf_vector() - Poisson probabilities for 0..max_goals goals.
    ScorelineModel       - fit(), expected_goals(), scoreline_matrix(),
                           outcome_probabilities(), marginal_goals(),
                           most_likely_score(), prob_over(), predict_proba().
"""

import numpy as np

REQUIRED_COLUMNS = ["League", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def poisson_pmf_vector(rate, max_goals):
    """P(X = k) for k = 0..max_goals when X ~ Poisson(rate), as one array.

    Uses only NumPy: k! comes from a cumulative product, so there is no loop.
    """
    goals = np.arange(max_goals + 1)
    factorials = np.cumprod(np.maximum(goals, 1))   # 1, 1, 2, 6, 24, ...
    return np.exp(-rate) * rate ** goals / factorials


def _as_matrix(matrix):
    """Validate a scoreline matrix and return it as a float array."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.size == 0:
        raise ValueError("scoreline matrix must be a non-empty 2-D array")
    if (m < 0).any() or m.sum() <= 0:
        raise ValueError("scoreline matrix must be non-negative and not all zero")
    return m


class ScorelineModel:
    """Attack/defence strengths per team and scoreline probabilities.

    Args:
        max_goals: the matrix covers 0..max_goals goals for each side. Results
            are renormalised over this range, so tiny tail probabilities are
            ignored.
        shrinkage: strengths are pulled towards 1.0 (the league average) as if
            each team had played this many extra average matches. This stops
            teams with few matches from getting extreme ratings.
    """

    def __init__(self, max_goals=6, shrinkage=5.0):
        if max_goals < 1:
            raise ValueError("max_goals must be at least 1")
        if shrinkage < 0:
            raise ValueError("shrinkage cannot be negative")
        self._max_goals = int(max_goals)
        self._shrinkage = float(shrinkage)
        self._attack = {}      # (league, team) -> attack strength
        self._defence = {}     # (league, team) -> defence strength
        self._league_of = {}   # team -> league
        self._mu_home = {}     # league -> average home goals per match
        self._mu_away = {}     # league -> average away goals per match
        self._fitted = False

    # ---- fitting ---------------------------------------------------------
    def fit(self, matches):
        """Estimate strengths from matches (DataFrame with several leagues allowed).

        Raises:
            ValueError: if columns are missing or there are no matches.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in matches.columns]
        if missing:
            raise ValueError(f"matches is missing columns: {missing}")
        if matches.empty:
            raise ValueError("cannot fit on an empty DataFrame")

        names = ["League", "Team"]
        home = matches.groupby(["League", "HomeTeam"]).agg(
            games=("FTHG", "size"), scored=("FTHG", "sum"),
            conceded=("FTAG", "sum"))
        away = matches.groupby(["League", "AwayTeam"]).agg(
            games=("FTAG", "size"), scored=("FTAG", "sum"),
            conceded=("FTHG", "sum"))
        home.index.names = names
        away.index.names = names
        totals = home.add(away, fill_value=0)   # every team: home + away games

        league_means = matches.groupby("League").agg(
            home_goals=("FTHG", "mean"), away_goals=("FTAG", "mean"))
        per_team = (league_means["home_goals"] + league_means["away_goals"]) / 2
        league_avg = per_team.reindex(
            totals.index.get_level_values("League")).to_numpy()

        games = totals["games"].to_numpy()
        attack = totals["scored"].to_numpy() / games / league_avg
        defence = totals["conceded"].to_numpy() / games / league_avg
        k = self._shrinkage
        attack = (games * attack + k) / (games + k)     # shrink towards 1.0
        defence = (games * defence + k) / (games + k)

        self._attack = dict(zip(totals.index, attack))
        self._defence = dict(zip(totals.index, defence))
        self._league_of = {team: league for league, team in totals.index}
        self._mu_home = league_means["home_goals"].to_dict()
        self._mu_away = league_means["away_goals"].to_dict()
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("call fit() first")

    # ---- information -----------------------------------------------------
    @property
    def max_goals(self):
        return self._max_goals

    def teams(self, league=None):
        """Sorted team names (of one league if given)."""
        self._check_fitted()
        return sorted(team for team, lg in self._league_of.items()
                      if league is None or lg == league)

    def strengths(self, team, league=None):
        """(attack, defence) of a known team."""
        self._check_fitted()
        league = league or self._league_of.get(team)
        key = (league, team)
        if key not in self._attack:
            raise ValueError(f"unknown team: {team}")
        return self._attack[key], self._defence[key]

    # ---- expected goals ----------------------------------------------------
    def _resolve_league(self, home_team, away_team, league):
        if league is not None:
            if league not in self._mu_home:
                raise ValueError(f"unknown league: {league}")
            return league
        known = {self._league_of[t] for t in (home_team, away_team)
                 if t in self._league_of}
        if not known:
            raise ValueError("neither team is known; pass league=")
        if len(known) > 1:
            raise ValueError("the two teams play in different leagues")
        return known.pop()

    def _team_strengths(self, league, team, strict):
        key = (league, team)
        if key in self._attack:
            return self._attack[key], self._defence[key]
        if strict:
            raise ValueError(f"unknown team for {league}: {team}")
        return 1.0, 1.0     # unseen team (e.g. newly promoted): league average

    def expected_goals(self, home_team, away_team, league=None, strict=True):
        """Expected goals (home, away) for a fixture.

        With strict=False an unseen team is treated as an average team
        instead of raising ValueError.
        """
        self._check_fitted()
        if home_team == away_team:
            raise ValueError("a team cannot play itself")
        league = self._resolve_league(home_team, away_team, league)
        att_h, def_h = self._team_strengths(league, home_team, strict)
        att_a, def_a = self._team_strengths(league, away_team, strict)
        lam_home = self._mu_home[league] * att_h * def_a
        lam_away = self._mu_away[league] * att_a * def_h
        return float(lam_home), float(lam_away)

    # ---- scoreline matrix ------------------------------------------------------
    def scoreline_matrix(self, home_team, away_team, league=None, strict=True):
        """(max_goals+1) x (max_goals+1) array; rows home goals, columns away goals."""
        lam_home, lam_away = self.expected_goals(home_team, away_team, league, strict)
        home_probs = poisson_pmf_vector(lam_home, self._max_goals)
        away_probs = poisson_pmf_vector(lam_away, self._max_goals)
        # Broadcasting: (n, 1) * (1, n) -> (n, n) outer product, no loops.
        return home_probs[:, np.newaxis] * away_probs[np.newaxis, :]

    @staticmethod
    def outcome_probabilities(matrix):
        """[P(home win), P(draw), P(away win)] from a scoreline matrix.

        Home wins are the cells below the diagonal (home goals > away goals),
        draws are the diagonal and away wins are the cells above it.
        """
        m = _as_matrix(matrix)
        home_win = np.tril(m, k=-1).sum()
        draw = np.trace(m)
        away_win = np.triu(m, k=1).sum()
        return np.array([home_win, draw, away_win]) / m.sum()

    @staticmethod
    def marginal_goals(matrix):
        """(home goal distribution, away goal distribution) via axis sums."""
        m = _as_matrix(matrix)
        return m.sum(axis=1) / m.sum(), m.sum(axis=0) / m.sum()

    @staticmethod
    def most_likely_score(matrix):
        """(home goals, away goals) of the single most probable scoreline."""
        m = _as_matrix(matrix)
        home_goals, away_goals = np.unravel_index(np.argmax(m), m.shape)
        return int(home_goals), int(away_goals)

    @staticmethod
    def prob_over(matrix, line=2.5):
        """Probability of more than `line` total goals (boolean-mask sum)."""
        m = _as_matrix(matrix)
        home_goals, away_goals = np.indices(m.shape)
        return float(m[(home_goals + away_goals) > line].sum() / m.sum())

    # ---- batch prediction ----------------------------------------------------
    def predict_proba(self, matches):
        """(n, 3) array of [home, draw, away] probabilities for many matches.

        `matches` needs League, HomeTeam and AwayTeam. Teams not seen in
        fit() are treated as average. Only the fitted strengths are used, never
        the goals of the matches being predicted.
        """
        self._check_fitted()
        rows = [
            self.outcome_probabilities(
                self.scoreline_matrix(home, away, league=league, strict=False))
            for league, home, away in zip(
                matches["League"], matches["HomeTeam"], matches["AwayTeam"])
        ]
        return np.array(rows, dtype=float).reshape(-1, 3)