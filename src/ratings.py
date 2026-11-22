"""Elo rating system for KickCast.

Rebuilds every team's strength from real match history. All teams start at
the same rating; after each match the winner takes points from the loser.
The size of the change depends on how surprising the result was, and a
home-advantage bonus is added to the home team's rating when predicting.

Contents:
    EloRating    - ratings for the teams of ONE league: expected_score(),
                   update(), history(), table().
    compute_elo  - replays a whole dataset in date order, one EloRating per
                   league, and records each match's PRE-match ratings
                   (the leakage-free input for the feature builder).
    elo_table    - the current ranking of a league's teams for one season.

Why one EloRating per league: EPL and La Liga teams never play each other
in this dataset, so their ratings are not comparable across leagues.
"""

import pandas as pd

INITIAL_RATING = 1500.0


class EloRating:
    """Elo ratings for the teams of one league.

    Args:
        k_factor: how fast ratings move (bigger = more reactive).
        home_advantage: Elo points added to the home team when computing the
            expected score. 50 points matches the roughly 57% average home
            score (wins plus half of draws) seen in EPL and La Liga data.
        initial_rating: rating given to a team the first time it is seen.
    """

    def __init__(self, k_factor=20.0, home_advantage=50.0,
                 initial_rating=INITIAL_RATING):
        if k_factor <= 0:
            raise ValueError("k_factor must be positive")
        self._k_factor = float(k_factor)
        self._home_advantage = float(home_advantage)
        self._initial_rating = float(initial_rating)
        self._ratings = {}   # team -> current rating
        self._history = []   # (date, team, rating after the match)

    # ---- read-only views -------------------------------------------------
    @property
    def k_factor(self):
        return self._k_factor

    @property
    def home_advantage(self):
        return self._home_advantage

    @property
    def ratings(self):
        """A copy of the current ratings (team -> float)."""
        return dict(self._ratings)

    def rating(self, team):
        """Current rating of a team (the initial rating if never seen)."""
        return self._ratings.get(team, self._initial_rating)

    # ---- core maths ------------------------------------------------------
    def expected_score(self, home_team, away_team):
        """Expected score of the home team, between 0 and 1.

        A win counts 1, a draw 0.5 and a loss 0, so this is "win probability
        plus half the draw probability". The home team gets the
        home-advantage bonus.
        """
        diff = (self.rating(away_team)
                - (self.rating(home_team) + self._home_advantage))
        return 1.0 / (1.0 + 10.0 ** (diff / 400.0))

    def update(self, home_team, away_team, home_goals, away_goals, date=None):
        """Update both teams after a match and return the home team's change.

        Raises:
            ValueError: if a team plays itself or goals are negative.
        """
        if home_team == away_team:
            raise ValueError("a team cannot play itself")
        if home_goals < 0 or away_goals < 0:
            raise ValueError("goals cannot be negative")

        expected = self.expected_score(home_team, away_team)
        if home_goals > away_goals:
            actual = 1.0
        elif home_goals == away_goals:
            actual = 0.5
        else:
            actual = 0.0

        delta = self._k_factor * (actual - expected)
        self._ratings[home_team] = self.rating(home_team) + delta
        self._ratings[away_team] = self.rating(away_team) - delta
        self._history.append((date, home_team, self._ratings[home_team]))
        self._history.append((date, away_team, self._ratings[away_team]))
        return delta

    # ---- reporting -------------------------------------------------------
    def history(self, team=None):
        """Rating after every match as a DataFrame (Date, Team, Rating).

        `team` may be a single name or a list of names to filter by.
        """
        df = pd.DataFrame(self._history, columns=["Date", "Team", "Rating"])
        if team is not None:
            wanted = [team] if isinstance(team, str) else list(team)
            df = df[df["Team"].isin(wanted)]
        return df.reset_index(drop=True)

    def table(self, teams=None, top=None):
        """Teams ranked by rating, best first, as (team, rating) tuples.

        Args:
            teams: optional collection of team names to restrict the table to.
            top: optional number of rows to keep.
        """
        items = list(self._ratings.items())
        if teams is not None:
            wanted = set(teams)
            items = [item for item in items if item[0] in wanted]
        ranked = sorted(items, key=lambda item: item[1], reverse=True)
        return ranked if top is None else ranked[:top]


def compute_elo(matches, k_factor=20.0, home_advantage=50.0):
    """Replay all matches in date order and rate every team.

    Args:
        matches: DataFrame with Date, League, HomeTeam, AwayTeam, FTHG, FTAG.

    Returns:
        (elos, pre_match) where elos maps league -> EloRating after the last
        match, and pre_match is a DataFrame aligned with `matches` holding
        HomeElo, AwayElo and EloExpected as they were BEFORE each match.
        Because every value is recorded before the update, no match ever
        influences its own pre-match features (no leakage).
    """
    ordered = matches.sort_values("Date", kind="stable")
    elos = {}
    rows = []
    for row in ordered.itertuples():
        elo = elos.setdefault(row.League, EloRating(k_factor, home_advantage))
        home_before = elo.rating(row.HomeTeam)
        away_before = elo.rating(row.AwayTeam)
        expected = elo.expected_score(row.HomeTeam, row.AwayTeam)
        elo.update(row.HomeTeam, row.AwayTeam, row.FTHG, row.FTAG, row.Date)
        rows.append((row.Index, home_before, away_before, expected))
    pre_match = pd.DataFrame(
        rows, columns=["Index", "HomeElo", "AwayElo", "EloExpected"]
    ).set_index("Index")
    pre_match.index.name = matches.index.name
    return elos, pre_match.reindex(matches.index)


def elo_table(elos, matches, league, season=None, top=None):
    """Current Elo ranking of the teams that played in one season of a league.

    Uses the latest season in the data when `season` is None, so relegated
    teams with stale ratings are left out.
    """
    if league not in elos:
        raise ValueError(f"unknown league: {league}")
    in_league = matches[matches["League"] == league]
    if season is None:
        season = in_league["Season"].max()
    in_season = in_league[in_league["Season"] == season]
    teams = set(in_season["HomeTeam"]) | set(in_season["AwayTeam"])
    return elos[league].table(teams=teams, top=top)