"""Leakage-free feature engineering for KickCast.

Builds the model inputs (Elo before the match, rolling form, goals and
shots) using ONLY matches played before each fixture. Every rolling window
is shifted by one match, so the current match never leaks into its own
features.

Contents:
    FeatureBuilder - build(matches) returns one row of features per match.

Features per match (16 in total):
    Elo (4):      HomeElo, AwayElo, EloDiff, EloExpected
    Rolling (12): Home_<stat> and Away_<stat> for each of
                  Points, GoalsFor, GoalsAgainst, Shots, ShotsAgainst,
                  ShotsOnTarget - the average over each team's previous
                  `window` matches (home and away games together).
"""

import pandas as pd

from src.ratings import compute_elo

ROLLING_STATS = ["Points", "GoalsFor", "GoalsAgainst",
                 "Shots", "ShotsAgainst", "ShotsOnTarget"]

REQUIRED_COLUMNS = ["Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                    "Target", "HS", "AS", "HST", "AST"]

POINTS_HOME = {0: 3, 1: 1, 2: 0}   # Target: 0 = home win, 1 = draw, 2 = away win
POINTS_AWAY = {0: 0, 1: 1, 2: 3}


class FeatureBuilder:
    """Turns a cleaned match DataFrame into model features.

    Args:
        window: how many previous matches the rolling averages cover.
        min_history: minimum number of previous matches a team needs before
            its rolling features are defined (otherwise they are NaN).
        k_factor, home_advantage: passed on to the Elo rating system.
    """

    def __init__(self, window=5, min_history=3, k_factor=20.0,
                 home_advantage=50.0):
        if window < 1:
            raise ValueError("window must be at least 1")
        if not 1 <= min_history <= window:
            raise ValueError("min_history must be between 1 and window")
        self._window = int(window)
        self._min_history = int(min_history)
        self._k_factor = k_factor
        self._home_advantage = home_advantage
        self._elos = None

    @property
    def window(self):
        return self._window

    @property
    def elos(self):
        """Per-league EloRating objects after the last build() (or None)."""
        return self._elos

    @property
    def feature_names(self):
        """The feature columns, in the order build() returns them."""
        return (["HomeElo", "AwayElo", "EloDiff", "EloExpected"]
                + [f"Home_{stat}" for stat in ROLLING_STATS]
                + [f"Away_{stat}" for stat in ROLLING_STATS])

    # ---- internals -------------------------------------------------------
    @staticmethod
    def _team_match_table(matches):
        """One row per team per match, seen from that team's point of view."""
        home = pd.DataFrame({
            "MatchId": matches.index.to_numpy(),
            "Side": "Home",
            "Date": matches["Date"].to_numpy(),
            "League": matches["League"].to_numpy(),
            "Team": matches["HomeTeam"].to_numpy(),
            "Points": matches["Target"].map(POINTS_HOME).to_numpy(),
            "GoalsFor": matches["FTHG"].to_numpy(),
            "GoalsAgainst": matches["FTAG"].to_numpy(),
            "Shots": matches["HS"].to_numpy(),
            "ShotsAgainst": matches["AS"].to_numpy(),
            "ShotsOnTarget": matches["HST"].to_numpy(),
        })
        away = pd.DataFrame({
            "MatchId": matches.index.to_numpy(),
            "Side": "Away",
            "Date": matches["Date"].to_numpy(),
            "League": matches["League"].to_numpy(),
            "Team": matches["AwayTeam"].to_numpy(),
            "Points": matches["Target"].map(POINTS_AWAY).to_numpy(),
            "GoalsFor": matches["FTAG"].to_numpy(),
            "GoalsAgainst": matches["FTHG"].to_numpy(),
            "Shots": matches["AS"].to_numpy(),
            "ShotsAgainst": matches["HS"].to_numpy(),
            "ShotsOnTarget": matches["AST"].to_numpy(),
        })
        return pd.concat([home, away], ignore_index=True)

    def _rolling_features(self, matches):
        """Rolling averages of each team's PREVIOUS matches (shift, then roll)."""
        long = self._team_match_table(matches)
        long = long.sort_values(["League", "Team", "Date", "MatchId"],
                                kind="stable")
        window, min_history = self._window, self._min_history
        # shift(1) first: a row's window then ends at the previous match,
        # so the current match can never be part of its own features.
        rolled = long.groupby(["League", "Team"], sort=False)[ROLLING_STATS] \
            .transform(lambda g: g.shift(1).rolling(
                window, min_periods=min_history).mean())
        rolled["MatchId"] = long["MatchId"]
        rolled["Side"] = long["Side"]

        parts = []
        for side in ("Home", "Away"):
            part = (rolled[rolled["Side"] == side]
                    .set_index("MatchId")[ROLLING_STATS]
                    .add_prefix(f"{side}_"))
            parts.append(part)
        return parts[0].join(parts[1])

    # ---- public API ------------------------------------------------------
    def build(self, matches, dropna=False):
        """Return a feature DataFrame aligned with the index of `matches`.

        Args:
            matches: cleaned matches (see dataprep.build_dataset).
            dropna: if True, drop rows where a team has too little history.

        Raises:
            ValueError: if a required column is missing.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in matches.columns]
        if missing:
            raise ValueError(f"matches is missing columns: {missing}")

        self._elos, pre_match = compute_elo(
            matches, self._k_factor, self._home_advantage)

        features = pre_match[["HomeElo", "AwayElo", "EloExpected"]].copy()
        features["EloDiff"] = features["HomeElo"] - features["AwayElo"]
        rolling = self._rolling_features(matches)
        rolling.index.name = features.index.name
        features = features.join(rolling)[self.feature_names]

        if dropna:
            features = features.dropna()
        return features