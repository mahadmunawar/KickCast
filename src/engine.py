"""Application logic for KickCast: everything the menu needs, without printing.

main.py only talks to the user (menus, validated input, printing); this module
loads the data, builds features and models and answers the menu's questions.
Keeping the two apart makes the logic testable without a console.

Contents:
    load_data()        - web, then data/raw cache, then data/sample snapshot.
    LoadedData         - what load_data() returns (matches, FPL feed, notes).
    KickCastEngine     - elo_ranking(), predict_fixture(), availability_report(),
                         evaluation() and the figure savers.
    FixturePrediction  - result of predict_fixture().
    season_label(), describe_age(), FPL_TEAM_ALIASES - small helpers.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.dataprep import build_dataset
from src.features import FeatureBuilder
from src.models import (MatchPredictor, accuracy_margin, calibration_inputs,
                        score, time_split)
from src.plots import (plot_calibration, plot_elo_trajectories,
                       plot_scoreline_heatmap)
from src.poisson import ScorelineModel
from src.ratings import elo_table
from src.sources import DataFetchError, FootballCsvSource, FplApiSource
from src.textproc import (AvailabilityClassifier, prepare_news,
                          summarize_availability)

LEAGUES = ("EPL", "LaLiga")
LEAGUE_NAMES = {"EPL": "Premier League", "LaLiga": "La Liga"}
POISSON_NAME = "Poisson (goals only)"

# FPL spells some clubs differently from football-data.co.uk.
FPL_TEAM_ALIASES = {"Spurs": "Tottenham", "Man Utd": "Man United"}


# ---- small helpers ---------------------------------------------------------------
def season_label(season):
    """'2526' -> '2025/26'."""
    season = str(season)
    return f"20{season[:2]}/{season[2:]}" if len(season) == 4 else season


def describe_age(hours):
    """Human-readable age of a cached file."""
    if hours is None:
        return "of unknown age"
    if hours < 1:
        return "less than an hour old"
    if hours < 48:
        return f"{hours:.0f} hours old"
    return f"{hours / 24:.0f} days old"


# ---- loading ------------------------------------------------------------------------
@dataclass
class LoadedData:
    """Result of load_data()."""
    matches: pd.DataFrame
    fpl: object              # FPL bootstrap dict, or None if unavailable
    origin: str              # "web", "cache", "mixed" or "sample"
    notes: list              # plain-English lines for the user


class _SnapshotSource:
    """Adapter so build_dataset() can load the offline sample snapshot."""

    def __init__(self, league):
        self.league = league

    def load(self):
        return FootballCsvSource.load_sample(self.league)


def _csv_origin(sources):
    """Summarise where each season file came from -> (origin, notes)."""
    counts = {"web": 0, "cache": 0}
    oldest = None
    for source in sources:
        for season, origin in source.status.items():
            counts[origin] = counts.get(origin, 0) + 1
            if origin == "cache":
                age = source.season_cache_age_hours(season)
                if age is not None:
                    oldest = age if oldest is None else max(oldest, age)
    notes = []
    if counts["web"]:
        notes.append(f"Match data: {counts['web']} season file(s) downloaded just now.")
    if counts["cache"]:
        notes.append(f"Match data: {counts['cache']} season file(s) loaded from the "
                     f"offline cache (oldest copy {describe_age(oldest)}).")
    origin = "web" if not counts["cache"] else "cache" if not counts["web"] else "mixed"
    return origin, notes


def _load_fpl():
    """FPL feed -> (data or None, note)."""
    source = FplApiSource()
    try:
        data = source.load()
    except DataFetchError as error:
        try:
            return (FplApiSource.load_sample(),
                    f"Injury news: live feed and cache unavailable ({error}); "
                    "using the small sample snapshot.")
        except DataFetchError:
            return None, (f"Injury news: unavailable ({error}). "
                          "The availability report is disabled.")
    if source.origin == "cache":
        return data, ("Injury news: loaded from the offline cache "
                      f"({describe_age(source.cache_age_hours())}).")
    return data, "Injury news: downloaded just now."


def load_data():
    """Load matches and FPL news, falling back web -> data/raw -> data/sample.

    The football-data CSVs and the FPL feed fall back independently. If no
    full match data can be found at all, the small 2025/26 snapshot kept in
    data/sample/ is used, so a fresh clone still runs offline.

    Raises:
        DataFetchError: only if not even the sample snapshot is available.
    """
    sources = [FootballCsvSource(league) for league in LEAGUES]
    try:
        matches, _report = build_dataset(sources)
        origin, notes = _csv_origin(sources)
    except DataFetchError as error:
        matches, _report = build_dataset([_SnapshotSource(l) for l in LEAGUES])
        origin = "sample"
        notes = [f"Match data: full data unavailable ({error}). Using the small "
                 "sample snapshot (2025/26 only): predictions work, but the "
                 "evaluation needs earlier seasons."]
    fpl, fpl_note = _load_fpl()
    return LoadedData(matches, fpl, origin, notes + [fpl_note])


# ---- prediction result ---------------------------------------------------------------
@dataclass
class FixturePrediction:
    """Everything predict_fixture() knows about one fixture."""
    league: str
    home: str
    away: str
    model_name: str
    model_proba: np.ndarray      # [home, draw, away]
    poisson_proba: np.ndarray    # [home, draw, away]
    expected_goals: tuple        # (home, away) from the Poisson model
    likely_score: tuple          # most probable (home goals, away goals)
    over_2_5: float
    matrix: np.ndarray           # scoreline probabilities, rows = home goals
    features: pd.Series          # the 16 model inputs used


# ---- engine ----------------------------------------------------------------------------
class KickCastEngine:
    """Models and reports built from one loaded dataset.

    Args:
        matches: the cleaned dataset from dataprep.build_dataset().
        fpl: the FPL bootstrap dict, or None (the availability report is then
            unavailable).
        notes: plain-English lines about where the data came from.
    """

    def __init__(self, matches, fpl=None, notes=None):
        if matches is None or len(matches) == 0:
            raise ValueError("no matches to work with")
        self._matches = matches.reset_index(drop=True)
        self._fpl = fpl
        self._notes = list(notes or [])
        self._builder = FeatureBuilder()
        self._features = self._builder.build(self._matches)
        self._predictor = None
        self._poisson = None
        self._cache = {}

    # ---- information ---------------------------------------------------------------
    @property
    def notes(self):
        return list(self._notes)

    @property
    def matches(self):
        return self._matches

    @property
    def leagues(self):
        return sorted(self._matches["League"].unique())

    @property
    def seasons(self):
        return sorted(self._matches["Season"].unique())

    @property
    def window(self):
        return self._builder.window

    @property
    def has_availability(self):
        return self._fpl is not None

    def _check_league(self, league):
        if league not in self.leagues:
            raise ValueError(f"unknown league: {league}")

    def teams(self, league):
        """Teams that played in the league's latest season, sorted."""
        self._check_league(league)
        in_league = self._matches[self._matches["League"] == league]
        latest = in_league[in_league["Season"] == in_league["Season"].max()]
        return sorted(set(latest["HomeTeam"]) | set(latest["AwayTeam"]))

    # ---- Elo ---------------------------------------------------------------------------
    def elo_ranking(self, league, top=10, season=None):
        """Elo ranking after the last match in the data (DataFrame Rank, Team, Elo)."""
        self._check_league(league)
        if top < 1:
            raise ValueError("top must be at least 1")
        if season is not None and season not in set(self._matches["Season"]):
            raise ValueError(f"unknown season: {season}")
        rows = elo_table(self._builder.elos, self._matches, league,
                         season=season, top=top)
        table = pd.DataFrame(rows, columns=["Team", "Elo"])
        table["Elo"] = table["Elo"].round(0).astype(int)
        table.insert(0, "Rank", range(1, len(table) + 1))
        return table

    def save_elo_figure(self, league, top=5, path=None):
        """Figure 1 style chart of the top teams' Elo history; returns the Path."""
        self._check_league(league)
        rows = elo_table(self._builder.elos, self._matches, league, top=top)
        teams = [team for team, _ in rows]
        title = f"Elo rating trajectories: top {len(teams)} {LEAGUE_NAMES.get(league, league)} teams"
        return plot_elo_trajectories(self._builder.elos[league], teams,
                                     title=title, save_path=path)

    # ---- fixture prediction ----------------------------------------------------------------
    def _ensure_models(self):
        """Train the deployed models on ALL matches (evaluation is reported separately)."""
        if self._predictor is None:
            complete = self._features.dropna()
            self._predictor = MatchPredictor("logistic").fit(
                complete, self._matches.loc[complete.index, "Target"])
            self._poisson = ScorelineModel().fit(self._matches)

    def predict_fixture(self, league, home, away):
        """Predict a future fixture between two teams of one league.

        Raises:
            ValueError: for an unknown league or team, the same team twice, or
                a team with too little recent history.
        """
        self._check_league(league)
        known = self.teams(league)
        for team in (home, away):
            if team not in known:
                raise ValueError(f"{team} is not a {league} team in the latest season")
        if home == away:
            raise ValueError("a team cannot play itself")

        row = self._builder.fixture_features(self._matches, league, home, away)
        for side, team in (("Home", home), ("Away", away)):
            if row.filter(like=f"{side}_").isna().any(axis=None):
                raise ValueError(f"not enough recent matches for {team} "
                                 f"(need at least {self._builder.min_history})")

        self._ensure_models()
        matrix = self._poisson.scoreline_matrix(home, away, league=league)
        return FixturePrediction(
            league=league, home=home, away=away,
            model_name=self._predictor.name,
            model_proba=self._predictor.predict_proba(row)[0],
            poisson_proba=ScorelineModel.outcome_probabilities(matrix),
            expected_goals=self._poisson.expected_goals(home, away, league=league),
            likely_score=ScorelineModel.most_likely_score(matrix),
            over_2_5=ScorelineModel.prob_over(matrix, 2.5),
            matrix=matrix, features=row.iloc[0])

    def save_heatmap(self, prediction, path=None):
        """Figure 2 for a prediction; returns the Path of the saved image."""
        return plot_scoreline_heatmap(prediction.matrix, prediction.home,
                                      prediction.away, save_path=path)

    # ---- injury news -------------------------------------------------------------------------
    def _prepared_news(self):
        news = FplApiSource().get_news(data=self._fpl)
        prepared = prepare_news(news)
        prepared["team"] = prepared["team"].replace(FPL_TEAM_ALIASES)
        return prepared

    def availability_report(self, team=None):
        """Per-player availability (out-of-fold predictions), worst news first.

        Raises:
            ValueError: if no FPL data is loaded, there is too little news to
                classify, or `team` has no news.
        """
        if self._fpl is None:
            raise ValueError("no injury news is loaded, so there is no availability report")
        if "report" not in self._cache:
            self._cache["report"] = AvailabilityClassifier().report(self._prepared_news())
        report = self._cache["report"]
        if team is None:
            return report
        part = report[report["Team"] == team]
        if part.empty:
            raise ValueError(f"no injury news for team: {team}")
        return part.reset_index(drop=True)

    def availability_teams(self):
        """Teams that have injury news, sorted."""
        return sorted(set(self.availability_report()["Team"]))

    def availability_summary(self):
        """Per-team counts of predicted out / doubtful / fit players."""
        return summarize_availability(self.availability_report())

    def availability_honesty(self):
        """Cross-validated accuracy of the text model next to the trivial baseline.

        Returns a dict (n, accuracy, majority_baseline, label_counts), or None
        if there is no news or too few labels to evaluate.
        """
        if self._fpl is None:
            return None
        if "honesty" not in self._cache:
            prepared = self._prepared_news()
            try:
                result = AvailabilityClassifier().evaluate(prepared["news"], prepared["label"])
            except ValueError:
                self._cache["honesty"] = None
            else:
                self._cache["honesty"] = {
                    "n": result["n"], "accuracy": result["accuracy"],
                    "majority_baseline": result["majority_baseline"],
                    "label_counts": prepared["label"].value_counts().to_dict()}
        return self._cache["honesty"]

    def fixture_availability(self, league, home, away):
        """Injury news for the two teams of an EPL fixture (None if not applicable)."""
        if league != "EPL" or self._fpl is None:
            return None
        try:
            report = self.availability_report()
        except ValueError:
            return None
        return {team: report[report["Team"] == team].reset_index(drop=True)
                for team in (home, away)}

    def unmatched_fpl_teams(self):
        """FPL team names with no EPL team of that name in the latest season.

        Expected: clubs that joined the league after the last football-data
        season. Anything else points to a missing entry in FPL_TEAM_ALIASES.
        """
        if self._fpl is None:
            return []
        news = FplApiSource().get_news(data=self._fpl, only_with_news=False)
        names = set(news["team"].replace(FPL_TEAM_ALIASES))
        return sorted(names - set(self.teams("EPL")))

    # ---- evaluation ----------------------------------------------------------------------------
    def evaluation(self):
        """Test-season results of every method (earlier seasons train, latest tests).

        Returns:
            (table, test_season). The table is MatchPredictor.compare() plus
            Poisson rows scored on exactly the same test matches.

        Raises:
            ValueError: if there is no earlier season to train on.
        """
        if "evaluation" not in self._cache:
            split = time_split(self._matches, self._features)
            table = MatchPredictor.compare(self._matches, self._features)

            poisson = ScorelineModel().fit(
                self._matches[self._matches["Season"] < split.test_season])
            teams = self._matches.loc[split.test.index, ["League", "HomeTeam", "AwayTeam"]]
            proba = poisson.predict_proba(teams)
            y = split.test["Target"].to_numpy()
            league_column = split.test["League"].to_numpy()
            rows = []
            for test_set in sorted(set(league_column)) + ["All"]:
                mask = (np.ones(len(y), dtype=bool) if test_set == "All"
                        else league_column == test_set)
                result = score(y[mask], proba[mask])
                rows.append({"Method": POISSON_NAME, "Training": "Combined",
                             "TestSet": test_set, "N": int(mask.sum()),
                             "Accuracy": result["accuracy"],
                             "AccMargin": accuracy_margin(result["accuracy"], int(mask.sum())),
                             "LogLoss": result["log_loss"]})
            table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
            self._cache["evaluation"] = (table, split.test_season)
        return self._cache["evaluation"]

    def save_calibration_figure(self, path=None):
        """Figure 3 (model vs bookmaker calibration); returns the Path."""
        y_true, probabilities = calibration_inputs(self._matches, self._features)
        return plot_calibration(y_true, probabilities, save_path=path)