"""Match outcome models and evaluation for KickCast.

Trains and compares two scikit-learn models (Logistic Regression and
Random Forest) against two baselines (always pick the home team, and the
bookmakers' odds). Uses a time-based split: train on earlier seasons,
test on the latest completed season. Reports accuracy and log loss for
league-specific, combined and cross-league ("transfer") training.

Outcome coding everywhere: 0 = home win, 1 = draw, 2 = away win, and every
probability array has its columns in that order (home, draw, away).

Contents:
    time_split()          - earlier seasons for training, latest season for testing.
    score()               - accuracy and log loss of a probability array.
    HomeBaseline          - the "always pick the home team" benchmark.
    MatchPredictor        - wraps one scikit-learn model: fit(), predict_proba(),
                            evaluate(), feature_importance(), compare().
    calibration_inputs()  - test probabilities of the model and the bookmaker,
                            ready for the calibration chart (Figure 3).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataprep import PROB_COLUMNS

CLASSES = [0, 1, 2]
MODEL_NAMES = {"logistic": "Logistic Regression", "forest": "Random Forest"}
HOME_BASELINE = "Home-team baseline"
BOOKMAKER = "Bookmaker odds"


# ---- data split -----------------------------------------------------------
@dataclass
class Split:
    """Train/test DataFrames (features + Target + League + bookmaker probs)."""
    train: pd.DataFrame
    test: pd.DataFrame
    feature_names: list
    test_season: str


def time_split(matches, features, test_season=None):
    """Train on all seasons before `test_season`, test on `test_season`.

    Rows with any missing feature or missing bookmaker probability are dropped
    from BOTH sets, so every method is evaluated on exactly the same matches.
    The test season defaults to the latest season in the data.

    Raises:
        ValueError: if either side ends up empty.
    """
    extra = ["League", "Season", "Date", "Target"] + PROB_COLUMNS
    data = features.join(matches[extra])
    data = data[data.notna().all(axis=1)]
    if test_season is None:
        test_season = data["Season"].max()
    train = data[data["Season"] < test_season]
    test = data[data["Season"] == test_season]
    if train.empty or test.empty:
        raise ValueError(f"no data to train on or test on for season {test_season}")
    return Split(train, test, list(features.columns), test_season)


# ---- metrics ----------------------------------------------------------------
def score(y_true, proba):
    """Accuracy (most likely outcome) and log loss of (n, 3) probabilities."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    return {
        "accuracy": float(accuracy_score(y_true, proba.argmax(axis=1))),
        "log_loss": float(log_loss(y_true, proba, labels=CLASSES)),
    }


def accuracy_margin(accuracy, n):
    """Approximate 95% margin of error of an accuracy measured on n matches."""
    return 1.96 * float(np.sqrt(accuracy * (1.0 - accuracy) / n))


# ---- baselines --------------------------------------------------------------
class HomeBaseline:
    """The "always pick the home team" benchmark.

    Home wins are the most common result, so predicting the most likely
    outcome from the training class frequencies is the same as always
    predicting a home win. Those frequencies are also used as probabilities
    so that log loss is finite and comparable with the other methods.
    """

    def __init__(self):
        self._prior = None

    def fit(self, y):
        counts = np.bincount(np.asarray(y, dtype=int), minlength=3)
        self._prior = counts / counts.sum()
        return self

    def predict_proba(self, n_rows):
        if self._prior is None:
            raise RuntimeError("call fit() first")
        return np.tile(self._prior, (n_rows, 1))


# ---- models -----------------------------------------------------------------
def make_model(kind, random_state=42):
    """Create an untrained scikit-learn model ("logistic" or "forest")."""
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if kind == "forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                      random_state=random_state)
    raise ValueError(f"kind must be one of {list(MODEL_NAMES)}, got {kind!r}")


class MatchPredictor:
    """Wraps one scikit-learn model that predicts home / draw / away."""

    def __init__(self, kind="logistic", random_state=42):
        self._kind = kind
        self._model = make_model(kind, random_state)
        self._random_state = random_state
        self._feature_names = None

    @property
    def kind(self):
        return self._kind

    @property
    def name(self):
        return MODEL_NAMES[self._kind]

    def _check_fitted(self):
        if self._feature_names is None:
            raise RuntimeError("call fit() first")

    def fit(self, X, y):
        """Train on a feature DataFrame X and outcome codes y."""
        self._feature_names = list(X.columns)
        self._model.fit(X, np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X):
        """(n, 3) array of probabilities ordered home, draw, away."""
        self._check_fitted()
        raw = self._model.predict_proba(X[self._feature_names])
        proba = np.zeros((len(X), 3))
        proba[:, np.asarray(self._model.classes_, dtype=int)] = raw
        return proba

    def predict(self, X):
        """Most likely outcome code for every row."""
        return self.predict_proba(X).argmax(axis=1)

    def evaluate(self, X, y):
        """Accuracy and log loss on (X, y)."""
        return score(y, self.predict_proba(X))

    def feature_importance(self):
        """Relative importance of each feature (sums to 1), largest first."""
        self._check_fitted()
        if self._kind == "forest":
            values = self._model.feature_importances_
        else:
            coefficients = self._model.named_steps["logisticregression"].coef_
            values = np.abs(coefficients).mean(axis=0)
            values = values / values.sum()
        return pd.Series(values, index=self._feature_names).sort_values(ascending=False)

    @staticmethod
    def compare(matches, features, test_season=None,
                kinds=("logistic", "forest"), random_state=42):
        """Evaluate every method and training set-up on the test season.

        Training set-ups for each model:
            League-specific - train on one league, test on the same league.
            Combined        - train on both leagues, test on each and on both.
            Transfer        - train on one league, test on the OTHER league.

        Returns:
            DataFrame with columns Method, Training, TestSet, N, Accuracy,
            AccMargin (95% margin) and LogLoss.
        """
        split = time_split(matches, features, test_season)
        names = split.feature_names
        train, test = split.train, split.test
        leagues = sorted(test["League"].unique())
        test_sets = {league: test[test["League"] == league] for league in leagues}
        test_sets["All"] = test
        train_sets = {league: train[train["League"] == league] for league in leagues}
        train_sets["All"] = train

        rows = []

        def add(method, training, test_set, y, proba):
            result = score(y, proba)
            rows.append({
                "Method": method, "Training": training, "TestSet": test_set,
                "N": len(y), "Accuracy": result["accuracy"],
                "AccMargin": accuracy_margin(result["accuracy"], len(y)),
                "LogLoss": result["log_loss"],
            })

        # Baselines (they depend only on the test set).
        for test_set, df in test_sets.items():
            y = df["Target"].to_numpy()
            baseline = HomeBaseline().fit(train_sets[test_set]["Target"])
            add(HOME_BASELINE, "-", test_set, y, baseline.predict_proba(len(df)))
            add(BOOKMAKER, "-", test_set, y, df[PROB_COLUMNS].to_numpy())

        # Models.
        for kind in kinds:
            def fitted(train_df):
                return MatchPredictor(kind, random_state).fit(
                    train_df[names], train_df["Target"])

            def run(model, training, test_set):
                df = test_sets[test_set]
                add(model.name, training, test_set, df["Target"].to_numpy(),
                    model.predict_proba(df[names]))

            for league in leagues:                        # league-specific
                run(fitted(train_sets[league]), "League-specific", league)
            combined = fitted(train_sets["All"])          # combined
            for test_set in test_sets:
                run(combined, "Combined", test_set)
            for source in leagues:                        # transfer
                model = fitted(train_sets[source])
                for target in leagues:
                    if target != source:
                        run(model, f"Transfer from {source}", target)

        return pd.DataFrame(rows)


def calibration_inputs(matches, features, kind="logistic", test_season=None,
                       random_state=42):
    """Test-season outcomes and probabilities for the calibration chart.

    The model is trained on both leagues' earlier seasons. Returns
    (y_true, {method name: (n, 3) probabilities}).
    """
    split = time_split(matches, features, test_season)
    names = split.feature_names
    model = MatchPredictor(kind, random_state).fit(
        split.train[names], split.train["Target"])
    probabilities = {
        model.name: model.predict_proba(split.test[names]),
        BOOKMAKER: split.test[PROB_COLUMNS].to_numpy(),
    }
    return split.test["Target"].to_numpy(), probabilities