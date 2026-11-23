"""Text processing for KickCast (Track A: Text / NLP).

Cleans the injury news from the FPL feed with our own code (lowercase,
strip punctuation, tokenize, remove stop words), converts it to TF-IDF
features and classifies each player's availability as fit, doubtful or
out.

Contents:
    tokenize(), clean_text()  - the own text cleaner.
    label_availability()      - official label from FPL's numeric fields.
    prepare_news()            - filters FPL news rows and adds that label.
    AvailabilityClassifier    - TF-IDF + logistic regression: fit(), predict(),
                                predict_proba(), evaluate(), report().
    summarize_availability()  - per-team counts of out / doubtful / fit.

Honest-evaluation note (see the proposal's Risk 2): the labels are derived
from FPL's numeric chance-of-playing field, and FPL's own news text is often
written from that same number ("75% chance of playing"). A text model can
therefore partly "read the answer" from the text. To make the size of that
effect visible, the cleaner can drop every token containing a digit
(remove_numbers=True), and evaluation always uses out-of-fold predictions
with the TF-IDF vocabulary learned only from each fold's training rows.
"""

import string
import warnings
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict as sk_cross_val_predict
from sklearn.pipeline import Pipeline

LABELS = ["fit", "doubtful", "out"]
SEVERITY = {"out": 0, "doubtful": 1, "fit": 2}   # sort order: worst news first

# Deliberately small: words such as "out", "not", "no", "until", "back" carry
# meaning for availability, so they are NOT stop words.
STOP_WORDS = frozenset(
    "a an and are as at be been by for from has have he in is it its of on or "
    "that the their this to was were will with".split())

# Every punctuation mark becomes a space, except apostrophes, which are
# deleted so that "doesn't" becomes "doesnt" instead of two fragments.
_PUNCTUATION = {ord(ch): " " for ch in string.punctuation + "\u2013\u2014\u2212\u201c\u201d\u2026"}
for _apostrophe in "'\u2019\u2018":
    _PUNCTUATION[ord(_apostrophe)] = None


# ---- own text cleaner -------------------------------------------------------
def tokenize(text, remove_numbers=False, stop_words=STOP_WORDS):
    """Lowercase, strip punctuation, split into words and drop stop words.

    Args:
        text: the news string (anything that is not a string gives []).
        remove_numbers: also drop every word that contains a digit
            ("75", "20", "u21"), so the classifier cannot read FPL's
            percentage directly from the text.
        stop_words: words to remove.

    Returns:
        A list of tokens.
    """
    if not isinstance(text, str):
        return []
    words = text.lower().translate(_PUNCTUATION).split()
    words = list(filter(lambda word: word not in stop_words, words))
    if remove_numbers:
        words = [w for w in words if not any(ch.isdigit() for ch in w)]
    return words


def clean_text(text, remove_numbers=False, stop_words=STOP_WORDS):
    """tokenize() joined back into one cleaned string."""
    return " ".join(tokenize(text, remove_numbers, stop_words))


# ---- labels ------------------------------------------------------------------
def label_availability(chance, status):
    """Official availability label from FPL's numeric fields.

    chance is chance_of_playing_next_round (0, 25, 50, 75, 100 or missing) and
    status is FPL's letter code (a, d, i, s, u, n). The percentage decides when
    it is present; otherwise the status letter is used.
    """
    if not pd.isna(chance):
        if chance <= 0:
            return "out"
        if chance >= 100:
            return "fit"
        return "doubtful"
    return {"a": "fit", "d": "doubtful"}.get(status, "out")


def prepare_news(news, exclude_statuses=("u",)):
    """Keep usable FPL news rows and add the official `label` column.

    Players with status "u" (unavailable because they LEFT the club, e.g. "Has
    joined Al Hilal permanently") are excluded by default: FPL gives them a 0%
    chance of playing, so they would fill the "out" class with transfer news
    instead of real injuries and suspensions.

    Args:
        news: DataFrame from FplApiSource.get_news().
        exclude_statuses: status letters to drop.

    Raises:
        ValueError: if a required column is missing.
    """
    required = ["player", "team", "status", "news", "chance_of_playing"]
    missing = [c for c in required if c not in news.columns]
    if missing:
        raise ValueError(f"news is missing columns: {missing}")
    keep = (~news["status"].isin(exclude_statuses)) & (news["news"].str.strip() != "")
    prepared = news[keep].copy()
    prepared["label"] = [label_availability(chance, status) for chance, status
                         in zip(prepared["chance_of_playing"], prepared["status"])]
    return prepared.reset_index(drop=True)


# ---- classifier ----------------------------------------------------------------
def _make_pipeline(remove_numbers, random_state):
    """TF-IDF (own tokenizer, word and two-word features) + logistic regression."""
    vectorizer = TfidfVectorizer(
        tokenizer=partial(tokenize, remove_numbers=remove_numbers),
        token_pattern=None, lowercase=False, ngram_range=(1, 2),
        sublinear_tf=True)
    classifier = LogisticRegression(C=10.0, max_iter=1000,
                                    class_weight="balanced",
                                    random_state=random_state)
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def _check_inputs(texts, labels):
    texts = np.asarray(list(texts), dtype=object)
    labels = np.asarray(list(labels), dtype=object)
    if len(texts) == 0:
        raise ValueError("no examples given")
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError(f"unknown labels {unknown}; expected {LABELS}")
    return texts, labels


class AvailabilityClassifier:
    """Classifies injury news as fit, doubtful or out.

    Args:
        remove_numbers: drop digit-containing tokens before TF-IDF (default
            True). This blocks the shortcut of reading FPL's own percentage
            from the text; use False to measure how much that shortcut helps.
        random_state: seed for the classifier and the cross-validation folds.
    """

    def __init__(self, remove_numbers=True, random_state=42):
        self._remove_numbers = remove_numbers
        self._random_state = random_state
        self._pipeline = None

    @property
    def remove_numbers(self):
        return self._remove_numbers

    @property
    def classes(self):
        """The labels seen in fit(), or an empty list before fitting."""
        return [] if self._pipeline is None else list(self._pipeline.classes_)

    def _check_fitted(self):
        if self._pipeline is None:
            raise RuntimeError("call fit() first")

    def fit(self, texts, labels):
        """Train on news strings and their labels (at least two classes)."""
        texts, labels = _check_inputs(texts, labels)
        if len(set(labels)) < 2:
            raise ValueError("need examples of at least two different labels")
        self._pipeline = _make_pipeline(self._remove_numbers, self._random_state)
        self._pipeline.fit(texts, labels)
        return self

    def predict(self, texts):
        """Predicted label for every news string."""
        self._check_fitted()
        return self._pipeline.predict(np.asarray(list(texts), dtype=object))

    def predict_proba(self, texts):
        """DataFrame of class probabilities, one column per class."""
        self._check_fitted()
        proba = self._pipeline.predict_proba(np.asarray(list(texts), dtype=object))
        return pd.DataFrame(proba, columns=self.classes)

    # ---- evaluation ------------------------------------------------------------
    def out_of_fold_predictions(self, texts, labels, n_splits=5):
        """Predict every row with a model that never saw that row.

        Stratified cross-validation. The whole pipeline, including the TF-IDF
        vocabulary, is re-fitted inside each fold, so nothing leaks from the
        held-out rows. The number of folds is reduced if a class is small.

        Raises:
            ValueError: if fewer than two classes have at least 2 examples.
        """
        texts, labels = _check_inputs(texts, labels)
        usable = [c for c in Counter(labels).values() if c >= 2]
        if len(usable) < 2:
            raise ValueError("need at least two labels with 2 or more examples")
        folds = StratifiedKFold(n_splits=min(n_splits, min(usable)), shuffle=True,
                                random_state=self._random_state)
        pipeline = _make_pipeline(self._remove_numbers, self._random_state)
        with warnings.catch_warnings():
            # A label with a single example cannot be stratified; it is reported
            # in evaluate() as a rare class, so the warning adds nothing.
            warnings.filterwarnings("ignore", message="The least populated class")
            return sk_cross_val_predict(pipeline, texts, labels, cv=folds)

    def evaluate(self, texts, labels, n_splits=5):
        """Cross-validated accuracy, macro F1, confusion matrix and baseline.

        Returns:
            dict with n, accuracy, macro_f1, majority_baseline (accuracy of
            always predicting the most common label), confusion (DataFrame),
            predictions (out-of-fold, in input order) and rare_classes (labels
            with fewer than 2 examples, which cannot be learned reliably).
        """
        texts, labels = _check_inputs(texts, labels)
        predictions = self.out_of_fold_predictions(texts, labels, n_splits)
        counts = Counter(labels)
        present = [label for label in LABELS if label in counts]
        matrix = confusion_matrix(labels, predictions, labels=present)
        return {
            "n": len(labels),
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=present,
                                       average="macro", zero_division=0)),
            "majority_baseline": max(counts.values()) / len(labels),
            "confusion": pd.DataFrame(
                matrix, index=[f"true {c}" for c in present],
                columns=[f"pred {c}" for c in present]),
            "predictions": predictions,
            "rare_classes": [c for c in present if counts[c] < 2],
        }

    def report(self, news, team=None, n_splits=5):
        """Per-player availability report from prepared news (see prepare_news).

        The Predicted column comes from out-of-fold predictions, so no player
        is classified by a model that trained on that same player. Rows are
        sorted by team, then worst news first.

        Raises:
            ValueError: if `team` is not in the data.
        """
        predicted = self.out_of_fold_predictions(news["news"], news["label"], n_splits)
        table = pd.DataFrame({
            "Team": news["team"].to_numpy(),
            "Player": news["player"].to_numpy(),
            "News": news["news"].to_numpy(),
            "Official": news["label"].to_numpy(),
            "Predicted": predicted,
        })
        table["Agree"] = table["Official"] == table["Predicted"]
        if team is not None:
            if team not in set(table["Team"]):
                raise ValueError(f"no news for team: {team}")
            table = table[table["Team"] == team]
        table = table.sort_values(
            ["Team", "Predicted", "Player"],
            key=lambda col: col.map(SEVERITY) if col.name == "Predicted" else col)
        return table.reset_index(drop=True)


def summarize_availability(report):
    """Per-team counts of predicted out / doubtful / fit players."""
    counts = pd.crosstab(report["Team"], report["Predicted"])
    return counts.reindex(columns=LABELS, fill_value=0)[["out", "doubtful", "fit"]]