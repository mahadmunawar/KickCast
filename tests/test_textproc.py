"""Tests for the text cleaner and availability classifier. No network access."""

import numpy as np
import pandas as pd
import pytest

from src.textproc import (LABELS, STOP_WORDS, AvailabilityClassifier,
                          clean_text, label_availability, prepare_news,
                          summarize_availability, tokenize)

PARTS = ["Hamstring", "Knee", "Ankle", "Calf", "Thigh", "Groin", "Illness", "Knock"]


# ---- synthetic FPL-style news -------------------------------------------------
def make_news(seed=0, n=160, teams=("Arsenal", "Chelsea", "Spurs", "Wolves")):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        part = rng.choice(PARTS)
        kind = rng.choice(["out", "doubtful", "fit"], p=[0.55, 0.35, 0.10])
        if kind == "out":
            text = rng.choice([
                f"{part} injury - Expected back {rng.integers(1, 28)} Dec",
                f"{part} injury - Unknown return date",
                f"Suspended until {rng.integers(1, 28)} Dec",
                f"Ruled out for the season with a {part.lower()} injury"])
            chance, status = 0.0, "i"
        elif kind == "doubtful":
            chance = float(rng.choice([25, 50, 75]))
            text = f"{part} injury - {int(chance)}% chance of playing"
            status = "d"
        else:
            text = rng.choice([f"{part} knock but should be fine",
                               "Back in full training and available"])
            chance, status = 100.0, "a"
        rows.append([f"Player{k}", teams[k % len(teams)], status, text, chance])
    # a few players who left the club
    for k in range(5):
        rows.append([f"Leaver{k}", teams[0], "u", "Has joined Al Hilal permanently", 0.0])
    # a player with no news at all
    rows.append(["Healthy", teams[1], "a", "", np.nan])
    return pd.DataFrame(rows, columns=["player", "team", "status", "news",
                                       "chance_of_playing"])


@pytest.fixture(scope="module")
def prepared():
    return prepare_news(make_news())


# ---- tokenize / clean_text ----------------------------------------------------------
def test_clean_text_lowercases_strips_punctuation_and_stop_words():
    cleaned = clean_text("Hamstring injury - 75% chance of playing")
    assert cleaned == "hamstring injury 75 chance playing"


def test_remove_numbers_drops_digit_tokens_only():
    cleaned = clean_text("Knee injury - Expected back 20 Dec (u21 duty)", remove_numbers=True)
    assert cleaned == "knee injury expected back dec duty"


def test_apostrophes_are_deleted_and_unicode_dashes_split():
    assert tokenize("Player\u2019s knock \u2013 doesn't train") == ["players", "knock", "doesnt", "train"]


def test_non_strings_give_no_tokens():
    assert tokenize(None) == [] and tokenize(float("nan")) == [] and clean_text(42) == ""


def test_clean_text_equals_joined_tokens():
    text = "Ruled out for the season!"
    assert clean_text(text) == " ".join(tokenize(text))


def test_meaningful_words_are_not_stop_words():
    for word in ("out", "not", "no", "until", "back", "chance", "playing", "expected"):
        assert word not in STOP_WORDS
    assert tokenize("Out until January, not expected back") == [
        "out", "until", "january", "not", "expected", "back"]


# ---- labels and preparation --------------------------------------------------------------
@pytest.mark.parametrize("chance,status,expected", [
    (0, "i", "out"), (0.0, "u", "out"), (25, "d", "doubtful"), (50.0, "d", "doubtful"),
    (75, "d", "doubtful"), (100, "a", "fit"), (np.nan, "a", "fit"),
    (np.nan, "d", "doubtful"), (np.nan, "i", "out"), (None, "s", "out")])
def test_label_availability(chance, status, expected):
    assert label_availability(chance, status) == expected


def test_prepare_news_filters_and_labels():
    raw = make_news()
    prepared = prepare_news(raw)
    assert "u" not in set(prepared["status"])
    assert (prepared["news"].str.strip() != "").all()
    assert set(prepared["label"]) <= set(LABELS)
    assert len(prepared) == len(raw) - 5 - 1
    kept = prepare_news(raw, exclude_statuses=())
    assert len(kept) == len(raw) - 1            # leavers kept, empty news still dropped
    assert (kept[kept["status"] == "u"]["label"] == "out").all()


def test_prepare_news_requires_columns():
    with pytest.raises(ValueError):
        prepare_news(make_news().drop(columns=["status"]))


# ---- classifier basics ----------------------------------------------------------------------
def test_fit_predict_and_probabilities(prepared):
    model = AvailabilityClassifier().fit(prepared["news"], prepared["label"])
    assert set(model.classes) == set(LABELS)
    predictions = model.predict(["Hamstring injury - 50% chance of playing",
                                 "Suspended until 3 Dec"])
    assert set(predictions) <= set(LABELS)
    proba = model.predict_proba(prepared["news"].head(10))
    assert proba.shape == (10, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_classifier_input_validation(prepared):
    with pytest.raises(RuntimeError):
        AvailabilityClassifier().predict(["x"])
    with pytest.raises(ValueError):
        AvailabilityClassifier().fit(["a", "b"], ["out", "out"])      # one class
    with pytest.raises(ValueError):
        AvailabilityClassifier().fit(["a", "b"], ["out"])             # length mismatch
    with pytest.raises(ValueError):
        AvailabilityClassifier().fit(["a", "b"], ["out", "maybe"])    # unknown label
    with pytest.raises(ValueError):
        AvailabilityClassifier().fit([], [])


# ---- evaluation -----------------------------------------------------------------------------
def test_evaluate_returns_honest_metrics(prepared):
    result = AvailabilityClassifier().evaluate(prepared["news"], prepared["label"])
    assert result["n"] == len(prepared)
    assert 0.85 < result["accuracy"] <= 1.0           # templated text is easy
    assert result["accuracy"] > result["majority_baseline"]
    assert 0 < result["macro_f1"] <= 1
    assert result["confusion"].to_numpy().sum() == len(prepared)
    assert len(result["predictions"]) == len(prepared)
    assert result["rare_classes"] == []


def test_evaluation_is_reproducible(prepared):
    a = AvailabilityClassifier(random_state=3).evaluate(prepared["news"], prepared["label"])
    b = AvailabilityClassifier(random_state=3).evaluate(prepared["news"], prepared["label"])
    assert a["accuracy"] == b["accuracy"]
    assert list(a["predictions"]) == list(b["predictions"])


def test_cross_validation_does_not_leak():
    """Unique random words with random labels can be memorised in-sample but
    never predicted out-of-fold, so honest accuracy must stay near chance."""
    rng = np.random.default_rng(0)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    texts = ["".join(rng.choice(letters, 8)) + " " + "".join(rng.choice(letters, 8))
             for _ in range(90)]
    labels = ["fit"] * 30 + ["doubtful"] * 30 + ["out"] * 30
    # sanity: the model really can memorise this data when it is allowed to
    in_sample = AvailabilityClassifier(remove_numbers=False).fit(texts, labels).predict(texts)
    assert np.mean(in_sample == np.array(labels, dtype=object)) > 0.9
    # ...but out-of-fold it has nothing to learn from
    result = AvailabilityClassifier(remove_numbers=False).evaluate(texts, labels)
    assert result["accuracy"] < 0.55


def test_removing_numbers_exposes_the_percentage_shortcut():
    """Two classes whose texts differ ONLY by the percentage."""
    texts = (["Injury - 0% chance of playing"] * 30
             + ["Injury - 50% chance of playing"] * 30)
    labels = ["out"] * 30 + ["doubtful"] * 30
    with_numbers = AvailabilityClassifier(remove_numbers=False).evaluate(texts, labels)
    without = AvailabilityClassifier(remove_numbers=True).evaluate(texts, labels)
    assert with_numbers["accuracy"] > 0.95
    assert without["accuracy"] < 0.7


def test_rare_class_is_reported_not_crashing():
    texts = (["Knee injury - Unknown return date"] * 12
             + ["Ankle injury - 50% chance of playing"] * 12
             + ["Back in full training"])
    labels = ["out"] * 12 + ["doubtful"] * 12 + ["fit"]
    result = AvailabilityClassifier().evaluate(texts, labels)
    assert result["rare_classes"] == ["fit"]
    assert result["n"] == 25


def test_too_little_data_raises():
    with pytest.raises(ValueError):
        AvailabilityClassifier().out_of_fold_predictions(
            ["a", "b", "c"], ["out", "doubtful", "fit"])


# ---- report -------------------------------------------------------------------------------------
def test_report_structure_sorting_and_team_filter(prepared):
    model = AvailabilityClassifier()
    table = model.report(prepared)
    assert list(table.columns) == ["Team", "Player", "News", "Official",
                                   "Predicted", "Agree"]
    assert len(table) == len(prepared)
    assert table["Agree"].dtype == bool
    for _, group in table.groupby("Team"):
        order = group["Predicted"].map({"out": 0, "doubtful": 1, "fit": 2}).tolist()
        assert order == sorted(order)             # worst news first within each team
    one = model.report(prepared, team="Arsenal")
    assert set(one["Team"]) == {"Arsenal"}
    with pytest.raises(ValueError):
        model.report(prepared, team="Nowhere FC")


def test_summarize_availability_counts(prepared):
    table = AvailabilityClassifier().report(prepared)
    summary = summarize_availability(table)
    assert list(summary.columns) == ["out", "doubtful", "fit"]
    assert summary.to_numpy().sum() == len(table)
    assert set(summary.index) == set(prepared["team"])