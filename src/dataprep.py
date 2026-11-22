"""Cleaning and combination of match data for KickCast.

Takes the raw DataFrames produced by the data layer (sources.py) and turns
them into ONE clean, chronologically sorted dataset covering both leagues.
Also converts bookmaker odds into probabilities, which later serve as a
baseline and as model features.

Contents:
    implied_probabilities() - decimal odds -> probabilities (map/filter based).
    overround()             - the bookmaker's built-in margin for a set of odds.
    clean_matches()         - fixes types, drops bad rows, picks one odds set.
    build_dataset()         - loads any list of DataSource objects and combines them.
    summarize()             - per league and season summary table.
"""

import math
import numbers

import numpy as np
import pandas as pd

# Numeric encoding of the match result. Order matches (home, draw, away).
RESULT_CODES = {"H": 0, "D": 1, "A": 2}

NUMERIC_COLUMNS = ["FTHG", "FTAG", "HS", "AS", "HST", "AST",
                   "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA"]

B365_COLUMNS = ["B365H", "B365D", "B365A"]
AVG_COLUMNS = ["AvgH", "AvgD", "AvgA"]
ODDS_COLUMNS = ["OddsH", "OddsD", "OddsA"]
PROB_COLUMNS = ["ProbH", "ProbD", "ProbA"]


def _is_valid_odd(value):
    """True for a real number strictly greater than 1 (decimal odds)."""
    return (isinstance(value, numbers.Real)
            and not math.isnan(value)
            and value > 1.0)


def implied_probabilities(odds, normalize=True):
    """Convert decimal odds to implied probabilities.

    Args:
        odds: an iterable of decimal odds, e.g. (home, draw, away).
        normalize: if True, rescale so the probabilities sum to exactly 1
            (this removes the bookmaker's margin). If False, return the raw
            1/odds values, which sum to more than 1.

    Returns:
        A list of floats in the same order as the input.

    Raises:
        ValueError: if odds is empty or any value is missing or not > 1.
    """
    odds = list(odds)
    if not odds:
        raise ValueError("odds must not be empty")
    valid = list(filter(_is_valid_odd, odds))
    if len(valid) != len(odds):
        raise ValueError(f"all odds must be numbers greater than 1, got {odds}")
    probs = list(map(lambda o: 1.0 / o, odds))
    if normalize:
        total = sum(probs)
        probs = list(map(lambda p: p / total, probs))
    return probs


def overround(odds):
    """Bookmaker margin: how far the raw implied probabilities exceed 1."""
    return sum(implied_probabilities(odds, normalize=False)) - 1.0


def _row_probabilities(odds_tuple):
    """Normalised probabilities for one match, or NaNs if odds are unusable."""
    try:
        return implied_probabilities(odds_tuple)
    except ValueError:
        return [np.nan] * 3


def _pick_odds(df):
    """Choose one (H, D, A) odds triple per match.

    Market-average odds are preferred; Bet365 odds are the fallback. A triple
    is only used if all three of its values are valid, so odds from two
    different bookmakers are never mixed within one match.
    """
    out = pd.DataFrame(np.nan, index=df.index, columns=ODDS_COLUMNS)
    for columns in (B365_COLUMNS, AVG_COLUMNS):  # later entry takes priority
        if all(c in df.columns for c in columns):
            usable = (df[columns] > 1.0).all(axis=1)
            out.loc[usable, ODDS_COLUMNS] = df.loc[usable, columns].to_numpy()
    return out


def clean_matches(df):
    """Clean a combined match DataFrame.

    Steps: fix types, trim team names, drop rows with missing date/goals,
    drop invalid or inconsistent results, drop duplicates, pick one odds set,
    add implied probabilities and a numeric Target (0=home, 1=draw, 2=away).

    Returns:
        (clean_df, report) where report is a dict counting rows removed.
    """
    df = df.copy()
    report = {"input_rows": len(df)}

    for column in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in ("HomeTeam", "AwayTeam"):
        df[column] = df[column].astype(str).str.strip()
    df["FTR"] = df["FTR"].astype(str).str.strip().str.upper()

    before = len(df)
    df = df.dropna(subset=["Date", "FTHG", "FTAG"])
    report["missing_date_or_goals"] = before - len(df)

    before = len(df)
    df = df[df["FTR"].isin(RESULT_CODES)]
    report["bad_result_code"] = before - len(df)

    before = len(df)
    from_goals = np.where(df["FTHG"] > df["FTAG"], "H",
                          np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    df = df[df["FTR"].to_numpy() == from_goals]
    report["result_contradicts_goals"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["Date", "League", "HomeTeam", "AwayTeam"])
    report["duplicates"] = before - len(df)

    df = df.copy()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["Target"] = df["FTR"].map(RESULT_CODES).astype(int)

    df[ODDS_COLUMNS] = _pick_odds(df)
    probs = [_row_probabilities(t)
             for t in df[ODDS_COLUMNS].itertuples(index=False, name=None)]
    df[PROB_COLUMNS] = np.array(probs, dtype=float).reshape(-1, 3)

    report["output_rows"] = len(df)
    report["rows_without_odds"] = int(df["ProbH"].isna().sum())
    return df, report


def build_dataset(sources):
    """Load every DataSource in `sources`, combine and clean the result.

    `sources` can hold any objects with a load() method returning a match
    DataFrame (e.g. one FootballCsvSource per league).

    Returns:
        (dataset, report): the cleaned DataFrame sorted by date, and the
        cleaning report from clean_matches().
    """
    frames = [source.load() for source in sources]
    combined = pd.concat(frames, ignore_index=True)
    clean, report = clean_matches(combined)
    clean = (clean.sort_values(["Date", "League", "HomeTeam"], kind="stable")
                  .reset_index(drop=True))
    return clean, report


def summarize(df):
    """Per league and season: matches, result shares, goals and missing odds."""
    work = df.assign(TotalGoals=df["FTHG"] + df["FTAG"])
    summary = work.groupby(["League", "Season"]).agg(
        matches=("Target", "size"),
        home_win_pct=("Target", lambda s: (s == 0).mean() * 100),
        draw_pct=("Target", lambda s: (s == 1).mean() * 100),
        away_win_pct=("Target", lambda s: (s == 2).mean() * 100),
        goals_per_match=("TotalGoals", "mean"),
        missing_odds=("ProbH", lambda s: int(s.isna().sum())),
    )
    return summary.round(1).reset_index()