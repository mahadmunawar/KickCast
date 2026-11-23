"""Synthetic data shared by the Part 8 tests (no network, no real files)."""

import math

import numpy as np
import pandas as pd

from src.dataprep import PROB_COLUMNS

# Two synthetic leagues named like the real ones, with two EPL clubs named the way
# football-data spells them, so the FPL spelling map ("Spurs", "Man Utd") is exercised.
RENAMES = {"EPL_T0": "Tottenham", "EPL_T1": "Man United"}


def make_matches(seed=0, n_teams=12, seasons=4, leagues=("EPL", "LaLiga"), spread=0.5):
    """Round-robin leagues with hidden strengths and a bookmaker that knows them."""
    rng = np.random.default_rng(seed)
    rows = []
    for league in leagues:
        strengths = rng.normal(0, spread, n_teams)
        for s in range(seasons):
            season = f"{21 + s}{22 + s}"
            day = pd.Timestamp(year=2021 + s, month=8, day=1)
            busy = set()
            for i in range(n_teams):
                for j in range(n_teams):
                    if i == j:
                        continue
                    if i in busy or j in busy:
                        day = day + pd.offsets.Day(1)
                        busy = set()
                    busy.update((i, j))
                    d = strengths[i] - strengths[j]
                    hg = rng.poisson(math.exp(0.25 + d))
                    ag = rng.poisson(math.exp(-0.10 - d))
                    target = 0 if hg > ag else 2 if hg < ag else 1
                    weights = np.array([math.exp(0.3 + 1.5 * d), 1.1, math.exp(-1.5 * d)])
                    probs = weights / weights.sum()
                    rows.append([day, league, season, f"{league}_T{i}",
                                 f"{league}_T{j}", hg, ag, target,
                                 rng.poisson(13), rng.poisson(11),
                                 rng.poisson(4), rng.poisson(3.5), *probs])
    columns = (["Date", "League", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                "Target", "HS", "AS", "HST", "AST"] + PROB_COLUMNS)
    df = pd.DataFrame(rows, columns=columns)
    df[["HomeTeam", "AwayTeam"]] = df[["HomeTeam", "AwayTeam"]].replace(RENAMES)
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def make_fpl():
    """A tiny FPL bootstrap-static dict using FPL's own club spellings."""
    teams = [{"id": 1, "name": "Spurs"}, {"id": 2, "name": "Man Utd"},
             {"id": 3, "name": "EPL_T2"}, {"id": 4, "name": "Hull City"}]
    out_news = ["Knee injury - Unknown return date", "Hamstring injury - Expected back 29 Nov",
                "Calf injury - Unknown return date", "Back injury - Unknown return date"]
    doubtful_news = ["Knock - 75% chance of playing", "Illness - 50% chance of playing",
                     "Shoulder injury - 75% chance of playing"]
    elements = []
    for team in teams:
        for i in range(6):
            elements.append({"web_name": f"{team['name'][:3]}Out{i}", "team": team["id"],
                             "status": "i", "news": out_news[i % len(out_news)],
                             "chance_of_playing_next_round": 0})
        for i in range(4):
            elements.append({"web_name": f"{team['name'][:3]}Doubt{i}", "team": team["id"],
                             "status": "d", "news": doubtful_news[i % len(doubtful_news)],
                             "chance_of_playing_next_round": 75 if i % 2 == 0 else 50})
        elements.append({"web_name": f"{team['name'][:3]}Gone", "team": team["id"],
                         "status": "u", "news": "Has joined Al Hilal permanently",
                         "chance_of_playing_next_round": 0})
        elements.append({"web_name": "Ødegaard", "team": team["id"], "status": "a",
                         "news": "", "chance_of_playing_next_round": None})
    return {"teams": teams, "elements": elements}


def raw_frame(league, seasons=("2425", "2526"), seed=1):
    """A frame shaped like FootballCsvSource.load() output (before cleaning)."""
    rng = np.random.default_rng(seed)
    names = [f"{league}_X{i}" for i in range(4)]
    rows = []
    for s, season in enumerate(seasons):
        day = pd.Timestamp(year=2024 + s, month=8, day=1)
        for i, home in enumerate(names):
            for j, away in enumerate(names):
                if i == j:
                    continue
                hg, ag = int(rng.poisson(1.5)), int(rng.poisson(1.1))
                result = "H" if hg > ag else "A" if hg < ag else "D"
                rows.append([day, home, away, hg, ag, result,
                             int(rng.poisson(13)), int(rng.poisson(11)),
                             int(rng.poisson(4)), int(rng.poisson(3.5)), league, season])
                day = day + pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                                       "FTR", "HS", "AS", "HST", "AST", "League", "Season"])