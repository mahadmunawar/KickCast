"""KickCast: a two-league (EPL and La Liga) football match outcome predictor.

Entry point of the program. Shows a console menu (update data, predict a
fixture, view the Elo table, view the FPL availability report, view the
model evaluation) and validates all user input.

All the real work is done in src/engine.py; this file only talks to the user:
it asks questions in loops until the answer is valid, prints results, and
turns every error into a message instead of a crash.

Author: Mahad Munawar (24I-0065), AL-2001 Programming for AI (Lab), Fall 2026.
"""

import re
import sys
from pathlib import Path

from src.engine import (LEAGUE_NAMES, POISSON_NAME, KickCastEngine,
                        load_data, season_label)
from src.sources import DataFetchError

FIGURE_DIR = Path(__file__).resolve().parent / "outputs" / "figures"

BANNER = """
==========================================================
 KickCast: EPL and La Liga match outcome predictor
 Educational project. Predictions are uncertain estimates,
 not betting advice. Please gamble responsibly, or not at all.
==========================================================""".strip("\n")


class UserQuit(Exception):
    """Raised when input ends (Ctrl+Z / Ctrl+C) so the program can exit cleanly."""


# ---- validated input ----------------------------------------------------------------
def read_line(prompt):
    """input() that turns end-of-input and Ctrl+C into UserQuit."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt, OSError):   # OSError: no console attached
        print()
        raise UserQuit from None


def ask_int(prompt, low, high, default=None):
    """Ask for a whole number in [low, high]; repeat until the answer is valid."""
    hint = f" [{default}]" if default is not None else ""
    while True:
        text = read_line(f"{prompt} ({low}-{high}){hint}: ")
        if text == "" and default is not None:
            return default
        try:
            value = int(text)
        except ValueError:
            print("  Please type a whole number.")
            continue
        if low <= value <= high:
            return value
        print(f"  Please choose a number between {low} and {high}.")


def ask_yes_no(prompt, default=False):
    """Ask a yes/no question; an empty answer gives `default`."""
    hint = "Y/n" if default else "y/N"
    while True:
        text = read_line(f"{prompt} [{hint}]: ").lower()
        if text == "":
            return default
        if text in ("y", "yes"):
            return True
        if text in ("n", "no"):
            return False
        print("  Please answer y or n.")


def ask_choice(prompt, options):
    """Show numbered options and return the chosen option's index."""
    for number, label in enumerate(options, start=1):
        print(f"  {number}. {label}")
    return ask_int(prompt, 1, len(options)) - 1


def match_team(text, teams):
    """Identify a team from what the user typed.

    Accepts the list number, the exact name, or a part of the name that fits
    exactly one team (case does not matter).

    Returns:
        (team, None) on success, or (None, reason) if nothing fits or several do.
    """
    text = text.strip()
    if not text:
        return None, "Nothing was typed."
    if text.isdigit():
        number = int(text)
        if 1 <= number <= len(teams):
            return teams[number - 1], None
        return None, f"Choose a number between 1 and {len(teams)}."
    lowered = text.lower()
    exact = [team for team in teams if team.lower() == lowered]
    if exact:
        return exact[0], None
    partial = [team for team in teams if lowered in team.lower()]
    if len(partial) == 1:
        return partial[0], None
    if partial:
        return None, "Did you mean: " + ", ".join(partial) + "?"
    return None, f"No team matches '{text}'."


def print_teams(teams, columns=3):
    """Print a numbered list of teams in a few columns."""
    cells = [f"{number:>2}. {team:<18}" for number, team in enumerate(teams, start=1)]
    for start in range(0, len(cells), columns):
        print("  " + "".join(cells[start:start + columns]))


def ask_team(prompt, teams):
    """Ask for a team (number or name) until it is valid."""
    while True:
        team, problem = match_team(read_line(prompt), teams)
        if team is not None:
            return team
        print(f"  {problem}")


def ask_league(engine):
    """Ask which league; returns its key (e.g. "EPL")."""
    leagues = engine.leagues
    labels = [LEAGUE_NAMES.get(league, league) for league in leagues]
    return leagues[ask_choice("League", labels)]


def slug(text):
    """File-name-safe version of a name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


# ---- session ----------------------------------------------------------------------------
class Session:
    """Holds the loader and the currently loaded engine (None until data loads)."""

    def __init__(self, loader=load_data):
        self.loader = loader
        self.engine = None

    def require_engine(self):
        """The loaded engine, or None after telling the user how to load data."""
        if self.engine is None:
            print("No data is loaded yet. Choose option 1 (Update data) first.")
        return self.engine


def print_data_summary(engine):
    for note in engine.notes:
        print(f"  - {note}")
    matches = engine.matches
    first, last = engine.seasons[0], engine.seasons[-1]
    print(f"  {len(matches)} matches, {', '.join(engine.leagues)}, "
          f"seasons {season_label(first)} to {season_label(last)}.")


# ---- menu actions --------------------------------------------------------------------------
def update_data(session):
    """1. Load the data (web, then cache, then sample) and rebuild the models."""
    print("Loading data (a few seconds if it has to be downloaded)...")
    try:
        data = session.loader()
        engine = KickCastEngine(data.matches, data.fpl, data.notes)
    except DataFetchError as error:
        print(f"Could not load any data: {error}")
        if session.engine is None:
            print("Connect to the internet once so the data can be cached, and try again.")
        else:
            print("The data that was already loaded stays in use.")
        return
    session.engine = engine
    print("Data ready:")
    print_data_summary(engine)


def format_percent(value):
    return f"{value:.1%}"


def print_prediction(engine, prediction):
    p = prediction
    title = (f"{LEAGUE_NAMES.get(p.league, p.league)}: "
             f"{p.home} (home) vs {p.away} (away)")
    print("\n" + title)
    print("-" * len(title))
    print(f"{'Method':<24}{'Home win':>10}{'Draw':>9}{'Away win':>10}")
    for name, proba in ((p.model_name, p.model_proba), (POISSON_NAME, p.poisson_proba)):
        print(f"{name:<24}{format_percent(proba[0]):>10}"
              f"{format_percent(proba[1]):>9}{format_percent(proba[2]):>10}")
    outcomes = [f"{p.home} win", "Draw", f"{p.away} win"]
    best = int(p.model_proba.argmax())
    print(f"Most likely result ({p.model_name}): {outcomes[best]} "
          f"({format_percent(p.model_proba[best])})")
    goals_home, goals_away = p.expected_goals
    print(f"Expected goals: {goals_home:.2f} - {goals_away:.2f} | most likely score "
          f"{p.likely_score[0]}-{p.likely_score[1]} | over 2.5 goals "
          f"{format_percent(p.over_2_5)}")
    f = p.features
    print(f"Elo rating: {p.home} {f['HomeElo']:.0f} | {p.away} {f['AwayElo']:.0f}")
    print(f"Form (average per match over the last {engine.window} matches):")
    for side, team in (("Home", p.home), ("Away", p.away)):
        print(f"  {team}: {f[side + '_Points']:.1f} points, "
              f"{f[side + '_GoalsFor']:.1f} scored, "
              f"{f[side + '_GoalsAgainst']:.1f} conceded")

    availability = engine.fixture_availability(p.league, p.home, p.away)
    if availability is not None:
        print("Injury news (FPL, labelled by the text classifier):")
        for team, players in availability.items():
            if players.empty:
                print(f"  {team}: no injury news listed")
                continue
            out = players[players["Predicted"] == "out"]["Player"].tolist()
            doubtful = players[players["Predicted"] == "doubtful"]["Player"].tolist()
            print(f"  {team}: out {len(out)}, doubtful {len(doubtful)}")
            if out:
                print(f"    out: {', '.join(out[:8])}")
            if doubtful:
                print(f"    doubtful: {', '.join(doubtful[:8])}")
    print("Estimates only: football is very hard to predict. Not betting advice.")


def predict_fixture(session):
    """2. Choose a league and two teams, then show the prediction."""
    engine = session.require_engine()
    if engine is None:
        return
    league = ask_league(engine)
    teams = engine.teams(league)
    print(f"\nTeams in the latest {LEAGUE_NAMES.get(league, league)} season:")
    print_teams(teams)
    home = ask_team("Home team (number or name): ", teams)
    away = ask_team("Away team (number or name): ", teams)
    while away == home:
        print("  A team cannot play itself. Choose a different away team.")
        away = ask_team("Away team (number or name): ", teams)

    print("Working it out...")
    prediction = engine.predict_fixture(league, home, away)
    print_prediction(engine, prediction)
    if ask_yes_no("Save the scoreline heatmap as a picture?"):
        path = FIGURE_DIR / f"menu_scoreline_{slug(home)}_vs_{slug(away)}.png"
        print(f"Saved: {engine.save_heatmap(prediction, path)}")


def show_elo_table(session):
    """3. Elo ranking of one league."""
    engine = session.require_engine()
    if engine is None:
        return
    league = ask_league(engine)
    top = ask_int("How many teams to show", 1, 20, default=10)
    table = engine.elo_ranking(league, top=top)
    print(f"\nElo ratings after the last match in the data "
          f"({LEAGUE_NAMES.get(league, league)}, teams of the latest season):")
    print(table.to_string(index=False))
    if ask_yes_no("Save an Elo trajectory chart of the top 5?"):
        path = FIGURE_DIR / f"menu_elo_{slug(league)}.png"
        print(f"Saved: {engine.save_elo_figure(league, top=5, path=path)}")


def print_availability_honesty(engine):
    info = engine.availability_honesty()
    if info is None:
        return
    counts = ", ".join(f"{label} {count}" for label, count in info["label_counts"].items())
    print(f"\nHow far to trust this: {info['n']} players with news ({counts}). "
          f"Cross-validated accuracy {info['accuracy']:.1%}, versus "
          f"{info['majority_baseline']:.1%} for always guessing the most common label.")
    print("FPL writes its news from fixed templates (for example '... chance of playing' "
          "or 'Unknown return date'), so a very high score means the model recognises "
          "those phrases, not that it understands injuries. There is no 'fit' class "
          "because fit players have no news, and players who left their club are excluded.")


def show_availability(session):
    """4. FPL injury-news availability report (EPL)."""
    engine = session.require_engine()
    if engine is None:
        return
    if not engine.has_availability:
        print("No injury news is available (the FPL feed could not be loaded).")
        return
    choice = ask_choice("Report", ["Summary for every team", "Players of one team"])
    if choice == 0:
        print("\nPredicted availability per team (players with news only):")
        print(engine.availability_summary().to_string())
    else:
        teams = engine.availability_teams()
        print_teams(teams)
        team = ask_team("Team (number or name): ", teams)
        report = engine.availability_report(team)
        shown = report[["Player", "News", "Official", "Predicted"]].copy()
        shown["News"] = shown["News"].str.slice(0, 55)
        print(f"\n{team}:")
        print(shown.to_string(index=False))
    print_availability_honesty(engine)


def show_evaluation(session):
    """5. Test-season results of every method."""
    engine = session.require_engine()
    if engine is None:
        return
    print("Evaluating (trains several models, this can take 10-30 seconds)...")
    try:
        table, season = engine.evaluation()
    except ValueError as error:
        print(f"Cannot evaluate: {error}")
        print("The evaluation needs at least two seasons: earlier ones to train on and "
              "the latest to test on. Choose option 1 to load the full data.")
        return

    shown = table.copy()
    shown["Accuracy"] = shown["Accuracy"].map(lambda v: f"{v:.1%}")
    shown["AccMargin"] = shown["AccMargin"].map(lambda v: f"+/-{v * 100:.1f}")
    shown["LogLoss"] = shown["LogLoss"].map(lambda v: f"{v:.3f}")
    shown = shown.rename(columns={"AccMargin": "Margin(pts)", "TestSet": "Test set"})
    print(f"\nResults on the {season_label(season)} season (trained on earlier seasons only):")
    print(shown[shown["Test set"] == "All"].drop(columns="Test set").to_string(index=False))
    if ask_yes_no("Show every league and training set-up too?"):
        print(shown.to_string(index=False))
    print("Lower log loss is better. 'Margin' is the 95% margin of error of the "
          "accuracy, so differences smaller than it are not meaningful.")
    if ask_yes_no("Save the calibration chart (Figure 3)?"):
        path = FIGURE_DIR / "menu_calibration.png"
        print(f"Saved: {engine.save_calibration_figure(path)}")


MENU = [
    ("Update data", update_data),
    ("Predict a fixture", predict_fixture),
    ("Elo table", show_elo_table),
    ("FPL availability report", show_availability),
    ("Model evaluation", show_evaluation),
]


# ---- main loop ---------------------------------------------------------------------------------
def run_action(action, session):
    """Run one menu action; every ordinary error becomes a message, not a crash."""
    try:
        action(session)
    except UserQuit:
        raise
    except DataFetchError as error:
        print(f"Data problem: {error}")
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Could not complete that: {error}")
    except Exception as error:  # last resort: the menu must never crash
        print(f"Unexpected problem ({type(error).__name__}: {error}). "
              "Returning to the menu.")


def run_menu(session):
    while True:
        print("\nMain menu" + ("" if session.engine else " (no data loaded)"))
        for number, (label, _action) in enumerate(MENU, start=1):
            print(f"  {number}. {label}")
        print("  0. Quit")
        choice = ask_int("Choose an option", 0, len(MENU))
        if choice == 0:
            print("Goodbye.")
            return
        run_action(MENU[choice - 1][1], session)


def _make_output_safe():
    """Never crash on a character the console cannot show (player names, etc.)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(loader=load_data):
    """Start the program: load the data once, then run the menu."""
    _make_output_safe()
    print(BANNER)
    session = Session(loader)
    try:
        run_action(update_data, session)
        run_menu(session)
    except UserQuit:
        print("Goodbye.")


if __name__ == "__main__":
    main()