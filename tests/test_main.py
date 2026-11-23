"""Tests for the console menu: validated input, error handling, a scripted session."""

import builtins

import pytest

import main
from src.engine import KickCastEngine, LoadedData
from src.sources import DataFetchError
from tests.synthetic import make_fpl, make_matches


# ---- helpers ---------------------------------------------------------------------------------
def feed(monkeypatch, answers):
    """Make input() return the given answers in order, then end of input."""
    answers = iter(answers)

    def fake_input(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", fake_input)


@pytest.fixture(scope="module")
def loaded():
    return LoadedData(make_matches(), make_fpl(), "web", ["test data"])


def loader_for(loaded):
    return lambda: loaded


# ---- match_team ----------------------------------------------------------------------------------
TEAMS = ["Arsenal", "Aston Villa", "Chelsea", "Man City", "Man United"]


@pytest.mark.parametrize("text, expected", [
    ("1", "Arsenal"), ("5", "Man United"), ("chelsea", "Chelsea"), ("CHEL", "Chelsea"),
    ("  man city ", "Man City"), ("villa", "Aston Villa"),
])
def test_match_team_accepts(text, expected):
    assert main.match_team(text, TEAMS) == (expected, None)


@pytest.mark.parametrize("text", ["", "   ", "0", "6", "99", "zzz", "man", "a", "ar!"])
def test_match_team_rejects_with_a_reason(text):
    team, reason = main.match_team(text, TEAMS)
    assert team is None and reason


def test_match_team_ambiguous_names_the_options():
    _, reason = main.match_team("man", TEAMS)
    assert "Man City" in reason and "Man United" in reason


# ---- input helpers ---------------------------------------------------------------------------------
def test_ask_int_repeats_until_valid(monkeypatch, capsys):
    feed(monkeypatch, ["abc", "", "7.5", "99", "-1", "3"])
    assert main.ask_int("Pick", 1, 5) == 3
    assert capsys.readouterr().out.count("Please") == 5


def test_ask_int_default_and_end_of_input(monkeypatch):
    feed(monkeypatch, [""])
    assert main.ask_int("Pick", 1, 20, default=10) == 10
    feed(monkeypatch, [])
    with pytest.raises(main.UserQuit):
        main.ask_int("Pick", 1, 5)


def test_ctrl_c_is_a_clean_quit(monkeypatch):
    def interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", interrupt)
    with pytest.raises(main.UserQuit):
        main.read_line("> ")


def test_ask_yes_no(monkeypatch):
    feed(monkeypatch, ["maybe", "YES"])
    assert main.ask_yes_no("Ok?") is True
    feed(monkeypatch, ["", ])
    assert main.ask_yes_no("Ok?", default=True) is True
    feed(monkeypatch, ["no"])
    assert main.ask_yes_no("Ok?", default=True) is False


def test_ask_team_repeats_until_valid(monkeypatch, capsys):
    feed(monkeypatch, ["nope", "man", "chel"])
    assert main.ask_team("Team: ", TEAMS) == "Chelsea"
    out = capsys.readouterr().out
    assert "No team matches 'nope'" in out and "Did you mean" in out


def test_ask_choice_returns_a_zero_based_index(monkeypatch):
    feed(monkeypatch, ["9", "2"])
    assert main.ask_choice("Option", ["a", "b", "c"]) == 1


def test_slug():
    assert main.slug("Nott'm Forest") == "Nott_m_Forest"


# ---- no data ------------------------------------------------------------------------------------------
def test_start_without_any_data_does_not_crash(monkeypatch, capsys):
    def nothing():
        raise DataFetchError("no network and no cache")

    feed(monkeypatch, ["2", "3", "4", "5", "0"])
    main.main(loader=nothing)
    out = capsys.readouterr().out
    assert "Could not load any data" in out
    assert out.count("No data is loaded yet") == 4
    assert "Goodbye." in out


def test_failed_update_keeps_the_old_data(monkeypatch, capsys, loaded):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > 1:
            raise DataFetchError("went offline")
        return loaded

    feed(monkeypatch, ["1", "0"])
    main.main(loader=flaky)
    out = capsys.readouterr().out
    assert "already loaded stays in use" in out


# ---- scripted sessions ---------------------------------------------------------------------------------------
def test_garbage_input_never_crashes_the_menu(monkeypatch, capsys, loaded):
    feed(monkeypatch, ["", "banana", "-3", "9", "2.5", "  ", "0"])
    main.main(loader=loader_for(loaded))
    out = capsys.readouterr().out
    assert "Traceback" not in out and "Goodbye." in out


def test_predict_fixture_session(monkeypatch, capsys, loaded, tmp_path):
    monkeypatch.setattr(main, "FIGURE_DIR", tmp_path)
    feed(monkeypatch, [
        "2",                      # predict a fixture
        "x", "3", "1",            # league: two bad answers, then EPL
        "zzz", "Tott",            # home: unknown name, then partial match
        "Tottenham", "Man Un",    # away: same as home is refused, then Man United
        "y",                      # save the heatmap
        "0"])
    main.main(loader=loader_for(loaded))
    out = capsys.readouterr().out
    assert "Tottenham (home) vs Man United (away)" in out
    assert "A team cannot play itself" in out
    assert "No team matches 'zzz'" in out
    assert "Logistic Regression" in out and "Poisson (goals only)" in out
    assert "Not betting advice" in out
    assert "Injury news" in out
    assert (tmp_path / "menu_scoreline_Tottenham_vs_Man_United.png").exists()


def test_laliga_fixture_has_no_injury_section(monkeypatch, capsys, loaded):
    feed(monkeypatch, ["2", "2", "1", "2", "n", "0"])
    main.main(loader=loader_for(loaded))
    out = capsys.readouterr().out
    assert "LaLiga_T0 (home) vs LaLiga_T1 (away)" in out
    assert "Injury news" not in out


def test_elo_table_session(monkeypatch, capsys, loaded, tmp_path):
    monkeypatch.setattr(main, "FIGURE_DIR", tmp_path)
    feed(monkeypatch, ["3", "2", "abc", "0", "4", "y", "0"])
    main.main(loader=loader_for(loaded))
    out = capsys.readouterr().out
    assert "Rank" in out and "Elo" in out
    assert (tmp_path / "menu_elo_LaLiga.png").exists()


def test_availability_session(monkeypatch, capsys, loaded):
    feed(monkeypatch, ["4", "1", "4", "2", "tott", "0"])
    main.main(loader=loader_for(loaded))
    out = capsys.readouterr().out
    assert "Predicted availability per team" in out
    assert out.count("How far to trust this") == 2
    assert "recognises those phrases" in out
    assert "Tottenham" in out


def test_evaluation_session(monkeypatch, capsys, loaded, tmp_path):
    monkeypatch.setattr(main, "FIGURE_DIR", tmp_path)
    feed(monkeypatch, ["5", "maybe", "y", "y", "0"])
    main.main(loader=loader_for(loaded))
    out = capsys.readouterr().out
    assert "Results on the 2024/25 season" in out
    assert "Poisson (goals only)" in out and "Bookmaker odds" in out
    assert "Transfer from" in out                      # the "show everything" table
    assert (tmp_path / "menu_calibration.png").exists()


def test_evaluation_with_a_single_season_explains_itself(monkeypatch, capsys):
    one = make_matches(seasons=1)
    feed(monkeypatch, ["5", "0"])
    main.main(loader=lambda: LoadedData(one, None, "sample", ["small"]))
    out = capsys.readouterr().out
    assert "Cannot evaluate" in out and "at least two seasons" in out


def test_availability_without_fpl_data(monkeypatch, capsys):
    feed(monkeypatch, ["4", "0"])
    main.main(loader=lambda: LoadedData(make_matches(), None, "web", []))
    assert "No injury news is available" in capsys.readouterr().out


# ---- error handling ----------------------------------------------------------------------------------------------
def test_ordinary_errors_become_messages(monkeypatch, capsys, loaded):
    def broken(self, league, home, away):
        raise ValueError("model exploded")

    monkeypatch.setattr(KickCastEngine, "predict_fixture", broken)
    feed(monkeypatch, ["2", "1", "1", "2", "0"])
    main.main(loader=loader_for(loaded))
    out = capsys.readouterr().out
    assert "Could not complete that: model exploded" in out and "Goodbye." in out


def test_unexpected_errors_still_return_to_the_menu(monkeypatch, capsys, loaded):
    def broken(self, league, top=10, season=None):
        raise KeyError("surprise")

    monkeypatch.setattr(KickCastEngine, "elo_ranking", broken)
    feed(monkeypatch, ["3", "1", "", "0"])
    main.main(loader=loader_for(loaded))
    out = capsys.readouterr().out
    assert "Unexpected problem (KeyError" in out and "Goodbye." in out


def test_end_of_input_in_the_middle_of_an_action_quits_cleanly(monkeypatch, capsys, loaded):
    feed(monkeypatch, ["2", "1"])          # input ends while choosing the home team
    main.main(loader=loader_for(loaded))
    assert "Goodbye." in capsys.readouterr().out