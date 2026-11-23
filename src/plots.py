"""Visualisations for KickCast.

Produces the three labelled figures used in the evaluation report:
    1. Elo rating trajectories of selected teams over time.
    2. Seaborn heatmap of scoreline probabilities for a fixture.
    3. Calibration chart comparing the model with the bookmaker.

Figures are saved to outputs/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.poisson import ScorelineModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


def plot_elo_trajectories(elo, teams, title="Elo rating trajectories",
                          save_path=None, show=False):
    """Figure 1: line chart of each team's Elo rating after every match.

    Args:
        elo: an EloRating that has already been updated with matches.
        teams: list of team names to draw.
        title: chart title.
        save_path: where to save the PNG (default outputs/figures/fig1_...).
        show: if True, also open the figure in a window.

    Returns:
        The Path of the saved image.

    Raises:
        ValueError: if a team has no rating history.
    """
    save_path = Path(save_path) if save_path else FIGURE_DIR / "fig1_elo_trajectories.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    try:
        for team in teams:
            history = elo.history(team)
            if history.empty:
                raise ValueError(f"no rating history for team: {team}")
            ax.plot(history["Date"], history["Rating"], label=team, linewidth=1.8)
        ax.axhline(1500, color="grey", linestyle="--", linewidth=1,
                   label="Starting rating (1500)")
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Elo rating")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
    finally:
        plt.close(fig)
    return save_path


def plot_calibration(y_true, probabilities, bins=8,
                     title="Calibration on the test season",
                     save_path=None, show=False):
    """Figure 3: reliability chart of predicted probabilities vs. reality.

    Home, draw and away probabilities are pooled: each of the three
    "will this outcome happen?" predictions per match is one point. Points
    are sorted into `bins` groups of equal size; for each group the average
    predicted probability (x) is plotted against how often the outcome really
    happened (y). A perfectly calibrated method lies on the diagonal.

    Args:
        y_true: outcome codes (0 = home, 1 = draw, 2 = away) for each match.
        probabilities: dict mapping a method name to an (n, 3) probability array.

    Returns:
        The Path of the saved image.
    """
    y_true = np.asarray(y_true)
    save_path = Path(save_path) if save_path else FIGURE_DIR / "fig3_calibration.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    happened = (y_true[:, None] == np.arange(3)[None, :]).ravel()

    fig, ax = plt.subplots(figsize=(7, 7))
    try:
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
                label="Perfect calibration")
        for name, proba in probabilities.items():
            proba = np.asarray(proba)
            if proba.shape != (len(y_true), 3):
                raise ValueError(f"{name}: expected shape {(len(y_true), 3)}, "
                                 f"got {proba.shape}")
            frame = pd.DataFrame({"predicted": proba.ravel(), "happened": happened})
            frame["bin"] = pd.qcut(frame["predicted"], bins, duplicates="drop")
            curve = frame.groupby("bin", observed=True).agg(
                predicted=("predicted", "mean"), happened=("happened", "mean"))
            ax.plot(curve["predicted"], curve["happened"], marker="o",
                    linewidth=1.8, label=name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("Predicted probability (home, draw and away pooled)")
        ax.set_ylabel("Observed frequency")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
    finally:
        plt.close(fig)
    return save_path


def plot_scoreline_heatmap(matrix, home_team, away_team, save_path=None,
                           show=False):
    """Figure 2: Seaborn heatmap of scoreline probabilities for one fixture.

    Rows are the home team's goals and columns the away team's goals, so the
    cell in row 2, column 1 is the probability of a 2-1 home win. Cells below
    the diagonal are home wins, the diagonal is draws, above it away wins.

    Args:
        matrix: 2-D array from ScorelineModel.scoreline_matrix().
        home_team, away_team: names used in the labels.

    Returns:
        The Path of the saved image.

    Raises:
        ValueError: if matrix is not a non-empty 2-D array.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.size == 0:
        raise ValueError("matrix must be a non-empty 2-D array")
    save_path = Path(save_path) if save_path else FIGURE_DIR / "fig2_scoreline_heatmap.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    percent = m / m.sum() * 100
    home_win, draw, away_win = ScorelineModel.outcome_probabilities(m)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    try:
        sns.heatmap(percent, annot=True, fmt=".1f", cmap="YlOrRd",
                    linewidths=0.5, linecolor="white",
                    annot_kws={"size": 9},
                    cbar_kws={"label": "Probability (%)"}, ax=ax)
        ax.set_title(f"Scoreline probabilities (%): {home_team} vs {away_team}\n"
                     f"{home_team} win {home_win:.0%}  |  Draw {draw:.0%}  |  "
                     f"{away_team} win {away_win:.0%}")
        ax.set_xlabel(f"{away_team} goals")
        ax.set_ylabel(f"{home_team} goals")
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
    finally:
        plt.close(fig)
    return save_path