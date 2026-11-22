"""Visualisations for KickCast.

Produces the three labelled figures used in the evaluation report:
    1. Elo rating trajectories of selected teams over time.
    2. Seaborn heatmap of scoreline probabilities for a fixture. (added later)
    3. Calibration chart comparing the model with the bookmaker.

Figures are saved to outputs/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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