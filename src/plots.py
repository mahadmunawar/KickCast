"""Visualisations for KickCast.

Produces the three labelled figures used in the evaluation report:
    1. Elo rating trajectories of selected teams over time.   (this file, now)
    2. Seaborn heatmap of scoreline probabilities for a fixture. (added later)
    3. Calibration chart comparing the model with the bookmaker. (added later)

Figures are saved to outputs/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt

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