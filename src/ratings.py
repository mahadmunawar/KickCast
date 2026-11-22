"""Elo rating system for KickCast.

Rebuilds every team's strength from real match history. Ratings start
equal and are updated match by match, with a home-advantage term and a
K-factor controlling how fast ratings move.

Planned contents:
    EloRating - keeps ratings (team -> float) and provides expected_score(),
                update(), history() and table().
"""