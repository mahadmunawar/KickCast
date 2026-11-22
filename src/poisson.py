"""Poisson scoreline model for KickCast.

Estimates each team's attack and defence strength and turns them into a
2-D matrix of scoreline probabilities using NumPy broadcasting (outer
product of two Poisson probability vectors). Home win, draw and away win
probabilities are obtained by summing the lower triangle, diagonal and
upper triangle of the matrix, with no element loops.

Planned contents:
    ScorelineModel - scoreline_matrix(home, away), outcome_probabilities().
"""