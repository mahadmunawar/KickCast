"""Match outcome models and evaluation for KickCast.

Trains and compares two scikit-learn models (Logistic Regression and
Random Forest) against two baselines (always pick the home team, and the
bookmakers' odds). Uses a time-based split: train on earlier seasons,
test on the latest completed season. Reports accuracy and log loss, for
per-league and combined configurations.

Planned contents:
    MatchPredictor - fit(), predict_proba(), evaluate(), compare().
    Baseline helpers for the home-team and bookmaker predictions.
"""