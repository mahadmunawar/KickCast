"""Leakage-free feature engineering for KickCast.

Builds the model inputs (Elo before the match, rolling form, goals and
shots) using ONLY matches played before each fixture. Every rolling
window is shifted by one match so the current match never leaks into
its own features.

Planned contents:
    FeatureBuilder - build(matches) returns a feature DataFrame.
"""