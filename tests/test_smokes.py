"""Smoke test: confirms the environment and required libraries are installed."""


def test_imports():
    """All required third-party libraries should import without errors."""
    import pandas, numpy, matplotlib, seaborn, sklearn, requests
    assert True