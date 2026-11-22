"""Data layer for KickCast.

Downloads, validates and caches the data the project depends on:
season CSV files for the EPL and La Liga (football-data.co.uk) and the
live Fantasy Premier League (FPL) feed. Every download is saved under
data/raw/ so the program can fall back to the cache when offline.

Planned contents:
    DataFetchError      - custom exception for network failures, bad status
                          codes or a missing cache.
    DataSource          - abstract base class with load() and cache helpers.
    FootballCsvSource   - loads and cleans season CSVs into a DataFrame.
    FplApiSource        - loads the FPL bootstrap-static JSON and exposes
                          injury news and chance of playing.
"""