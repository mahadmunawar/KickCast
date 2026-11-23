# KickCast

KickCast is a menu-driven console application that predicts the result of English Premier League (EPL) and La Liga matches as **home win, draw or away win**. It rebuilds team strength from real match history with its own Elo rating system and rolling-form features, and compares two machine-learning models against two benchmarks: *always pick the home team* and *the bookmakers' odds*.

For any fixture it also shows the probability of every scoreline as a heatmap (Poisson model), and an FPL availability report classifies live injury news as **doubtful** or **out**. There is no "fit" class: fit players have no news, so the classifier never sees one.

Individual project for AL-2001 Programming for Artificial Intelligence (Lab), FAST-NUCES Karachi, Fall 2026. Author: Mahad Munawar (24I-0065). Specialisation track: A (Text / NLP).

> Educational project. Predictions are uncertain estimates, **not betting advice**.

## Quick start

Requires Python 3.13. From a fresh clone, in Windows PowerShell:

```powershell
git clone https://github.com/mahadmunawar/KickCast.git
cd KickCast
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

On macOS or Linux, activate with `source venv/bin/activate` instead.

Run the test suite with:

```powershell
python -m pytest -q
```

## Using the menu

| Option | What it does |
|---|---|
| 1. Update data | Downloads the season CSVs and the FPL feed, and refreshes the local cache |
| 2. Predict a fixture | Pick a league, a home team and an away team. Shows home/draw/away probabilities, the bookmaker's view for comparison, and a scoreline heatmap (optionally saved to `outputs/figures/`) |
| 3. Elo table | Current Elo ratings for a league, with an optional trajectory chart |
| 4. FPL availability report | Classifies injury news for EPL players as doubtful or out, with a per-team summary or a per-player list |
| 5. Model evaluation | Baselines versus Logistic Regression, Random Forest and a goals-only Poisson model on the held-out season, with league-specific, combined and transfer set-ups |
| 0. Quit | |

Input is validated everywhere: bad numbers, unknown team names and network failures are reported in plain language and never crash the program.

## Offline use

Every download is saved under `data/raw/` with a timestamp (this folder is git-ignored). If the network is unavailable, KickCast loads the cached copy and tells you how old it is. A fresh clone has no cache, so a small snapshot of one season (`data/sample/`) is committed and used as a last resort. Results from the snapshot are for demonstration only.

## Project layout

```
main.py            console menu and validated input
src/sources.py     DataSource (ABC), FootballCsvSource, FplApiSource, DataFetchError, cache
src/dataprep.py    cleaning, implied probabilities, combined dataset
src/ratings.py     EloRating, Elo table
src/features.py    FeatureBuilder (leakage-free Elo and rolling-form features)
src/models.py      baselines, MatchPredictor, time-based split, evaluation
src/poisson.py     ScorelineModel (NumPy broadcasting, scoreline matrix)
src/textproc.py    own text cleaner, TF-IDF availability classifier
src/engine.py      KickCastEngine: ties the modules together for the menu
src/plots.py       Figure 1 (Elo), Figure 2 (scoreline heatmap), Figure 3 (calibration)
tests/             pytest suite
data/sample/       offline snapshot
outputs/figures/   saved figures
```

## Method in brief

- **No leakage.** Every rolling feature is shifted by one match, and Elo is read *before* each match is applied. A test checks that a match's features never use that match or later ones.
- **Honest evaluation.** Models train on earlier seasons and are tested on the latest season (2025/26). The test season is never used for tuning.
- **Metrics.** Accuracy and log loss, measured on the same test matches for every method.

## Results

Test season 2025/26, 754 matches (both leagues), trained on 2021/22 to 2024/25 only. The accuracy margin of error is about ±3.6 points (95%).

| Method | Accuracy | Log loss (lower is better) |
|---|---|---|
| Always pick the home team | 45.5% | 1.068 |
| Bookmaker odds | 52.0% | 0.991 |
| Logistic Regression (combined) | 49.7% | 1.015 |
| Random Forest (combined) | 48.9% | 1.020 |
| Poisson, goals only (combined) | 48.1% | 1.018 |

Both machine-learning models are ahead of the home-team baseline by 3 to 4 points, which is about the size of the margin of error, so the evidence is suggestive and not conclusive. Neither reaches the bookmaker, whose log loss is lower in every row of the full table (`Show every league and training set-up` in option 5). Models trained on one league and tested on the other scored below their league-specific versions in all four runs, though the gaps (1.6 to 4.5 points) are within the ±5 point margin for a single league.

Football is highly random. Even the bookmaker is right only about half the time, so small gaps in accuracy are within sampling noise.

The injury-text classifier reaches 100% cross-validated accuracy on 88 players with news (out 70, doubtful 18), against 79.5% for always guessing "out". This is not a sign of understanding: FPL writes its news from fixed templates, so the model recognises those phrases.

## Data sources and terms of use

- **Match results, shots and odds:** [football-data.co.uk](https://www.football-data.co.uk/) season CSVs (e.g. `https://www.football-data.co.uk/mmz4281/2526/E0.csv`). Free for private, non-commercial use. No formal licence is stated, so the data is cited here and not redistributed beyond the small sample snapshot.
- **Injury news:** the unofficial Fantasy Premier League API (`https://fantasy.premierleague.com/api/bootstrap-static/`). It is undocumented and has no formal licence. KickCast uses it read-only, for non-commercial purposes, with very few requests and a local cache.

## Ethics and limitations

- **Gambling harm.** Football betting can be addictive and financially harmful. KickCast gives probabilities for learning and discussion. It does not give betting advice, staking strategies or tips, and its bookmaker figures are used only as a benchmark. If gambling is a problem for you, please seek support.
- **Uncertainty.** A probability of 50% for a home win means the model is unsure, not that the home team will win. The models did not reach the bookmaker's accuracy or log loss, and a wrong prediction is the normal case, not an exception.
- **Bias in the data.** Ratings reflect past results, so they favour established big clubs and are slow to react to new managers, transfers and promoted teams. Only two leagues are covered.
- **Bias in the injury labels.** The fit / doubtful / out labels are *derived from* FPL's numeric chance-of-playing field, so the text classifier learns to imitate that field rather than an independent medical judgement. Players who have left their club also show 0%, so they are excluded to avoid mislabelling them as injured. Treat the availability report as a rough guide.
- **Privacy.** The data is about teams and matches. The FPL feed names players, but only publicly available availability information is used and nothing personal is stored.
- **AI assistance.** AI tools were used while developing this project and are declared in the report.

## Licence and acknowledgements

Coursework project. Thanks to football-data.co.uk for the free data.