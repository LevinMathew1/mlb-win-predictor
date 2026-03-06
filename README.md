# MLB Win Predictor

I built this to predict which MLB team is more likely to win any given game using real, live data. It pulls directly from the MLB Stats API (totally free, no API key needed), trains on 4 years of historical games, and cross-references its predictions against dRatings.com to give you a more reliable final answer.

## What it does

- Pulls team stats, standings, and game results from the MLB Stats API
- Builds Elo ratings from 4 seasons of game data (same approach FiveThirtyEight used)
- Trains on ~9,700 games (2022–2025) using starting pitcher stats, Pythagorean win%, OPS, ERA, WHIP, and more
- Grabs the current week's schedule and runs win probability predictions
- Scrapes dRatings.com and compares their numbers against the model — flags where they agree (high confidence) and where they split (bet with caution)

## Files

| File | What it does |
|------|-------------|
| `fetch_data.py` | Pulls game results + team/pitcher stats from the MLB API, builds training CSVs |
| `elo.py` | Computes Elo ratings for all 30 teams across 2022–2025 |
| `train.py` | Trains and compares Logistic Regression, XGBoost, and LightGBM — picks the best one |
| `roster_stats.py` | Builds 2026 team projections using current rosters + 2025 player stats |
| `external_odds.py` | Scrapes dRatings.com and runs the ensemble comparison |
| `predict.py` | Puts everything together — run this to get predictions |

## How to run it

```bash
git clone https://github.com/LevinMathew1/mlb-win-predictor.git
cd mlb-win-predictor

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Grab 4 seasons of training data
python fetch_data.py --seasons 2022 2023 2024 2025

# Train the model
python train.py --seasons 2022 2023 2024 2025

# Run predictions (pulls live schedule + dRatings comparison automatically)
python predict.py
```

Just want predictions without retraining? Running `predict.py` directly will auto-fetch and train if no model exists.

## Options

```bash
# Predict the next 14 days
python predict.py --days 14

# Start from a specific date
python predict.py --date 2026-04-10

# Skip the dRatings scrape, show our model only
python predict.py --no-external

# Retrain on a single season
python fetch_data.py --season 2025
python train.py --seasons 2025
```

## How predictions work

The model pulls 2026 rosters, maps each player to their 2025 individual stats, and builds projected team batting/pitching lines. It then applies a 30% regression toward the league average — so outlier seasons don't inflate predictions. Elo ratings handle the dynamic team strength piece.

For each game it outputs:

- **Ours** — our model's home team win probability
- **dRate** — dRatings.com's probability (when available)
- **FINAL** — ensemble average of both
- **Signal** — STRONG (both agree, big margin), LEAN (both agree, close), or SPLIT (they disagree — proceed with caution)

## Model performance (2022–2025, 5-fold CV)

| Model | AUC | Log-Loss |
|-------|-----|---------|
| Baseline (always pick home) | 0.500 | — |
| Elo only | 0.585 | 0.680 |
| Logistic Regression (full features) | **0.639** | **0.661** |
| LightGBM | 0.612 | 0.673 |
| XGBoost | 0.612 | 0.673 |

For context, Vegas sportsbooks typically sit around 0.62–0.66 AUC on MLB games. Baseball is genuinely hard to predict — even the best teams only win about 60% of their games.

## Features the model uses

- Win% differential, home/away splits, last-10 form, run differential
- Pythagorean win% (strips out luck from the raw win%)
- OPS, runs per game, HR per game
- Team ERA, WHIP, K/9
- Starting pitcher ERA, FIP, WHIP, K/9 (home and away)
- Elo rating differential

## Known limitations

- Early in the season (April), stats are based on small samples so predictions are wider. Things tighten up by late May.
- Starting pitcher stats are season-level averages, not the actual guy taking the mound that day. That would require day-of lineup data.
- dRatings only posts games close to game time, so for future dates you'll see "OUR MODEL ONLY" — that's expected.

## License

MIT
