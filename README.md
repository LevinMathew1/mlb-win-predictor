# MLB Win Predictor

Predicts the probability of each MLB team winning their upcoming games using
live data from the **free MLB Stats API** (no API key required).

Trained on the full 2025 regular season (~2,430 games) and updated with
current 2026 season stats as they accumulate.

## How it works

1. **`fetch_data.py`** — pulls completed game results + team batting/pitching
   stats from `statsapi.mlb.com` and builds a training CSV.
2. **`train.py`** — trains a calibrated XGBoost classifier on team stat
   differentials (win%, OPS, ERA, WHIP, last-10 form, run differential, etc.)
   and compares against Logistic Regression baseline.
3. **`predict.py`** — fetches this week's schedule, grabs current 2026 season
   stats for each team, and outputs win probabilities in a formatted table.

## Features used

| Category | Features |
|----------|----------|
| Win rate | Win%, home win%, away win%, last-10 win%, run differential |
| Batting  | OPS, runs/game, HR/game (differentials) |
| Pitching | ERA, WHIP, K/9 (differentials) |

## Quickstart

```bash
# 1. Clone & set up environment
git clone https://github.com/LevinMathew1/mlb-win-predictor.git
cd mlb-win-predictor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Fetch 2025 season training data (~20 seconds)
python fetch_data.py

# 3. Train the model
python train.py

# 4. Predict this week's games (auto-fetches live 2026 data)
python predict.py
```

`predict.py` will **auto-run steps 2 and 3** if no model is found, so you can
also just run `python predict.py` directly.

## Options

```bash
# Predict next 14 days
python predict.py --days 14

# Predict from a specific date
python predict.py --date 2026-04-10

# Train on a different season
python fetch_data.py --season 2024
python train.py --data data/mlb_games_2024.csv
```

## Example output

```
===========================================================================
  MLB WIN PROBABILITY PREDICTIONS
  Model: XGBoost  |  CV AUC: 0.621  |  Home win rate: 54.1%
===========================================================================

  Thursday, April 03 2026
  Away Team                    Home Team                    Away%  Home%  Result
  ---------------------------- ---------------------------- ------ ------  ------
  New York Yankees             Boston Red Sox               41.2%  58.8%  Scheduled
  Los Angeles Dodgers          San Francisco Giants         55.3%  44.7%  Scheduled
  Houston Astros               Texas Rangers                48.1%  51.9%  Scheduled
```

## Model performance (2025 season, 5-fold CV)

| Metric | XGBoost | Logistic Regression |
|--------|---------|---------------------|
| AUC    | ~0.62   | ~0.60               |
| Log-Loss | ~0.68 | ~0.69               |

> **Note:** MLB is notoriously hard to predict — even Vegas lines rarely exceed
> 0.65 AUC. A ~0.62 AUC reflects genuine predictive signal, not a bug.

## Limitations & future improvements

- **Small-sample early season**: predictions are less reliable in April when
  teams have played fewer games. Stats stabilize by late May.
- **No starting pitcher data**: adding that day's starter ERA would
  significantly improve accuracy.
- **No data leakage guard**: training uses full-season stats, not
  rolling/in-game stats. A proper implementation would use stats accumulated
  only up to each game's date.

## License

MIT
