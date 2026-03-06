"""
Fetch this week's MLB schedule and predict win probabilities.

Automatically trains the model if no saved model is found.

Usage:
    python predict.py                    # predictions for the next 7 days
    python predict.py --days 14          # next 14 days
    python predict.py --date 2026-04-05  # start from a specific date
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"


def get(endpoint: str, params: dict = None) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def load_model():
    if not os.path.exists("models/model.pkl"):
        print("No trained model found. Running training pipeline...")
        # Fetch 2025 data if not already present
        if not os.path.exists("data/mlb_games_2025.csv"):
            subprocess.run([sys.executable, "fetch_data.py", "--season", "2025"], check=True)
        subprocess.run([sys.executable, "train.py"], check=True)

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open("models/metadata.json") as f:
        metadata = json.load(f)
    return model, feature_cols, metadata


def fetch_week_schedule(start_date: str, days: int = 7) -> list[dict]:
    """Return scheduled/live/final games in the date window."""
    end = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=days - 1)).strftime("%Y-%m-%d")
    data = get("schedule", {
        "sportId": 1,
        "gameType": "R",
        "startDate": start_date,
        "endDate": end,
        "hydrate": "linescore",
        "fields": (
            "dates,date,games,gamePk,status,statusCode,abstractGameState,"
            "teams,home,away,team,id,name,isWinner,score"
        ),
    })

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            status = game.get("status", {})
            games.append({
                "game_pk":      game["gamePk"],
                "date":         date_entry["date"],
                "state":        status.get("abstractGameState", "Preview"),
                "status_code":  status.get("statusCode", ""),
                "home_team_id": home["team"]["id"],
                "home_team":    home["team"]["name"],
                "away_team_id": away["team"]["id"],
                "away_team":    away["team"]["name"],
                "home_score":   home.get("score"),
                "away_score":   away.get("score"),
                "home_won":     home.get("isWinner"),
            })
    return games


def fetch_current_team_stats(team_ids: list[int]) -> dict:
    """
    Build team stat projections from 2026 rosters + 2025 individual player stats.
    This accounts for offseason trades, free agent signings, and roster changes.
    """
    from roster_stats import build_roster_projections
    print("Building roster-aware projections (2026 rosters + 2025 player stats)...")
    team_stats, standings = build_roster_projections(
        roster_season=2026, stats_season=2025, verbose=True
    )
    return team_stats, standings


def build_features_for_game(htid: int, atid: int,
                              team_stats: dict, standings: dict) -> dict:
    from fetch_data import build_features
    hs  = team_stats.get(htid, {})
    as_ = team_stats.get(atid, {})
    hst = standings.get(htid, {})
    ast = standings.get(atid, {})
    return build_features(hs, as_, hst, ast)


def print_predictions(results: list[dict], metadata: dict) -> None:
    print("\n" + "=" * 75)
    print(f"  MLB WIN PROBABILITY PREDICTIONS")
    print(f"  Model: {metadata['model_name']}  |  CV AUC: {metadata['cv_auc']:.3f}  |  "
          f"Home win rate: {metadata['home_win_rate']:.1%}")
    print("=" * 75)

    current_date = None
    for r in results:
        if r["date"] != current_date:
            current_date = r["date"]
            dt = datetime.strptime(current_date, "%Y-%m-%d")
            print(f"\n  {dt.strftime('%A, %B %d %Y')}")
            print(f"  {'Away Team':<28} {'Home Team':<28} {'Away%':>6} {'Home%':>6}  Result")
            print(f"  {'-'*28} {'-'*28} {'-'*6} {'-'*6}  ------")

        away_pct = r["away_win_prob"] * 100
        home_pct = r["home_win_prob"] * 100

        if r["state"] == "Final":
            if r["home_won"]:
                result_str = f"HOME WON  {r['away_score']}-{r['home_score']}"
            else:
                result_str = f"AWAY WON  {r['away_score']}-{r['home_score']}"
        elif r["state"] == "Live":
            result_str = f"LIVE  {r['away_score']}-{r['home_score']}"
        else:
            result_str = "Scheduled"

        print(f"  {r['away_team']:<28} {r['home_team']:<28} {away_pct:>5.1f}% {home_pct:>5.1f}%  {result_str}")

    print("\n" + "=" * 75)
    print("  Note: Probabilities use season-to-date team stats. Early in the")
    print("  season, predictions are less reliable due to small sample sizes.")
    print("=" * 75 + "\n")


def predict(start_date: str = None, days: int = 7) -> None:
    if start_date is None:
        start_date = datetime.today().strftime("%Y-%m-%d")

    print(f"Loading model...")
    model, feature_cols, metadata = load_model()

    print(f"Fetching schedule: {start_date} + {days} days...")
    games = fetch_week_schedule(start_date, days)

    if not games:
        print(f"No MLB games found between {start_date} and the next {days} days.")
        print("The 2026 regular season typically starts in late March.")
        return

    print(f"Found {len(games)} games. Fetching current team stats...")
    all_team_ids = list({g["home_team_id"] for g in games} | {g["away_team_id"] for g in games})
    team_stats, standings = fetch_current_team_stats(all_team_ids)
    time.sleep(0.3)

    results = []
    skipped = 0
    for g in games:
        htid, atid = g["home_team_id"], g["away_team_id"]
        if htid not in team_stats or atid not in team_stats:
            skipped += 1
            continue

        features = build_features_for_game(htid, atid, team_stats, standings)
        X = [[features.get(col, 0) for col in feature_cols]]

        probs = model.predict_proba(X)[0]  # [prob_away_win, prob_home_win]
        results.append({
            **g,
            "home_win_prob": float(probs[1]),
            "away_win_prob": float(probs[0]),
        })

    if skipped:
        print(f"  ({skipped} games skipped — missing team stats)")

    if not results:
        print("No predictions could be generated.")
        return

    # Sort by date then game_pk
    results.sort(key=lambda x: (x["date"], x["game_pk"]))
    print_predictions(results, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict MLB game outcomes for the current week")
    parser.add_argument("--date", default=None,
                        help="Start date YYYY-MM-DD (default: today)")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days to look ahead (default: 7)")
    args = parser.parse_args()
    predict(start_date=args.date, days=args.days)
