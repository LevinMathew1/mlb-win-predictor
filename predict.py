"""
Main prediction script. Grabs the upcoming MLB schedule, runs win
probability predictions, then cross-references against dRatings.com
and outputs an ensemble with signal strength for each game.

Usage:
    python predict.py                    # next 7 days
    python predict.py --days 14
    python predict.py --date 2026-04-05
    python predict.py --no-external      # skip dRatings, show our model only
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
    """Return scheduled/live/final games in the date window, including probable starter IDs."""
    end = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=days - 1)).strftime("%Y-%m-%d")
    data = get("schedule", {
        "sportId": 1,
        "gameType": "R",
        "startDate": start_date,
        "endDate": end,
        "hydrate": "linescore,probablePitcher",
        "fields": (
            "dates,date,games,gamePk,status,statusCode,abstractGameState,"
            "teams,home,away,team,id,name,isWinner,score,probablePitcher"
        ),
    })

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            status = game.get("status", {})
            games.append({
                "game_pk":         game["gamePk"],
                "date":            date_entry["date"],
                "state":           status.get("abstractGameState", "Preview"),
                "status_code":     status.get("statusCode", ""),
                "home_team_id":    home["team"]["id"],
                "home_team":       home["team"]["name"],
                "away_team_id":    away["team"]["id"],
                "away_team":       away["team"]["name"],
                "home_score":      home.get("score"),
                "away_score":      away.get("score"),
                "home_won":        home.get("isWinner"),
                "home_pitcher_id": home.get("probablePitcher", {}).get("id"),
                "away_pitcher_id": away.get("probablePitcher", {}).get("id"),
            })
    return games


# League averages used for regression-to-mean shrinkage
_LEAGUE_AVG = {
    "bat_avg": 0.248, "bat_obp": 0.317, "bat_slg": 0.401, "bat_ops": 0.718,
    "runs_per_game": 4.60, "hr_per_game": 1.05, "k_per_game": 8.80,
    "era": 4.20, "whip": 1.28, "k9": 8.90, "runs_allowed_per_game": 4.60,
}
_REGRESSION_FACTOR = 0.30  # blend 30% toward league average


def _regress_to_mean(team_stats: dict) -> dict:
    """
    Shrink each team's stats 30% toward the league average.
    Prevents extreme predictions caused by small-sample or outlier seasons.
    This is standard in sports analytics (Marcel projections, PECOTA, etc.).
    """
    regressed = {}
    for tid, stats in team_stats.items():
        new_stats = dict(stats)
        for key, lg_avg in _LEAGUE_AVG.items():
            if key in stats:
                new_stats[key] = (1 - _REGRESSION_FACTOR) * stats[key] + _REGRESSION_FACTOR * lg_avg
        regressed[tid] = new_stats
    return regressed


def fetch_current_team_stats(team_ids: list[int]) -> tuple:
    """
    Build team stat projections from 2026 rosters + 2025 individual player stats,
    with regression-to-mean shrinkage to prevent overconfident predictions.
    """
    from roster_stats import build_roster_projections
    from fetch_data import fetch_all_player_pitching_stats

    print("Building roster-aware projections (2026 rosters + 2025 player stats)...")
    team_stats, standings = build_roster_projections(
        roster_season=2026, stats_season=2025, verbose=True
    )
    print("Applying 30% regression-to-mean shrinkage...")
    team_stats = _regress_to_mean(team_stats)

    print("Fetching 2025 player pitching stats for probable starter lookup...")
    player_pitching = fetch_all_player_pitching_stats(2025)
    return team_stats, standings, player_pitching


def build_features_for_game(htid: int, atid: int,
                              team_stats: dict, standings: dict,
                              home_pitcher_id: int = None,
                              away_pitcher_id: int = None,
                              player_pitching: dict = None) -> dict:
    from fetch_data import build_features, pitcher_features
    hs  = team_stats.get(htid, {})
    as_ = team_stats.get(atid, {})
    hst = standings.get(htid, {})
    ast = standings.get(atid, {})

    pp = player_pitching or {}
    home_sp = pitcher_features(pp.get(home_pitcher_id)) if home_pitcher_id else None
    away_sp = pitcher_features(pp.get(away_pitcher_id)) if away_pitcher_id else None

    return build_features(hs, as_, hst, ast, home_sp=home_sp, away_sp=away_sp)


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
    team_stats, standings, player_pitching = fetch_current_team_stats(all_team_ids)
    time.sleep(0.3)

    # Load Elo ratings for dynamic team-strength feature
    elo_ratings = {}
    elo_path = "models/elo_ratings.json"
    if os.path.exists(elo_path):
        with open(elo_path) as f:
            elo_ratings = json.load(f)
    else:
        print("  Building Elo ratings (first run)...")
        from elo import build_elo_ratings
        elo_ratings, _ = build_elo_ratings()

    from elo import elo_win_prob

    sp_found = sum(1 for g in games if g.get("home_pitcher_id") or g.get("away_pitcher_id"))
    print(f"  Probable starters announced for {sp_found}/{len(games)} games.")

    results = []
    skipped = 0
    for g in games:
        htid, atid = g["home_team_id"], g["away_team_id"]
        if htid not in team_stats or atid not in team_stats:
            skipped += 1
            continue

        features = build_features_for_game(
            htid, atid, team_stats, standings,
            home_pitcher_id=g.get("home_pitcher_id"),
            away_pitcher_id=g.get("away_pitcher_id"),
            player_pitching=player_pitching,
        )

        # Add Elo features
        home_name = g["home_team"]
        away_name = g["away_team"]
        home_elo  = elo_ratings.get(home_name, 1500.0)
        away_elo  = elo_ratings.get(away_name, 1500.0)
        features["elo_diff"]      = home_elo - away_elo
        features["elo_home_prob"] = elo_win_prob(home_elo, away_elo)

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

    # --- Fetch dRatings and build ensemble ---
    print("Fetching dRatings.com predictions for comparison...")
    from external_odds import fetch_dratings, build_comparison_table, print_comparison
    dratings = fetch_dratings()

    if dratings:
        matched = sum(1 for g in results
                      if (g["away_team"].lower(), g["home_team"].lower()) in
                      {(d["away_team"].lower(), d["home_team"].lower()) for d in dratings})
        print(f"  Matched {matched}/{len(results)} games with dRatings data.")
        combined = build_comparison_table(results, dratings)
        print_comparison(combined, metadata)
    else:
        print("  dRatings unavailable — showing our model only.")
        print_predictions(results, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict MLB game outcomes for the current week")
    parser.add_argument("--date", default=None,
                        help="Start date YYYY-MM-DD (default: today)")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days to look ahead (default: 7)")
    parser.add_argument("--no-external", action="store_true",
                        help="Skip dRatings scraping, show our model only")
    args = parser.parse_args()
    predict(start_date=args.date, days=args.days)
