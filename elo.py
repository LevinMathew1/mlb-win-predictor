"""
Builds Elo ratings for all 30 MLB teams by processing 4 seasons of game
results chronologically. Same basic approach FiveThirtyEight used for their
MLB predictions — teams gain/lose points after every game, and ratings get
pulled back toward 1500 each offseason to account for roster changes.

Usage:
    python elo.py
    from elo import build_elo_ratings, elo_win_prob
"""

import json
import os

import pandas as pd

K_FACTOR     = 5.0    # rating change per game
HOME_ADV     = 30.0   # Elo points of home field advantage
OFFSEASON_REG = 0.25  # fraction regressed toward 1500 each offseason
BASE_RATING  = 1500.0


def elo_win_prob(home_elo: float, away_elo: float) -> float:
    """Expected win probability for the HOME team given Elo ratings."""
    return 1.0 / (1.0 + 10 ** ((away_elo - home_elo - HOME_ADV) / 400))


def _regress(ratings: dict) -> dict:
    return {
        team: OFFSEASON_REG * BASE_RATING + (1 - OFFSEASON_REG) * elo
        for team, elo in ratings.items()
    }


def build_elo_ratings(seasons: list[int] = None,
                      save: bool = True) -> tuple[dict, pd.DataFrame]:
    """
    Process game CSVs chronologically to build Elo ratings.

    Returns
    -------
    final_ratings : dict  {team_name: elo_rating}
        Ratings at the END of the last season, with one round of offseason
        regression applied — ready to use for next-season predictions.
    game_elos : pd.DataFrame
        Pre-game elo_diff and elo_home_prob for every historical game,
        keyed on game_pk. Merge into training data for richer features.
    """
    seasons = seasons or [2022, 2023, 2024, 2025]
    ratings: dict[str, float] = {}
    game_records = []

    for i, season in enumerate(seasons):
        csv_path = f"data/mlb_games_{season}.csv"
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found — skipping {season}")
            continue

        # Apply offseason regression between seasons
        if i > 0 and ratings:
            ratings = _regress(ratings)

        df = pd.read_csv(csv_path).sort_values("date").reset_index(drop=True)

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            if home not in ratings:
                ratings[home] = BASE_RATING
            if away not in ratings:
                ratings[away] = BASE_RATING

            home_elo = ratings[home]
            away_elo = ratings[away]
            exp      = elo_win_prob(home_elo, away_elo)
            actual   = float(row["home_win"])

            game_records.append({
                "game_pk":       int(row["game_pk"]),
                "elo_diff":      round(home_elo - away_elo, 2),
                "elo_home_prob": round(exp, 4),
            })

            ratings[home] = home_elo + K_FACTOR * (actual - exp)
            ratings[away] = away_elo + K_FACTOR * ((1 - actual) - (1 - exp))

    # Final offseason regression → these are the 2026 projection ratings
    final_ratings = _regress(ratings)
    game_elos_df  = pd.DataFrame(game_records)

    if save:
        os.makedirs("models", exist_ok=True)
        os.makedirs("data",   exist_ok=True)
        with open("models/elo_ratings.json", "w") as f:
            json.dump(final_ratings, f, indent=2)
        game_elos_df.to_csv("data/game_elos.csv", index=False)

        top5 = sorted(final_ratings.items(), key=lambda x: -x[1])[:5]
        bot5 = sorted(final_ratings.items(), key=lambda x:  x[1])[:5]
        print("  Top-5 Elo entering 2026:")
        for team, elo in top5:
            print(f"    {team:<30} {elo:.1f}")
        print("  Bottom-5 Elo entering 2026:")
        for team, elo in bot5:
            print(f"    {team:<30} {elo:.1f}")

    return final_ratings, game_elos_df


if __name__ == "__main__":
    print("Building Elo ratings from 2022-2025 game data...")
    ratings, _ = build_elo_ratings()
    print(f"\nAll 30 team Elo ratings (entering 2026):")
    for team, elo in sorted(ratings.items(), key=lambda x: -x[1]):
        bar = "#" * int((elo - 1450) / 5)
        print(f"  {team:<30} {elo:6.1f}  {bar}")
