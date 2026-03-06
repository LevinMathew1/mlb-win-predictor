"""
Fetch MLB game results and team stats from the free MLB Stats API.
Builds a training CSV from historical season data.

Usage:
    python fetch_data.py              # builds 2025 season dataset
    python fetch_data.py --season 2024
"""

import argparse
import os
import time

import pandas as pd
import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"


def get(endpoint: str, params: dict = None) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_team_batting_stats(season: int) -> dict:
    """Return {team_id: batting_stat_dict} for the given season."""
    data = get("teams/stats", {
        "stats": "season",
        "group": "hitting",
        "season": season,
        "sportId": 1,
    })
    result = {}
    for split in data.get("stats", [{}])[0].get("splits", []):
        tid = split["team"]["id"]
        s = split["stat"]
        gp = max(int(s.get("gamesPlayed", 1)), 1)
        result[tid] = {
            "team_name": split["team"]["name"],
            "bat_avg":        float(s.get("avg", 0)),
            "bat_obp":        float(s.get("obp", 0)),
            "bat_slg":        float(s.get("slg", 0)),
            "bat_ops":        float(s.get("ops", 0)),
            "runs_per_game":  float(s.get("runs", 0)) / gp,
            "hr_per_game":    float(s.get("homeRuns", 0)) / gp,
            "k_per_game":     float(s.get("strikeOuts", 0)) / gp,
        }
    return result


def fetch_team_pitching_stats(season: int) -> dict:
    """Return {team_id: pitching_stat_dict} for the given season."""
    data = get("teams/stats", {
        "stats": "season",
        "group": "pitching",
        "season": season,
        "sportId": 1,
    })
    result = {}
    for split in data.get("stats", [{}])[0].get("splits", []):
        tid = split["team"]["id"]
        s = split["stat"]
        gp = max(int(s.get("gamesPlayed", 1)), 1)
        result[tid] = {
            "era":                   float(s.get("era", 4.50)),
            "whip":                  float(s.get("whip", 1.30)),
            "k9":                    float(s.get("strikeoutsPer9Inn", 8.0)),
            "runs_allowed_per_game": float(s.get("runs", 0)) / gp,
        }
    return result


def fetch_standings(season: int) -> dict:
    """Return {team_id: standings_dict} with win%, home%, away%, last-10%."""
    data = get("standings", {"leagueId": "103,104", "season": season})

    def pct(rec):
        w, l = rec.get("wins", 0), rec.get("losses", 0)
        return w / max(w + l, 1)

    result = {}
    for division in data.get("records", []):
        for tr in division.get("teamRecords", []):
            tid = tr["team"]["id"]
            splits = {r["type"]: r for r in tr.get("records", {}).get("splitRecords", [])}
            result[tid] = {
                "win_pct":       pct(tr),
                "home_win_pct":  pct(splits.get("home", {})),
                "away_win_pct":  pct(splits.get("away", {})),
                "last10_win_pct": pct(splits.get("lastTen", {})),
                "run_diff":      tr.get("runDifferential", 0),
            }
    return result


def fetch_completed_games(season: int) -> list[dict]:
    """Return list of completed regular-season games with home/away IDs and result."""
    data = get("schedule", {
        "sportId": 1,
        "season": season,
        "gameType": "R",
        "startDate": f"{season}-03-01",
        "endDate":   f"{season}-10-15",
        "hydrate":   "linescore",
        "fields":    (
            "dates,date,games,gamePk,status,statusCode,"
            "teams,home,away,team,id,isWinner,score"
        ),
    })

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("statusCode") != "F":
                continue
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            games.append({
                "game_pk":      game["gamePk"],
                "date":         date_entry["date"],
                "home_team_id": home["team"]["id"],
                "away_team_id": away["team"]["id"],
                "home_score":   home.get("score", 0),
                "away_score":   away.get("score", 0),
                "home_win":     int(home.get("isWinner", False)),
            })
    return games


def build_features(home_stats: dict, away_stats: dict,
                   home_stand: dict, away_stand: dict) -> dict:
    """Combine raw team stats into model features (differentials + absolutes)."""
    return {
        # Win-rate features
        "win_pct_diff":       home_stand.get("win_pct", 0.5)        - away_stand.get("win_pct", 0.5),
        "home_win_pct":       home_stand.get("home_win_pct", 0.5),
        "away_win_pct_road":  away_stand.get("away_win_pct", 0.5),
        "last10_diff":        home_stand.get("last10_win_pct", 0.5) - away_stand.get("last10_win_pct", 0.5),
        "run_diff_diff":      home_stand.get("run_diff", 0)          - away_stand.get("run_diff", 0),
        # Batting differentials
        "ops_diff":           home_stats.get("bat_ops", 0)           - away_stats.get("bat_ops", 0),
        "runs_pg_diff":       home_stats.get("runs_per_game", 0)     - away_stats.get("runs_per_game", 0),
        "hr_pg_diff":         home_stats.get("hr_per_game", 0)       - away_stats.get("hr_per_game", 0),
        # Pitching differentials (lower ERA/WHIP is better, so positive = home disadvantage)
        "era_diff":           home_stats.get("era", 4.5)             - away_stats.get("era", 4.5),
        "whip_diff":          home_stats.get("whip", 1.3)            - away_stats.get("whip", 1.3),
        "k9_diff":            home_stats.get("k9", 8.0)              - away_stats.get("k9", 8.0),
        # Absolute home stats
        "home_era":           home_stats.get("era", 4.5),
        "home_ops":           home_stats.get("bat_ops", 0),
        "home_win_pct_abs":   home_stand.get("win_pct", 0.5),
        # Absolute away stats
        "away_era":           away_stats.get("era", 4.5),
        "away_ops":           away_stats.get("bat_ops", 0),
        "away_win_pct_abs":   away_stand.get("win_pct", 0.5),
    }


def build_dataset(season: int = 2025) -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)

    print(f"[1/4] Fetching {season} batting stats...")
    batting  = fetch_team_batting_stats(season)
    time.sleep(0.3)

    print(f"[2/4] Fetching {season} pitching stats...")
    pitching = fetch_team_pitching_stats(season)
    time.sleep(0.3)

    # Merge batting + pitching into one dict
    team_stats = {}
    for tid, b in batting.items():
        team_stats[tid] = {**b, **pitching.get(tid, {})}

    print(f"[3/4] Fetching {season} standings...")
    standings = fetch_standings(season)
    time.sleep(0.3)

    print(f"[4/4] Fetching {season} completed games (this takes ~20s)...")
    games = fetch_completed_games(season)
    print(f"      Found {len(games)} completed games.")

    rows = []
    for g in games:
        htid, atid = g["home_team_id"], g["away_team_id"]
        if htid not in team_stats or atid not in team_stats:
            continue
        features = build_features(
            team_stats[htid], team_stats[atid],
            standings.get(htid, {}), standings.get(atid, {}),
        )
        row = {
            "game_pk":   g["game_pk"],
            "date":      g["date"],
            "home_team": team_stats[htid]["team_name"],
            "away_team": team_stats[atid]["team_name"],
            **features,
            "home_win":  g["home_win"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = f"data/mlb_games_{season}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows -> {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025,
                        help="Season to fetch (default: 2025)")
    args = parser.parse_args()
    build_dataset(season=args.season)
