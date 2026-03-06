"""
Pulls game results, team stats, standings, and starting pitcher data
from the MLB Stats API and saves it as a CSV for training.

Usage:
    python fetch_data.py                          # grabs 2025 season
    python fetch_data.py --season 2024
    python fetch_data.py --seasons 2022 2023 2024 2025
"""

import argparse
import os
import time

import pandas as pd
import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"

FIP_CONSTANT = 3.10  # MLB league-average FIP constant

# Fallback when a game has no probable pitcher on record
LEAGUE_AVG_SP = {"sp_era": 4.20, "sp_fip": 4.10, "sp_whip": 1.28, "sp_k9": 8.50}


def get(endpoint: str, params: dict = None) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def parse_ip(ip_str) -> float:
    """Convert MLB innings-pitched format ('45.2' = 45 IP + 2 outs) to true innings."""
    try:
        parts = str(ip_str).split(".")
        return int(parts[0]) + (int(parts[1]) if len(parts) > 1 else 0) / 3
    except Exception:
        return 0.0


def pitcher_features(stat: dict) -> dict:
    """
    Compute ERA, FIP, WHIP, and K/9 from a raw API pitching stat dict.
    Returns league-average values when the sample is too small (<10 IP).
    """
    if not stat:
        return LEAGUE_AVG_SP.copy()
    ip = parse_ip(stat.get("inningsPitched", "0"))
    if ip < 10:
        return LEAGUE_AVG_SP.copy()
    hr  = int(stat.get("homeRuns", 0))
    bb  = int(stat.get("baseOnBalls", 0))
    hbp = int(stat.get("hitByPitch", 0))
    so  = int(stat.get("strikeOuts", 0))
    h   = int(stat.get("hits", 0))
    er  = int(stat.get("earnedRuns", 0))
    return {
        "sp_era":  round(9 * er / ip, 2),
        "sp_fip":  round(((13*hr + 3*(bb+hbp) - 2*so) / ip) + FIP_CONSTANT, 2),
        "sp_whip": round((h + bb) / ip, 3),
        "sp_k9":   round(9 * so / ip, 2),
    }


def fetch_team_batting_stats(season: int) -> dict:
    """Return {team_id: batting_stat_dict} for the given season."""
    data = get("teams/stats", {
        "stats": "season", "group": "hitting", "season": season, "sportId": 1,
    })
    result = {}
    for split in data.get("stats", [{}])[0].get("splits", []):
        tid = split["team"]["id"]
        s = split["stat"]
        gp = max(int(s.get("gamesPlayed", 1)), 1)
        result[tid] = {
            "team_name":     split["team"]["name"],
            "bat_avg":       float(s.get("avg", 0)),
            "bat_obp":       float(s.get("obp", 0)),
            "bat_slg":       float(s.get("slg", 0)),
            "bat_ops":       float(s.get("ops", 0)),
            "runs_per_game": float(s.get("runs", 0)) / gp,
            "hr_per_game":   float(s.get("homeRuns", 0)) / gp,
            "k_per_game":    float(s.get("strikeOuts", 0)) / gp,
        }
    return result


def fetch_team_pitching_stats(season: int) -> dict:
    """Return {team_id: pitching_stat_dict} for the given season."""
    data = get("teams/stats", {
        "stats": "season", "group": "pitching", "season": season, "sportId": 1,
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
    """Return {team_id: standings_dict} with win%, home%, away%, last-10%, and Pythagorean W%."""
    data = get("standings", {"leagueId": "103,104", "season": season})

    def pct(rec):
        w, l = rec.get("wins", 0), rec.get("losses", 0)
        return w / max(w + l, 1)

    result = {}
    for division in data.get("records", []):
        for tr in division.get("teamRecords", []):
            tid = tr["team"]["id"]
            splits = {r["type"]: r for r in tr.get("records", {}).get("splitRecords", [])}
            rs = tr.get("runsScored", 0)
            ra = tr.get("runsAllowed", 1)
            # Pythagorean expectation: truer measure of team quality than raw W%
            pyth = rs ** 1.83 / max(rs ** 1.83 + ra ** 1.83, 1)
            result[tid] = {
                "win_pct":        pct(tr),
                "home_win_pct":   pct(splits.get("home", {})),
                "away_win_pct":   pct(splits.get("away", {})),
                "last10_win_pct": pct(splits.get("lastTen", {})),
                "run_diff":       tr.get("runDifferential", 0),
                "pyth_win_pct":   round(pyth, 4),
            }
    return result


def fetch_all_player_pitching_stats(season: int) -> dict:
    """
    Bulk fetch pitching stats for all players in one API call.
    Returns {player_id: raw_stat_dict}.
    """
    data = get("stats", {
        "stats": "season", "group": "pitching",
        "season": season, "sportId": 1, "limit": 2000,
    })
    result = {}
    for split in data.get("stats", [{}])[0].get("splits", []):
        pid = split["player"]["id"]
        result[pid] = split["stat"]
    return result


def fetch_probable_pitchers(season: int) -> dict:
    """
    Return {game_pk: {"home": player_id_or_None, "away": player_id_or_None}}
    for all regular-season games (probable pitchers are stored even for completed games).
    """
    data = get("schedule", {
        "sportId": 1,
        "season": season,
        "gameType": "R",
        "startDate": f"{season}-03-01",
        "endDate":   f"{season}-10-15",
        "hydrate":   "probablePitcher",
        "fields":    "dates,games,gamePk,teams,home,away,probablePitcher,id",
    })
    result = {}
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            pk = game["gamePk"]
            home_pid = game["teams"]["home"].get("probablePitcher", {}).get("id")
            away_pid = game["teams"]["away"].get("probablePitcher", {}).get("id")
            result[pk] = {"home": home_pid, "away": away_pid}
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
                   home_stand: dict, away_stand: dict,
                   home_sp: dict = None, away_sp: dict = None) -> dict:
    """
    Combine raw team stats + starting pitcher stats into model features.
    home_sp / away_sp are dicts with keys: sp_era, sp_fip, sp_whip, sp_k9.
    Falls back to league average when not provided.
    """
    hsp = home_sp or LEAGUE_AVG_SP
    asp = away_sp or LEAGUE_AVG_SP

    return {
        # --- Team win-rate features ---
        "win_pct_diff":      home_stand.get("win_pct", 0.5)        - away_stand.get("win_pct", 0.5),
        "home_win_pct":      home_stand.get("home_win_pct", 0.5),
        "away_win_pct_road": away_stand.get("away_win_pct", 0.5),
        "last10_diff":       home_stand.get("last10_win_pct", 0.5) - away_stand.get("last10_win_pct", 0.5),
        "run_diff_diff":     home_stand.get("run_diff", 0)          - away_stand.get("run_diff", 0),
        # --- Pythagorean expectation (truer measure of quality than raw W%) ---
        "pyth_diff":         home_stand.get("pyth_win_pct", 0.5)   - away_stand.get("pyth_win_pct", 0.5),
        "home_pyth":         home_stand.get("pyth_win_pct", 0.5),
        "away_pyth":         away_stand.get("pyth_win_pct", 0.5),
        # --- Team batting differentials ---
        "ops_diff":          home_stats.get("bat_ops", 0)           - away_stats.get("bat_ops", 0),
        "runs_pg_diff":      home_stats.get("runs_per_game", 0)     - away_stats.get("runs_per_game", 0),
        "hr_pg_diff":        home_stats.get("hr_per_game", 0)       - away_stats.get("hr_per_game", 0),
        # --- Team pitching differentials ---
        "era_diff":          home_stats.get("era", 4.5)             - away_stats.get("era", 4.5),
        "whip_diff":         home_stats.get("whip", 1.3)            - away_stats.get("whip", 1.3),
        "k9_diff":           home_stats.get("k9", 8.0)              - away_stats.get("k9", 8.0),
        # --- Absolute team stats ---
        "home_era":          home_stats.get("era", 4.5),
        "home_ops":          home_stats.get("bat_ops", 0),
        "home_win_pct_abs":  home_stand.get("win_pct", 0.5),
        "away_era":          away_stats.get("era", 4.5),
        "away_ops":          away_stats.get("bat_ops", 0),
        "away_win_pct_abs":  away_stand.get("win_pct", 0.5),
        # --- Starting pitcher features (the key new addition) ---
        "home_sp_era":       hsp["sp_era"],
        "away_sp_era":       asp["sp_era"],
        "sp_era_diff":       hsp["sp_era"]  - asp["sp_era"],
        "home_sp_fip":       hsp["sp_fip"],
        "away_sp_fip":       asp["sp_fip"],
        "sp_fip_diff":       hsp["sp_fip"]  - asp["sp_fip"],
        "home_sp_whip":      hsp["sp_whip"],
        "away_sp_whip":      asp["sp_whip"],
        "sp_whip_diff":      hsp["sp_whip"] - asp["sp_whip"],
        "home_sp_k9":        hsp["sp_k9"],
        "away_sp_k9":        asp["sp_k9"],
        "sp_k9_diff":        hsp["sp_k9"]   - asp["sp_k9"],
    }


def build_dataset(season: int = 2025) -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)

    print(f"[1/6] Fetching {season} team batting stats...")
    batting  = fetch_team_batting_stats(season)
    time.sleep(0.3)

    print(f"[2/6] Fetching {season} team pitching stats...")
    pitching = fetch_team_pitching_stats(season)
    time.sleep(0.3)

    team_stats = {}
    for tid, b in batting.items():
        team_stats[tid] = {**b, **pitching.get(tid, {})}

    print(f"[3/6] Fetching {season} standings...")
    standings = fetch_standings(season)
    time.sleep(0.3)

    print(f"[4/6] Fetching {season} player pitching stats (bulk)...")
    player_pitching = fetch_all_player_pitching_stats(season)
    time.sleep(0.3)

    print(f"[5/6] Fetching {season} probable starters...")
    probable = fetch_probable_pitchers(season)
    time.sleep(0.3)

    print(f"[6/6] Fetching {season} completed games...")
    games = fetch_completed_games(season)
    print(f"      Found {len(games)} completed games.")

    rows = []
    missing_sp = 0
    for g in games:
        htid, atid = g["home_team_id"], g["away_team_id"]
        if htid not in team_stats or atid not in team_stats:
            continue

        # Look up probable starters for this game
        sp_info   = probable.get(g["game_pk"], {})
        home_pid  = sp_info.get("home")
        away_pid  = sp_info.get("away")

        home_sp = pitcher_features(player_pitching.get(home_pid)) if home_pid else None
        away_sp = pitcher_features(player_pitching.get(away_pid)) if away_pid else None
        if not home_pid or not away_pid:
            missing_sp += 1

        features = build_features(
            team_stats[htid], team_stats[atid],
            standings.get(htid, {}), standings.get(atid, {}),
            home_sp=home_sp, away_sp=away_sp,
        )
        rows.append({
            "game_pk":      g["game_pk"],
            "date":         g["date"],
            "season":       season,
            "home_team_id": htid,
            "away_team_id": atid,
            "home_team":    team_stats[htid]["team_name"],
            "away_team":    team_stats[atid]["team_name"],
            **features,
            "home_win":  g["home_win"],
        })

    df = pd.DataFrame(rows)
    out_path = f"data/mlb_games_{season}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows -> {out_path}  ({missing_sp} games missing SP data, used league avg)")
    return df


def build_multi_season_dataset(seasons: list[int]) -> pd.DataFrame:
    """Fetch and combine multiple seasons into one CSV for richer training data."""
    dfs = []
    for season in seasons:
        csv_path = f"data/mlb_games_{season}.csv"
        if os.path.exists(csv_path):
            print(f"Loading cached {season} data from {csv_path}...")
            dfs.append(pd.read_csv(csv_path))
        else:
            print(f"\n{'='*50}")
            print(f"Fetching {season} season...")
            print(f"{'='*50}")
            dfs.append(build_dataset(season))
            time.sleep(1)

    combined = pd.concat(dfs, ignore_index=True)
    out_path = "data/mlb_games_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nCombined {len(combined)} games ({len(seasons)} seasons) -> {out_path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None,
                        help="Single season to fetch (e.g. 2025)")
    parser.add_argument("--seasons", type=int, nargs="+", default=None,
                        help="Multiple seasons to fetch and combine (e.g. --seasons 2022 2023 2024 2025)")
    args = parser.parse_args()

    if args.seasons:
        build_multi_season_dataset(args.seasons)
    else:
        build_dataset(season=args.season or 2025)
