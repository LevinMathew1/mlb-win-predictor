"""
Builds projected team stats for 2026 based on who's actually on each roster
right now — not last year's team as a whole. Pulls every team's current 40-man
roster, maps each player to their 2025 stats, and aggregates it all into
team-level batting and pitching lines.

This way trades, free agent signings, and departures are all accounted for.

Usage:
    from roster_stats import build_roster_projections
    team_stats, standings = build_roster_projections()
"""

import time
from typing import Optional

import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"

# League-average fallback for players with no 2025 stats (rookies, injuries, etc.)
LEAGUE_AVG_BATTING = {
    "bat_avg": 0.248, "bat_obp": 0.317, "bat_slg": 0.401, "bat_ops": 0.718,
    "runs_per_game": 0.55, "hr_per_game": 0.16, "k_per_game": 1.10,
}
LEAGUE_AVG_PITCHING = {
    "era": 4.20, "whip": 1.28, "k9": 8.90, "runs_allowed_per_game": 4.60,
}


def _get(endpoint: str, params: dict = None) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def parse_ip(ip_str) -> float:
    """Convert MLB innings-pitched format ('45.2' = 45 IP + 2 outs) to true innings."""
    try:
        parts = str(ip_str).split(".")
        return int(parts[0]) + (int(parts[1]) if len(parts) > 1 else 0) / 3
    except Exception:
        return 0.0


def fetch_all_teams(season: int = 2026) -> dict:
    """Return {team_id: team_name} for all MLB teams."""
    data = _get("teams", {"sportId": 1, "season": season})
    return {t["id"]: t["name"] for t in data.get("teams", [])
            if t.get("sport", {}).get("id") == 1}


def fetch_roster(team_id: int, season: int = 2026) -> list[dict]:
    """Return 40-man roster entries for a team."""
    try:
        data = _get(f"teams/{team_id}/roster",
                    {"rosterType": "40Man", "season": season})
        return data.get("roster", [])
    except Exception:
        return []


def fetch_all_player_hitting_stats(season: int = 2025) -> dict:
    """
    Bulk fetch hitting stats for all players in one API call.
    Returns {player_id: stat_dict}.
    """
    data = _get("stats", {
        "stats": "season",
        "group": "hitting",
        "season": season,
        "sportId": 1,
        "limit": 2000,
    })
    result = {}
    for split in data.get("stats", [{}])[0].get("splits", []):
        pid = split["player"]["id"]
        result[pid] = split["stat"]
    return result


def fetch_all_player_pitching_stats(season: int = 2025) -> dict:
    """
    Bulk fetch pitching stats for all players in one API call.
    Returns {player_id: stat_dict}.
    """
    data = _get("stats", {
        "stats": "season",
        "group": "pitching",
        "season": season,
        "sportId": 1,
        "limit": 2000,
    })
    result = {}
    for split in data.get("stats", [{}])[0].get("splits", []):
        pid = split["player"]["id"]
        result[pid] = split["stat"]
    return result


def aggregate_batting(player_stats: list[dict]) -> dict:
    """
    Aggregate individual batter counting stats into team-level rate stats.
    Uses 162-game season assumption for per-game rates.
    """
    totals = dict(ab=0, h=0, doubles=0, triples=0, hr=0,
                  r=0, bb=0, hbp=0, sf=0, so=0, pa=0)

    for s in player_stats:
        totals["ab"]      += int(s.get("atBats", 0))
        totals["h"]       += int(s.get("hits", 0))
        totals["doubles"] += int(s.get("doubles", 0))
        totals["triples"] += int(s.get("triples", 0))
        totals["hr"]      += int(s.get("homeRuns", 0))
        totals["r"]       += int(s.get("runs", 0))
        totals["bb"]      += int(s.get("baseOnBalls", 0))
        totals["hbp"]     += int(s.get("hitByPitch", 0))
        totals["sf"]      += int(s.get("sacFlies", 0))
        totals["so"]      += int(s.get("strikeOuts", 0))
        totals["pa"]      += int(s.get("plateAppearances", 0))

    ab  = max(totals["ab"], 1)
    obp_denom = max(ab + totals["bb"] + totals["hbp"] + totals["sf"], 1)
    tb = (totals["h"] + totals["doubles"] + 2 * totals["triples"] + 3 * totals["hr"])

    avg = totals["h"] / ab
    obp = (totals["h"] + totals["bb"] + totals["hbp"]) / obp_denom
    slg = tb / ab
    ops = obp + slg

    # Scale to per-game over 162-game season
    games = 162
    return {
        "bat_avg":       round(avg, 3),
        "bat_obp":       round(obp, 3),
        "bat_slg":       round(slg, 3),
        "bat_ops":       round(ops, 3),
        "runs_per_game": round(totals["r"] / games, 3),
        "hr_per_game":   round(totals["hr"] / games, 3),
        "k_per_game":    round(totals["so"] / games, 3),
    }


def aggregate_pitching(player_stats: list[dict]) -> dict:
    """
    Aggregate individual pitcher counting stats into team-level rate stats.
    """
    total_ip = 0.0
    total_er = 0
    total_h  = 0
    total_bb = 0
    total_so = 0
    total_r  = 0

    for s in player_stats:
        ip = parse_ip(s.get("inningsPitched", "0"))
        total_ip += ip
        total_er += int(s.get("earnedRuns", 0))
        total_h  += int(s.get("hits", 0))
        total_bb += int(s.get("baseOnBalls", 0))
        total_so += int(s.get("strikeOuts", 0))
        total_r  += int(s.get("runs", 0))

    ip = max(total_ip, 1.0)
    games = 162
    return {
        "era":                   round(9 * total_er / ip, 2),
        "whip":                  round((total_h + total_bb) / ip, 3),
        "k9":                    round(9 * total_so / ip, 2),
        "runs_allowed_per_game": round(total_r / games, 3),
    }


def build_roster_projections(roster_season: int = 2026,
                              stats_season: int = 2025,
                              verbose: bool = True) -> tuple[dict, dict]:
    """
    Main entry point. Returns (team_stats, standings) dicts keyed by team_id,
    where team_stats is built from 2026 rosters + 2025 individual player stats.
    """
    if verbose:
        print(f"[1/4] Fetching all {stats_season} player batting stats (bulk)...")
    all_hitting  = fetch_all_player_hitting_stats(stats_season)
    time.sleep(0.3)

    if verbose:
        print(f"[2/4] Fetching all {stats_season} player pitching stats (bulk)...")
    all_pitching = fetch_all_player_pitching_stats(stats_season)
    time.sleep(0.3)

    if verbose:
        print(f"[3/4] Fetching {roster_season} rosters for all 30 teams...")
    teams = fetch_all_teams(roster_season)

    team_stats = {}
    missing_batters = 0
    missing_pitchers = 0

    for team_id, team_name in teams.items():
        roster = fetch_roster(team_id, roster_season)
        time.sleep(0.05)  # be polite to the API

        batter_stats  = []
        pitcher_stats = []

        for player in roster:
            pid      = player["person"]["id"]
            pos_type = player.get("position", {}).get("type", "")
            pos_code = player.get("position", {}).get("code", "")

            if pos_type == "Pitcher" or pos_code == "1":
                # Pitcher
                if pid in all_pitching:
                    pitcher_stats.append(all_pitching[pid])
                else:
                    missing_pitchers += 1
            else:
                # Position player
                if pid in all_hitting:
                    batter_stats.append(all_hitting[pid])
                else:
                    missing_batters += 1

        # Fall back to league average if roster data is thin
        if len(batter_stats) < 5:
            batting_proj = LEAGUE_AVG_BATTING.copy()
        else:
            batting_proj = aggregate_batting(batter_stats)

        if len(pitcher_stats) < 3:
            pitching_proj = LEAGUE_AVG_PITCHING.copy()
        else:
            pitching_proj = aggregate_pitching(pitcher_stats)

        team_stats[team_id] = {
            "team_name": team_name,
            **batting_proj,
            **pitching_proj,
        }

    if verbose:
        print(f"      Built projections for {len(team_stats)} teams "
              f"({missing_batters} batters / {missing_pitchers} pitchers "
              f"without 2025 stats -> league avg used)")

    # Fetch standings for win%, home/away splits, last-10
    if verbose:
        print(f"[4/4] Fetching standings (2025 season-end for baseline)...")
    from fetch_data import fetch_standings
    standings = fetch_standings(2025)

    return team_stats, standings


if __name__ == "__main__":
    team_stats, standings = build_roster_projections()
    print("\nSample projections:")
    for tid, s in list(team_stats.items())[:5]:
        print(f"  {s['team_name']:<30} OPS={s['bat_ops']:.3f}  ERA={s['era']:.2f}  WHIP={s['whip']:.3f}")
