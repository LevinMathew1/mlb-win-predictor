"""
Scrapes win probabilities from dRatings.com and compares them against
our model game by game. Outputs an ensemble (straight average of both)
with a signal rating so you know how much to trust each prediction.

  STRONG = both models agree and the margin is meaningful — good pick
  LEAN   = both agree but it's close — slight edge
  SPLIT  = they disagree — too uncertain to act on confidently

Usage:
    from external_odds import fetch_dratings, build_comparison_table
"""

import re
import time

import requests
from bs4 import BeautifulSoup

DRATINGS_URL = "https://www.dratings.com/predictor/mlb-baseball-predictions/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Normalize team name variations between dRatings and MLB Stats API
_NAME_MAP = {
    "Arizona D-Backs":           "Arizona Diamondbacks",
    "AZ Diamondbacks":           "Arizona Diamondbacks",
    "Chi White Sox":             "Chicago White Sox",
    "Chi Cubs":                  "Chicago Cubs",
    "LA Dodgers":                "Los Angeles Dodgers",
    "LA Angels":                 "Los Angeles Angels",
    "NY Yankees":                "New York Yankees",
    "NY Mets":                   "New York Mets",
    "SF Giants":                 "San Francisco Giants",
    "SD Padres":                 "San Diego Padres",
    "TB Rays":                   "Tampa Bay Rays",
    "KC Royals":                 "Kansas City Royals",
    "Oakland Athletics":         "Athletics",
    "Colorado":                  "Colorado Rockies",
    "Miami":                     "Miami Marlins",
    "Cleveland":                 "Cleveland Guardians",
    "Minnesota":                 "Minnesota Twins",
    "Milwaukee":                 "Milwaukee Brewers",
    "Baltimore":                 "Baltimore Orioles",
    "Boston":                    "Boston Red Sox",
    "Cincinnati":                "Cincinnati Reds",
    "Pittsburgh":                "Pittsburgh Pirates",
    "Philadelphia":              "Philadelphia Phillies",
    "Washington":                "Washington Nationals",
    "Detroit":                   "Detroit Tigers",
    "Houston":                   "Houston Astros",
    "Seattle":                   "Seattle Mariners",
    "Atlanta":                   "Atlanta Braves",
    "Toronto":                   "Toronto Blue Jays",
    "Texas":                     "Texas Rangers",
    "St. Louis":                 "St. Louis Cardinals",
}


def _normalize(name: str) -> str:
    name = name.strip()
    return _NAME_MAP.get(name, name)


def _parse_teams_cell(cell_text: str) -> tuple[str, str] | None:
    """
    Parse 'New York Yankees(94-68)San Francisco Giants(81-81)'
    → ('New York Yankees', 'San Francisco Giants')
    First team = away, second = home.
    """
    match = re.match(r"(.+?)\(\d+-\d+\)(.+?)\(\d+-\d+\)", cell_text.strip())
    if match:
        return _normalize(match.group(1).strip()), _normalize(match.group(2).strip())
    # Fallback: no record in parentheses
    parts = cell_text.strip().split("\n")
    parts = [p.strip() for p in parts if p.strip() and not re.match(r"\(\d+-\d+\)", p.strip())]
    if len(parts) >= 2:
        return _normalize(parts[0]), _normalize(parts[1])
    return None


def _parse_win_pcts(cell_text: str) -> tuple[float, float] | None:
    """
    Parse '52.8%47.2%' → (0.528, 0.472)
    """
    pcts = re.findall(r"(\d+\.?\d*)%", cell_text)
    if len(pcts) >= 2:
        return float(pcts[0]) / 100, float(pcts[1]) / 100
    return None


def fetch_dratings(retries: int = 2) -> list[dict]:
    """
    Scrape today's MLB predictions from dratings.com.
    Returns list of:
        {away_team, home_team, away_win_prob, home_win_prob, source: 'dratings'}
    """
    for attempt in range(retries + 1):
        try:
            resp = requests.get(DRATINGS_URL, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == retries:
                print(f"  [dRatings] Failed to fetch after {retries+1} attempts: {e}")
                return []
            time.sleep(2)

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table")
    if not tables:
        print("  [dRatings] No tables found on page.")
        return []

    # Table 0 = upcoming games
    rows = tables[0].find_all("tr")[1:]  # skip header
    games = []

    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < 4:
            continue

        teams = _parse_teams_cell(cells[1].get_text(separator=""))
        pcts  = _parse_win_pcts(cells[3].get_text())

        if not teams or not pcts:
            continue

        away_team, home_team = teams
        away_prob, home_prob = pcts

        games.append({
            "away_team":    away_team,
            "home_team":    home_team,
            "away_win_prob": away_prob,
            "home_win_prob": home_prob,
            "source":       "dratings",
        })

    return games


def _signal(our_home: float, dr_home: float) -> str:
    """Classify the agreement between the two models."""
    ensemble_home = (our_home + dr_home) / 2
    agree = (our_home > 0.5) == (dr_home > 0.5)  # both favor same team

    if not agree:
        return "SPLIT"
    margin = abs(ensemble_home - 0.5)
    if margin >= 0.07:
        return "STRONG"
    return "LEAN"


def build_comparison_table(our_results: list[dict],
                           dratings_results: list[dict]) -> list[dict]:
    """
    Match our model's predictions to dRatings by team names.
    Returns a combined list with ensemble probabilities and signal strength.
    """
    # Index dratings by (away, home) — case-insensitive
    dr_index = {}
    for g in dratings_results:
        key = (g["away_team"].lower(), g["home_team"].lower())
        dr_index[key] = g

    combined = []
    for g in our_results:
        away = g["away_team"]
        home = g["home_team"]
        key  = (away.lower(), home.lower())

        dr = dr_index.get(key)

        if dr:
            ens_home = (g["home_win_prob"] + dr["home_win_prob"]) / 2
            ens_away = 1 - ens_home
            sig = _signal(g["home_win_prob"], dr["home_win_prob"])
            combined.append({
                **g,
                "dr_home_prob":       dr["home_win_prob"],
                "dr_away_prob":       dr["away_win_prob"],
                "ens_home_prob":      round(ens_home, 4),
                "ens_away_prob":      round(ens_away, 4),
                "model_diff":         round(abs(g["home_win_prob"] - dr["home_win_prob"]), 4),
                "signal":             sig,
                "dratings_available": True,
            })
        else:
            # No dRatings match — use our model alone
            combined.append({
                **g,
                "dr_home_prob":       None,
                "dr_away_prob":       None,
                "ens_home_prob":      round(g["home_win_prob"], 4),
                "ens_away_prob":      round(g["away_win_prob"], 4),
                "model_diff":         None,
                "signal":             "OUR MODEL ONLY",
                "dratings_available": False,
            })

    return combined


def print_comparison(combined: list[dict], metadata: dict) -> None:
    from datetime import datetime

    dr_count = sum(1 for r in combined if r["dratings_available"])
    total    = len(combined)
    print("\n" + "=" * 95)
    print("  MLB WIN PROBABILITY — MODEL vs dRATINGS ENSEMBLE")
    print(f"  Our model AUC: {metadata['cv_auc']:.3f}  |  "
          f"Home win rate: {metadata['home_win_rate']:.1%}  |  "
          f"Ensemble = avg(Our Model, dRatings)")
    print(f"  dRatings coverage: {dr_count}/{total} games  "
          f"(dRatings posts games closer to game date — full coverage on game day)")
    print("=" * 95)
    print(f"  {'Away Team':<26} {'Home Team':<26} "
          f"{'Ours':>6} {'dRate':>6} {'FINAL':>6}  Signal     Result")
    print(f"  {'-'*26} {'-'*26} "
          f"{'-'*6} {'-'*6} {'-'*6}  ---------  ------")

    current_date = None
    for r in combined:
        if r.get("date") != current_date:
            current_date = r.get("date")
            if current_date:
                try:
                    dt = datetime.strptime(current_date, "%Y-%m-%d")
                    print(f"\n  {dt.strftime('%A, %B %d %Y')}")
                    print(f"  {'Away Team':<26} {'Home Team':<26} "
                          f"{'Ours':>6} {'dRate':>6} {'FINAL':>6}  Signal     Result")
                    print(f"  {'-'*26} {'-'*26} "
                          f"{'-'*6} {'-'*6} {'-'*6}  ---------  ------")
                except Exception:
                    pass

        our_h  = r["home_win_prob"] * 100
        dr_h   = r["dr_home_prob"] * 100 if r["dr_home_prob"] is not None else None
        ens_h  = r["ens_home_prob"] * 100
        ens_a  = r["ens_away_prob"] * 100
        signal = r["signal"]

        dr_str  = f"{dr_h:5.1f}%" if dr_h is not None else "  N/A "

        # Result column
        state = r.get("state", "Preview")
        if state == "Final":
            result_str = f"HOME WON {r['away_score']}-{r['home_score']}" if r.get("home_won") \
                         else f"AWAY WON {r['away_score']}-{r['home_score']}"
        elif state == "Live":
            result_str = f"LIVE {r['away_score']}-{r['home_score']}"
        else:
            # Show which team the ensemble favors
            if ens_h > ens_a:
                result_str = f">> {r['home_team'].split()[-1]} ({ens_h:.1f}%)"
            else:
                result_str = f">> {r['away_team'].split()[-1]} ({ens_a:.1f}%)"

        print(f"  {r['away_team']:<26} {r['home_team']:<26} "
              f"{our_h:5.1f}% {dr_str} {ens_h:5.1f}%  {signal:<10} {result_str}")

    print("\n" + "=" * 95)
    print("  SIGNAL KEY:")
    print("  STRONG  = both models agree, margin >7%  (high confidence)")
    print("  LEAN    = both models agree, smaller margin (moderate confidence)")
    print("  SPLIT   = models favor opposite teams    (bet with caution)")
    print("=" * 95 + "\n")
