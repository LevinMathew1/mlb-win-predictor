"""
Trains and compares multiple models on the historical MLB game data.
Runs Logistic Regression, XGBoost, LightGBM, and a pure Elo benchmark
then saves whichever one scores best on AUC.

Usage:
    python train.py                              # trains on 2022-2025 combined
    python train.py --seasons 2022 2023 2024 2025
    python train.py --data data/mlb_games_combined.csv
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    # Win-rate
    "win_pct_diff", "home_win_pct", "away_win_pct_road", "last10_diff", "run_diff_diff",
    # Pythagorean expectation (removes luck from win%)
    "pyth_diff", "home_pyth", "away_pyth",
    # Team batting
    "ops_diff", "runs_pg_diff", "hr_pg_diff",
    # Team pitching
    "era_diff", "whip_diff", "k9_diff",
    # Absolute stats
    "home_era", "home_ops", "home_win_pct_abs",
    "away_era", "away_ops", "away_win_pct_abs",
    # Starting pitcher (ERA, FIP, WHIP, K9)
    "home_sp_era", "away_sp_era", "sp_era_diff",
    "home_sp_fip", "away_sp_fip", "sp_fip_diff",
    "home_sp_whip", "away_sp_whip", "sp_whip_diff",
    "home_sp_k9", "away_sp_k9", "sp_k9_diff",
    # Elo rating differential (dynamic team strength)
    "elo_diff", "elo_home_prob",
]

os.makedirs("models", exist_ok=True)


def evaluate_model(name: str, model, X, y, cv) -> dict:
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    ll_scores  = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")
    print(f"  {name:<32} AUC={auc_scores.mean():.4f} ± {auc_scores.std():.4f}  "
          f"LogLoss={-ll_scores.mean():.4f}")
    return {"auc": float(auc_scores.mean()), "log_loss": float(-ll_scores.mean())}


def ensure_data(seasons: list[int]) -> str:
    os.makedirs("data", exist_ok=True)
    for s in seasons:
        if not os.path.exists(f"data/mlb_games_{s}.csv"):
            print(f"Fetching {s} season data...")
            subprocess.run([sys.executable, "fetch_data.py", "--season", str(s)], check=True)

    combined_path = "data/mlb_games_combined.csv"
    if len(seasons) > 1 and not os.path.exists(combined_path):
        subprocess.run(
            [sys.executable, "fetch_data.py", "--seasons"] + [str(s) for s in seasons],
            check=True,
        )
    return combined_path if len(seasons) > 1 else f"data/mlb_games_{seasons[0]}.csv"


def train(data_path: str = None, seasons: list[int] = None) -> None:
    seasons = seasons or [2022, 2023, 2024, 2025]
    if data_path is None:
        data_path = ensure_data(seasons)

    print(f"Loading training data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  {len(df)} games | Seasons: {sorted(df['season'].dropna().unique().astype(int))}")

    # --- Merge Elo features ---
    elo_path = "data/game_elos.csv"
    if not os.path.exists(elo_path):
        print("Building Elo ratings...")
        from elo import build_elo_ratings
        build_elo_ratings(seasons=seasons)

    elo_df = pd.read_csv(elo_path)
    df = df.merge(elo_df, on="game_pk", how="left")

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"  Note: {len(missing)} features not in data: {missing}")

    X = df[available_features].fillna(0).values
    y = df["home_win"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\nComparing models (5-fold CV, {len(available_features)} features):")
    print(f"  {'Model':<32} {'AUC':>10}   {'LogLoss':>10}")
    print(f"  {'-'*32} {'-'*10}   {'-'*10}")

    # Baseline: always predict home win
    dummy = DummyClassifier(strategy="most_frequent")
    evaluate_model("Baseline (always home)",   dummy, X, y, cv)

    # Elo-only model (industry standard benchmark)
    elo_only_col = [c for c in ["elo_home_prob"] if c in df.columns]
    if elo_only_col:
        X_elo = df[elo_only_col].fillna(0.5).values
        evaluate_model("Elo-only (FiveThirtyEight)", LogisticRegression(), X_elo, y, cv)

    # Full models
    lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    lr_metrics = evaluate_model("Logistic Regression",      lr,  X, y, cv)

    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )
    lgbm_cal = CalibratedClassifierCV(lgbm, method="isotonic", cv=3)
    lgbm_metrics = evaluate_model("LightGBM (calibrated)",  lgbm_cal, X, y, cv)

    xgb = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    xgb_cal = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
    xgb_metrics = evaluate_model("XGBoost (calibrated)",    xgb_cal, X, y, cv)

    # Pick winner
    candidates = [
        ("LogisticRegression", lr,      lr_metrics),
        ("LightGBM",           lgbm_cal, lgbm_metrics),
        ("XGBoost",            xgb_cal,  xgb_metrics),
    ]
    best_name, best_model, best_metrics = max(candidates, key=lambda x: x[2]["auc"])

    print(f"\n  Winner: {best_name}  (AUC={best_metrics['auc']:.4f})")
    print("Fitting winner on full dataset...")
    best_model.fit(X, y)

    home_win_rate = float(y.mean())

    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/feature_cols.pkl", "wb") as f:
        pickle.dump(available_features, f)

    metadata = {
        "model_name":    best_name,
        "data_path":     data_path,
        "seasons":       [int(s) for s in seasons],
        "n_games":       len(df),
        "n_features":    len(available_features),
        "features":      available_features,
        "cv_auc":        best_metrics["auc"],
        "cv_log_loss":   best_metrics["log_loss"],
        "home_win_rate": home_win_rate,
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to models/")
    print(f"  CV AUC       : {best_metrics['auc']:.4f}")
    print(f"  CV Log-Loss  : {best_metrics['log_loss']:.4f}")
    print(f"  Home win rate: {home_win_rate:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()
    train(data_path=args.data, seasons=args.seasons)
