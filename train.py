"""
Train an XGBoost win-probability model on historical MLB game data.

Usage:
    python train.py                        # uses data/mlb_games_2025.csv
    python train.py --data data/mlb_games_2024.csv
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "win_pct_diff", "home_win_pct", "away_win_pct_road", "last10_diff",
    "run_diff_diff", "ops_diff", "runs_pg_diff", "hr_pg_diff",
    "era_diff", "whip_diff", "k9_diff",
    "home_era", "home_ops", "home_win_pct_abs",
    "away_era", "away_ops", "away_win_pct_abs",
]

os.makedirs("models", exist_ok=True)


def evaluate_model(name: str, model, X, y) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores  = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    ll_scores   = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")
    print(f"  {name:<30} AUC={auc_scores.mean():.4f} ± {auc_scores.std():.4f}  "
          f"LogLoss={-ll_scores.mean():.4f}")
    return {"auc": float(auc_scores.mean()), "log_loss": float(-ll_scores.mean())}


def train(data_path: str = "data/mlb_games_2025.csv") -> None:
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  {len(df)} games loaded.")

    X = df[FEATURE_COLS].fillna(0).values
    y = df["home_win"].values

    print("\nComparing models (5-fold CV):")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    # Calibrate XGBoost probabilities using Platt scaling
    xgb_cal = CalibratedClassifierCV(xgb, method="sigmoid", cv=3)

    lr = LogisticRegression(max_iter=500, random_state=42)

    xgb_metrics = evaluate_model("XGBoost (calibrated)", xgb_cal, X, y)
    lr_metrics  = evaluate_model("Logistic Regression",  lr,      X, y)

    # Pick best by AUC, fall back to XGBoost
    best_model  = xgb_cal if xgb_metrics["auc"] >= lr_metrics["auc"] else lr
    best_name   = "XGBoost" if best_model is xgb_cal else "LogisticRegression"
    best_metrics = xgb_metrics if best_model is xgb_cal else lr_metrics

    print(f"\nBest model: {best_name}")
    print("Fitting on full dataset...")
    best_model.fit(X, y)

    # Home-field baseline
    home_win_rate = float(y.mean())
    print(f"  Home team win rate in training data: {home_win_rate:.3f}")

    # Save artifacts
    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/feature_cols.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    metadata = {
        "model_name":      best_name,
        "data_path":       data_path,
        "n_games":         len(df),
        "features":        FEATURE_COLS,
        "cv_auc":          best_metrics["auc"],
        "cv_log_loss":     best_metrics["log_loss"],
        "home_win_rate":   home_win_rate,
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to models/")
    print(f"  CV AUC       : {best_metrics['auc']:.4f}")
    print(f"  CV Log-Loss  : {best_metrics['log_loss']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mlb_games_2025.csv",
                        help="Path to training CSV (default: data/mlb_games_2025.csv)")
    args = parser.parse_args()
    train(data_path=args.data)
