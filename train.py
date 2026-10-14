"""Train two LightGBM models that forecast tomorrow's income for every shop.

- typical model (quantile 0.5 = median): our best guess.
- safe model    (quantile 0.2):          a careful guess. Real income should be above it on
                                         ~80% of days, so savings planned from it rarely fail.

Time-based split (never train on the future):
    train: Jan 2024 - Mar 2025 | validation: Apr - Jun 2025 | test: Jul - Dec 2025

Outputs: models/typical.pkl, models/safe.pkl, results/model_metrics.json
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, build_features

ROOT = Path(__file__).parent
TRAIN_END, VALID_END = "2025-03-31", "2025-06-30"


def load_features():
    shops = pd.read_csv(ROOT / "data" / "shops.csv")
    income = pd.read_csv(ROOT / "data" / "daily_income.csv", parse_dates=["date"])
    df = income.merge(shops[["shop_id", "shop_type"]], on="shop_id")
    return shops, build_features(df).dropna(subset=["base_income"])  # first 28 days have no history


def error(pred, actual):
    return {"avg_error_rs": round(float(np.mean(np.abs(pred - actual))), 1),
            "error_pct_of_income": round(float(np.abs(pred - actual).sum() / actual.sum() * 100), 1)}


def main():
    _, df = load_features()
    train = df[df["date"] <= TRAIN_END]
    valid = df[(df["date"] > TRAIN_END) & (df["date"] <= VALID_END)]
    test = df[df["date"] > VALID_END]
    print(f"rows -> train {len(train):,} | valid {len(valid):,} | test {len(test):,}")

    (ROOT / "models").mkdir(exist_ok=True)
    importance = {}
    for name, alpha in [("typical", 0.5), ("safe", 0.2)]:
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=2000,
                                  learning_rate=0.05, num_leaves=31, min_child_samples=100,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  random_state=42, verbose=-1)
        model.fit(train[FEATURES], train["target"],
                  eval_X=(valid[FEATURES],), eval_y=(valid["target"],),
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        joblib.dump(model, ROOT / "models" / f"{name}.pkl")
        gain = model.booster_.feature_importance("gain")
        importance[name] = {f: round(float(g / gain.sum() * 100), 1)
                            for f, g in sorted(zip(FEATURES, gain), key=lambda x: -x[1])}
        print(f"{name} model: stopped at {model.best_iteration_} trees")

    from forecast import Forecaster  # load the saved models exactly like the app does
    typical, safe = Forecaster().predict(test)
    actual, base = test["income"].to_numpy(), test["base_income"].to_numpy()

    metrics = {
        "test_period": "2025-07-01 to 2025-12-31",
        "test_shop_days": len(test),
        "forecast_error": {
            "ai_model_typical": error(typical, actual),
            "simple_same_day_last_week": error(test["lag_7"].to_numpy() * base, actual),
            "simple_last_7_day_average": error(test["mean_7"].to_numpy() * base, actual),
        },
        "safe_model_income_above_estimate_pct": round(float(np.mean(actual >= safe) * 100), 1),
        "feature_importance_pct_typical": importance["typical"],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
