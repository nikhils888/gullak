"""Replay Jul-Dec 2025 for all 300 shops and compare ways of choosing the daily saving.

1. fixed          - the same amount every day (how a normal daily auto-debit works)
2. pct_yesterday  - a fixed % of YESTERDAY's income (the amount must be set a day ahead)
3. simple_rule    - careful income estimate = 20th percentile of the last 14 days' income
4. ai             - careful income estimate = the LightGBM 'safe' forecast
5. pct_today      - a fixed % of TODAY's income. Impossible in real life (today's income isn't
                    known when the amount is set), shown only as a reference point.

Strategies 3 and 4 use the exact same saving rule (planner.py); only the forecast differs.
So any difference between them is purely the value added by the AI model.

Output: results/backtest.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import Forecaster
from planner import MAX_MULTIPLE, MIN_SAVING, ROUND_TO, plan_amount, settle
from train import VALID_END, load_features

ROOT = Path(__file__).parent
STRATEGIES = ["fixed", "pct_yesterday", "simple_rule", "ai", "pct_today"]


def percent_of_income(goal_per_day, income_ratio):
    """Save in proportion to income: a normal day saves the goal, a 2x day saves 2x goal."""
    amount = int(min(goal_per_day * income_ratio, MAX_MULTIPLE * goal_per_day) // ROUND_TO * ROUND_TO)
    return amount if amount >= MIN_SAVING else 0


def replay(shop, days, strategy):
    """Go day by day for one shop. Returns a list of (status, planned, saved)."""
    out, saved_total = [], 0
    for n, day in enumerate(days.itertuples()):
        if strategy == "fixed":
            amount = shop.goal_per_day
        elif strategy == "pct_yesterday":
            amount = percent_of_income(shop.goal_per_day, day.lag_1)
        elif strategy == "pct_today":
            amount = percent_of_income(shop.goal_per_day, day.income / day.base_income)
        else:
            safe_income = day.safe_ai if strategy == "ai" else day.safe_rule
            behind_by = shop.goal_per_day * n - saved_total
            amount = plan_amount(safe_income, shop.goal_per_day, shop.margin, shop.home_expense, behind_by)
        status, saved = settle(amount, day.income, shop.margin, shop.home_expense)
        saved_total += saved
        out.append((status, amount, saved))
    return out


def summarize(rows, n_shops, n_months):
    df = pd.DataFrame(rows, columns=["shop_type", "goal", "status", "planned", "saved"])
    attempts = (df["status"] != "rest").sum()
    failed = (df["status"] == "failed").sum()
    return {
        "failed_payment_rate_pct": round(failed / attempts * 100, 1),
        "failed_payments_per_shop_per_month": round(failed / n_shops / n_months, 1),
        "rest_days_per_shop_per_month": round((df["status"] == "rest").sum() / n_shops / n_months, 1),
        "goal_reached_pct": round(df["saved"].sum() / df["goal"].sum() * 100, 1),
        "total_saved_rs": int(df["saved"].sum()),
        "failed_rate_by_shop_type_pct": {
            t: round((g["status"] == "failed").sum() / (g["status"] != "rest").sum() * 100, 1)
            for t, g in df.groupby("shop_type")},
    }


def main():
    shops, df = load_features()
    # simple rule: 20th percentile of the 14 days before each day
    df["safe_rule"] = df.groupby("shop_id")["income"].transform(
        lambda s: s.shift(1).rolling(14, min_periods=14).quantile(0.2))
    test = df[df["date"] > VALID_END].copy()
    _, test["safe_ai"] = Forecaster().predict(test)

    rows = {s: [] for s in STRATEGIES}
    for shop in shops.itertuples():
        days = test[test["shop_id"] == shop.shop_id]
        for s in STRATEGIES:
            rows[s] += [(shop.shop_type, shop.goal_per_day, *r) for r in replay(shop, days, s)]

    n_months = test["date"].dt.to_period("M").nunique()
    results = {"period": "2025-07-01 to 2025-12-31", "shops": len(shops),
               **{s: summarize(rows[s], len(shops), n_months) for s in STRATEGIES}}
    fixed, ai = results["fixed"], results["ai"]
    results["ai_vs_fixed"] = {
        "fewer_failed_payments_pct": round(float(1 - ai["failed_payments_per_shop_per_month"]
                                                 / fixed["failed_payments_per_shop_per_month"]) * 100, 1),
        "more_money_saved_pct": round(float(ai["total_saved_rs"] / fixed["total_saved_rs"] - 1) * 100, 1),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "backtest.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'':38}" + "".join(f"{s:>15}" for s in STRATEGIES))
    for key in ["failed_payment_rate_pct", "failed_payments_per_shop_per_month",
                "rest_days_per_shop_per_month", "goal_reached_pct"]:
        print(f"{key:38}" + "".join(f"{results[s][key]:>15}" for s in STRATEGIES))
    print("\nAI vs fixed:", results["ai_vs_fixed"])


if __name__ == "__main__":
    main()
