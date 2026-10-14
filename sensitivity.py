"""Is the AI actually needed? It depends on how much leftover money stays in the bank.

The main backtest assumes each day's saving must come out of THAT day's spare money
(nothing carries over, because shopkeepers spend leftover cash on stock and family).
Here we relax that:

    keep = share of leftover money still in the bank the next day
    keep = 0   -> nothing carries over (same as backtest.py)
    keep = 1   -> every unspent rupee stays in the bank

Strategies compared:
    fixed          - same amount every day
    pct_yesterday  - in proportion to yesterday's income ("earn first, save next day")
    ai             - AI forecast only (backtest.py)
    ai_balance     - AI forecast + money already in the bank (cap = balance + careful spare)
    balance_only   - money already in the bank only, no forecast

Output: results/sensitivity.json
"""
import json
from pathlib import Path

from backtest import percent_of_income
from forecast import Forecaster
from planner import CATCH_UP_SHARE, MAX_MULTIPLE, MIN_SAVING, ROUND_TO, spare_money
from train import VALID_END, load_features

ROOT = Path(__file__).parent
KEEPS = [0, 0.25, 0.5, 0.75, 1.0]
STRATEGIES = ["fixed", "pct_yesterday", "ai", "ai_balance", "balance_only"]


def amount_for(strategy, shop, day, balance, behind_by):
    goal = shop.goal_per_day
    if strategy == "fixed":
        return goal
    if strategy == "pct_yesterday":
        return percent_of_income(goal, day.lag_1)
    safe_spare = spare_money(day.safe_ai, shop.margin, shop.home_expense)
    cap = {"ai": safe_spare, "ai_balance": balance + safe_spare, "balance_only": balance}[strategy]
    amount = int(min(goal + CATCH_UP_SHARE * max(behind_by, 0), cap, MAX_MULTIPLE * goal) // ROUND_TO * ROUND_TO)
    return amount if amount >= MIN_SAVING else 0


def simulate(strategy, keep, shops, groups):
    failed = saved = goal = 0
    for shop in shops.itertuples():
        balance, saved_shop = 0.0, 0
        for n, day in enumerate(groups[shop.shop_id].itertuples()):
            amount = amount_for(strategy, shop, day, balance, shop.goal_per_day * n - saved_shop)
            balance += spare_money(day.income, shop.margin, shop.home_expense)   # the day's leftover money
            if amount > 0:
                if balance >= amount:
                    balance -= amount
                    saved_shop += amount
                else:
                    failed += 1
            balance = max(balance, 0) * keep    # overnight: the rest goes to stock and family
            goal += shop.goal_per_day
        saved += saved_shop
    months = 6
    return {"failed_per_shop_per_month": round(failed / len(shops) / months, 1),
            "goal_reached_pct": round(saved / goal * 100, 1)}


def main():
    shops, df = load_features()
    test = df[df["date"] > VALID_END].copy()
    _, test["safe_ai"] = Forecaster().predict(test)
    groups = dict(tuple(test.groupby("shop_id")))

    results = {str(k): {s: simulate(s, k, shops, groups) for s in STRATEGIES} for k in KEEPS}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "sensitivity.json").write_text(json.dumps(results, indent=2))

    print("failed payments per shop per month / % of savings goal reached\n")
    print(f"{'keep':>5}" + "".join(f"{s:>16}" for s in STRATEGIES))
    for k in KEEPS:
        r = results[str(k)]
        print(f"{k:>5}" + "".join(f"{r[s]['failed_per_shop_per_month']:>8} /{r[s]['goal_reached_pct']:>5}%" for s in STRATEGIES))


if __name__ == "__main__":
    main()
