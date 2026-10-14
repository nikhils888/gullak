"""Generate 2 years of daily income for 300 small Indian shops (synthetic data).

There is no public dataset of Indian shopkeepers' daily sales, so we simulate one.

    daily income = shop's normal income
                   x weekday effect      (e.g. clothing shops are busy on weekends)
                   x month-start effect  (salaried customers just got paid)
                   x festival effect     (rush before Diwali, Holi, ...; quiet on the day itself)
                   x rain effect         (monsoon hurts open-air vendors, helps chai stalls)
                   x growth trend
                   x random noise
    + closed days (weekly off-day for some shops, random sick/family days)

Each shop also gets:
    margin        - share of sales that is profit (e.g. 0.15 = Rs 15 profit per Rs 100 sales)
    home_expense  - money the family needs every day
    goal_per_day  - a realistic daily savings goal
Money the shopkeeper can spare on a day = income x margin - home_expense.

Outputs: data/shops.csv, data/daily_income.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from features import FESTIVALS, SHOP_TYPES

ROOT = Path(__file__).parent
SEED = 42
N_SHOPS = 300
DATES = pd.date_range("2024-01-01", "2025-12-31", freq="D")

# weekday effect is Mon..Sun
PROFILES = {
    "kirana":        dict(label="Kirana Store",  income=6000, margin=0.15, share=0.35, noise=0.18,
                          week=[1.0, 0.95, 0.95, 1.0, 1.05, 1.15, 1.2], festival=1.35, rain=0.92,
                          month_start=1.15, weekly_off=0.0),
    "sabzi_vendor":  dict(label="Sabzi Vendor",  income=2500, margin=0.25, share=0.20, noise=0.25,
                          week=[0.95, 0.95, 1.0, 1.0, 1.05, 1.15, 1.25], festival=1.2, rain=0.7,
                          month_start=1.05, weekly_off=0.0),
    "chai_stall":    dict(label="Chai Stall",    income=1500, margin=0.40, share=0.15, noise=0.20,
                          week=[1.1, 1.1, 1.1, 1.1, 1.05, 0.85, 0.6], festival=0.9, rain=1.15,
                          month_start=1.0, weekly_off=0.0),
    "clothing":      dict(label="Clothing Shop", income=9000, margin=0.25, share=0.15, noise=0.40,
                          week=[0.8, 0.8, 0.85, 0.9, 1.0, 1.35, 1.5], festival=2.2, rain=0.8,
                          month_start=1.1, weekly_off=0.7),
    "mobile_repair": dict(label="Mobile Repair", income=3000, margin=0.35, share=0.15, noise=0.50,
                          week=[1.0, 1.0, 1.0, 1.0, 1.05, 1.1, 1.1], festival=1.25, rain=0.9,
                          month_start=1.05, weekly_off=0.4),
}
assert list(PROFILES) == SHOP_TYPES


def city_calendar(rng):
    """Effects shared by every shop in the city: rain and festival rush."""
    month, day = DATES.month.values, DATES.day.values
    monsoon = (month == 7) | (month == 8) | ((month == 9) & (day <= 15))
    rainy = rng.random(len(DATES)) < np.where(monsoon, 0.45, 0.04)

    rush = np.zeros(len(DATES))            # 0..1, how strong the pre-festival rush is
    festival_day = np.zeros(len(DATES), dtype=bool)
    for date, _, strength in FESTIVALS:
        days_before = (pd.Timestamp(date) - DATES).days.values
        window = int(7 + 14 * strength)    # Diwali rush lasts ~3 weeks, smaller festivals ~1-2
        ramp = np.where((days_before > 0) & (days_before <= window), 1 - days_before / (window + 1), 0)
        rush = np.maximum(rush, ramp * strength)
        festival_day |= days_before == 0
    return rainy, rush, festival_day


def main():
    rng = np.random.default_rng(SEED)
    rainy, rush, festival_day = city_calendar(rng)
    weekday = DATES.dayofweek.values
    t = np.arange(len(DATES))
    in_2024 = DATES.year == 2024

    types = rng.choice(SHOP_TYPES, size=N_SHOPS, p=[PROFILES[s]["share"] for s in SHOP_TYPES])
    shops, incomes = [], []
    for shop_id, shop_type in enumerate(types, start=1):
        p = PROFILES[shop_type]
        scale = p["income"] * rng.lognormal(0, 0.35)          # some shops are bigger than others
        trend = (1 + rng.uniform(-0.10, 0.25)) ** (t / 365)   # yearly growth between -10% and +25%
        week = np.array(p["week"])[weekday]
        month_start = np.where(DATES.day.values <= 7, p["month_start"], 1.0)
        festival = np.where(festival_day, 0.4, 1 + (p["festival"] - 1) * rush)
        rain = np.where(rainy, p["rain"], 1.0)
        noise = rng.lognormal(-p["noise"] ** 2 / 2, p["noise"], len(DATES))  # average 1.0

        income = scale * trend * week * month_start * festival * rain * noise
        closed = rng.random(len(DATES)) < 0.02                # sick day, family function, ...
        weekly_off_day = -1
        if rng.random() < p["weekly_off"]:
            weekly_off_day = int(rng.choice([0, 1]))          # Delhi markets often close Mon or Tue
            closed |= weekday == weekly_off_day
        income = np.where(closed, 0, np.round(income, -1))

        margin = round(p["margin"] * rng.uniform(0.85, 1.15), 3)
        avg_profit = income[in_2024].mean() * margin
        home_expense = round(avg_profit * rng.uniform(0.4, 0.7), -1)
        goal = max(20, int(0.4 * (avg_profit - home_expense) // 10 * 10))  # save 40% of avg spare money

        shops.append(dict(shop_id=shop_id, name=f"{p['label']} #{shop_id}", shop_type=shop_type,
                          margin=margin, home_expense=home_expense, goal_per_day=goal,
                          weekly_off_day=weekly_off_day))
        incomes.append(pd.DataFrame({"shop_id": shop_id, "date": DATES, "income": income}))

    (ROOT / "data").mkdir(exist_ok=True)
    shops = pd.DataFrame(shops)
    income = pd.concat(incomes, ignore_index=True)
    shops.to_csv(ROOT / "data" / "shops.csv", index=False)
    income.to_csv(ROOT / "data" / "daily_income.csv", index=False, date_format="%Y-%m-%d")

    print(f"{len(shops)} shops, {len(income):,} shop-days ({DATES[0].date()} to {DATES[-1].date()})")
    print(shops.groupby("shop_type")[["goal_per_day"]].agg(["count", "median"]))


if __name__ == "__main__":
    main()
