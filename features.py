"""Turns daily income history into features for the forecasting model.

Key idea: we predict tomorrow's income as a RATIO of the shop's normal income
(its average over the last 28 days). A ratio of 1.3 means "30% busier than
usual". This lets one model work for a Rs 1,500/day chai stall and a
Rs 9,000/day clothing shop.

Every feature only uses what is known the evening before the day we predict
(past income + the calendar), so the model never peeks into the future.
"""
import numpy as np
import pandas as pd

SHOP_TYPES = ["kirana", "sabzi_vendor", "chai_stall", "clothing", "mobile_repair"]

# Festival dates are known in advance, so the model is allowed to use them.
# strength = how much extra shopping the festival brings (1.0 = Diwali).
FESTIVALS = [
    ("2024-03-25", "Holi", 0.5),
    ("2024-04-11", "Eid", 0.4),
    ("2024-08-19", "Raksha Bandhan", 0.5),
    ("2024-10-12", "Dussehra", 0.6),
    ("2024-11-01", "Diwali", 1.0),
    ("2024-12-25", "Christmas", 0.3),
    ("2025-03-14", "Holi", 0.5),
    ("2025-03-31", "Eid", 0.4),
    ("2025-08-09", "Raksha Bandhan", 0.5),
    ("2025-10-02", "Dussehra", 0.6),
    ("2025-10-20", "Diwali", 1.0),
    ("2025-12-25", "Christmas", 0.3),
    ("2026-03-04", "Holi", 0.5),
]

FEATURES = [
    "shop_type", "weekday", "day_of_month", "month",
    "days_to_festival", "festival_strength",
    "lag_1", "lag_7", "lag_14", "mean_7", "std_7", "min_7",
    "same_weekday_avg", "same_weekday_zeros",
]

# Features grouped into reasons a shopkeeper would understand (used to explain a forecast).
REASON_GROUPS = {
    "weekday": ["weekday", "lag_7", "lag_14", "same_weekday_avg", "same_weekday_zeros"],
    "festival": ["days_to_festival", "festival_strength"],
    "month_start": ["day_of_month"],
    "season": ["month"],
    "recent": ["lag_1", "mean_7", "std_7", "min_7"],
}

MAX_FESTIVAL_DAYS = 60


def festival_info(dates):
    """For each date: days until the next festival (0 = festival day), its strength and name."""
    fest = sorted((pd.Timestamp(d), name, s) for d, name, s in FESTIVALS)
    fest_dates = pd.DatetimeIndex([f[0] for f in fest])
    fest_strength = np.array([f[2] for f in fest])
    fest_names = np.array([f[1] for f in fest], dtype=object)

    dates = pd.DatetimeIndex(dates)
    idx = fest_dates.searchsorted(dates)          # first festival on or after each date
    has_next = idx < len(fest)
    idx = np.minimum(idx, len(fest) - 1)
    days = np.where(has_next, (fest_dates[idx] - dates).days, MAX_FESTIVAL_DAYS)
    days = np.minimum(days, MAX_FESTIVAL_DAYS)
    near = has_next & (days < MAX_FESTIVAL_DAYS)
    strength = np.where(near, fest_strength[idx], 0.0)
    names = np.where(near, fest_names[idx], "")
    return days, strength, names


def build_features(df):
    """df: one row per shop per day, columns shop_id, date, income, shop_type.

    Days must be consecutive for each shop. The income of the day being
    predicted may be NaN (that is the row we want a forecast for).
    """
    df = df.sort_values(["shop_id", "date"]).reset_index(drop=True)
    g = df.groupby("shop_id")["income"]

    def past(stat, window):
        """A statistic over the `window` days BEFORE each row (shift(1) skips the row itself)."""
        return g.transform(lambda s: getattr(s.shift(1).rolling(window, min_periods=window), stat)())

    base = past("mean", 28).clip(lower=1.0)       # the shop's "normal" daily income
    df["base_income"] = base

    for k in (1, 7, 14):
        df[f"lag_{k}"] = g.shift(k) / base         # income k days ago, relative to normal
    df["mean_7"] = past("mean", 7) / base
    df["std_7"] = past("std", 7) / base            # how jumpy the last week was
    df["min_7"] = past("min", 7) / base

    same_weekday = [g.shift(k) for k in (7, 14, 21, 28)]   # same weekday in the last 4 weeks
    df["same_weekday_avg"] = sum(same_weekday) / 4 / base
    df["same_weekday_zeros"] = sum((s == 0).astype(int) for s in same_weekday)  # spots weekly off-days

    df["weekday"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    days, strength, names = festival_info(df["date"])
    df["days_to_festival"] = days
    df["festival_strength"] = strength
    df["festival_name"] = names
    df["shop_type"] = pd.Categorical(df["shop_type"], categories=SHOP_TYPES)

    df["target"] = df["income"] / base             # what the model learns to predict
    return df
