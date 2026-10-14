"""Loads the two trained models and turns their output into rupees + a plain-English reason."""
from pathlib import Path

import joblib
import numpy as np

from features import FEATURES, REASON_GROUPS

MODEL_DIR = Path(__file__).parent / "models"
FESTIVAL_RUSH_DAYS = 21   # festival shopping starts at most ~3 weeks before (Diwali)


class Forecaster:
    def __init__(self):
        self.typical = joblib.load(MODEL_DIR / "typical.pkl")   # median: best guess
        self.safe = joblib.load(MODEL_DIR / "safe.pkl")         # 20th percentile: careful guess

    def predict(self, feats):
        """feats: rows from build_features(). Returns (typical, safe) income in rupees."""
        X = feats[FEATURES]
        base = feats["base_income"].to_numpy()
        typical = np.clip(self.typical.predict(X), 0, None) * base
        safe = np.clip(self.safe.predict(X), 0, None) * base
        return typical, np.minimum(safe, typical)   # careful guess can't be above best guess

    def explain(self, row, direction):
        """Main reason behind the careful forecast for ONE row, in words a shopkeeper understands.

        LightGBM can split a prediction into how much each feature pushed it up or down
        (SHAP values, via pred_contrib=True). We add these up per reason group and pick the
        biggest one that points the same way as the plan (down = saving less, up = saving more).
        """
        contrib = self.safe.predict(row[FEATURES], pred_contrib=True)[0][:-1]  # last value = baseline
        per_feature = dict(zip(FEATURES, contrib))
        groups = {g: sum(per_feature[f] for f in feats) for g, feats in REASON_GROUPS.items()}
        if row["days_to_festival"].iloc[0] > FESTIVAL_RUSH_DAYS and direction != "down":
            del groups["festival"]   # a festival weeks away is not a reason to save more
        if direction == "down":
            group = min(groups, key=groups.get)
        elif direction == "up":
            group = max(groups, key=groups.get)
        else:
            group = max(groups, key=lambda g: abs(groups[g]))
        return _reason_text(group, groups[group] > 0, row.iloc[0])


def _reason_text(group, up, r):
    day = r["date"].day_name()
    if group == "weekday":
        if r["same_weekday_zeros"] >= 2:
            return f"your shop is usually closed on {day}s"
        return f"{day}s are usually {'busy' if up else 'slow'} for you"
    if group == "festival":
        if r["days_to_festival"] == 0:
            return f"it's {r['festival_name']} and most shops close early"
        if r["festival_name"] and up and r["days_to_festival"] <= FESTIVAL_RUSH_DAYS:
            when = "tomorrow" if r["days_to_festival"] == 1 else f"{r['days_to_festival']} days away"
            return f"{r['festival_name']} is {when}, festival shopping has started"
        return "there is no festival rush right now"
    if group == "month_start":
        return ("it's the start of the month and customers have just been paid" if up
                else "it's late in the month and customers have less money")
    if group == "season":
        if r["month"] in (7, 8, 9) and not up:
            return "monsoon rain usually slows sales this time of year"
        return f"this time of year is usually {'busy' if up else 'slow'}"
    return f"your sales over the last few days have been {'strong' if up else 'weak'}"
