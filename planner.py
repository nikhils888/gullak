"""The savings rule: how much to save tomorrow.

Plain idea:
  1. Take a CAREFUL estimate of tomorrow's income (income should beat it ~80% of days).
  2. Work out how much money that leaves after shop costs and home expenses.
  3. Save the daily goal (plus a bit extra if we are behind) - but never more than
     that careful amount. If it's too small, make it a "rest day" and save nothing.

A saving FAILS when the shopkeeper did not actually have that much spare money that day.
"""
MIN_SAVING = 20         # below Rs 20 we skip the day instead ("rest day")
CATCH_UP_SHARE = 0.25   # each day, try to recover 25% of what we are behind on
MAX_MULTIPLE = 3        # never ask for more than 3x the normal daily goal
ROUND_TO = 10           # amounts are rounded down to the nearest Rs 10


def spare_money(income, margin, home_expense):
    """Money left after shop costs (1 - margin) and the family's daily needs."""
    return income * margin - home_expense


def plan_amount(safe_income, goal_per_day, margin, home_expense, behind_by):
    """Tomorrow's saving, in rupees. 0 means rest day."""
    safe_spare = spare_money(safe_income, margin, home_expense)
    wanted = goal_per_day + CATCH_UP_SHARE * max(behind_by, 0)
    amount = min(wanted, safe_spare, MAX_MULTIPLE * goal_per_day)
    amount = int(amount // ROUND_TO * ROUND_TO)
    return amount if amount >= MIN_SAVING else 0


def settle(amount, income, margin, home_expense):
    """What actually happened at the end of the day: (status, amount saved)."""
    if amount == 0:
        return "rest", 0
    if amount <= spare_money(income, margin, home_expense):
        return "saved", amount
    return "failed", 0


def plan_message(amount, goal_per_day, reason):
    if amount == 0:
        return f"Rest day: {reason}, so no saving tomorrow."
    if amount < goal_per_day:
        return f"Saving a bit less: {reason}."
    if amount > goal_per_day:
        return f"Saving a bit more to catch up: {reason}."
    return f"Normal saving day: {reason}."
