"""Gullak web app: Flask + SQLite.

Every evening the shopkeeper's income for the day is recorded. The app then:
  1. settles today's planned saving (saved / failed / rest day),
  2. forecasts tomorrow's income with the AI model,
  3. plans tomorrow's saving and explains the reason in plain words.

All of this happens inside ONE database transaction, so it either fully
happens or not at all. A day can never be recorded (or saved) twice.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import sqlite3
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import build_features
from forecast import Forecaster
from planner import plan_amount, plan_message, settle

ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "gullak.db"
DEMO_TODAY = pd.Timestamp("2025-10-05")   # demo starts ~2 weeks before Diwali (20 Oct 2025)
JOINED_DAYS = 60                          # demo shops "joined" Gullak 60 days before that
HISTORY_DAYS = 40                         # past days needed to build features for one forecast
# 12 typical shops (2-3 of each type) shown in the demo. Backtest numbers cover all 300 shops.
DEMO_SHOP_IDS = [1, 2, 4, 8, 23, 24, 39, 18, 19, 30, 34, 98]

SCHEMA = """
CREATE TABLE shops (
    id INTEGER PRIMARY KEY, name TEXT, shop_type TEXT,
    margin REAL, home_expense REAL, goal_per_day INTEGER
);
CREATE TABLE daily_income (
    shop_id INTEGER, date TEXT, income REAL,
    PRIMARY KEY (shop_id, date)           -- one income entry per shop per day
);
CREATE TABLE plans (
    shop_id INTEGER, date TEXT, amount INTEGER,
    typical_income REAL, safe_income REAL, message TEXT,
    PRIMARY KEY (shop_id, date)
);
CREATE TABLE savings (
    shop_id INTEGER, date TEXT, planned INTEGER, saved INTEGER, status TEXT,
    PRIMARY KEY (shop_id, date)           -- a day can never be saved twice
);
"""

app = Flask(__name__)
forecaster = Forecaster()
# Only used by the demo "simulate next day" button, to replay real (generated) future income.
dataset = pd.read_csv(ROOT / "data" / "daily_income.csv", parse_dates=["date"])


class DayError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def day_str(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def next_day(date):
    return day_str(pd.Timestamp(date) + pd.Timedelta(days=1))


def get_shop(conn, shop_id):
    shop = conn.execute("SELECT * FROM shops WHERE id = ?", (shop_id,)).fetchone()
    if shop is None:
        raise DayError("Shop not found", 404)
    return shop


def last_income_date(conn, shop_id):
    return conn.execute("SELECT MAX(date) FROM daily_income WHERE shop_id = ?", (shop_id,)).fetchone()[0]


def make_plan(conn, shop, date):
    """Forecast income for `date` and store how much to save that day."""
    rows = conn.execute(
        "SELECT date, income FROM daily_income WHERE shop_id = ? AND date < ? ORDER BY date DESC LIMIT ?",
        (shop["id"], date, HISTORY_DAYS)).fetchall()
    hist = pd.DataFrame([tuple(r) for r in reversed(rows)] + [(date, None)], columns=["date", "income"])
    hist["date"] = pd.to_datetime(hist["date"])
    hist["income"] = hist["income"].astype(float)
    hist["shop_id"], hist["shop_type"] = shop["id"], shop["shop_type"]
    row = build_features(hist).iloc[[-1]]

    typical, safe = forecaster.predict(row)
    saved, days = conn.execute(
        "SELECT COALESCE(SUM(saved), 0), COUNT(*) FROM savings WHERE shop_id = ?", (shop["id"],)).fetchone()
    goal = shop["goal_per_day"]
    amount = plan_amount(safe[0], goal, shop["margin"], shop["home_expense"], behind_by=goal * days - saved)
    direction = "down" if amount < goal else "up" if amount > goal else "any"
    message = plan_message(amount, goal, forecaster.explain(row, direction))
    conn.execute("INSERT OR REPLACE INTO plans VALUES (?, ?, ?, ?, ?, ?)",
                 (shop["id"], date, amount, round(typical[0]), round(safe[0]), message))


def record_day(conn, shop, date, income):
    """Record one day's income, settle that day's saving, and plan the next day."""
    last = last_income_date(conn, shop["id"])
    if last and date <= last:
        raise DayError(f"Income for {date} is already recorded", 409)
    if last and date != next_day(last):
        raise DayError(f"Days must be recorded in order. Next day to record is {next_day(last)}")

    conn.execute("INSERT INTO daily_income VALUES (?, ?, ?)", (shop["id"], date, income))
    plan = conn.execute("SELECT amount FROM plans WHERE shop_id = ? AND date = ?", (shop["id"], date)).fetchone()
    if plan is None:                       # first day after joining: no plan was made the evening before
        make_plan(conn, shop, date)
        plan = conn.execute("SELECT amount FROM plans WHERE shop_id = ? AND date = ?", (shop["id"], date)).fetchone()

    status, saved = settle(plan["amount"], income, shop["margin"], shop["home_expense"])
    conn.execute("INSERT INTO savings VALUES (?, ?, ?, ?, ?)", (shop["id"], date, plan["amount"], saved, status))
    make_plan(conn, shop, next_day(date))
    return {"date": date, "income": income, "planned": plan["amount"], "saved": saved, "status": status}


def setup_demo_db():
    """Create the database with 12 demo shops that joined 60 days before DEMO_TODAY."""
    print("Setting up demo database (one time, ~20s)...")
    DB_PATH.unlink(missing_ok=True)
    shops = pd.read_csv(ROOT / "data" / "shops.csv")
    demo = shops[shops["shop_id"].isin(DEMO_SHOP_IDS)]
    join_date = DEMO_TODAY - pd.Timedelta(days=JOINED_DAYS - 1)

    conn = connect()
    conn.executescript(SCHEMA)
    with conn:
        for s in demo.itertuples():
            conn.execute("INSERT INTO shops VALUES (?, ?, ?, ?, ?, ?)",
                         (s.shop_id, s.name, s.shop_type, s.margin, s.home_expense, s.goal_per_day))
            days = dataset[(dataset["shop_id"] == s.shop_id) & (dataset["date"] <= DEMO_TODAY)]
            before = days[days["date"] < join_date]
            conn.executemany("INSERT INTO daily_income VALUES (?, ?, ?)",
                             [(s.shop_id, day_str(d), float(i)) for d, i in zip(before["date"], before["income"])])
            shop = get_shop(conn, s.shop_id)
            for d, i in zip(days["date"][days["date"] >= join_date], days["income"][days["date"] >= join_date]):
                record_day(conn, shop, day_str(d), float(i))
    conn.close()


# ---------------------------------------------------------------- API

@app.errorhandler(DayError)
def handle_day_error(e):
    return jsonify(error=str(e)), e.status


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/shops")
def list_shops():
    conn = connect()
    rows = conn.execute("SELECT id, name, shop_type, goal_per_day FROM shops ORDER BY id").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.get("/api/shops/<int:shop_id>")
def shop_detail(shop_id):
    conn = connect()
    shop = get_shop(conn, shop_id)
    today = last_income_date(conn, shop_id)
    plan = conn.execute("SELECT * FROM plans WHERE shop_id = ? AND date = ?", (shop_id, next_day(today))).fetchone()
    s = conn.execute("""SELECT COUNT(*) AS days, COALESCE(SUM(saved), 0) AS saved,
                               SUM(status = 'failed') AS failed, SUM(status = 'rest') AS rest
                        FROM savings WHERE shop_id = ?""", (shop_id,)).fetchone()
    history = conn.execute("""SELECT d.date, d.income, s.planned, s.saved, s.status
                              FROM daily_income d JOIN savings s USING (shop_id, date)
                              WHERE d.shop_id = ? ORDER BY d.date DESC LIMIT 30""", (shop_id,)).fetchall()
    conn.close()
    return jsonify(
        shop=dict(shop), today=today, tomorrow=dict(plan),
        summary={**dict(s), "goal_so_far": shop["goal_per_day"] * s["days"]},
        history=[dict(h) for h in history],
    )


@app.post("/api/shops/<int:shop_id>/income")
def add_income(shop_id):
    body = request.get_json(silent=True) or {}
    try:
        date, income = day_str(body["date"]), float(body["income"])
    except (KeyError, ValueError, TypeError):
        raise DayError("Send JSON like {\"date\": \"2025-10-06\", \"income\": 5400}")
    if income < 0:
        raise DayError("Income cannot be negative")
    return jsonify(save_day(shop_id, date, income)), 201


@app.post("/api/shops/<int:shop_id>/simulate-day")
def simulate_day(shop_id):
    """Demo helper: record the next day using the shop's actual income from the dataset."""
    conn = connect()
    date = next_day(last_income_date(conn, shop_id))
    conn.close()
    match = dataset[(dataset["shop_id"] == shop_id) & (dataset["date"] == date)]
    if match.empty:
        raise DayError("No more demo data for this shop")
    return jsonify(save_day(shop_id, date, float(match["income"].iloc[0]))), 201


def save_day(shop_id, date, income):
    conn = connect()
    try:
        with conn:                         # one transaction: all steps happen, or none do
            return record_day(conn, get_shop(conn, shop_id), date, income)
    except sqlite3.IntegrityError:         # the PRIMARY KEY blocked a duplicate day
        raise DayError(f"Income for {date} is already recorded", 409)
    finally:
        conn.close()


if __name__ == "__main__":
    if not DB_PATH.exists():
        setup_demo_db()
    app.run(debug=False)
