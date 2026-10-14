# Gullak: AI Daily Savings Planner for Shopkeepers

Small shopkeepers (kirana stores, sabzi vendors, chai stalls) earn a different amount every day.
A fixed daily auto-saving often **fails** on slow days, when the money just isn't there, and
users who see failed payments tend to stop saving.

Gullak forecasts tomorrow's income with LightGBM and sets tomorrow's saving to match:
it saves more on busy days, less on slow days, and takes a rest day when the shop is
likely closed. It also explains why in plain words ("your shop is usually closed on Mondays").

## Results (6-month backtest, 300 shops, Jul–Dec 2025)

| | Fixed daily saving | % of yesterday's income | Simple rule | **Gullak (AI)** | % of today's income* |
|---|---|---|---|---|---|
| Failed payments per shop per month | 5.8 | 5.4 | 4.3 | **3.3** (43% fewer) | 3.2 |
| Failed payment rate | 19.1% | 18.6% | 16.8% | **12.4%** | 11.0% |
| Share of savings goal reached | 79.2% | 75.2% | 66.6% | **86.7%** (9.5% more money saved) | 92.9% |

\* Impossible in practice: the saving amount must be set a day in advance, before today's
income is known. Shown as a reference. The AI almost matches it.

**Assumption check (`sensitivity.py`):** the table above assumes leftover money is spent
(on stock and family) rather than kept in the bank. If most of it stays in the bank, even a
fixed amount rarely fails and the forecast adds little. Combining the AI forecast with the
current bank balance beats the fixed amount on both failures and money saved at every level.

Forecast error (average error as % of income): **25.0%** for the AI model,
vs 34.2% for "last 7-day average" and 38.4% for "same day last week".

> Data is synthetic: there is no public dataset of Indian shopkeepers' daily sales, so
> `generate_data.py` simulates realistic patterns (weekdays, weekly off-days, festivals,
> month-start salary days, monsoon, random closures).

## How it works

```
generate_data.py  ->  data/daily_income.csv  (300 shops x 2 years)
features.py       ->  turns past income + calendar into model inputs
train.py          ->  2 LightGBM models: "typical" (median) and "safe" (20th percentile)
planner.py        ->  savings rule: save goal (+ catch-up), capped by the SAFE income forecast
backtest.py       ->  replays 6 months: fixed vs simple rule vs AI
app.py            ->  Flask + SQLite web app (record income -> settle saving -> plan tomorrow)
```

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt

python generate_data.py   # ~5 s
python train.py           # ~5 s
python backtest.py        # ~4 s
python app.py             # first run builds the demo database (~20 s), then open http://127.0.0.1:5000
```

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/shops` | List demo shops |
| GET | `/api/shops/<id>` | Shop summary, tomorrow's plan, last 30 days |
| POST | `/api/shops/<id>/income` | Record a day's income `{"date": "2025-10-06", "income": 5400}` |
| POST | `/api/shops/<id>/simulate-day` | Demo: record the next day using the dataset |

Recording a day settles that day's saving and plans the next day in **one database
transaction**. `PRIMARY KEY (shop_id, date)` makes sure a day can never be recorded or saved
twice (returns `409`).
