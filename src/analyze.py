"""Project 1: Maize and bean retail prices in Kenya (WFP data).

Steps: load -> standardise units -> keep retail maize and beans -> flag outliers
-> fixed market panel -> seasonality -> regional comparison -> forecast vs baselines.

Run:
    python src/analyze.py --data data/raw/wfp_kenya_food_prices.csv \
        --start 2015-01 --end 2020-12 --compare-start 2024-09 --compare-min-months 1
"""
import argparse, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--start", default="2015-01")
p.add_argument("--end", default="2020-12")
p.add_argument("--min-coverage", type=float, default=0.8, help="share of months a market must report")
p.add_argument("--compare-start", default="2024-09")
p.add_argument("--compare-min-months", type=int, default=1)
p.add_argument("--pricetype", default="Retail")
p.add_argument("--out", default="outputs")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
R = {}  # results written to results.json

# 1. Load and standardise units -------------------------------------------------
df = pd.read_csv(a.data)
R["raw_rows"] = len(df)
R["raw_markets"] = int(df.market.nunique())
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M")


def to_base(unit):
    """'90 KG' -> (90, 'kg'); '500 ML' -> (0.5, 'l'); returns (None, None) if not a mass/volume."""
    m = re.match(r"^\s*([\d.]+)?\s*(KG|G|L|ML)\s*$", str(unit).upper())
    if not m:
        return (None, None)
    q = float(m.group(1) or 1)
    u = m.group(2)
    return {"KG": (q, "kg"), "G": (q / 1000, "kg"), "L": (q, "l"), "ML": (q / 1000, "l")}[u]


conv = df["unit"].map(to_base)
df["qty"] = conv.map(lambda t: t[0])
df["base"] = conv.map(lambda t: t[1])
R["rows_convertible_units"] = int(df.qty.notna().sum())
R["rows_non_weight_units"] = int(df.qty.isna().sum())
df["price_per_base"] = df["price"] / df["qty"]

# 2. Keep retail maize (grain) and beans, price per kg ---------------------------
is_grain = df.commodity.str.match(r"^(Maize( \(white(, dry)?\))?|Beans.*)$")
df["product"] = np.where(df.commodity.str.startswith("Maize"), "Maize", "Beans")
R["rows_wholesale_removed"] = int(((df.pricetype != a.pricetype) & is_grain).sum())
g = df[is_grain & (df.pricetype == a.pricetype) & (df.base == "kg")].copy()
R["grain_retail_rows"] = len(g)
R["grain_retail_markets"] = int(g.market.nunique())

# monthly value per market and product (median if several varieties)
mk = g.groupby(["market", "admin1", "product", "month"], as_index=False)["price_per_base"].median()

# 3. Outliers: compare with the local 7-month median of the same series ----------
flags = []
for (m, pr), s in mk.groupby(["market", "product"]):
    s = s.sort_values("month").set_index("month")["price_per_base"]
    s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
    loc = s.rolling(7, center=True, min_periods=3).median()
    bad = (s > 2 * loc) | (s < 0.5 * loc)
    for mo in s.index[bad.fillna(False)]:
        flags.append((m, pr, mo))
fl = pd.DataFrame(flags, columns=["market", "product", "month"])
fl["flag"] = True
mk = mk.merge(fl, on=["market", "product", "month"], how="left")
mk["flag"] = mk["flag"].fillna(False).astype(bool)
R["outliers_flagged"] = int(mk.flag.sum())
clean = mk[~mk.flag].copy()
R["clean_market_months"] = len(clean)
mk[mk.flag].to_csv(f"{a.out}/flagged_outliers.csv", index=False)

# 4. Fixed panel of markets, normalised per market -------------------------------
w = clean[(clean.month >= pd.Period(a.start)) & (clean.month <= pd.Period(a.end))]
n_months = len(pd.period_range(a.start, a.end, freq="M"))
panel = {}
for pr in ["Maize", "Beans"]:
    cov = w[w["product"] == pr].groupby("market").month.nunique() / n_months
    panel[pr] = sorted(cov[cov >= a.min_coverage].index)
R["panel_markets"] = {k: v for k, v in panel.items()}
R["panel_window"] = [a.start, a.end]

series = {}
for pr in ["Maize", "Beans"]:
    x = w[(w["product"] == pr) & (w.market.isin(panel[pr]))].copy()
    x["idx"] = x["price_per_base"] / x.groupby("market")["price_per_base"].transform("mean") * 100
    series[pr] = x.groupby("month")["idx"].mean()
    series[pr].index = series[pr].index.to_timestamp()
pd.DataFrame(series).to_csv(f"{a.out}/panel_monthly_index.csv")

# chart 1: price index
fig, ax = plt.subplots(figsize=(9, 4))
for pr, c in [("Maize", "#1f77b4"), ("Beans", "#d95f02")]:
    ax.plot(series[pr].index, series[pr].values, label=f"{pr} ({len(panel[pr])} markets)", color=c)
ax.set_title(f"Retail price index, same markets only ({a.start} to {a.end}; each market's average = 100)")
ax.set_ylabel("Index"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{a.out}/01_price_trends.png", dpi=130); plt.close(fig)

# 5. Seasonality (maize and beans) ----------------------------------------------
season = {}
for pr in ["Maize", "Beans"]:
    s = series[pr]
    yr = s.groupby(s.index.year).transform("mean")
    rel = (s / yr * 100).groupby(s.index.month).mean()
    season[pr] = rel
R["seasonality"] = {}
for pr, rel in season.items():
    R["seasonality"][pr] = {
        "peak_month": int(rel.idxmax()), "peak_value": round(float(rel.max()), 1),
        "low_month": int(rel.idxmin()), "low_value": round(float(rel.min()), 1),
        "swing_pct": round(float((rel.max() - rel.min()) / rel.min() * 100), 1)}
fig, ax = plt.subplots(figsize=(8, 4))
mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for pr, c in [("Maize", "#1f77b4"), ("Beans", "#d95f02")]:
    ax.plot(mn, season[pr].values, marker="o", label=pr, color=c)
ax.axhline(100, color="grey", lw=.8)
ax.set_title("Typical month vs the year's average (100)"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{a.out}/02_seasonality.png", dpi=130); plt.close(fig)

# 6. Regional comparison in the latest period -----------------------------------
cmp_ = clean[(clean.month >= pd.Period(a.compare_start)) & (clean["product"] == "Maize")]
reg = cmp_.groupby("admin1").agg(median_price=("price_per_base", "median"),
                                  markets=("market", "nunique")).sort_values("median_price")
reg = reg[reg.markets >= a.compare_min_months]
reg["median_price"] = reg["median_price"].round(1)
reg.to_csv(f"{a.out}/market_comparison.csv")
R["region_compare"] = {"from": a.compare_start, "table": reg.reset_index().to_dict("records")}
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(reg.index, reg.median_price, color="#1f77b4")
for i, (v, n) in enumerate(zip(reg.median_price, reg.markets)):
    ax.text(v + .5, i, f"{v} ({n} mkts)", va="center", fontsize=8)
ax.set_title(f"Median retail maize price by region, since {a.compare_start} (KES/kg)")
fig.tight_layout(); fig.savefig(f"{a.out}/03_market_comparison.png", dpi=130); plt.close(fig)

# 7. Forecast: Ridge vs naive and seasonal-naive on 12 held-out months ----------
y = series["Maize"].asfreq("MS").interpolate(limit=2)
X = pd.DataFrame({f"lag{k}": y.shift(k) for k in (1, 2, 3, 12)})
for k in range(1, 12):
    X[f"m{k}"] = (y.index.month == k).astype(int)
d = pd.concat([y.rename("y"), X], axis=1).dropna()
test = d.iloc[-12:]; train = d.iloc[:-12]
model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(train.drop(columns="y"), train.y)
pred = pd.Series(model.predict(test.drop(columns="y")), index=test.index)
naive = test["lag1"]; snaive = test["lag12"]
mae = lambda p_: float(np.mean(np.abs(test.y - p_)))
R["forecast"] = {"test_months": 12, "test_from": str(test.index[0].date()), "test_to": str(test.index[-1].date()),
                 "mae_ridge": round(mae(pred), 2), "mae_naive": round(mae(naive), 2),
                 "mae_seasonal_naive": round(mae(snaive), 2), "unit": "index points (index = 100 is each market's average)"}
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(d.index[-36:], d.y.iloc[-36:], label="Actual", color="black")
ax.plot(pred.index, pred, label="Ridge (one step ahead)", color="#d95f02")
ax.plot(naive.index, naive, label="Naive (last month)", color="#1f77b4", ls="--")
ax.axvspan(test.index[0], test.index[-1], color="grey", alpha=.12, label="Test months")
ax.set_title("Maize price index: forecast vs actual"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{a.out}/04_forecast.png", dpi=130); plt.close(fig)

json.dump(R, open(f"{a.out}/results.json", "w"), indent=2, default=str)
print(json.dumps(R, indent=2, default=str))
