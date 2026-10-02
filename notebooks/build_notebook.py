import nbformat as nbf, sys, json
NAMES_JSON = sys.argv[1] if len(sys.argv) > 1 else "{}"
nb = nbf.v4.new_notebook(); C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t))
def code(t): C.append(nbf.v4.new_code_cell(t))

md("""# Customer Segmentation and Churn Risk (Online Retail II)
Segment customers with RFM + K-Means, predict who will stop buying, and size the revenue at risk.

**Data:** UCI Online Retail II (CC BY 4.0), a UK online gift retailer, 2009-12-01 to 2011-12-09. This is **UK data**, not Australian.

**Churn definition (no label exists in the data):** a customer who has bought before the snapshot date is *churned* if they make **no purchase in the next 90 days**.
**Leakage control:** features use only invoices before the snapshot; the label uses only the 90 days after it. Training uses earlier snapshots, testing uses a later one, so the test period is never seen in training.""")

code("""import warnings, json, platform, sqlite3
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (silhouette_score, roc_auc_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, f1_score, roc_curve, precision_recall_curve)
warnings.filterwarnings("ignore")
SEED, HORIZON = 42, 90
RAW = "../data/online_retail_II.xlsx"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})""")

md("## 1. Load and clean (every step is counted)")
code("""sheets = pd.read_excel(RAW, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
raw = pd.concat(sheets.values(), ignore_index=True).rename(columns={"Customer ID": "CustomerID"})
steps = [("Raw rows", len(raw))]
df = raw.drop_duplicates(); steps.append(("After dropping exact duplicate rows", len(df)))
df = df[df.CustomerID.notna()].copy(); steps.append(("After dropping rows with no CustomerID", len(df)))
df["CustomerID"] = df.CustomerID.astype(int)
df["is_cancel"] = df.Invoice.str.startswith("C")
product = df.StockCode.str.match(r"^\\d{5}")          # real product codes; drops POST, DOT, M, BANK CHARGES, etc.
sales = df[~df.is_cancel & (df.Quantity > 0) & (df.Price > 0) & product].copy()
steps.append(("Sales lines (not cancelled, qty>0, price>0, product code)", len(sales)))
returns = df[df.is_cancel & (df.Quantity < 0) & product].copy()
steps.append(("Return lines (cancelled invoices, product code)", len(returns)))
sales["Revenue"] = sales.Quantity * sales.Price
returns["ReturnValue"] = -returns.Quantity * returns.Price
print(pd.DataFrame(steps, columns=["Step", "Rows"]).to_string(index=False))
print("\\nCustomers with sales:", sales.CustomerID.nunique(), "| invoices:", sales.Invoice.nunique())
print("Date range:", sales.InvoiceDate.min(), "to", sales.InvoiceDate.max())
print("UK share of revenue: %.1f%%" % (100 * sales.loc[sales.Country == "United Kingdom", "Revenue"].sum() / sales.Revenue.sum()))
pd.DataFrame(steps, columns=["step", "rows"]).to_csv("../outputs/cleaning_log.csv", index=False)""")

md("""## 2. Features and churn label at a snapshot
Snapshots are 90 days apart. **Test snapshot: 2011-09-09** (label window 2011-09-10 to 2011-12-08).
**Training snapshots:** 2011-06-11, 2011-03-13 and 2010-12-12, whose label windows all end before the test snapshot.""")
code("""def build(snap):
    cutoff = pd.Timestamp(snap) + pd.Timedelta(days=1)          # history = strictly before this
    end = cutoff + pd.Timedelta(days=HORIZON)                    # label window = [cutoff, end)
    h = sales[sales.InvoiceDate < cutoff]
    assert h.InvoiceDate.max() < cutoff                          # leakage guard
    inv = h.groupby(["CustomerID", "Invoice"]).agg(date=("InvoiceDate", "max"), value=("Revenue", "sum")).reset_index()
    f = inv.groupby("CustomerID").agg(frequency=("Invoice", "count"), monetary=("value", "sum"),
                                      last=("date", "max"), first=("date", "min"))
    f["recency_days"] = (cutoff - f["last"]).dt.days
    f["tenure_days"] = (cutoff - f["first"]).dt.days
    f["avg_order_value"] = f.monetary / f.frequency
    for d in (90, 180, 365):
        w = inv[inv.date >= cutoff - pd.Timedelta(days=d)].groupby("CustomerID")
        f[f"orders_{d}d"] = w.Invoice.count(); f[f"revenue_{d}d"] = w.value.sum()
    f[["orders_90d", "orders_180d", "orders_365d", "revenue_90d", "revenue_180d", "revenue_365d"]] = \\
        f[["orders_90d", "orders_180d", "orders_365d", "revenue_90d", "revenue_180d", "revenue_365d"]].fillna(0)
    f["single_order"] = (f.frequency == 1).astype(int)
    f["avg_gap_days"] = ((f["last"] - f["first"]).dt.days / (f.frequency - 1).replace(0, np.nan)).fillna(0)
    f["n_products"] = h.groupby("CustomerID").StockCode.nunique()
    f["uk"] = (h.groupby("CustomerID").Country.agg(lambda x: x.mode().iat[0]) == "United Kingdom").astype(int)
    r = returns[returns.InvoiceDate < cutoff].groupby("CustomerID").agg(return_invoices=("Invoice", "nunique"), return_value=("ReturnValue", "sum"))
    f = f.join(r).fillna({"return_invoices": 0, "return_value": 0})
    f["return_ratio"] = f.return_value / f.monetary
    active = set(sales[(sales.InvoiceDate >= cutoff) & (sales.InvoiceDate < end)].CustomerID)
    f["churn"] = (~f.index.isin(active)).astype(int)
    f["snapshot"] = pd.Timestamp(snap)
    return f.drop(columns=["last", "first"])

TEST_SNAP = "2011-09-09"
TRAIN_SNAPS = ["2010-12-12", "2011-03-13", "2011-06-11"]
tr = {s: build(s) for s in TRAIN_SNAPS}; te = build(TEST_SNAP)
for s, d in {**tr, TEST_SNAP: te}.items():
    print(s, "| customers:", len(d), "| churn rate: %.1f%%" % (100 * d.churn.mean()))
FEATS = ["recency_days", "frequency", "monetary", "tenure_days", "avg_order_value", "orders_90d", "orders_180d", "orders_365d",
         "revenue_90d", "revenue_180d", "revenue_365d", "single_order", "avg_gap_days", "n_products", "uk",
         "return_invoices", "return_ratio"]""")

md("""## 3. SQL check of RFM (SQLite)
The same Recency, Frequency and Monetary values are computed in SQL with CTEs and window functions (`NTILE`) and must match pandas exactly.""")
code("""con = sqlite3.connect(":memory:")
cut = pd.Timestamp(TEST_SNAP) + pd.Timedelta(days=1)
sales[sales.InvoiceDate < cut][["CustomerID", "Invoice", "InvoiceDate", "Revenue"]].assign(
    InvoiceDate=lambda x: x.InvoiceDate.dt.strftime("%Y-%m-%d %H:%M:%S")).to_sql("sales", con, index=False)
SQL = '''
WITH inv AS (
  SELECT CustomerID, Invoice, MAX(InvoiceDate) AS d, SUM(Revenue) AS v
  FROM sales GROUP BY CustomerID, Invoice),
rfm AS (
  SELECT CustomerID,
         CAST(julianday(:cutoff) - julianday(MAX(d)) AS INTEGER) AS recency_days,
         COUNT(*) AS frequency, SUM(v) AS monetary
  FROM inv GROUP BY CustomerID),
scored AS (
  SELECT *, NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
            NTILE(5) OVER (ORDER BY frequency, monetary)  AS f_score,
            NTILE(5) OVER (ORDER BY monetary)             AS m_score
  FROM rfm)
SELECT *, r_score * 100 + f_score * 10 + m_score AS rfm_code FROM scored;'''
sql_rfm = pd.read_sql(SQL, con, params={"cutoff": cut.strftime("%Y-%m-%d %H:%M:%S")}).set_index("CustomerID")
chk = te[["recency_days", "frequency", "monetary"]].join(sql_rfm, rsuffix="_sql", how="inner")
assert len(sql_rfm) == len(te)
assert (chk.recency_days == chk.recency_days_sql).all() and (chk.frequency == chk.frequency_sql).all()
assert np.allclose(chk.monetary, chk.monetary_sql, atol=0.01)
print("SQL and pandas RFM match for all", len(chk), "customers.")
print(sql_rfm.rfm_code.value_counts().head(5).rename("customers").to_frame().T.to_string())
open("../sql_rfm.sql", "w").write(SQL.strip() + "\\n")""")

md("""## 4. Segmentation (RFM + K-Means) at the test snapshot
Churn outcomes are **not** used to build segments. They are only used afterwards to check whether the segments mean something.""")
code("""rfm = te[["recency_days", "frequency", "monetary"]]
X = StandardScaler().fit_transform(np.log1p(rfm))
rows = []
for k in range(3, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    rows.append((k, km.inertia_, silhouette_score(X, km.labels_, random_state=SEED)))
ks = pd.DataFrame(rows, columns=["k", "inertia", "silhouette"]); print(ks.round(3).to_string(index=False))
K = int(ks.loc[ks.silhouette.idxmax(), "k"]); print("Chosen k (highest silhouette):", K)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(ks.k, ks.inertia, "o-"); ax[0].set_title("Inertia"); ax[1].plot(ks.k, ks.silhouette, "o-"); ax[1].set_title("Silhouette")
for a in ax: a.set_xlabel("k")
plt.tight_layout(); plt.savefig("../images/choose_k.png"); plt.show()
km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(X)
te["cluster"] = km.labels_""")
code("""prof = te.groupby("cluster").agg(customers=("frequency", "size"), median_recency_days=("recency_days", "median"),
        median_orders=("frequency", "median"), median_spend=("monetary", "median"), total_spend=("monetary", "sum"),
        observed_churn_rate=("churn", "mean"))
prof["pct_customers"] = 100 * prof.customers / prof.customers.sum(); prof["pct_spend"] = 100 * prof.total_spend / prof.total_spend.sum()
print(prof.round(1).sort_values("median_recency_days").to_string())""")

code("""# Rule-based names from each cluster's medians, so the labels survive a re-run
med = prof[["median_recency_days", "median_spend"]]
NAMES = {c: f"Segment {c}" for c in prof.index}
if K == 4:
    champ, dormant = med.median_spend.idxmax(), med.median_recency_days.idxmax()
    rest = [c for c in med.index if c not in (champ, dormant)]
    recent = min(rest, key=lambda c: med.loc[c, "median_recency_days"]); atrisk = [c for c in rest if c != recent][0]
    NAMES.update({champ: "Champions", atrisk: "At-risk regulars", dormant: "Dormant one-time buyers", recent: "Recent light buyers"})
te["segment"] = te.cluster.map(NAMES)
prof.index = prof.index.map(NAMES); prof.index.name = "segment"
prof = prof.sort_values("pct_spend", ascending=False)
print(prof.round(1).to_string()); prof.round(2).to_csv("../outputs/segment_profile.csv")""")

md("""## 5. Churn models
Train on the three earlier snapshots, test on the later snapshot. Model choice is made on a validation step (train on the first two snapshots, validate on the third) before the test snapshot is touched.""")
code("""def lr_model(): return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED))
def rf_model(): return RandomForestClassifier(n_estimators=400, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=SEED)
def fit_eval(make, train_df, test_df):
    m = make().fit(train_df[FEATS], train_df.churn); p = m.predict_proba(test_df[FEATS])[:, 1]
    return m, p
# validation: first two snapshots -> third
val_tr = pd.concat([tr[TRAIN_SNAPS[0]], tr[TRAIN_SNAPS[1]]]); val_te = tr[TRAIN_SNAPS[2]]
val_auc = {n: roc_auc_score(val_te.churn, fit_eval(f, val_tr, val_te)[1]) for n, f in [("Logistic regression", lr_model), ("Random forest", rf_model)]}
print("Validation AUC:", {k: round(v, 3) for k, v in val_auc.items()})
CHOSEN = max(val_auc, key=val_auc.get); print("Model chosen on validation:", CHOSEN)
train_all = pd.concat(tr.values())
print("Training rows:", len(train_all), "| training churn rate: %.1f%%" % (100 * train_all.churn.mean()))
models, probs = {}, {}
for n, f in [("Logistic regression", lr_model), ("Random forest", rf_model)]:
    models[n], probs[n] = fit_eval(f, train_all, te)
probs["Recency only (baseline)"] = te.recency_days / te.recency_days.max()""")

code("""def top_k_stats(y, p, frac=0.2):
    n = int(len(y) * frac); idx = np.argsort(-np.asarray(p))[:n]; yy = np.asarray(y)[idx]
    return yy.mean(), yy.sum() / np.sum(y)
res = []
for n, p in probs.items():
    prec20, rec20 = top_k_stats(te.churn, p)
    row = {"model": n, "AUC": roc_auc_score(te.churn, p), "avg_precision": average_precision_score(te.churn, p),
           "precision_top20pct": prec20, "recall_top20pct": rec20, "lift_top20pct": prec20 / te.churn.mean()}
    if n != "Recency only (baseline)":
        pred = (np.asarray(p) >= 0.5).astype(int)
        row.update({"brier": brier_score_loss(te.churn, p), "precision@0.5": precision_score(te.churn, pred),
                    "recall@0.5": recall_score(te.churn, pred), "f1@0.5": f1_score(te.churn, pred)})
    res.append(row)
res = pd.DataFrame(res).set_index("model")
print("Test snapshot %s | customers: %d | churn rate: %.1f%%" % (TEST_SNAP, len(te), 100 * te.churn.mean()))
print(res.round(3).to_string()); res.round(4).to_csv("../outputs/model_metrics.csv")""")

code("""fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for n, p in probs.items():
    fpr, tpr, _ = roc_curve(te.churn, p); ax[0].plot(fpr, tpr, label=f"{n} (AUC {roc_auc_score(te.churn, p):.2f})")
    pr, rc, _ = precision_recall_curve(te.churn, p); ax[1].plot(rc, pr, label=n)
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_title("ROC (test)"); ax[0].legend(frameon=False, fontsize=8)
ax[1].axhline(te.churn.mean(), color="k", ls=":"); ax[1].set_title("Precision-recall (test)"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
best_p = probs[CHOSEN]; dec = pd.qcut(pd.Series(best_p).rank(method="first"), 10, labels=False)
ax[2].bar(range(1, 11), pd.Series(te.churn.values).groupby(dec.values).mean().values * 100)
ax[2].axhline(te.churn.mean() * 100, color="k", ls=":"); ax[2].set_title(f"Observed churn % by risk decile ({CHOSEN})"); ax[2].set_xlabel("Risk decile (10 = highest)")
plt.tight_layout(); plt.savefig("../images/model_performance.png"); plt.show()
imp = pd.Series(models["Random forest"].feature_importances_, index=FEATS).sort_values()
imp.plot.barh(figsize=(6, 4.5), title="Random forest feature importance"); plt.tight_layout(); plt.savefig("../images/feature_importance.png"); plt.show()
print(imp.sort_values(ascending=False).head(6).round(3).to_string())""")

md("""## 6. Revenue at risk
Expected quarterly value = trailing 365-day revenue x 90/365. Revenue at risk = predicted churn probability x expected quarterly value.
Check: compare the predicted total with the value of the customers who actually churned.""")
code("""te["p_churn"] = probs[CHOSEN]
te["quarterly_value"] = te.revenue_365d * HORIZON / 365
te["revenue_at_risk"] = te.p_churn * te.quarterly_value
te["realised_at_risk"] = te.churn * te.quarterly_value
seg = te.groupby("segment").agg(customers=("churn", "size"), observed_churn_rate=("churn", "mean"), mean_predicted_churn=("p_churn", "mean"),
        quarterly_value=("quarterly_value", "sum"), predicted_revenue_at_risk=("revenue_at_risk", "sum"), realised_revenue_at_risk=("realised_at_risk", "sum"))
seg["pct_of_total_at_risk"] = 100 * seg.predicted_revenue_at_risk / seg.predicted_revenue_at_risk.sum()
seg = seg.sort_values("predicted_revenue_at_risk", ascending=False)
print(seg.round(2).to_string()); seg.round(2).to_csv("../outputs/revenue_at_risk_by_segment.csv")
print("\\nTotal predicted revenue at risk: %.0f | realised (value of customers who actually churned): %.0f | ratio %.2f" %
      (te.revenue_at_risk.sum(), te.realised_at_risk.sum(), te.revenue_at_risk.sum() / te.realised_at_risk.sum()))
print("Total expected quarterly value of all customers: %.0f" % te.quarterly_value.sum())
fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(seg)); wd = 0.38
ax.bar(x - wd/2, seg.predicted_revenue_at_risk / 1e3, wd, label="Predicted"); ax.bar(x + wd/2, seg.realised_revenue_at_risk / 1e3, wd, label="Realised")
ax.set_xticks(x); ax.set_xticklabels(seg.index, rotation=20, ha="right"); ax.set_ylabel("GBP thousand per quarter"); ax.legend(frameon=False)
ax.set_title("Revenue at risk by segment (test snapshot)"); plt.tight_layout(); plt.savefig("../images/revenue_at_risk.png"); plt.show()
te.reset_index()[["CustomerID", "segment", "recency_days", "frequency", "monetary", "p_churn", "quarterly_value", "revenue_at_risk", "churn"]].round(3).to_csv("../outputs/customers_scored_test_snapshot.csv", index=False)""")

code("""cfg = {"seed": SEED, "horizon_days": HORIZON, "test_snapshot": TEST_SNAP, "train_snapshots": TRAIN_SNAPS, "k": K, "model_chosen_on_validation": CHOSEN,
       "validation_auc": val_auc, "features": FEATS, "segment_names": {str(k): v for k, v in NAMES.items()},
       "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__}}
json.dump(cfg, open("../outputs/run_config.json", "w"), indent=2); print(json.dumps(cfg, indent=2)[:900])""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/analysis.ipynb")
