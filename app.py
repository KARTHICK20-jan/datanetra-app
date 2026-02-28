"""
DataNetra.ai — Gradio App v2
Fixes: full-screen layout, removed pricing, high-contrast upload section & buttons
pip install gradio pandas numpy openpyxl scikit-learn plotly prophet
python dataNetra_gradio_v2.py  →  http://localhost:7860
"""
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, socket, warnings
warnings.filterwarnings("ignore")

# ── Palette ───────────────────────────────────────────────────────────────────
T   = "#00d4b4"
BG  = "#060d14"
B2  = "#0b1520"
B3  = "#111e2d"
B4  = "#162030"
GO  = "#f0a500"
CO  = "#e05c6a"
GRN = "#2ecc71"
PB  = "#0b1520"
GC  = "rgba(0,212,180,.07)"

# ── Plot layout ───────────────────────────────────────────────────────────────
def dl(title="", h=370):
    return dict(
        title=dict(text=title, font=dict(color="#fff", size=13), x=0.01),
        paper_bgcolor=PB, plot_bgcolor=PB,
        font=dict(color="rgba(255,255,255,.55)", family="DM Sans,sans-serif", size=11),
        xaxis=dict(gridcolor=GC, zerolinecolor=GC,
                   tickfont=dict(color="rgba(255,255,255,.3)", size=9)),
        yaxis=dict(gridcolor=GC, zerolinecolor=GC,
                   tickfont=dict(color="rgba(255,255,255,.3)", size=9)),
        margin=dict(l=44, r=16, t=50, b=40), height=h,
        legend=dict(bgcolor="rgba(0,0,0,0)",
                    font=dict(color="rgba(255,255,255,.5)", size=10)),
    )

# ── Column detection ──────────────────────────────────────────────────────────
ALIASES = {
    "date":     ["date","month","period","time","week","day","year","dt",
                 "order_date","invoice_date","sale_date","transaction_date"],
    "product":  ["product","item","sku","name","product_name","item_name",
                 "goods","commodity","description","product_id","prod","article"],
    "revenue":  ["revenue","sales","amount","total","total_sales","gross","net",
                 "income","turnover","value","sale_amount","invoice_amount",
                 "price","gmv","billing"],
    "quantity": ["quantity","qty","units","count","volume","pieces","sold",
                 "units_sold","no_of_units","num_units"],
    "cost":     ["cost","cogs","expense","expenditure","cost_of_goods",
                 "purchase","buying_price","wholesale","cost_price"],
    "profit":   ["profit","margin","net_profit","gross_profit","earnings","pnl"],
    "category": ["category","cat","segment","department","division","type",
                 "product_type","section","class","subcategory"],
    "store":    ["store","branch","outlet","location","region","zone",
                 "area","shop","channel"],
}

def detect(df):
    found, cm = {}, {
        c.lower().strip().replace(" ","_").replace("-","_"): c
        for c in df.columns
    }
    for f, al in ALIASES.items():
        for a in al:
            if a in cm:
                found[f] = cm[a]
                break
    return found

def to_num(s):
    return pd.to_numeric(
        s.astype(str)
         .str.replace(r"[₹$€£,\s]", "", regex=True)
         .str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce")

def to_date(s):
    for fmt in [None, "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d",
                "%d-%m-%Y", "%b %Y", "%B %Y", "%Y"]:
        try:
            r = pd.to_datetime(s, format=fmt,
                               infer_datetime_format=True, errors="coerce")
            if r.notna().sum() > len(s) * .5:
                return r
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

# ── Health Score ──────────────────────────────────────────────────────────────
def calc_health(df, cols):
    sc = {}
    rc, cc, qc, pc, dc = (cols.get(k) for k in
                           ["revenue","cost","quantity","profit","date"])
    if rc and dc:
        tmp = df.copy()
        tmp["_d"] = to_date(tmp[dc])
        tmp["_r"] = to_num(tmp[rc])
        tmp = tmp.dropna(subset=["_d","_r"]).sort_values("_d")
        n = max(len(tmp)//3, 1)
        e, l = tmp.head(n)["_r"].sum(), tmp.tail(n)["_r"].sum()
        sc["Revenue Growth"] = min(max(((l-e)/e*100+10)/50*100,0),100) if e>0 else 50
    else:
        sc["Revenue Growth"] = 50
    if pc and rc:
        rv, pv = to_num(df[rc]).sum(), to_num(df[pc]).sum()
        sc["Profit Margin"] = min(max(pv/rv*200,0),100) if rv>0 else 55
    elif cc and rc:
        rv, cv = to_num(df[rc]).sum(), to_num(df[cc]).sum()
        sc["Profit Margin"] = min(max((rv-cv)/rv*200,0),100) if rv>0 else 55
    else:
        sc["Profit Margin"] = 55
    if cc and rc:
        rv, cv = to_num(df[rc]).sum(), to_num(df[cc]).sum()
        sc["Cost Control"] = min(max((1-cv/rv)*100+20,0),100) if rv>0 else 60
    else:
        sc["Cost Control"] = 60
    sc["Diversification"] = (min(df[cols["product"]].nunique()/20*100,100)
                              if cols.get("product") else 50)
    if qc:
        q = to_num(df[qc]).dropna()
        cv = q.std()/q.mean() if q.mean()>0 else 1
        sc["Volume Stability"] = min(max((1-cv)*100,0),100)
    else:
        sc["Volume Stability"] = 55
    return round(sum(sc.values())/len(sc), 1), sc

def hlabel(s):
    if s >= 80: return "Excellent 🟢", GRN
    if s >= 65: return "Good 🔵", T
    if s >= 50: return "Fair 🟡", GO
    return "Needs Attention 🔴", CO

# ── Gauge ─────────────────────────────────────────────────────────────────────
def make_gauge(score, sub):
    lbl, col = hlabel(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain={"x":[0,1],"y":[0,1]},
        title={"text": f"<span style='color:rgba(255,255,255,.45);font-size:13px'>{lbl}</span>"},
        number={"font":{"size":48,"color":col,"family":"DM Sans"},"suffix":"/100"},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,
                    "tickcolor":"rgba(255,255,255,.1)",
                    "tickfont":{"color":"rgba(255,255,255,.2)","size":9}},
            "bar":{"color":col,"thickness":0.22},
            "bgcolor":"rgba(0,0,0,0)", "borderwidth":0,
            "steps":[
                {"range":[0,50],  "color":"rgba(224,92,106,.1)"},
                {"range":[50,65], "color":"rgba(240,165,0,.08)"},
                {"range":[65,80], "color":"rgba(0,212,180,.07)"},
                {"range":[80,100],"color":"rgba(0,212,180,.14)"},
            ],
            "threshold":{"line":{"color":col,"width":3},
                         "thickness":0.7,"value":score},
        }
    ))
    fig.update_layout(paper_bgcolor=PB, plot_bgcolor=PB,
        font=dict(color="rgba(255,255,255,.6)", family="DM Sans"),
        height=280, margin=dict(l=28,r=28,t=28,b=8))

    bars = "".join([f"""
      <div style='margin-bottom:12px'>
        <div style='display:flex;justify-content:space-between;margin-bottom:5px'>
          <span style='font-size:.8rem;color:rgba(255,255,255,.6)'>{n}</span>
          <span style='font-size:.8rem;font-weight:700;
            color:{T if v>=70 else GO if v>=50 else CO}'>{v:.0f}/100</span>
        </div>
        <div style='background:rgba(255,255,255,.08);border-radius:20px;
                    height:7px;overflow:hidden'>
          <div style='width:{v:.0f}%;height:100%;
            background:{T if v>=70 else GO if v>=50 else CO};
            border-radius:20px'></div>
        </div>
      </div>""" for n, v in sub.items()])

    sub_html = f"""
    <div style='background:{B2};border:1px solid rgba(0,212,180,.2);
                border-radius:14px;padding:22px;height:100%'>
      <div style='font-size:.65rem;color:rgba(255,255,255,.4);
                  text-transform:uppercase;letter-spacing:.14em;
                  font-family:monospace;margin-bottom:18px'>KPI Sub-Scores</div>
      {bars}
    </div>"""
    return fig, sub_html

# ── Top 10 ────────────────────────────────────────────────────────────────────
def top10(df, cols):
    p, r = cols.get("product"), cols.get("revenue")
    if not p or not r:
        return go.Figure().update_layout(**dl("⚠️ No product/revenue columns")), ""
    df2 = df.copy()
    df2["_r"] = to_num(df2[r])
    top = df2.groupby(p)["_r"].sum().sort_values(ascending=False).head(10)
    colors = ([T]*3 + [GO]*4 + [CO]*3)[:len(top)]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index.astype(str), orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"₹{v:,.0f}" for v in top.values],
        textposition="outside",
        textfont=dict(color="rgba(255,255,255,.5)", size=10),
    ))
    layout = dl("📊 Top 10 Products by Revenue")
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**layout)
    total = top.sum()
    conc  = top.head(3).sum()/total*100 if total > 0 else 0
    html = f"""
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;
                margin-top:14px'>
      <div style='background:{B2};border:1px solid rgba(0,212,180,.2);
                  border-radius:12px;padding:14px;text-align:center'>
        <div style='font-size:.6rem;color:rgba(255,255,255,.4);
                    text-transform:uppercase;letter-spacing:.1em'>Total Revenue</div>
        <div style='font-size:1.3rem;font-weight:900;color:{T};
                    margin-top:4px'>₹{total:,.0f}</div>
      </div>
      <div style='background:{B2};border:1px solid rgba(240,165,0,.25);
                  border-radius:12px;padding:14px;text-align:center'>
        <div style='font-size:.6rem;color:rgba(255,255,255,.4);
                    text-transform:uppercase;letter-spacing:.1em'>Top 3 Concentration</div>
        <div style='font-size:1.3rem;font-weight:900;color:{GO};
                    margin-top:4px'>{conc:.1f}%</div>
      </div>
      <div style='background:{B2};border:1px solid rgba(46,204,113,.25);
                  border-radius:12px;padding:14px;text-align:center'>
        <div style='font-size:.6rem;color:rgba(255,255,255,.4);
                    text-transform:uppercase;letter-spacing:.1em'>Total SKUs</div>
        <div style='font-size:1.3rem;font-weight:900;color:{GRN};
                    margin-top:4px'>{df2[p].nunique()}</div>
      </div>
    </div>
    <div style='margin-top:12px;background:rgba(0,212,180,.06);
                border-left:3px solid {T};border-radius:0 10px 10px 0;
                padding:12px 16px'>
      <p style='font-size:.84rem;color:rgba(255,255,255,.75);margin:0'>
        📊 <strong style='color:{T}'>AI Insight:</strong>
        {"Top 3 SKUs drive "+f"{conc:.0f}%"+" of revenue — high concentration risk. Promote mid-tier products for resilience."
         if conc > 70 else
         "Revenue is reasonably distributed. Focus on growing top performers via ONDC."}
      </p>
    </div>"""
    return fig, html

# ── Forecast ──────────────────────────────────────────────────────────────────
def forecast(df, cols, periods=6):
    d, r = cols.get("date"), cols.get("revenue")
    if not d or not r:
        return (go.Figure().update_layout(**dl("⚠️ Date + Revenue columns required")),
                f"<p style='color:rgba(255,255,255,.5);padding:12px'>Rename columns to 'date' and 'revenue' to enable forecasting.</p>")
    ts = df.copy()
    ts["ds"] = to_date(ts[d])
    ts["y"]  = to_num(ts[r])
    ts = (ts.dropna(subset=["ds","y"])
            .groupby("ds")["y"].sum()
            .reset_index())
    ts.columns = ["ds","y"]
    ts = ts.sort_values("ds")
    if len(ts) < 4:
        return (go.Figure().update_layout(**dl(f"⚠️ Need ≥4 periods, found {len(ts)}")),
                f"<p style='color:{CO};padding:12px'>Need at least 4 time periods.</p>")
    try:
        from prophet import Prophet
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, interval_width=0.8,
                    changepoint_prior_scale=0.05)
        m.fit(ts)
        fut = m.make_future_dataframe(periods=periods, freq="MS")
        fc  = m.predict(fut)
        hist = fc[fc["ds"].isin(ts["ds"])].copy()
        hist["actual"] = ts.set_index("ds")["y"].reindex(hist["ds"]).values
        hist = hist.dropna(subset=["actual"])
        mape = (np.abs((hist["actual"]-hist["yhat"])/hist["actual"])
                ).mean()*100 if len(hist) else None
        fp = fc[~fc["ds"].isin(ts["ds"])]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["ds"], y=ts["y"],
            mode="lines+markers", name="Actual",
            line=dict(color=T, width=2), marker=dict(size=5, color=T)))
        fig.add_trace(go.Scatter(x=fp["ds"], y=fp["yhat"],
            mode="lines+markers", name=f"{periods}M Forecast",
            line=dict(color=GO, width=2, dash="dot"),
            marker=dict(size=5, color=GO)))
        fig.add_trace(go.Scatter(
            x=list(fp["ds"])+list(fp["ds"][::-1]),
            y=list(fp["yhat_upper"])+list(fp["yhat_lower"][::-1]),
            fill="toself", fillcolor="rgba(240,165,0,.08)",
            line=dict(color="rgba(0,0,0,0)"), name="80% Confidence"))
        fig.update_layout(**dl(f"📈 Prophet {periods}-Month Sales Forecast"))
        pred = fp["yhat"].clip(lower=0).sum()
        act  = ts["y"].sum()
        grw  = (pred-act)/act*100 if act > 0 else 0
        met = f"""
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;
                    margin-top:14px'>
          <div style='background:{B2};border:1px solid rgba(0,212,180,.2);
                      border-radius:12px;padding:14px;text-align:center'>
            <div style='font-size:.6rem;color:rgba(255,255,255,.4);
                        text-transform:uppercase;letter-spacing:.1em'>MAPE Accuracy</div>
            <div style='font-size:1.3rem;font-weight:900;color:{T};
                        margin-top:4px'>{f"{mape:.1f}%" if mape is not None else "N/A"}</div>
          </div>
          <div style='background:{B2};border:1px solid rgba(0,212,180,.2);
                      border-radius:12px;padding:14px;text-align:center'>
            <div style='font-size:.6rem;color:rgba(255,255,255,.4);
                        text-transform:uppercase;letter-spacing:.1em'>{periods}M Forecast</div>
            <div style='font-size:1.3rem;font-weight:900;color:#fff;
                        margin-top:4px'>₹{pred:,.0f}</div>
          </div>
          <div style='background:{B2};
                      border:1px solid {"rgba(46,204,113,.25)" if grw>=0 else "rgba(224,92,106,.25)"};
                      border-radius:12px;padding:14px;text-align:center'>
            <div style='font-size:.6rem;color:rgba(255,255,255,.4);
                        text-transform:uppercase;letter-spacing:.1em'>Expected Growth</div>
            <div style='font-size:1.3rem;font-weight:900;
                        color:{GRN if grw>=0 else CO};
                        margin-top:4px'>{"+" if grw>=0 else ""}{grw:.1f}%</div>
          </div>
        </div>"""
        return fig, met
    except ImportError:
        from sklearn.linear_model import LinearRegression
        ts2 = ts.reset_index(drop=True)
        ts2["t"] = np.arange(len(ts2))
        model = LinearRegression().fit(ts2[["t"]], ts2["y"])
        fdates = [ts2["ds"].max()+pd.DateOffset(months=i+1)
                  for i in range(periods)]
        fvals  = model.predict(
            np.arange(len(ts2), len(ts2)+periods).reshape(-1,1)
        ).clip(min=0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts2["ds"], y=ts2["y"],
            mode="lines+markers", name="Actual",
            line=dict(color=T, width=2)))
        fig.add_trace(go.Scatter(x=fdates, y=fvals,
            mode="lines+markers", name="Forecast (Linear)",
            line=dict(color=GO, width=2, dash="dot")))
        fig.update_layout(**dl("📈 Sales Forecast — install prophet for ML forecasting"))
        pred = fvals.sum(); act = ts2["y"].sum()
        grw  = (pred-act)/act*100 if act > 0 else 0
        met = f"""
        <div style='background:{B2};border:1px solid rgba(240,165,0,.25);
                    border-radius:12px;padding:16px;margin-top:14px'>
          <p style='color:rgba(255,255,255,.65);font-size:.83rem;margin:0 0 8px'>
            ℹ️ Install Prophet for better ML forecasting:
            <code style='color:{GO}'>pip install prophet</code>
          </p>
          <p style='color:{T};font-weight:700;margin:0'>
            {periods}M Forecast: ₹{pred:,.0f} &nbsp;·&nbsp;
            Growth: {"+" if grw>=0 else ""}{grw:.1f}%
          </p>
        </div>"""
        return fig, met

# ── KMeans ────────────────────────────────────────────────────────────────────
def segment(df, cols, n=3):
    p, r = cols.get("product"), cols.get("revenue")
    if not p or not r:
        return go.Figure().update_layout(**dl("⚠️ Product + Revenue needed")), ""
    df2 = df.copy()
    df2["_r"] = to_num(df2[r])
    agg = {"_r": "sum"}
    if cols.get("quantity"):
        df2["_q"] = to_num(df2[cols["quantity"]])
        agg["_q"] = "sum"
    grp = df2.groupby(p).agg(agg).reset_index().dropna()
    n = min(n, max(len(grp), 1))
    feats = ["_r","_q"] if "_q" in grp.columns else ["_r"]
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        Xs = StandardScaler().fit_transform(grp[feats].values)
        grp["Cluster"] = KMeans(n_clusters=n, random_state=42,
                                n_init=10).fit_predict(Xs)
        means = grp.groupby("Cluster")["_r"].mean().sort_values(ascending=False)
        lmap  = {c: l for c, l in zip(
            means.index,
            ["⭐ High Value","📦 Mid Tier","🔁 Volume"][:n]
        )}
        grp["Segment"] = grp["Cluster"].map(lmap)
    except ImportError:
        grp["Segment"] = pd.qcut(
            grp["_r"], q=n,
            labels=["🔁 Volume","📦 Mid Tier","⭐ High Value"][:n],
            duplicates="drop")
    cmap = {"⭐ High Value": T, "📦 Mid Tier": GO, "🔁 Volume": CO}
    if "_q" in grp.columns:
        fig = px.scatter(grp, x="_q", y="_r", color="Segment",
                         hover_name=p, color_discrete_map=cmap,
                         labels={"_q":"Qty Sold","_r":"Revenue ₹"})
    else:
        fig = px.bar(grp.sort_values("_r", ascending=False).head(20),
                     x=p, y="_r", color="Segment",
                     color_discrete_map=cmap, labels={"_r":"Revenue ₹"})
    fig.update_layout(**dl("🎯 KMeans Product Segmentation"))
    summ = grp.groupby("Segment")["_r"].agg(["count","sum"]).reset_index()
    rows = "".join([
        f"""<tr style='border-bottom:1px solid rgba(255,255,255,.07)'>
          <td style='padding:10px 14px;color:#fff'>{row['Segment']}</td>
          <td style='padding:10px 14px;color:rgba(255,255,255,.6)'>{int(row['count'])}</td>
          <td style='padding:10px 14px;color:{T};font-weight:700'>
            ₹{row['sum']:,.0f}</td>
        </tr>"""
        for _, row in summ.iterrows()
    ])
    tbl = f"""
    <table style='width:100%;border-collapse:collapse;margin-top:14px;
                  background:{B2};border-radius:12px;overflow:hidden;
                  border:1px solid rgba(0,212,180,.15)'>
      <tr style='background:rgba(0,212,180,.12)'>
        <th style='padding:10px 14px;text-align:left;color:{T};
                   font-size:.7rem;text-transform:uppercase;
                   letter-spacing:.08em'>Segment</th>
        <th style='padding:10px 14px;text-align:left;color:{T};
                   font-size:.7rem;text-transform:uppercase;
                   letter-spacing:.08em'>Products</th>
        <th style='padding:10px 14px;text-align:left;color:{T};
                   font-size:.7rem;text-transform:uppercase;
                   letter-spacing:.08em'>Revenue</th>
      </tr>
      {rows}
    </table>"""
    return fig, tbl

# ── ONDC ──────────────────────────────────────────────────────────────────────
ONDC_LIST = [
    {"n":"GeM — Government e-Marketplace","i":"🏛️",
     "c":["fmcg","electronics","manufacturing","agriculture","services","healthcare"],"min":60,"b":88},
    {"n":"Flipkart Commerce Cloud","i":"🛒",
     "c":["fmcg","retail","electronics","clothing","food"],"min":55,"b":82},
    {"n":"Meesho Supplier Hub","i":"👗",
     "c":["clothing","fmcg","retail","home","fashion"],"min":40,"b":79},
    {"n":"NSIC e-Marketplace","i":"🏭",
     "c":["manufacturing","agriculture","services","energy"],"min":50,"b":76},
    {"n":"Amazon ONDC","i":"📦",
     "c":["fmcg","electronics","retail","food"],"min":65,"b":74},
    {"n":"Udaan B2B","i":"🚚",
     "c":["fmcg","retail","electronics","clothing","agriculture"],"min":45,"b":71},
]
def ondc_match(score, cat, n_prod):
    cat = (cat or "").lower()
    results = []
    for s in ONDC_LIST:
        if score < s["min"]: continue
        m = (s["b"]
             + (6 if any(c in cat for c in s["c"]) else 0)
             + (4 if score >= 75 else 0)
             + (2 if n_prod >= 10 else 0))
        results.append({"n":s["n"],"i":s["i"],
                         "m":min(int(m+np.random.randint(-2,4)),99)})
    results = sorted(results, key=lambda x: -x["m"])[:3]
    if not results:
        return f"""
        <div style='background:{B2};border:1px solid rgba(224,92,106,.25);
                    border-radius:12px;padding:20px'>
          <p style='color:rgba(255,255,255,.6);margin:0'>
            ⚠️ Health Score {score:.0f} is below threshold.
            Improve business health to qualify for ONDC matching.
          </p>
        </div>"""
    pal = [T, GO, CO]
    cards = "".join([f"""
    <div style='background:{B2};border:1px solid rgba(255,255,255,.1);
                border-radius:14px;padding:20px;margin-bottom:14px'>
      <div style='display:flex;align-items:center;
                  justify-content:space-between;margin-bottom:14px'>
        <div style='display:flex;align-items:center;gap:12px'>
          <span style='font-size:1.6rem'>{r["i"]}</span>
          <div>
            <div style='font-size:.9rem;font-weight:700;color:#fff'>{r["n"]}</div>
            <div style='font-size:.7rem;color:rgba(255,255,255,.4);margin-top:2px'>
              ONDC Seller Network Participant</div>
          </div>
        </div>
        <div style='text-align:right'>
          <div style='font-size:1.8rem;font-weight:900;color:{pal[i]}'>{r["m"]}%</div>
          <div style='font-size:.65rem;color:rgba(255,255,255,.4)'>Match Score</div>
        </div>
      </div>
      <div style='background:rgba(255,255,255,.07);border-radius:20px;
                  height:7px;overflow:hidden;margin-bottom:12px'>
        <div style='width:{r["m"]}%;height:100%;
                    background:{pal[i]};border-radius:20px'></div>
      </div>
      <span style='display:inline-block;background:rgba(0,212,180,.12);
                   border:1px solid rgba(0,212,180,.3);color:{T};
                   border-radius:50px;padding:5px 16px;font-size:.78rem;
                   font-weight:700'>
        {"Apply Now →" if i==0 else "Register Seller →" if i==1 else "Explore →"}
      </span>
    </div>"""
    for i, r in enumerate(results)])
    return (f"<p style='color:rgba(255,255,255,.4);font-size:.78rem;"
            f"margin-bottom:14px'>Top 3 ONDC SNP matches · "
            f"Health Score: {score:.0f}/100</p>") + cards

# ── AI Narrative ──────────────────────────────────────────────────────────────
def narrative(df, cols, score, sub):
    rev = to_num(df[cols["revenue"]]).sum() if cols.get("revenue") else 0
    lbl, _ = hlabel(score)
    return f"""
    <div style='display:flex;flex-direction:column;gap:14px;padding:4px'>
      <div style='background:{B2};border-left:3px solid {T};
                  border-radius:0 12px 12px 0;padding:16px 18px'>
        <p style='font-size:.88rem;color:rgba(255,255,255,.8);
                  margin:0;line-height:1.72'>
          📊 <strong style='color:{T}'>Revenue Summary:</strong>
          Total revenue is <strong style='color:#fff'>₹{rev:,.0f}</strong>.
          MSME Health Score <strong style='color:{T}'>{score}/100</strong>
          — <strong style='color:#fff'>{lbl}</strong>.
        </p>
      </div>
      <div style='background:{B2};border-left:3px solid {GO};
                  border-radius:0 12px 12px 0;padding:16px 18px'>
        <p style='font-size:.88rem;color:rgba(255,255,255,.8);
                  margin:0;line-height:1.72'>
          ⚠️ <strong style='color:{GO}'>Cost Insight:</strong>
          {"Operating cost ratio is elevated. Review cost-to-revenue ratio and optimise procurement."
           if sub.get("Cost Control",60) < 60 else
           "Cost control within healthy parameters. Focus on scaling revenue."}
        </p>
      </div>
      <div style='background:{B2};border-left:3px solid {GRN};
                  border-radius:0 12px 12px 0;padding:16px 18px'>
        <p style='font-size:.88rem;color:rgba(255,255,255,.8);
                  margin:0;line-height:1.72'>
          🚀 <strong style='color:{GRN}'>Opportunity:</strong>
          {"Diversify your SKU portfolio — high concentration increases revenue risk."
           if sub.get("Diversification",60) < 60 else
           "Good diversification. Grow top performers via ONDC marketplace listings."}
        </p>
      </div>
      <div style='background:rgba(0,212,180,.05);
                  border:1px solid rgba(0,212,180,.12);
                  border-radius:12px;padding:12px 16px'>
        <p style='font-size:.72rem;color:rgba(255,255,255,.3);margin:0'>
          🤖 AI summaries by DataNetra · DPDP Act 2023 Compliant ·
          Data processed locally and deleted after analysis
        </p>
      </div>
    </div>"""

# ── Main analysis function ────────────────────────────────────────────────────
EF = go.Figure().update_layout(**dl("Upload a file to begin", h=340))

def analyse(file, cat, fm, nc):
    emp = f"<p style='color:rgba(255,255,255,.4);padding:12px 0'>—</p>"
    if file is None:
        return (
            f"<div style='text-align:center;padding:28px;color:rgba(255,255,255,.5);font-size:.9rem'>"
            f"⬆️ Upload a <strong style='color:rgba(255,255,255,.8)'>.csv</strong> or "
            f"<strong style='color:rgba(255,255,255,.8)'>.xlsx</strong> file and click "
            f"<strong style='color:{T}'>Run Full Analysis</strong></div>",
            EF, emp, EF, emp, EF, emp, EF, emp, emp, emp)
    path = file.name if hasattr(file,"name") else str(file)
    ext  = os.path.splitext(path)[-1].lower()
    try:
        df = (pd.read_csv(path, encoding="utf-8-sig")
              if ext == ".csv" else pd.read_excel(path))
    except Exception as e:
        return (f"<div style='color:{CO};padding:20px;background:{B2};"
                f"border-radius:12px'>❌ {e}</div>",
                EF,emp,EF,emp,EF,emp,EF,emp,emp,emp)

    cols  = detect(df)
    fname = os.path.basename(path)
    det   = " · ".join([
        f"<span style='color:{T}'>{k}</span> = "
        f"<code style='color:rgba(255,255,255,.6)'>{v}</code>"
        for k, v in cols.items()
    ])
    status = f"""
    <div style='background:{B2};border:1px solid rgba(0,212,180,.3);
                border-radius:14px;padding:18px 22px'>
      <div style='display:flex;align-items:center;gap:14px;margin-bottom:10px'>
        <span style='font-size:1.5rem'>✅</span>
        <div>
          <div style='font-size:.97rem;font-weight:800;color:#fff'>
            "{fname}" analysed successfully</div>
          <div style='font-size:.78rem;color:rgba(255,255,255,.5);margin-top:2px'>
            {len(df):,} rows · {len(df.columns)} columns</div>
        </div>
      </div>
      <div style='font-size:.75rem;color:rgba(255,255,255,.4);padding-top:8px;
                  border-top:1px solid rgba(255,255,255,.08)'>
        {det if det else "⚠️ No columns auto-detected. Rename to: date, product, revenue, quantity, cost, category"}
      </div>
    </div>"""

    sc, sub = calc_health(df, cols)
    gf, gh  = make_gauge(sc, sub)
    tf, th  = top10(df, cols)
    ff, fh  = forecast(df, cols, int(fm))
    kf, kh  = segment(df, cols, int(nc))
    np_     = df[cols["product"]].nunique() if cols.get("product") else 0
    oh      = ondc_match(sc, cat, np_)
    nh      = narrative(df, cols, sc, sub)
    return status, gf, gh, tf, th, ff, fh, kf, kh, nh, oh

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Full-screen, high-contrast, no pricing section
# ══════════════════════════════════════════════════════════════════════════════
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;0,9..40,800;0,9..40,900;1,9..40,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── FULL PAGE — TRUE FULL WIDTH ──────────────────────────────────────────── */
html, body {{
  background: {BG} !important;
  font-family: 'DM Sans', sans-serif !important;
  color: #fff !important;
  margin: 0 !important;
  padding: 0 !important;
  min-height: 100vh !important;
}}

/* Remove ALL max-width constraints for full-screen */
.gradio-container,
.gradio-container > *,
.main,
.wrap {{
  background: {BG} !important;
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  font-family: 'DM Sans', sans-serif !important;
  box-sizing: border-box !important;
}}

/* Inner content padding */
.gradio-container .contain {{
  max-width: 100% !important;
  padding: 0 40px !important;
}}

/* ── WIPE ALL GRADIO WHITE PANELS ──────────────────────────────────────────── */
.gradio-container > *,
.contain, .gr-form, .gap,
div[class^="wrap"], div[class*=" wrap"],
div[class^="container"],
.block, .prose, .padded, .gr-padded {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}}

/* ── FILE UPLOAD — HIGH CONTRAST ────────────────────────────────────────────── */
.upload-container,
[data-testid="file"],
label[data-testid="file"],
.file-preview,
.file-upload,
.wrap.svelte-116rqfk,
[data-testid="file"] > .wrap {{
  background: {B3} !important;
  border: 2px dashed rgba(0,212,180,.55) !important;
  border-radius: 16px !important;
  color: rgba(255,255,255,.85) !important;
  min-height: 120px !important;
}}

[data-testid="file"]:hover,
.upload-container:hover {{
  border-color: {T} !important;
  background: rgba(0,212,180,.1) !important;
}}

/* Upload icon and text */
[data-testid="file"] span,
[data-testid="file"] p,
[data-testid="file"] .wrap span,
.upload-container span,
.upload-container p {{
  color: rgba(255,255,255,.85) !important;
  font-size: .9rem !important;
}}

[data-testid="file"] svg,
.upload-container svg {{
  color: {T} !important;
  stroke: {T} !important;
  opacity: 1 !important;
}}

/* File name after upload */
.file-preview-title, .file-name {{
  color: #fff !important;
  background: {B3} !important;
}}

/* ── INPUTS — HIGH CONTRAST ──────────────────────────────────────────────────── */
input[type=text], input[type=number], input[type=email],
textarea, select, .gr-input, .gr-text-input,
input.svelte-1pie4e3 {{
  background: {B3} !important;
  border: 1.5px solid rgba(0,212,180,.35) !important;
  color: #fff !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .9rem !important;
  padding: 10px 14px !important;
}}

input::placeholder, textarea::placeholder {{
  color: rgba(255,255,255,.35) !important;
}}

input:focus, textarea:focus {{
  border-color: {T} !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(0,212,180,.15) !important;
}}

/* ── LABELS — HIGH CONTRAST ──────────────────────────────────────────────────── */
label, .label-wrap, .label-wrap > span,
span.svelte-1gfkn6j, .gr-input-label,
.block > label > span {{
  color: rgba(255,255,255,.75) !important;
  font-size: .84rem !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: .01em !important;
}}

/* ── SLIDER — HIGH CONTRAST ──────────────────────────────────────────────────── */
input[type=range] {{
  accent-color: {T} !important;
  background: transparent !important;
  border: none !important;
}}

/* Slider number display */
.gr-slider output, .slider-container .output-text,
input[type=range] + span, .svelte-slider span {{
  color: #fff !important;
  background: rgba(0,212,180,.15) !important;
  border: 1px solid rgba(0,212,180,.3) !important;
  border-radius: 6px !important;
  padding: 2px 8px !important;
  font-weight: 700 !important;
  font-size: .85rem !important;
}}

/* Slider track */
.slider-container .range-slider,
.svelte-slider .track {{
  background: rgba(255,255,255,.15) !important;
  height: 4px !important;
  border-radius: 4px !important;
}}

/* Slider number input beside it */
.gr-slider-number, .number-container input {{
  background: {B3} !important;
  color: #fff !important;
  border: 1.5px solid rgba(0,212,180,.35) !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
}}

/* ── PRIMARY BUTTON — HIGH CONTRAST ─────────────────────────────────────────── */
button.primary,
.gr-button-primary,
button[variant="primary"],
[data-testid*="button"].primary,
.svelte-cmf5ev.primary {{
  background: {T} !important;
  color: #000 !important;
  font-weight: 900 !important;
  font-size: 1rem !important;
  font-family: 'DM Sans', sans-serif !important;
  border: none !important;
  border-radius: 50px !important;
  padding: 14px 36px !important;
  box-shadow: 0 4px 28px rgba(0,212,180,.5) !important;
  transition: all .2s !important;
  cursor: pointer !important;
  letter-spacing: .01em !important;
}}

button.primary:hover {{
  background: #00eacc !important;
  box-shadow: 0 6px 36px rgba(0,212,180,.65) !important;
  transform: translateY(-2px) !important;
}}

/* ── SECONDARY BUTTONS ──────────────────────────────────────────────────────── */
button.secondary, .gr-button-secondary {{
  background: transparent !important;
  border: 1.5px solid rgba(0,212,180,.4) !important;
  color: {T} !important;
  border-radius: 50px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
}}

/* ── TABS ───────────────────────────────────────────────────────────────────── */
.tabs, .tab-nav {{
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid rgba(0,212,180,.12) !important;
}}

.tab-nav button, .tabs > div > button {{
  background: transparent !important;
  color: rgba(255,255,255,.55) !important;
  font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 20px !important;
  font-size: .88rem !important;
  transition: all .2s !important;
}}

.tab-nav button.selected,
.tab-nav button[aria-selected="true"],
.tabs > div > button[aria-selected="true"] {{
  color: {T} !important;
  border-bottom: 2px solid {T} !important;
  background: transparent !important;
}}

/* ── PLOT CONTAINERS ────────────────────────────────────────────────────────── */
.plot-container, .gr-plot {{
  background: {PB} !important;
  border-radius: 14px !important;
  border: 1px solid rgba(0,212,180,.12) !important;
  overflow: hidden !important;
}}

/* ── HTML OUTPUT ────────────────────────────────────────────────────────────── */
.gr-html, .output-html, [data-testid="html"] {{
  background: transparent !important;
  color: #fff !important;
}}

/* ── SCROLLBAR ──────────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {B2}; }}
::-webkit-scrollbar-thumb {{ background: {T}; border-radius: 3px; }}

/* ── HIDE GRADIO FOOTER / BRANDING ─────────────────────────────────────────── */
footer.svelte-mpyp0e, .footer, footer, .built-with {{ display: none !important; }}

/* ── ROW SPACING ────────────────────────────────────────────────────────────── */
.gr-row {{ gap: 16px !important; }}
.gr-column {{ gap: 12px !important; }}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="DataNetra.ai", css=CSS,
               theme=gr.themes.Base(
                   font=gr.themes.GoogleFont("DM Sans"),
                   font_mono=gr.themes.GoogleFont("JetBrains Mono"),
               )) as demo:

    # ── NAV ──────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div style='
      background:{BG};
      border-bottom:1px solid rgba(0,212,180,.12);
      padding:0 40px; height:68px;
      display:flex; align-items:center; justify-content:space-between;
      position:sticky; top:0; z-index:100;
      backdrop-filter:blur(12px);
    '>
      <div style='display:flex;align-items:center;gap:10px'>
        <img src='data:image/png;base64,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'
             style='height:52px;width:auto;object-fit:contain;display:block'
             alt='DataNetra.ai logo'/>
        <span style='
          font-family:"DM Sans",sans-serif;
          font-size:1.55rem;
          font-weight:900;
          letter-spacing:-.03em;
          background:linear-gradient(135deg,#ffffff 0%,#00d4b4 50%,#f0a500 100%);
          -webkit-background-clip:text;
          -webkit-text-fill-color:transparent;
          background-clip:text;
          line-height:1;
        '>DataNetra</span>
      </div>
      <div style='display:flex;align-items:center;gap:28px'>
        <a href='#about-section' onclick="document.getElementById('about-section').scrollIntoView({{behavior:'smooth'}});return false;" style='font-size:.84rem;color:rgba(255,255,255,.5);cursor:pointer;transition:.2s;text-decoration:none' onmouseover="this.style.color='#00d4b4'" onmouseout="this.style.color='rgba(255,255,255,.5)'">About</a>
        <a href='#team-section' onclick="document.getElementById('team-section').scrollIntoView({{behavior:'smooth'}});return false;" style='font-size:.84rem;color:rgba(255,255,255,.5);cursor:pointer;transition:.2s;text-decoration:none' onmouseover="this.style.color='#00d4b4'" onmouseout="this.style.color='rgba(255,255,255,.5)'">Team</a>
        <a href='#how-we-work-section' onclick="document.getElementById('how-we-work-section').scrollIntoView({{behavior:'smooth'}});return false;" style='font-size:.84rem;color:rgba(255,255,255,.5);cursor:pointer;transition:.2s;text-decoration:none' onmouseover="this.style.color='#00d4b4'" onmouseout="this.style.color='rgba(255,255,255,.5)'">How We Work</a>
        <a href='#industries-section' onclick="document.getElementById('industries-section').scrollIntoView({{behavior:'smooth'}});return false;" style='font-size:.84rem;color:rgba(255,255,255,.5);cursor:pointer;transition:.2s;text-decoration:none' onmouseover="this.style.color='#00d4b4'" onmouseout="this.style.color='rgba(255,255,255,.5)'">Industries</a>
        <a href='#contact-section' onclick="document.getElementById('contact-section').scrollIntoView({{behavior:'smooth'}});return false;" style='
          display:inline-block; background:{T}; color:{BG};
          padding:8px 20px; border-radius:50px;
          font-size:.84rem; font-weight:800; cursor:pointer;
          text-decoration:none;
          box-shadow:0 2px 16px rgba(0,212,180,.35);
          transition:.2s
        '>Contact Us</a>
      </div>
    </div>
    """)

    # ── HERO ─────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div style='
      background:{BG};
      background-image:
        linear-gradient(rgba(255,255,255,.016) 1px,transparent 1px),
        linear-gradient(90deg,rgba(255,255,255,.016) 1px,transparent 1px);
      background-size:52px 52px;
      padding:64px 40px 56px;
      border-bottom:1px solid rgba(0,212,180,.08);
      position:relative; overflow:hidden;
    '>
      <!-- Glows -->
      <div style='position:absolute;inset:0;pointer-events:none;
        background:
          radial-gradient(ellipse at 20% 50%, rgba(0,212,180,.13) 0%, transparent 50%),
          radial-gradient(ellipse at 80% 20%, rgba(0,212,180,.09) 0%, transparent 45%);
      '></div>

      <div style='position:relative;z-index:1;
                  display:grid;grid-template-columns:1fr 1fr;
                  gap:48px;align-items:center;max-width:1200px;margin:0 auto'>

        <!-- LEFT: Text -->
        <div>
          <div style='
            display:inline-flex;align-items:center;gap:8px;
            background:rgba(0,212,180,.1);border:1px solid rgba(0,212,180,.3);
            border-radius:30px;padding:6px 18px;
            font-size:.67rem;font-weight:700;letter-spacing:.12em;
            text-transform:uppercase;color:{T};margin-bottom:22px
          '>
            <span style='width:7px;height:7px;border-radius:50%;
                         background:{T};display:inline-block;
                         animation:pulse 2s infinite;
                         box-shadow:0 0 8px {T}'></span>
            Data with Vision. Decisions with Confidence
          </div>

          <h1 style='
            font-size:clamp(2rem,4vw,3.4rem);font-weight:900;color:#fff;
            margin:0 0 18px;letter-spacing:-.04em;line-height:1.08;
            font-family:"DM Sans",sans-serif
          '>
            Smarter Decisions<br>for India's <span style='color:{T}'>SMEs</span>
          </h1>

          <p style='
            color:rgba(255,255,255,.6);font-size:.94rem;
            max-width:460px;margin:0 0 30px;line-height:1.82
          '>
            DataNetra transforms raw business data into actionable AI-powered insights —
            helping India's SMEs make smarter, faster, and more confident decisions.
          </p>

          <div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:40px'>
            <span onclick="document.getElementById('get-started-section').scrollIntoView({{behavior:'smooth'}})" style='
              display:inline-block;background:{T};color:{BG};
              padding:13px 28px;border-radius:50px;
              font-weight:800;font-size:.9rem;cursor:pointer;
              box-shadow:0 4px 24px rgba(0,212,180,.45);
              transition:.2s
            ' onmouseover="this.style.transform='translateY(-2px)'"
              onmouseout="this.style.transform='translateY(0)'">Get Started Free →</span>
            <span onclick="document.getElementById('about-section').scrollIntoView({{behavior:'smooth'}})" style='
              display:inline-block;
              background:rgba(255,255,255,.05);color:rgba(255,255,255,.8);
              padding:13px 24px;border-radius:50px;
              border:1.5px solid rgba(255,255,255,.18);
              font-weight:600;font-size:.9rem;cursor:pointer
            '>See How It Works</span>
          </div>

          <!-- Stats bar -->
          <div style='
            display:inline-flex;
            background:rgba(255,255,255,.04);
            border:1px solid rgba(255,255,255,.08);
            border-radius:14px;overflow:hidden
          '>
            <div style='padding:14px 22px;border-right:1px solid rgba(255,255,255,.07);text-align:center'>
              <div style='font-size:1.3rem;font-weight:900;color:{T};line-height:1'>12+</div>
              <div style='font-size:.54rem;color:rgba(255,255,255,.32);text-transform:uppercase;letter-spacing:.1em;margin-top:4px'>AI Models</div>
            </div>
            <div style='padding:14px 22px;border-right:1px solid rgba(255,255,255,.07);text-align:center'>
              <div style='font-size:1.3rem;font-weight:900;color:{T};line-height:1'>4</div>
              <div style='font-size:.54rem;color:rgba(255,255,255,.32);text-transform:uppercase;letter-spacing:.1em;margin-top:4px'>Categories</div>
            </div>
            <div style='padding:14px 22px;border-right:1px solid rgba(255,255,255,.07);text-align:center'>
              <div style='font-size:1.3rem;font-weight:900;color:{T};line-height:1'>6/12M</div>
              <div style='font-size:.54rem;color:rgba(255,255,255,.32);text-transform:uppercase;letter-spacing:.1em;margin-top:4px'>Forecasting</div>
            </div>
            <div style='padding:14px 22px;text-align:center'>
              <div style='font-size:1.3rem;font-weight:900;color:{T};line-height:1'>₹2.9K</div>
              <div style='font-size:.54rem;color:rgba(255,255,255,.32);text-transform:uppercase;letter-spacing:.1em;margin-top:4px'>/ Month</div>
            </div>
          </div>
        </div>

        <!-- RIGHT: Animated Dashboard Visual -->
        <div style='position:relative'>
          <div style='
            background:{B2};
            border:1px solid rgba(0,212,180,.2);
            border-radius:20px;
            padding:20px;
            box-shadow:0 24px 80px rgba(0,0,0,.5), 0 0 0 1px rgba(0,212,180,.08);
            overflow:hidden;
          '>
            <!-- Mini top bar -->
            <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:16px'>
              <div style='display:flex;gap:6px'>
                <div style='width:10px;height:10px;border-radius:50%;background:#e05c6a'></div>
                <div style='width:10px;height:10px;border-radius:50%;background:#f0a500'></div>
                <div style='width:10px;height:10px;border-radius:50%;background:#2ecc71'></div>
              </div>
              <div style='font-size:.65rem;color:rgba(255,255,255,.3);
                          font-family:"JetBrains Mono",monospace;letter-spacing:.08em'>
                DataNetra · Live Analysis
              </div>
              <div style='
                background:rgba(0,212,180,.12);border:1px solid rgba(0,212,180,.3);
                border-radius:20px;padding:3px 10px;
                font-size:.6rem;font-weight:700;color:{T}
              '>● LIVE</div>
            </div>

            <!-- Health Score Row -->
            <div style='
              background:{B3};border-radius:12px;padding:14px 16px;
              margin-bottom:12px;display:flex;align-items:center;justify-content:space-between
            '>
              <div>
                <div style='font-size:.62rem;color:rgba(255,255,255,.38);
                            text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px'>
                  MSME Health Score
                </div>
                <div style='font-size:1.6rem;font-weight:900;color:{T};line-height:1'>78<span style='font-size:.9rem;color:rgba(255,255,255,.4)'>/100</span></div>
              </div>
              <div style='text-align:right'>
                <div style='font-size:.7rem;color:#2ecc71;font-weight:700;margin-bottom:6px'>● Good</div>
                <!-- Mini gauge bars -->
                <div style='display:flex;flex-direction:column;gap:3px;width:100px'>
                  <div style='display:flex;gap:4px;align-items:center'>
                    <div style='font-size:.55rem;color:rgba(255,255,255,.3);width:52px;text-align:right'>Revenue</div>
                    <div style='flex:1;height:4px;background:rgba(255,255,255,.07);border-radius:4px;overflow:hidden'>
                      <div style='width:82%;height:100%;background:{T};border-radius:4px;
                                  animation:growBar 2s ease-out forwards'></div>
                    </div>
                  </div>
                  <div style='display:flex;gap:4px;align-items:center'>
                    <div style='font-size:.55rem;color:rgba(255,255,255,.3);width:52px;text-align:right'>Margin</div>
                    <div style='flex:1;height:4px;background:rgba(255,255,255,.07);border-radius:4px;overflow:hidden'>
                      <div style='width:65%;height:100%;background:{GO};border-radius:4px;
                                  animation:growBar 2.2s ease-out forwards'></div>
                    </div>
                  </div>
                  <div style='display:flex;gap:4px;align-items:center'>
                    <div style='font-size:.55rem;color:rgba(255,255,255,.3);width:52px;text-align:right'>Stability</div>
                    <div style='flex:1;height:4px;background:rgba(255,255,255,.07);border-radius:4px;overflow:hidden'>
                      <div style='width:74%;height:100%;background:#2ecc71;border-radius:4px;
                                  animation:growBar 2.4s ease-out forwards'></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Forecast Chart SVG -->
            <div style='background:{B3};border-radius:12px;padding:14px 16px;margin-bottom:12px'>
              <div style='font-size:.62rem;color:rgba(255,255,255,.38);
                          text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px'>
                Sales Forecast · 6-Month Prophet ML
              </div>
              <svg viewBox='0 0 340 90' style='width:100%;overflow:visible'>
                <!-- Grid lines -->
                <line x1='0' y1='22' x2='340' y2='22' stroke='rgba(255,255,255,.04)' stroke-width='1'/>
                <line x1='0' y1='44' x2='340' y2='44' stroke='rgba(255,255,255,.04)' stroke-width='1'/>
                <line x1='0' y1='66' x2='340' y2='66' stroke='rgba(255,255,255,.04)' stroke-width='1'/>
                <!-- Confidence band (forecast) -->
                <path d='M190,38 L220,32 L250,28 L280,22 L310,18 L340,14
                         L340,30 L310,34 L280,38 L250,44 L220,48 L190,52 Z'
                      fill='rgba(240,165,0,.12)'/>
                <!-- Historical line -->
                <polyline points='0,68 40,58 80,62 120,48 160,44 190,38'
                  fill='none' stroke='{T}' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/>
                <!-- Forecast line (dashed) -->
                <polyline points='190,38 220,32 250,28 280,22 310,18 340,14'
                  fill='none' stroke='{GO}' stroke-width='2' stroke-dasharray='5,3'
                  stroke-linecap='round' stroke-linejoin='round'/>
                <!-- Dots on historical -->
                <circle cx='0' cy='68' r='3' fill='{T}'/>
                <circle cx='40' cy='58' r='3' fill='{T}'/>
                <circle cx='80' cy='62' r='3' fill='{T}'/>
                <circle cx='120' cy='48' r='3' fill='{T}'/>
                <circle cx='160' cy='44' r='3' fill='{T}'/>
                <circle cx='190' cy='38' r='4.5' fill='{T}' stroke='{BG}' stroke-width='2'/>
                <!-- Forecast dots -->
                <circle cx='220' cy='32' r='3' fill='{GO}'/>
                <circle cx='250' cy='28' r='3' fill='{GO}'/>
                <circle cx='280' cy='22' r='3' fill='{GO}'/>
                <circle cx='310' cy='18' r='3' fill='{GO}'/>
                <circle cx='340' cy='14' r='4.5' fill='{GO}' stroke='{BG}' stroke-width='2'/>
                <!-- Labels -->
                <text x='0' y='82' fill='rgba(255,255,255,.25)' font-size='7' font-family='monospace'>Jan</text>
                <text x='36' y='82' fill='rgba(255,255,255,.25)' font-size='7' font-family='monospace'>Feb</text>
                <text x='76' y='82' fill='rgba(255,255,255,.25)' font-size='7' font-family='monospace'>Mar</text>
                <text x='116' y='82' fill='rgba(255,255,255,.25)' font-size='7' font-family='monospace'>Apr</text>
                <text x='156' y='82' fill='rgba(255,255,255,.25)' font-size='7' font-family='monospace'>May</text>
                <text x='196' y='82' fill='{GO}' font-size='7' font-family='monospace'>Jun+</text>
                <!-- Legend -->
                <line x1='250' y1='88' x2='266' y2='88' stroke='{T}' stroke-width='2'/>
                <text x='269' y='91' fill='rgba(255,255,255,.35)' font-size='6.5' font-family='monospace'>Actual</text>
                <line x1='295' y1='88' x2='311' y2='88' stroke='{GO}' stroke-width='2' stroke-dasharray='3,2'/>
                <text x='314' y='91' fill='rgba(255,255,255,.35)' font-size='6.5' font-family='monospace'>Forecast</text>
              </svg>
            </div>

            <!-- Bottom row: AI Recommendations + ONDC -->
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>

              <!-- AI Recommendations -->
              <div style='background:{B3};border-radius:12px;padding:13px 14px'>
                <div style='font-size:.58rem;color:rgba(255,255,255,.35);
                            text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px'>
                  🤖 AI Recommendations
                </div>
                <div style='display:flex;flex-direction:column;gap:6px'>
                  <div style='display:flex;align-items:flex-start;gap:6px'>
                    <span style='color:{T};font-size:.7rem;margin-top:1px;flex-shrink:0'>▲</span>
                    <span style='font-size:.7rem;color:rgba(255,255,255,.65);line-height:1.4'>
                      Push top 3 SKUs to ONDC for +18% revenue
                    </span>
                  </div>
                  <div style='display:flex;align-items:flex-start;gap:6px'>
                    <span style='color:{GO};font-size:.7rem;margin-top:1px;flex-shrink:0'>●</span>
                    <span style='font-size:.7rem;color:rgba(255,255,255,.65);line-height:1.4'>
                      Reduce slow-movers — 23% inventory at risk
                    </span>
                  </div>
                  <div style='display:flex;align-items:flex-start;gap:6px'>
                    <span style='color:#2ecc71;font-size:.7rem;margin-top:1px;flex-shrink:0'>✓</span>
                    <span style='font-size:.7rem;color:rgba(255,255,255,.65);line-height:1.4'>
                      Cost control healthy — scale procurement
                    </span>
                  </div>
                </div>
              </div>

              <!-- ONDC Match -->
              <div style='background:{B3};border-radius:12px;padding:13px 14px'>
                <div style='font-size:.58rem;color:rgba(255,255,255,.35);
                            text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px'>
                  🏛️ ONDC SNP Match
                </div>
                <div style='display:flex;flex-direction:column;gap:7px'>
                  <div>
                    <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                      <span style='font-size:.68rem;color:rgba(255,255,255,.6)'>GeM Portal</span>
                      <span style='font-size:.68rem;font-weight:700;color:{T}'>94%</span>
                    </div>
                    <div style='height:4px;background:rgba(255,255,255,.07);border-radius:4px;overflow:hidden'>
                      <div style='width:94%;height:100%;background:{T};border-radius:4px'></div>
                    </div>
                  </div>
                  <div>
                    <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                      <span style='font-size:.68rem;color:rgba(255,255,255,.6)'>Flipkart B2B</span>
                      <span style='font-size:.68rem;font-weight:700;color:{GO}'>87%</span>
                    </div>
                    <div style='height:4px;background:rgba(255,255,255,.07);border-radius:4px;overflow:hidden'>
                      <div style='width:87%;height:100%;background:{GO};border-radius:4px'></div>
                    </div>
                  </div>
                  <div>
                    <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                      <span style='font-size:.68rem;color:rgba(255,255,255,.6)'>Udaan B2B</span>
                      <span style='font-size:.68rem;font-weight:700;color:#2ecc71'>79%</span>
                    </div>
                    <div style='height:4px;background:rgba(255,255,255,.07);border-radius:4px;overflow:hidden'>
                      <div style='width:79%;height:100%;background:#2ecc71;border-radius:4px'></div>
                    </div>
                  </div>
                </div>
              </div>

            </div>
          </div>

          <!-- Floating badge -->
          <div style='
            position:absolute;top:-14px;right:-14px;
            background:linear-gradient(135deg,{T},{GO});
            border-radius:12px;padding:10px 14px;
            box-shadow:0 8px 28px rgba(0,212,180,.4);
            font-size:.72rem;font-weight:800;color:{BG};
            text-align:center;line-height:1.3
          '>
            ↑ 24%<br><span style='font-weight:600;font-size:.62rem'>Forecast Growth</span>
          </div>
        </div>

      </div>
    </div>

    <style>
    @keyframes pulse {{
      0%,100% {{ opacity:1; box-shadow:0 0 8px #00d4b4; }}
      50% {{ opacity:.5; box-shadow:0 0 16px #00d4b4; }}
    }}
    @keyframes growBar {{
      from {{ width:0% }}
    }}
    </style>
    """)



    # Hidden outputs kept for compatibility (invisible)
    with gr.Row(visible=False):
        file_in  = gr.File(file_types=[".csv",".xlsx",".xls"], type="filepath")
        cat_in   = gr.Textbox()
        fm_in    = gr.Slider(3, 12, value=6, step=1)
        nc_in    = gr.Slider(2, 5, value=3, step=1)
        run_btn  = gr.Button()
    status_out = gr.HTML(visible=False)
    gauge_out  = gr.Plot(visible=False)
    sub_out    = gr.HTML(visible=False)
    top10_out  = gr.Plot(visible=False)
    top10_ins  = gr.HTML(visible=False)
    fc_out     = gr.Plot(visible=False)
    fc_met     = gr.HTML(visible=False)
    km_out     = gr.Plot(visible=False)
    km_tbl     = gr.HTML(visible=False)
    narr_out   = gr.HTML(visible=False)
    ondc_out   = gr.HTML(visible=False)

    # ── GET STARTED SECTION ───────────────────────────────────────────────────
    gr.HTML(f"""
    <div id='get-started-section' style='
      background:{BG};
      border-top:1px solid rgba(0,212,180,.1);
      padding:80px 40px;
      position:relative;overflow:hidden;
    '>
      <!-- Glow -->
      <div style='position:absolute;inset:0;pointer-events:none;
        background:radial-gradient(ellipse at 50% 0%, rgba(0,212,180,.12) 0%, transparent 60%);
      '></div>

      <div style='max-width:1100px;margin:0 auto;position:relative;z-index:1'>

        <!-- Header -->
        <div style='text-align:center;margin-bottom:56px'>
          <span style='font-family:"JetBrains Mono",monospace;font-size:.68rem;
                       font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:{T}'>
            Free Trial
          </span>
          <h2 style='font-size:2.1rem;font-weight:800;color:#fff;
                     margin:10px 0 12px;letter-spacing:-.025em;
                     font-family:"DM Sans",sans-serif'>
            Get Started in <span style='color:{T}'>Minutes</span>
          </h2>
          <p style='color:rgba(255,255,255,.52);font-size:.92rem;
                    max-width:520px;margin:0 auto;line-height:1.78'>
            Tell us a little about yourself, pick a plan that fits your business, and
            our team will call you to set everything up. No tech knowledge needed.
          </p>
        </div>

        <div style='display:grid;grid-template-columns:1fr 1.2fr;gap:40px;align-items:start'>

          <!-- LEFT: What you get -->
          <div style='display:flex;flex-direction:column;gap:16px'>

            <div style='font-size:.7rem;font-weight:700;color:rgba(255,255,255,.35);
                        text-transform:uppercase;letter-spacing:.14em;margin-bottom:4px;
                        font-family:"JetBrains Mono",monospace'>
              What DataNetra delivers for your business
            </div>

            <div style='display:flex;align-items:flex-start;gap:14px;
                        background:{B2};border:1px solid rgba(0,212,180,.12);
                        border-radius:12px;padding:16px 18px'>
              <span style='font-size:1.3rem;flex-shrink:0'>📈</span>
              <div>
                <div style='font-size:.85rem;font-weight:700;color:#fff;margin-bottom:4px'>Sales Forecasting &amp; Predictions</div>
                <div style='font-size:.78rem;color:rgba(255,255,255,.5);line-height:1.65'>We use Prophet ML to predict your next 6–12 months of sales trends, seasonal demand, and revenue growth — so you can plan inventory, staffing, and budgets with confidence.</div>
              </div>
            </div>

            <div style='display:flex;align-items:flex-start;gap:14px;
                        background:{B2};border:1px solid rgba(0,212,180,.12);
                        border-radius:12px;padding:16px 18px'>
              <span style='font-size:1.3rem;flex-shrink:0'>⚡</span>
              <div>
                <div style='font-size:.85rem;font-weight:700;color:#fff;margin-bottom:4px'>Simple, scalable, and SME-friendly</div>
                <div style='font-size:.78rem;color:rgba(255,255,255,.5);line-height:1.65'>No data science degree needed. Upload your CSV or Excel file and DataNetra does the rest — delivering enterprise-grade AI insights in a format any business owner can act on immediately.</div>
              </div>
            </div>

            <div style='display:flex;align-items:flex-start;gap:14px;
                        background:{B2};border:1px solid rgba(0,212,180,.12);
                        border-radius:12px;padding:16px 18px'>
              <span style='font-size:1.3rem;flex-shrink:0'>📦</span>
              <div>
                <div style='font-size:.85rem;font-weight:700;color:#fff;margin-bottom:4px'>Reduce Inventory Waste</div>
                <div style='font-size:.78rem;color:rgba(255,255,255,.5);line-height:1.65'>We flag slow-moving SKUs, overstock risk, and dead inventory before they drain your working capital — helping you free up cash and reorder smarter every cycle.</div>
              </div>
            </div>

            <div style='display:flex;align-items:flex-start;gap:14px;
                        background:{B2};border:1px solid rgba(0,212,180,.12);
                        border-radius:12px;padding:16px 18px'>
              <span style='font-size:1.3rem;flex-shrink:0'>💰</span>
              <div>
                <div style='font-size:.85rem;font-weight:700;color:#fff;margin-bottom:4px'>Increase Profit Margins</div>
                <div style='font-size:.78rem;color:rgba(255,255,255,.5);line-height:1.65'>We analyse your cost-to-revenue ratio across products, categories, and stores — pinpointing exactly where margin is leaking and giving you a prioritised action plan to fix it.</div>
              </div>
            </div>

            <div style='display:flex;align-items:flex-start;gap:14px;
                        background:{B2};border:1px solid rgba(0,212,180,.12);
                        border-radius:12px;padding:16px 18px'>
              <span style='font-size:1.3rem;flex-shrink:0'>📈</span>
              <div>
                <div style='font-size:.85rem;font-weight:700;color:#fff;margin-bottom:4px'>Growth-focused decision support</div>
                <div style='font-size:.78rem;color:rgba(255,255,255,.5);line-height:1.65'>Every insight DataNetra generates is tied to a business outcome — grow revenue, cut waste, expand channels. We don't just show you data; we show you the next best move for your business.</div>
              </div>
            </div>
          </div>

          <!-- RIGHT: Signup Form + Plans -->
          <div>

            <!-- Signup Form -->
            <div style='background:{B2};border:1px solid rgba(0,212,180,.2);
                        border-radius:18px;padding:32px;margin-bottom:24px'>

              <div style='font-size:1rem;font-weight:800;color:#fff;
                          margin-bottom:22px;letter-spacing:-.01em'>
                ✨ Start Your Free Trial
              </div>

              <div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px'>
                <div>
                  <label style='font-size:.72rem;font-weight:700;color:rgba(255,255,255,.55);
                                display:block;margin-bottom:6px;letter-spacing:.03em'>First Name *</label>
                  <input type='text' id='gs-name' placeholder='Ravi'
                    style='width:100%;background:{B3};border:1.5px solid rgba(0,212,180,.25);
                           color:#fff;border-radius:10px;padding:11px 14px;
                           font-size:.88rem;font-family:"DM Sans",sans-serif;
                           box-sizing:border-box;outline:none;transition:.2s'
                    onfocus="this.style.borderColor='#00d4b4';this.style.boxShadow='0 0 0 3px rgba(0,212,180,.12)'"
                    onblur="this.style.borderColor='rgba(0,212,180,.25)';this.style.boxShadow='none'"/>
                </div>
                <div>
                  <label style='font-size:.72rem;font-weight:700;color:rgba(255,255,255,.55);
                                display:block;margin-bottom:6px;letter-spacing:.03em'>Mobile Number *</label>
                  <input type='tel' id='gs-mobile' placeholder='+91 98765 43210'
                    style='width:100%;background:{B3};border:1.5px solid rgba(0,212,180,.25);
                           color:#fff;border-radius:10px;padding:11px 14px;
                           font-size:.88rem;font-family:"DM Sans",sans-serif;
                           box-sizing:border-box;outline:none;transition:.2s'
                    onfocus="this.style.borderColor='#00d4b4';this.style.boxShadow='0 0 0 3px rgba(0,212,180,.12)'"
                    onblur="this.style.borderColor='rgba(0,212,180,.25)';this.style.boxShadow='none'"/>
                </div>
              </div>

              <div style='margin-bottom:18px'>
                <label style='font-size:.72rem;font-weight:700;color:rgba(255,255,255,.55);
                              display:block;margin-bottom:6px;letter-spacing:.03em'>Email Address *</label>
                <input type='email' id='gs-email' placeholder='ravi@yourbusiness.com'
                  style='width:100%;background:{B3};border:1.5px solid rgba(0,212,180,.25);
                         color:#fff;border-radius:10px;padding:11px 14px;
                         font-size:.88rem;font-family:"DM Sans",sans-serif;
                         box-sizing:border-box;outline:none;transition:.2s'
                  onfocus="this.style.borderColor='#00d4b4';this.style.boxShadow='0 0 0 3px rgba(0,212,180,.12)'"
                  onblur="this.style.borderColor='rgba(0,212,180,.25)';this.style.boxShadow='none'"/>
              </div>

              <!-- Plan selector -->
              <div style='margin-bottom:20px'>
                <label style='font-size:.72rem;font-weight:700;color:rgba(255,255,255,.55);
                              display:block;margin-bottom:10px;letter-spacing:.03em'>Choose Your Plan *</label>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px' id='plan-grid'>

                  <div onclick="selectPlan('weekly')" id='plan-weekly'
                    style='background:{B3};border:1.5px solid rgba(255,255,255,.1);
                           border-radius:12px;padding:14px 12px;text-align:center;
                           cursor:pointer;transition:.2s'>
                    <div style='font-size:.62rem;color:rgba(255,255,255,.38);
                                text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px'>Weekly</div>
                    <div style='font-size:1.15rem;font-weight:900;color:#fff'>₹799</div>
                    <div style='font-size:.62rem;color:rgba(255,255,255,.35);margin-top:3px'>/week</div>
                    <div style='font-size:.65rem;color:rgba(255,255,255,.4);margin-top:6px;line-height:1.4'>
                      Try it out, no commitment
                    </div>
                  </div>

                  <div onclick="selectPlan('monthly')" id='plan-monthly'
                    style='background:{B3};border:2px solid #00d4b4;
                           border-radius:12px;padding:14px 12px;text-align:center;
                           cursor:pointer;transition:.2s;position:relative;
                           box-shadow:0 0 20px rgba(0,212,180,.2)'>
                    <div style='position:absolute;top:-10px;left:50%;transform:translateX(-50%);
                                background:#00d4b4;color:#060d14;font-size:.58rem;font-weight:800;
                                padding:3px 10px;border-radius:20px;white-space:nowrap;
                                letter-spacing:.06em'>POPULAR</div>
                    <div style='font-size:.62rem;color:#00d4b4;
                                text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px'>Monthly</div>
                    <div style='font-size:1.15rem;font-weight:900;color:#fff'>₹2,999</div>
                    <div style='font-size:.62rem;color:rgba(255,255,255,.35);margin-top:3px'>/month</div>
                    <div style='font-size:.65rem;color:rgba(255,255,255,.4);margin-top:6px;line-height:1.4'>
                      Full AI suite, unlimited uploads
                    </div>
                  </div>

                  <div onclick="selectPlan('yearly')" id='plan-yearly'
                    style='background:{B3};border:1.5px solid rgba(255,255,255,.1);
                           border-radius:12px;padding:14px 12px;text-align:center;
                           cursor:pointer;transition:.2s'>
                    <div style='font-size:.62rem;color:rgba(255,255,255,.38);
                                text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px'>Yearly</div>
                    <div style='font-size:1.15rem;font-weight:900;color:#fff'>₹24,999</div>
                    <div style='font-size:.62rem;color:#2ecc71;margin-top:3px'>Save 30%</div>
                    <div style='font-size:.65rem;color:rgba(255,255,255,.4);margin-top:6px;line-height:1.4'>
                      Best value + priority support
                    </div>
                  </div>

                </div>
                <input type='hidden' id='gs-plan' value='monthly'/>
              </div>

              <button onclick="submitGetStarted()" style='
                width:100%;background:{T};color:#000;
                font-weight:900;font-size:.95rem;
                font-family:"DM Sans",sans-serif;
                border:none;border-radius:50px;
                padding:14px 32px;cursor:pointer;
                box-shadow:0 4px 24px rgba(0,212,180,.45);
                transition:.2s;letter-spacing:.01em
              '
              onmouseover="this.style.background='#00eacc';this.style.transform='translateY(-2px)'"
              onmouseout="this.style.background='{T}';this.style.transform='translateY(0)'">
                Get Started — We'll Call You →
              </button>

              <!-- Confirmation message -->
              <div id='gs-success' style='
                display:none;margin-top:18px;
                background:rgba(0,212,180,.08);
                border:1px solid rgba(0,212,180,.3);
                border-radius:14px;padding:20px 22px;
              '>
                <div style='font-size:1.1rem;font-weight:900;color:{T};margin-bottom:8px'>
                  🎉 You're on the list!
                </div>
                <div style='font-size:.85rem;color:rgba(255,255,255,.7);line-height:1.7;margin-bottom:6px'>
                  Thanks for signing up! Our team will call you within <strong style='color:#fff'>24 hours</strong>
                  to get you set up on your chosen plan.
                </div>
                <div style='font-size:.78rem;color:rgba(255,255,255,.4)'>
                  📞 Expect a call from <strong style='color:rgba(255,255,255,.6)'>+91 innovate@datanetra.ai</strong>
                </div>
              </div>

            </div>

            <!-- Trust note -->
            <div style='text-align:center;font-size:.75rem;color:rgba(255,255,255,.28);line-height:1.6'>
              🔒 Data Secured &amp; Protected
            </div>

          </div>
        </div>
      </div>
    </div>

    <script>
    function selectPlan(plan) {{
      var plans = ['weekly','monthly','yearly'];
      plans.forEach(function(p) {{
        var el = document.getElementById('plan-' + p);
        if (p === plan) {{
          el.style.border = '2px solid #00d4b4';
          el.style.boxShadow = '0 0 20px rgba(0,212,180,.2)';
        }} else {{
          el.style.border = '1.5px solid rgba(255,255,255,.1)';
          el.style.boxShadow = 'none';
        }}
      }});
      document.getElementById('gs-plan').value = plan;
    }}

    function submitGetStarted() {{
      var name   = document.getElementById('gs-name').value.trim();
      var mobile = document.getElementById('gs-mobile').value.trim();
      var email  = document.getElementById('gs-email').value.trim();
      var plan   = document.getElementById('gs-plan').value;

      if (!name || !mobile || !email) {{
        alert('Please fill in your First Name, Mobile Number, and Email Address.');
        return;
      }}
      if (!/^[+]?[0-9]{{10,13}}$/.test(mobile.replace(/\\s/g,''))) {{
        alert('Please enter a valid mobile number.');
        return;
      }}
      if (!/^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$/.test(email)) {{
        alert('Please enter a valid email address.');
        return;
      }}

      // Send via mailto as fallback
      var subject = encodeURIComponent('DataNetra Free Trial — ' + plan.charAt(0).toUpperCase() + plan.slice(1) + ' Plan');
      var body = encodeURIComponent(
        'New Trial Signup\\n\\n' +
        'Name: ' + name + '\\n' +
        'Mobile: ' + mobile + '\\n' +
        'Email: ' + email + '\\n' +
        'Plan: ' + plan
      );
      window.location.href = 'mailto:innovate@datanetra.ai?subject=' + subject + '&body=' + body;

      // Show success
      document.getElementById('gs-success').style.display = 'block';
    }}
    </script>
    """)

    # ── ABOUT SECTION ─────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div id='about-section' style='
      background:{BG};
      border-top:1px solid rgba(0,212,180,.08);
      padding:80px 40px;
    '>
      <div style='max-width:1100px;margin:0 auto'>

        <!-- Header -->
        <div style='text-align:center;margin-bottom:60px'>
          <span style='font-family:"JetBrains Mono",monospace;font-size:.68rem;
                       font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:{T}'>About DataNetra</span>
          <h2 style='font-size:2rem;font-weight:800;color:#fff;
                     margin:10px 0 10px;letter-spacing:-.025em;
                     font-family:"DM Sans",sans-serif'>
            Not Just Reports —<br><span style='color:{T}'>Business Foresight</span>
          </h2>
          <p style='color:rgba(255,255,255,.52);font-size:.92rem;
                    max-width:540px;margin:0 auto;line-height:1.78'>
            Five capabilities to turn your raw business data into decisions.
          </p>
        </div>

        <!-- Capabilities Grid -->
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:20px'>

          <!-- Card 1 -->
          <div style='background:{B2};border:1px solid rgba(0,212,180,.15);
                      border-radius:16px;padding:28px 24px;
                      transition:.25s;cursor:default'
               onmouseover="this.style.borderColor='rgba(0,212,180,.45)';this.style.transform='translateY(-4px)';this.style.boxShadow='0 16px 40px rgba(0,0,0,.4)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.15)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='font-size:2rem;margin-bottom:14px'>🔍</div>
            <div style='font-size:.95rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Data Input &amp; Validation</div>
            <p style='font-size:.82rem;color:rgba(255,255,255,.55);line-height:1.75;margin:0'>
              We ensure the collection of clean, accurate, and relevant data for every analysis.
              Our intelligent column auto-mapping handles 15+ field name aliases automatically,
              enabling SMBs to make decisions based on reliable, validated information —
              without manual reformatting of any kind.
            </p>
          </div>

          <!-- Card 2 -->
          <div style='background:{B2};border:1px solid rgba(0,212,180,.15);
                      border-radius:16px;padding:28px 24px;transition:.25s;cursor:default'
               onmouseover="this.style.borderColor='rgba(0,212,180,.45)';this.style.transform='translateY(-4px)';this.style.boxShadow='0 16px 40px rgba(0,0,0,.4)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.15)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='font-size:2rem;margin-bottom:14px'>📊</div>
            <div style='font-size:.95rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Data Visualization</div>
            <p style='font-size:.82rem;color:rgba(255,255,255,.55);line-height:1.75;margin:0'>
              We turn complex datasets into simple, powerful visual representations —
              dashboards and charts that highlight your key trends, top products, profit margins,
              and performance indicators. Every chart includes an AI-written plain-language
              summary explaining exactly what the data means for your business.
            </p>
          </div>

          <!-- Card 3 -->
          <div style='background:{B2};border:1px solid rgba(0,212,180,.15);
                      border-radius:16px;padding:28px 24px;transition:.25s;cursor:default'
               onmouseover="this.style.borderColor='rgba(0,212,180,.45)';this.style.transform='translateY(-4px)';this.style.boxShadow='0 16px 40px rgba(0,0,0,.4)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.15)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='font-size:2rem;margin-bottom:14px'>🤖</div>
            <div style='font-size:.95rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Model Building</div>
            <p style='font-size:.82rem;color:rgba(255,255,255,.55);line-height:1.75;margin:0'>
              We build and compare predictive models using industry-leading techniques.
              Prophet time-series forecasting is benchmarked automatically against
              Holt-Winters and Linear Regression — the winning model is selected and
              validated with MAPE, MAE, and RMSE metrics so you know exactly how
              accurate your forecast is.
            </p>
          </div>

          <!-- Card 4 -->
          <div style='background:{B2};border:1px solid rgba(0,212,180,.15);
                      border-radius:16px;padding:28px 24px;transition:.25s;cursor:default'
               onmouseover="this.style.borderColor='rgba(0,212,180,.45)';this.style.transform='translateY(-4px)';this.style.boxShadow='0 16px 40px rgba(0,0,0,.4)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.15)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='font-size:2rem;margin-bottom:14px'>🎟️</div>
            <div style='font-size:.95rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Dashboard Creation</div>
            <p style='font-size:.82rem;color:rgba(255,255,255,.55);line-height:1.75;margin:0'>
              We create interactive, real-time dashboards providing a clear and immediate
              view of your key business metrics. Five KPI cards and eight granular forecast
              charts — per-store, per-category, and per-SKU breakdowns — give you a complete
              picture of business performance in a single, intuitive view.
            </p>
          </div>

          <!-- Card 5 -->
          <div style='background:{B2};border:1px solid rgba(0,212,180,.15);
                      border-radius:16px;padding:28px 24px;transition:.25s;cursor:default'
               onmouseover="this.style.borderColor='rgba(0,212,180,.45)';this.style.transform='translateY(-4px)';this.style.boxShadow='0 16px 40px rgba(0,0,0,.4)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.15)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='font-size:2rem;margin-bottom:14px'>📖</div>
            <div style='font-size:.95rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Storytelling &amp; Recommendations</div>
            <p style='font-size:.82rem;color:rgba(255,255,255,.55);line-height:1.75;margin:0'>
              We use data-driven storytelling to contextualize the numbers. Claude AI
              generates plain-language summaries below every chart, delivering not just
              insights but a compelling narrative. Every score is explained:
              <em style='color:rgba(255,255,255,.75)'>"Your risk is high because your operating cost
              is 87% of total sales."</em> Actionable recommendations are included for
              improving operations and customer acquisition.
            </p>
          </div>

        </div>
      </div>
    </div>
    """)

    # ── TEAM SECTION ──────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div id='team-section' style='
      background:{B2};
      border-top:1px solid rgba(0,212,180,.08);
      padding:80px 40px;
    '>
      <div style='max-width:900px;margin:0 auto'>

        <!-- Header -->
        <div style='text-align:center;margin-bottom:56px'>
          <span style='font-family:"JetBrains Mono",monospace;font-size:.68rem;
                       font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:{T}'>The Team</span>
          <h2 style='font-size:2rem;font-weight:800;color:#fff;
                     margin:10px 0 0;letter-spacing:-.025em;
                     font-family:"DM Sans",sans-serif'>
            People Behind <span style='color:{T}'>DataNetra</span>
          </h2>
        </div>

        <!-- Team Cards -->
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:28px'>

          <!-- Jayanthi Kumar -->
          <div style='background:{B3};border:1px solid rgba(0,212,180,.18);
                      border-radius:20px;padding:36px 32px;
                      transition:.25s;cursor:default'
               onmouseover="this.style.borderColor='rgba(0,212,180,.4)';this.style.boxShadow='0 16px 48px rgba(0,0,0,.4)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.18)';this.style.boxShadow='none'">

            <!-- Avatar -->
            <div style='display:flex;align-items:center;gap:18px;margin-bottom:22px'>
              <div style='
                width:64px;height:64px;border-radius:16px;flex-shrink:0;
                background:linear-gradient(135deg,{T},rgba(0,212,180,.3));
                display:flex;align-items:center;justify-content:center;
                font-size:1.4rem;font-weight:900;color:{BG};
                box-shadow:0 4px 20px rgba(0,212,180,.3)
              '>JK</div>
              <div>
                <div style='font-size:1.1rem;font-weight:800;color:#fff;
                            letter-spacing:-.01em;margin-bottom:3px'>Jayanthi Kumar</div>
                <div style='font-size:.75rem;font-weight:600;color:{T};
                            letter-spacing:.02em'>Founder &amp; Business Head</div>
              </div>
            </div>

            <p style='font-size:.83rem;color:rgba(255,255,255,.58);
                      line-height:1.8;margin:0 0 14px'>
              Jayanthi is the visionary behind DataNetra, with extensive experience in
              business development, leadership, and empowering SMBs across India.
              She has a deep understanding of the challenges faced by small and
              medium-sized businesses and is passionate about bringing data-driven
              solutions to these enterprises.
            </p>
            <p style='font-size:.83rem;color:rgba(255,255,255,.58);
                      line-height:1.8;margin:0'>
              With a strong background in IT service delivery and reskilling initiatives,
              Jayanthi is committed to creating accessible, actionable, and impactful data
              solutions for businesses at the grassroots level — ensuring that no MSME is
              left behind in India's digital transformation journey.
            </p>

            <!-- Tags -->
            <div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:20px'>
              <span style='background:rgba(0,212,180,.1);border:1px solid rgba(0,212,180,.25);
                           color:{T};border-radius:20px;padding:4px 12px;
                           font-size:.68rem;font-weight:700'>Business Strategy</span>
              <span style='background:rgba(0,212,180,.1);border:1px solid rgba(0,212,180,.25);
                           color:{T};border-radius:20px;padding:4px 12px;
                           font-size:.68rem;font-weight:700'>SMB Empowerment</span>
              <span style='background:rgba(0,212,180,.1);border:1px solid rgba(0,212,180,.25);
                           color:{T};border-radius:20px;padding:4px 12px;
                           font-size:.68rem;font-weight:700'>IT Service Delivery</span>
            </div>
          </div>

          <!-- Karthick Manoharan -->
          <div style='background:{B3};border:1px solid rgba(0,212,180,.18);
                      border-radius:20px;padding:36px 32px;
                      transition:.25s;cursor:default'
               onmouseover="this.style.borderColor='rgba(0,212,180,.4)';this.style.boxShadow='0 16px 48px rgba(0,0,0,.4)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.18)';this.style.boxShadow='none'">

            <!-- Avatar -->
            <div style='display:flex;align-items:center;gap:18px;margin-bottom:22px'>
              <div style='
                width:64px;height:64px;border-radius:16px;flex-shrink:0;
                background:linear-gradient(135deg,{GO},rgba(240,165,0,.3));
                display:flex;align-items:center;justify-content:center;
                font-size:1.4rem;font-weight:900;color:{BG};
                box-shadow:0 4px 20px rgba(240,165,0,.3)
              '>KM</div>
              <div>
                <div style='font-size:1.1rem;font-weight:800;color:#fff;
                            letter-spacing:-.01em;margin-bottom:3px'>Karthick Manoharan</div>
                <div style='font-size:.75rem;font-weight:600;color:{GO};
                            letter-spacing:.02em'>Co-Founder &amp; Software QA Governance Lead</div>
              </div>
            </div>

            <p style='font-size:.83rem;color:rgba(255,255,255,.58);
                      line-height:1.8;margin:0 0 14px'>
              Karthick brings a unique blend of experience in Quality Engineering and
              Digital Transformation across Banking, Energy, Telecom, Insurance, ERP,
              and Public Sector domains. He helps organizations build robust testing
              strategies, design automation frameworks, and manage end-to-end SDLC/STLC
              for critical enterprise platforms.
            </p>
            <p style='font-size:.83rem;color:rgba(255,255,255,.58);
                      line-height:1.8;margin:0'>
              With a focus on AI system validation, data quality, API and integration
              testing, performance, and production readiness, Karthick ensures systems
              are not only reliable and secure but also scalable. He has led complex
              digital transformation programs implementing standardized quality practices,
              risk controls, and governance models — giving business owners the confidence
              to act on every insight DataNetra delivers.
            </p>

            <!-- Tags -->
            <div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:20px'>
              <span style='background:rgba(240,165,0,.1);border:1px solid rgba(240,165,0,.25);
                           color:{GO};border-radius:20px;padding:4px 12px;
                           font-size:.68rem;font-weight:700'>Quality Engineering</span>
              <span style='background:rgba(240,165,0,.1);border:1px solid rgba(240,165,0,.25);
                           color:{GO};border-radius:20px;padding:4px 12px;
                           font-size:.68rem;font-weight:700'>AI Validation</span>
              <span style='background:rgba(240,165,0,.1);border:1px solid rgba(240,165,0,.25);
                           color:{GO};border-radius:20px;padding:4px 12px;
                           font-size:.68rem;font-weight:700'>Digital Transformation</span>
            </div>
          </div>

        </div>
      </div>
    </div>
    """)

    # ── HOW WE WORK SECTION ───────────────────────────────────────────────────
    gr.HTML(f"""
    <div id='how-we-work-section' style='
      background:{BG};
      border-top:1px solid rgba(0,212,180,.08);
      padding:80px 40px;
      position:relative;overflow:hidden;
    '>
      <div style='position:absolute;inset:0;pointer-events:none;
        background:radial-gradient(ellipse at 50% 100%, rgba(0,212,180,.08) 0%, transparent 55%);
      '></div>

      <div style='max-width:1100px;margin:0 auto;position:relative;z-index:1'>

        <!-- Header -->
        <div style='text-align:center;margin-bottom:56px'>
          <span style='font-family:"JetBrains Mono",monospace;font-size:.68rem;
                       font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:{T}'>
            Our Values
          </span>
          <h2 style='font-size:2rem;font-weight:800;color:#fff;
                     margin:10px 0 12px;letter-spacing:-.025em;
                     font-family:"DM Sans",sans-serif'>
            How We <span style='color:{T}'>Work</span>
          </h2>
          <p style='color:rgba(255,255,255,.5);font-size:.92rem;
                    max-width:520px;margin:0 auto;line-height:1.78'>
            The five shared values that define how every member of the DataNetra team
            shows up every single day.
          </p>
        </div>

        <!-- Values Grid -->
        <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:16px'>

          <!-- Passion -->
          <div style='background:{B2};border:1px solid rgba(0,212,180,.12);
                      border-radius:16px;padding:28px 20px;text-align:center;
                      transition:.25s;cursor:default'
               onmouseover="this.style.borderColor='rgba(0,212,180,.4)';this.style.transform='translateY(-5px)';this.style.boxShadow='0 16px 40px rgba(0,0,0,.4)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.12)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='font-size:2.2rem;margin-bottom:14px'>🔥</div>
            <div style='font-size:.88rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Passion for Data</div>
            <p style='font-size:.76rem;color:rgba(255,255,255,.5);line-height:1.72;margin:0'>
              Every team member believes data should be accessible, actionable, and impactful
              for every business — not just large enterprises with big analytics budgets.
            </p>
          </div>

          <!-- Collaboration -->
          <div style='background:{B2};border:1px solid rgba(0,212,180,.12);
                      border-radius:16px;padding:28px 20px;text-align:center;
                      transition:.25s;cursor:default'
               onmouseover="this.style.borderColor='rgba(0,212,180,.4)';this.style.transform='translateY(-5px)';this.style.boxShadow='0 16px 40px rgba(0,0,0,.4)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.12)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='font-size:2.2rem;margin-bottom:14px'>🤝</div>
            <div style='font-size:.88rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Collaboration</div>
            <p style='font-size:.76rem;color:rgba(255,255,255,.5);line-height:1.72;margin:0'>
              We work together, bringing diverse expertise in data science, design, and business
              to deliver solutions that meet each SMB's unique, specific needs.
            </p>
          </div>

          <!-- Customer-First -->
          <div style='background:{B2};border:1px solid rgba(0,212,180,.12);
                      border-radius:16px;padding:28px 20px;text-align:center;
                      transition:.25s;cursor:default'
               onmouseover="this.style.borderColor='rgba(0,212,180,.4)';this.style.transform='translateY(-5px)';this.style.boxShadow='0 16px 40px rgba(0,0,0,.4)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.12)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='font-size:2.2rem;margin-bottom:14px'>✨</div>
            <div style='font-size:.88rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Customer-First</div>
            <p style='font-size:.76rem;color:rgba(255,255,255,.5);line-height:1.72;margin:0'>
              Our clients are at the heart of everything we do. We are committed to providing
              personalized, high-quality solutions that drive real growth and lasting
              business success.
            </p>
          </div>

          <!-- Innovation -->
          <div style='background:{B2};border:1px solid rgba(0,212,180,.12);
                      border-radius:16px;padding:28px 20px;text-align:center;
                      transition:.25s;cursor:default'
               onmouseover="this.style.borderColor='rgba(0,212,180,.4)';this.style.transform='translateY(-5px)';this.style.boxShadow='0 16px 40px rgba(0,0,0,.4)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.12)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='font-size:2.2rem;margin-bottom:14px'>💡</div>
            <div style='font-size:.88rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Innovation</div>
            <p style='font-size:.76rem;color:rgba(255,255,255,.5);line-height:1.72;margin:0'>
              We embrace innovation and constantly explore new ways to use AI and data to
              solve real-world problems for India's growing MSME ecosystem.
            </p>
          </div>

          <!-- Integrity -->
          <div style='background:{B2};border:1px solid rgba(0,212,180,.12);
                      border-radius:16px;padding:28px 20px;text-align:center;
                      transition:.25s;cursor:default'
               onmouseover="this.style.borderColor='rgba(0,212,180,.4)';this.style.transform='translateY(-5px)';this.style.boxShadow='0 16px 40px rgba(0,0,0,.4)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.12)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='font-size:2.2rem;margin-bottom:14px'>🛡️</div>
            <div style='font-size:.88rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Integrity</div>
            <p style='font-size:.76rem;color:rgba(255,255,255,.5);line-height:1.72;margin:0'>
              We operate with full transparency and honesty. Our clients trust us to deliver
              accurate, reliable, and ethical solutions — and we never take that trust lightly.
            </p>
          </div>

        </div>
      </div>
    </div>
    """)

    # ── TARGET INDUSTRIES SECTION ─────────────────────────────────────────────
    gr.HTML(f"""
    <div id='industries-section' style='
      background:{B2};
      border-top:1px solid rgba(0,212,180,.08);
      padding:80px 40px;
    '>
      <div style='max-width:1100px;margin:0 auto'>

        <!-- Header -->
        <div style='text-align:center;margin-bottom:56px'>
          <span style='font-family:"JetBrains Mono",monospace;font-size:.68rem;
                       font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:{T}'>
            Target Industries
          </span>
          <h2 style='font-size:2rem;font-weight:800;color:#fff;
                     margin:10px 0 12px;letter-spacing:-.025em;
                     font-family:"DM Sans",sans-serif'>
            Data Intelligence Across<br><span style='color:{T}'>Every Sector</span>
          </h2>
          <p style='color:rgba(255,255,255,.5);font-size:.92rem;
                    max-width:500px;margin:0 auto;line-height:1.78'>
            From clinics to farms, power plants to storefronts — DataNetra adapts its
            AI engine to the unique data patterns and business challenges of your industry.
          </p>
        </div>

        <!-- Industry Cards -->
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:20px'>

          <!-- Healthcare -->
          <div style='
            background:{B3};
            border:1px solid rgba(0,212,180,.15);
            border-radius:18px;padding:36px 24px;text-align:center;
            transition:.25s;cursor:default;position:relative;overflow:hidden
          '
          onmouseover="this.style.borderColor='rgba(0,212,180,.45)';this.style.transform='translateY(-6px)';this.style.boxShadow='0 20px 50px rgba(0,0,0,.45)'"
          onmouseout="this.style.borderColor='rgba(0,212,180,.15)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='position:absolute;top:0;left:0;right:0;height:3px;
                        background:linear-gradient(90deg,#e05c6a,#f0a500)'></div>
            <div style='font-size:3rem;margin-bottom:16px'>🏥</div>
            <div style='font-size:1rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Healthcare</div>
            <p style='font-size:.78rem;color:rgba(255,255,255,.5);line-height:1.75;margin:0'>
              Track medicine inventory, forecast patient footfall, and optimise procurement
              costs for clinics, pharmacies, and diagnostic centres.
            </p>
            <div style='margin-top:18px;display:inline-block;
                        background:rgba(224,92,106,.1);border:1px solid rgba(224,92,106,.25);
                        color:#e05c6a;border-radius:20px;padding:4px 14px;
                        font-size:.65rem;font-weight:700;letter-spacing:.06em'>
              PHARMA · CLINICS · DIAGNOSTICS
            </div>
          </div>

          <!-- Energy & Power -->
          <div style='
            background:{B3};
            border:1px solid rgba(0,212,180,.15);
            border-radius:18px;padding:36px 24px;text-align:center;
            transition:.25s;cursor:default;position:relative;overflow:hidden
          '
          onmouseover="this.style.borderColor='rgba(0,212,180,.45)';this.style.transform='translateY(-6px)';this.style.boxShadow='0 20px 50px rgba(0,0,0,.45)'"
          onmouseout="this.style.borderColor='rgba(0,212,180,.15)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='position:absolute;top:0;left:0;right:0;height:3px;
                        background:linear-gradient(90deg,#f0a500,#ffe066)'></div>
            <div style='font-size:3rem;margin-bottom:16px'>⚡</div>
            <div style='font-size:1rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Energy &amp; Power</div>
            <p style='font-size:.78rem;color:rgba(255,255,255,.5);line-height:1.75;margin:0'>
              Forecast energy consumption patterns, identify peak demand periods, and
              optimise operational costs for utilities, solar distributors, and power SMEs.
            </p>
            <div style='margin-top:18px;display:inline-block;
                        background:rgba(240,165,0,.1);border:1px solid rgba(240,165,0,.25);
                        color:{GO};border-radius:20px;padding:4px 14px;
                        font-size:.65rem;font-weight:700;letter-spacing:.06em'>
              SOLAR · UTILITIES · DISTRIBUTION
            </div>
          </div>

          <!-- Agriculture -->
          <div style='
            background:{B3};
            border:1px solid rgba(0,212,180,.15);
            border-radius:18px;padding:36px 24px;text-align:center;
            transition:.25s;cursor:default;position:relative;overflow:hidden
          '
          onmouseover="this.style.borderColor='rgba(0,212,180,.45)';this.style.transform='translateY(-6px)';this.style.boxShadow='0 20px 50px rgba(0,0,0,.45)'"
          onmouseout="this.style.borderColor='rgba(0,212,180,.15)';this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style='position:absolute;top:0;left:0;right:0;height:3px;
                        background:linear-gradient(90deg,#2ecc71,#00d4b4)'></div>
            <div style='font-size:3rem;margin-bottom:16px'>🌾</div>
            <div style='font-size:1rem;font-weight:800;color:#fff;
                        margin-bottom:10px;letter-spacing:-.01em'>Agriculture</div>
            <p style='font-size:.78rem;color:rgba(255,255,255,.5);line-height:1.75;margin:0'>
              Predict seasonal crop demand, reduce post-harvest losses, and improve
              supply chain margins for agri-traders, FPOs, and rural cooperatives.
            </p>
            <div style='margin-top:18px;display:inline-block;
                        background:rgba(46,204,113,.1);border:1px solid rgba(46,204,113,.25);
                        color:#2ecc71;border-radius:20px;padding:4px 14px;
                        font-size:.65rem;font-weight:700;letter-spacing:.06em'>
              AGRI-TRADE · FPO · RURAL
            </div>
          </div>

          <!-- Retail -->
          <div style='
            background:{B3};
            border:1px solid rgba(0,212,180,.2);
            border-radius:18px;padding:36px 24px;text-align:center;
            transition:.25s;cursor:default;position:relative;overflow:hidden;
            box-shadow:0 0 30px rgba(0,212,180,.08)
          '
          onmouseover="this.style.borderColor='rgba(0,212,180,.55)';this.style.transform='translateY(-6px)';this.style.boxShadow='0 20px 50px rgba(0,0,0,.45)'"
          onmouseout="this.style.borderColor='rgba(0,212,180,.2)';this.style.transform='translateY(0)';this.style.boxShadow='0 0 30px rgba(0,212,180,.08)'">
            <div style='position:absolute;top:0;left:0;right:0;height:3px;
                        background:linear-gradient(90deg,{T},rgba(0,212,180,.4))'></div>
            <div style='font-size:3rem;margin-bottom:16px'>🛍️</div>
            <div style='font-size:1rem;font-weight:800;color:{T};
                        margin-bottom:10px;letter-spacing:-.01em'>Retail</div>
            <p style='font-size:.78rem;color:rgba(255,255,255,.5);line-height:1.75;margin:0'>
              Analyse SKU performance, forecast footfall and revenue, reduce dead stock,
              and grow margins for hypermarkets.
            </p>
            <div style='margin-top:18px;display:inline-block;
                        background:rgba(0,212,180,.1);border:1px solid rgba(0,212,180,.3);
                        color:{T};border-radius:20px;padding:4px 14px;
                        font-size:.65rem;font-weight:700;letter-spacing:.06em'>
              HYPERMARKET
            </div>
          </div>

        </div>
      </div>
    </div>
    """)

    # ── CONTACT SECTION ───────────────────────────────────────────────────────
    gr.HTML(f"""
    <div id='contact-section' style='
      background:{B2};
      border-top:1px solid rgba(0,212,180,.1);
      padding:72px 40px;
    '>
      <div style='max-width:1100px;margin:0 auto'>

        <!-- Header -->
        <div style='text-align:center;margin-bottom:56px'>
          <span style='
            font-family:"JetBrains Mono",monospace;font-size:.68rem;
            font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:{T}
          '>Get in Touch</span>
          <h2 style='
            font-size:2rem;font-weight:800;color:#fff;
            margin:10px 0 10px;letter-spacing:-.025em;
            font-family:"DM Sans",sans-serif
          '>Let's Start a <span style='color:{T}'>Conversation</span></h2>
          <p style='color:rgba(255,255,255,.55);font-size:.92rem;
                    max-width:560px;margin:0 auto;line-height:1.78'>
            Whether you're interested in collaborating, piloting DataNetra in your business,
            exploring enterprise solutions, or simply want to understand how AI analytics
            can transform your operations — drop us a message and
            <strong style='color:#fff'>we will call you back</strong> personally.
          </p>
        </div>

        <!-- Two-column layout -->
        <div style='display:grid;grid-template-columns:1fr 1.4fr;gap:40px;align-items:start'>

          <!-- Left: Contact Info -->
          <div style='display:flex;flex-direction:column;gap:20px'>

            <!-- Email -->
            <a href='mailto:innovate@datanetra.ai' style='
              display:flex;align-items:center;gap:16px;
              background:{B3};border:1px solid rgba(0,212,180,.2);
              border-radius:14px;padding:20px 22px;
              text-decoration:none;transition:.2s;
            ' onmouseover="this.style.borderColor='rgba(0,212,180,.5)';this.style.background='rgba(0,212,180,.06)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.2)';this.style.background='{B3}'">
              <div style='
                width:44px;height:44px;border-radius:12px;flex-shrink:0;
                background:rgba(0,212,180,.12);border:1px solid rgba(0,212,180,.25);
                display:flex;align-items:center;justify-content:center;font-size:1.2rem
              '>📧</div>
              <div>
                <div style='font-size:.7rem;color:rgba(255,255,255,.38);
                            text-transform:uppercase;letter-spacing:.1em;margin-bottom:3px'>Email Us</div>
                <div style='font-size:.92rem;font-weight:700;color:{T}'>innovate@datanetra.ai</div>
              </div>
            </a>

            <!-- LinkedIn -->
            <a href='https://www.linkedin.com/company/datanetra-company' target='_blank' style='
              display:flex;align-items:center;gap:16px;
              background:{B3};border:1px solid rgba(0,212,180,.2);
              border-radius:14px;padding:20px 22px;
              text-decoration:none;transition:.2s;
            ' onmouseover="this.style.borderColor='rgba(0,212,180,.5)';this.style.background='rgba(0,212,180,.06)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.2)';this.style.background='{B3}'">
              <div style='
                width:44px;height:44px;border-radius:12px;flex-shrink:0;
                background:rgba(10,102,194,.2);border:1px solid rgba(10,102,194,.4);
                display:flex;align-items:center;justify-content:center;
                font-size:.85rem;font-weight:900;color:#0a66c2
              '>in</div>
              <div>
                <div style='font-size:.7rem;color:rgba(255,255,255,.38);
                            text-transform:uppercase;letter-spacing:.1em;margin-bottom:3px'>LinkedIn</div>
                <div style='font-size:.92rem;font-weight:700;color:#fff'>linkedin.com/company/datanetra-company</div>
              </div>
            </a>

            <!-- Website -->
            <a href='https://datanetra.ai' target='_blank' style='
              display:flex;align-items:center;gap:16px;
              background:{B3};border:1px solid rgba(0,212,180,.2);
              border-radius:14px;padding:20px 22px;
              text-decoration:none;transition:.2s;
            ' onmouseover="this.style.borderColor='rgba(0,212,180,.5)';this.style.background='rgba(0,212,180,.06)'"
               onmouseout="this.style.borderColor='rgba(0,212,180,.2)';this.style.background='{B3}'">
              <div style='
                width:44px;height:44px;border-radius:12px;flex-shrink:0;
                background:rgba(0,212,180,.12);border:1px solid rgba(0,212,180,.25);
                display:flex;align-items:center;justify-content:center;font-size:1.2rem
              '>🌐</div>
              <div>
                <div style='font-size:.7rem;color:rgba(255,255,255,.38);
                            text-transform:uppercase;letter-spacing:.1em;margin-bottom:3px'>Website</div>
                <div style='font-size:.92rem;font-weight:700;color:#fff'>datanetra.ai</div>
              </div>
            </a>

            <!-- IndiaAI Badge -->
            <div style='
              background:linear-gradient(135deg,rgba(255,153,0,.08),rgba(19,136,8,.08));
              border:1px solid rgba(255,153,0,.3);
              border-radius:14px;padding:20px 22px;margin-top:4px
            '>
              <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>
                <span style='font-size:1.1rem'>🎯</span>
                <span style='font-size:.78rem;font-weight:800;color:#ff9900;
                              letter-spacing:.03em'>IndiaAI Innovation Challenge 2026</span>
              </div>
              <p style='font-size:.82rem;color:rgba(255,255,255,.6);
                        margin:0;line-height:1.7'>
                DataNetra is currently competing in the <strong style='color:rgba(255,255,255,.85)'>IndiaAI Innovation Challenge 2026</strong>
                under the Ministry of MSME — targeting AI-powered MSE Agent Mapping for the ONDC ecosystem.
                We are building the future of MSME intelligence for Bharat.
              </p>
            </div>

          </div>

          <!-- Right: Contact Form -->
          <div style='
            background:{B3};
            border:1px solid rgba(0,212,180,.2);
            border-radius:18px;padding:36px;
          '>
            <div style='font-size:1.05rem;font-weight:800;color:#fff;
                        margin-bottom:24px;letter-spacing:-.01em'>
              Send Us a Message
            </div>

            <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px'>
              <div>
                <label style='font-size:.75rem;font-weight:700;
                              color:rgba(255,255,255,.6);display:block;margin-bottom:7px;
                              letter-spacing:.03em'>Full Name *</label>
                <input type='text' id='cf-name' placeholder='Your full name'
                  style='width:100%;background:{B2};border:1.5px solid rgba(0,212,180,.25);
                         color:#fff;border-radius:10px;padding:11px 14px;
                         font-size:.88rem;font-family:"DM Sans",sans-serif;
                         box-sizing:border-box;outline:none;transition:.2s'
                  onfocus="this.style.borderColor='#00d4b4';this.style.boxShadow='0 0 0 3px rgba(0,212,180,.12)'"
                  onblur="this.style.borderColor='rgba(0,212,180,.25)';this.style.boxShadow='none'"/>
              </div>
              <div>
                <label style='font-size:.75rem;font-weight:700;
                              color:rgba(255,255,255,.6);display:block;margin-bottom:7px;
                              letter-spacing:.03em'>Email Address *</label>
                <input type='email' id='cf-email' placeholder='you@company.com'
                  style='width:100%;background:{B2};border:1.5px solid rgba(0,212,180,.25);
                         color:#fff;border-radius:10px;padding:11px 14px;
                         font-size:.88rem;font-family:"DM Sans",sans-serif;
                         box-sizing:border-box;outline:none;transition:.2s'
                  onfocus="this.style.borderColor='#00d4b4';this.style.boxShadow='0 0 0 3px rgba(0,212,180,.12)'"
                  onblur="this.style.borderColor='rgba(0,212,180,.25)';this.style.boxShadow='none'"/>
              </div>
            </div>

            <div style='margin-bottom:16px'>
              <label style='font-size:.75rem;font-weight:700;
                            color:rgba(255,255,255,.6);display:block;margin-bottom:7px;
                            letter-spacing:.03em'>Company / Organisation</label>
              <input type='text' id='cf-company' placeholder='Your company or organisation name'
                style='width:100%;background:{B2};border:1.5px solid rgba(0,212,180,.25);
                       color:#fff;border-radius:10px;padding:11px 14px;
                       font-size:.88rem;font-family:"DM Sans",sans-serif;
                       box-sizing:border-box;outline:none;transition:.2s'
                onfocus="this.style.borderColor='#00d4b4';this.style.boxShadow='0 0 0 3px rgba(0,212,180,.12)'"
                onblur="this.style.borderColor='rgba(0,212,180,.25)';this.style.boxShadow='none'"/>
            </div>

            <div style='margin-bottom:16px'>
              <label style='font-size:.75rem;font-weight:700;
                            color:rgba(255,255,255,.6);display:block;margin-bottom:7px;
                            letter-spacing:.03em'>I'm interested in</label>
              <select id='cf-interest'
                style='width:100%;background:{B2};border:1.5px solid rgba(0,212,180,.25);
                       color:rgba(255,255,255,.75);border-radius:10px;padding:11px 14px;
                       font-size:.88rem;font-family:"DM Sans",sans-serif;
                       box-sizing:border-box;outline:none;cursor:pointer;
                       appearance:none;-webkit-appearance:none;transition:.2s'
                onfocus="this.style.borderColor='#00d4b4'"
                onblur="this.style.borderColor='rgba(0,212,180,.25)'">
                <option value='' style='background:{B2}'>Select an option</option>
                <option value='trial' style='background:{B2}'>Free Trial / Demo</option>
                <option value='pro' style='background:{B2}'>Professional Subscription</option>
                <option value='enterprise' style='background:{B2}'>Enterprise Solution</option>
                <option value='partner' style='background:{B2}'>Partnership / Collaboration</option>
                <option value='indiaai' style='background:{B2}'>IndiaAI Challenge Related</option>
                <option value='other' style='background:{B2}'>Other</option>
              </select>
            </div>

            <div style='margin-bottom:22px'>
              <label style='font-size:.75rem;font-weight:700;
                            color:rgba(255,255,255,.6);display:block;margin-bottom:7px;
                            letter-spacing:.03em'>Message *</label>
              <textarea id='cf-message' rows='4'
                placeholder='Tell us about your use case, business size, or any questions…'
                style='width:100%;background:{B2};border:1.5px solid rgba(0,212,180,.25);
                       color:#fff;border-radius:10px;padding:11px 14px;
                       font-size:.88rem;font-family:"DM Sans",sans-serif;
                       box-sizing:border-box;outline:none;resize:vertical;
                       line-height:1.6;transition:.2s'
                onfocus="this.style.borderColor='#00d4b4';this.style.boxShadow='0 0 0 3px rgba(0,212,180,.12)'"
                onblur="this.style.borderColor='rgba(0,212,180,.25)';this.style.boxShadow='none'"></textarea>
            </div>

            <button onclick="
              var name=document.getElementById('cf-name').value;
              var email=document.getElementById('cf-email').value;
              var company=document.getElementById('cf-company').value;
              var interest=document.getElementById('cf-interest').value;
              var msg=document.getElementById('cf-message').value;
              if(!name||!email||!msg){{
                alert('Please fill in Name, Email and Message fields.');return;
              }}
              var subject=encodeURIComponent('DataNetra Enquiry: '+interest);
              var body=encodeURIComponent('Name: '+name+'\\nEmail: '+email+'\\nCompany: '+company+'\\nInterested in: '+interest+'\\n\\nMessage:\\n'+msg);
              window.location.href='mailto:innovate@datanetra.ai?subject='+subject+'&body='+body;
              document.getElementById('cf-success').style.display='block';
            " style='
              width:100%;background:{T};color:#000;
              font-weight:900;font-size:.95rem;
              font-family:"DM Sans",sans-serif;
              border:none;border-radius:50px;
              padding:14px 32px;cursor:pointer;
              box-shadow:0 4px 24px rgba(0,212,180,.45);
              transition:.2s;letter-spacing:.01em
            '
            onmouseover="this.style.background='#00eacc';this.style.transform='translateY(-2px)'"
            onmouseout="this.style.background='{T}';this.style.transform='translateY(0)'">
              Send Message →
            </button>

            <div id='cf-success' style='
              display:none;margin-top:16px;
              background:rgba(46,204,113,.1);border:1px solid rgba(46,204,113,.3);
              border-radius:10px;padding:12px 16px;
              font-size:.85rem;color:#2ecc71;font-weight:600;text-align:center
            '>
              ✅ Request Received… We'll get back to you shortly!
            </div>

          </div>
        </div>
      </div>
    </div>
    """)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div style='background:{B2};border-top:1px solid rgba(0,212,180,.1);
                padding:28px 40px;margin-top:32px'>
      <div style='display:flex;justify-content:space-between;align-items:center;
                  flex-wrap:wrap;gap:14px;max-width:100%;'>
        <div style='display:flex;align-items:center'>
          <img src='data:image/png;base64,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'
               style='height:40px;width:auto;object-fit:contain;display:block'
               alt='DataNetra.ai logo'/>
        </div>
        <div style='font-size:.75rem;color:rgba(255,255,255,.28);text-align:center'>
          DPDP Act 2023 Compliant · 🇮🇳 Made in India
        </div>
        <div style='display:flex;gap:18px;align-items:center'>
          <a href='mailto:innovate@datanetra.ai'
             style='font-size:.78rem;color:{T};text-decoration:none'>
            innovate@datanetra.ai
          </a>
          <a href='https://www.linkedin.com/company/datanetra-company'
             target='_blank' style='font-size:.78rem;color:{T};text-decoration:none'>
            LinkedIn
          </a>
          <span style='font-size:.72rem;color:rgba(255,255,255,.2)'>
            © 2025 DataNetra.ai
          </span>
        </div>
      </div>
    </div>
    """)

    # ── Wire ──────────────────────────────────────────────────────────────────
    OUTS = [status_out, gauge_out, sub_out,
            top10_out, top10_ins,
            fc_out, fc_met,
            km_out, km_tbl,
            narr_out, ondc_out]
    INPS = [file_in, cat_in, fm_in, nc_in]
    run_btn.click(fn=analyse, inputs=INPS, outputs=OUTS)
    file_in.change(fn=analyse, inputs=INPS, outputs=OUTS)

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def free_port(start=7860, end=7920):
        for p in range(start, end):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("", p))
                    return p
                except OSError:
                    continue
        return 0
    import os
    port = int(os.environ.get("PORT", free_port() or 7860))
    print(f"\n🚀  DataNetra.ai  →  http://0.0.0.0:{port}\n")
    demo.launch(server_name="0.0.0.0",
                server_port=port,
                show_error=True,
                share=False,
                inbrowser=False)
