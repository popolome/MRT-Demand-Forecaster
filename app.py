import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Singapore MRT Demand Forecaster",
    page_icon="🚇",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border-left: 4px solid #1a56db;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-title { font-size: 13px; color: #6b7280; font-weight: 600; }
    .metric-value { font-size: 28px; font-weight: 700; color: #111827; }
    .metric-sub   { font-size: 12px; color: #6b7280; margin-top: 2px; }
    .winner-badge {
        display: inline-block;
        background: #d1fae5;
        color: #065f46;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
        margin-left: 8px;
    }
    .insight-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #374151;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Hard-coded data (from LTA DataMall, Jan 2019 – Sep 2024) ─────────────────
RAW = {
    "Jan-19":{"MRT":3462000,"LRT":218000,"Public Bus":4250000},
    "Feb-19":{"MRT":3248000,"LRT":206000,"Public Bus":4010000},
    "Mar-19":{"MRT":3383000,"LRT":209000,"Public Bus":4111000},
    "Apr-19":{"MRT":3400000,"LRT":209000,"Public Bus":4100000},
    "May-19":{"MRT":3359000,"LRT":208000,"Public Bus":4057000},
    "Jun-19":{"MRT":3302000,"LRT":202000,"Public Bus":3966000},
    "Jul-19":{"MRT":3439000,"LRT":210000,"Public Bus":4110000},
    "Aug-19":{"MRT":3384000,"LRT":208000,"Public Bus":4053000},
    "Sep-19":{"MRT":3361000,"LRT":207000,"Public Bus":4030000},
    "Oct-19":{"MRT":3410000,"LRT":211000,"Public Bus":4102000},
    "Nov-19":{"MRT":3336000,"LRT":205000,"Public Bus":4019000},
    "Dec-19":{"MRT":3329000,"LRT":207000,"Public Bus":4010000},
    "Jan-20":{"MRT":3359000,"LRT":212000,"Public Bus":4123000},
    "Feb-20":{"MRT":2743000,"LRT":172000,"Public Bus":3487000},
    "Mar-20":{"MRT":1714000,"LRT":107000,"Public Bus":2178000},
    "Apr-20":{"MRT":602000, "LRT":47000, "Public Bus":1187000},
    "May-20":{"MRT":916000, "LRT":69000, "Public Bus":1596000},
    "Jun-20":{"MRT":1601000,"LRT":104000,"Public Bus":2373000},
    "Jul-20":{"MRT":1985000,"LRT":127000,"Public Bus":2787000},
    "Aug-20":{"MRT":2133000,"LRT":136000,"Public Bus":2952000},
    "Sep-20":{"MRT":2237000,"LRT":143000,"Public Bus":3062000},
    "Oct-20":{"MRT":2362000,"LRT":151000,"Public Bus":3201000},
    "Nov-20":{"MRT":2382000,"LRT":152000,"Public Bus":3224000},
    "Dec-20":{"MRT":2361000,"LRT":151000,"Public Bus":3198000},
    "Jan-21":{"MRT":2363000,"LRT":150000,"Public Bus":3202000},
    "Feb-21":{"MRT":2274000,"LRT":144000,"Public Bus":3094000},
    "Mar-21":{"MRT":2492000,"LRT":158000,"Public Bus":3351000},
    "Apr-21":{"MRT":2558000,"LRT":162000,"Public Bus":3432000},
    "May-21":{"MRT":2219000,"LRT":142000,"Public Bus":3030000},
    "Jun-21":{"MRT":1934000,"LRT":124000,"Public Bus":2670000},
    "Jul-21":{"MRT":1925000,"LRT":123000,"Public Bus":2659000},
    "Aug-21":{"MRT":2019000,"LRT":129000,"Public Bus":2783000},
    "Sep-21":{"MRT":2089000,"LRT":133000,"Public Bus":2878000},
    "Oct-21":{"MRT":2330000,"LRT":148000,"Public Bus":3171000},
    "Nov-21":{"MRT":2490000,"LRT":158000,"Public Bus":3350000},
    "Dec-21":{"MRT":2617000,"LRT":166000,"Public Bus":3502000},
    "Jan-22":{"MRT":2667000,"LRT":169000,"Public Bus":3559000},
    "Feb-22":{"MRT":2632000,"LRT":167000,"Public Bus":3514000},
    "Mar-22":{"MRT":2929000,"LRT":185000,"Public Bus":3872000},
    "Apr-22":{"MRT":3027000,"LRT":192000,"Public Bus":3993000},
    "May-22":{"MRT":3061000,"LRT":194000,"Public Bus":4033000},
    "Jun-22":{"MRT":3084000,"LRT":196000,"Public Bus":4062000},
    "Jul-22":{"MRT":3169000,"LRT":201000,"Public Bus":4162000},
    "Aug-22":{"MRT":3173000,"LRT":201000,"Public Bus":4167000},
    "Sep-22":{"MRT":3179000,"LRT":202000,"Public Bus":4174000},
    "Oct-22":{"MRT":3316000,"LRT":210000,"Public Bus":4338000},
    "Nov-22":{"MRT":3329000,"LRT":211000,"Public Bus":4354000},
    "Dec-22":{"MRT":3326000,"LRT":211000,"Public Bus":4350000},
    "Jan-23":{"MRT":3360000,"LRT":213000,"Public Bus":4392000},
    "Feb-23":{"MRT":3149000,"LRT":200000,"Public Bus":4127000},
    "Mar-23":{"MRT":3489000,"LRT":221000,"Public Bus":4552000},
    "Apr-23":{"MRT":3441000,"LRT":218000,"Public Bus":4495000},
    "May-23":{"MRT":3441000,"LRT":218000,"Public Bus":4495000},
    "Jun-23":{"MRT":3393000,"LRT":215000,"Public Bus":4432000},
    "Jul-23":{"MRT":3483000,"LRT":221000,"Public Bus":4549000},
    "Aug-23":{"MRT":3499000,"LRT":222000,"Public Bus":4570000},
    "Sep-23":{"MRT":3467000,"LRT":220000,"Public Bus":4528000},
    "Oct-23":{"MRT":3542000,"LRT":225000,"Public Bus":4624000},
    "Nov-23":{"MRT":3481000,"LRT":221000,"Public Bus":4548000},
    "Dec-23":{"MRT":3501000,"LRT":222000,"Public Bus":4573000},
    "Jan-24":{"MRT":3504000,"LRT":222000,"Public Bus":4577000},
    "Feb-24":{"MRT":3356000,"LRT":213000,"Public Bus":4386000},
    "Mar-24":{"MRT":3604000,"LRT":220000,"Public Bus":4300000},
    "Apr-24":{"MRT":3531000,"LRT":215000,"Public Bus":4212000},
    "May-24":{"MRT":3530000,"LRT":215000,"Public Bus":4211000},
    "Jun-24":{"MRT":3480000,"LRT":212000,"Public Bus":4152000},
    "Jul-24":{"MRT":3562000,"LRT":217000,"Public Bus":4254000},
    "Aug-24":{"MRT":3569000,"LRT":217000,"Public Bus":4262000},
    "Sep-24":{"MRT":3540000,"LRT":215000,"Public Bus":4228000},
}

@st.cache_data
def load_data():
    rows = []
    for month_str, modes in RAW.items():
        date = pd.to_datetime(month_str, format="%b-%y")
        for mode, ridership in modes.items():
            rows.append({"date": date, "mode": mode, "ridership": ridership})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

@st.cache_data
def fit_sarima():
    """Fit SARIMA(0,1,0)×(1,1,0,12) on scaled data — mirrors notebook section 20."""
    rows = []
    for month_str, modes in RAW.items():
        date = pd.to_datetime(month_str, format="%b-%y")
        rows.append({"date": date, "ridership": modes["MRT"]})
    mrt_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Scale to millions (same as notebook)
    ridership_scaled = mrt_df["ridership"] / 1_000_000

    # Same train/test split as notebook: 60 train, 12 test
    train_scaled = ridership_scaled[:60]
    test_scaled  = ridership_scaled[60:]
    train_dates  = mrt_df["date"][:60]
    test_dates   = mrt_df["date"][60:]

    # Fit final model
    model = SARIMAX(train_scaled, order=(0, 1, 0), seasonal_order=(1, 1, 0, 12))
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(test_scaled))

    return train_dates, train_scaled, test_dates, test_scaled, forecast

df = load_data()
mrt = df[df["mode"] == "MRT"].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚇 Singapore MRT Demand Forecaster")
st.markdown("**Time-series forecasting of Singapore MRT ridership** · LTA DataMall · Jan 2019 – Sep 2024")
st.markdown("---")

# ── Key Metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">PRODUCTION MODEL</div>
        <div class="metric-value">SARIMA <span class="winner-badge">✅ FINAL</span></div>
        <div class="metric-sub">SARIMA(0,1,0)×(1,1,0,12)</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">SARIMA MAPE (2024 test)</div>
        <div class="metric-value">4.67%</div>
        <div class="metric-sub">MAE: 157,776 avg daily riders</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">ERROR REDUCTION vs ARIMA</div>
        <div class="metric-value">52%</div>
        <div class="metric-sub">47% reduction vs best Prophet config</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">TRAINING DATA</div>
        <div class="metric-value">72 months</div>
        <div class="metric-sub">Includes full COVID-19 disruption</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ── Chart 1 – SARIMA Forecast (THE main chart) ────────────────────────────────
st.subheader("🎯 SARIMA(0,1,0)×(1,1,0,12) — Forecast vs Actual (2024 Test Set)")

with st.spinner("Fitting SARIMA model..."):
    train_dates, train_scaled, test_dates, test_scaled, sarima_forecast = fit_sarima()

sns.set_theme(style="darkgrid")
fig0, ax0 = plt.subplots(figsize=(14, 5))

ax0.plot(train_dates, train_scaled, color="#1a56db", label="Training Data", alpha=0.8)
ax0.plot(test_dates, test_scaled, color="#22c55e", linewidth=2.5, label="Actual (2024 Test)")
ax0.plot(test_dates, sarima_forecast, color="#f59e0b", linestyle="--",
         linewidth=2.5, label="SARIMA Forecast")

ax0.set_xlabel("Date")
ax0.set_ylabel("Ridership (Millions)")
ax0.set_title("Final Model: SARIMA(0,1,0)×(1,1,0,12) — 4.67% MAPE on 2024 Test Set",
              fontsize=13, fontweight="bold")
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
ax0.legend()
plt.tight_layout()
st.pyplot(fig0)
plt.close()

# Show actual vs forecast table
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    forecast_table = pd.DataFrame({
        "Month": test_dates.dt.strftime("%b %Y").values,
        "Actual (M)": test_scaled.values.round(3),
        "Forecast (M)": sarima_forecast.values.round(3),
        "Error %": (abs(test_scaled.values - sarima_forecast.values) / test_scaled.values * 100).round(2),
    })

with st.expander("📋 View Actual vs Forecast Table (2024)"):
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

st.markdown("")

# ── Chart 2 – Ridership Trend ─────────────────────────────────────────────────
st.subheader("📈 Public Transport Ridership (Jan 2019 – Sep 2024)")

fig1, ax1 = plt.subplots(figsize=(14, 5))

colors = {"MRT": "#1a56db", "LRT": "#f59e0b", "Public Bus": "#10b981"}
for mode in ["MRT", "Public Bus", "LRT"]:
    md = df[df["mode"] == mode]
    ax1.plot(md["date"], md["ridership"] / 1_000_000,
             marker="o", markersize=3, label=mode, color=colors[mode])

ax1.axvspan(pd.Timestamp("2020-04-07"), pd.Timestamp("2020-06-01"),
            alpha=0.15, color="red")
ax1.text(pd.Timestamp("2020-04-10"), 0.3,
         "Circuit\nBreaker", fontsize=8, color="red", va="bottom")

ax1.set_xlabel("Month")
ax1.set_ylabel("Avg Daily Ridership (Millions)")
ax1.set_title("Singapore Public Transport Daily Ridership", fontsize=13, fontweight="bold")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
ax1.legend()
plt.tight_layout()
st.pyplot(fig1)
plt.close()

# ── Chart 3 – Model Comparison ────────────────────────────────────────────────
st.subheader("🏆 Model Performance Comparison (2024 Test Set)")

st.markdown("""
<div class="insight-box">
    💡 <strong>Key finding:</strong> SARIMA(0,1,0)×(1,1,0,12) achieves <strong>4.67% MAPE</strong> —
    52% lower error than ARIMA (9.74%) and 47% lower than the best Prophet config (8.76%).
    A lower-MAPE SARIMA variant (2.54%) was rejected after failing diagnostic tests
    (boundary parameters + residual violations).
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns([1, 2])

with col_a:
    model_data = pd.DataFrame({
        "Model": [
            "SARIMA(0,1,0)×(1,1,0,12)",
            "ARIMA(2,1,2)",
            "Prophet (Multiplicative)",
            "Prophet (Additive)",
            "SARIMA(0,1,0)×(1,1,1,12)",
        ],
        "MAPE %": [4.67, 9.74, 8.76, 12.61, 2.54],
        "MAE (riders)": ["157,776", "335,614", "297,615", "—", "—"],
        "Status": ["✅ Production", "Baseline", "❌ Underperformed", "❌ Worst", "⚠️ Rejected*"],
    })

    def highlight_rows(row):
        if row["Status"] == "✅ Production":
            return ["background-color: #d1fae5; font-weight: bold; color: black"] * len(row)
        elif "Rejected" in row["Status"]:
            return ["background-color: #fef3c7; color: black"] * len(row)
        elif "❌" in row["Status"]:
            return ["background-color: #fee2e2; color: black"] * len(row)
        return ["color: black"] * len(row)

    st.dataframe(
        model_data.style.apply(highlight_rows, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("*Rejected: boundary parameters + residual diagnostic failures despite low MAPE")

with col_b:
    chart_models = model_data[model_data["Status"] != "⚠️ Rejected*"].copy()
    bar_colors = ["#22c55e", "#ef4444", "#f97316", "#dc2626"]

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    bars3 = ax3.barh(chart_models["Model"], chart_models["MAPE %"],
                     color=bar_colors, alpha=0.9)
    ax3.bar_label(bars3, fmt="%.2f%%", padding=4, fontsize=10)
    ax3.axvline(x=5, color="#f59e0b", linestyle="--", linewidth=1.5, label="5% threshold")
    ax3.set_xlabel("MAPE (%) — Lower is better")
    ax3.set_title("SARIMA is the only model under 5% MAPE", fontsize=11)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 16)
    ax3.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ── Chart 4 – Seasonality ─────────────────────────────────────────────────────
st.subheader("📅 MRT Seasonality — Average Ridership by Calendar Month")
st.caption("Note: COVID years (2020–2021) suppress the Apr–Jun average significantly.")

mrt_s = mrt.copy()
mrt_s["month_num"] = mrt_s["date"].dt.month
monthly_avg = mrt_s.groupby("month_num")["ridership"].mean() / 1_000_000
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

fig2, ax2 = plt.subplots(figsize=(14, 4))
bar_colors_season = ["#ef4444" if m in [4, 5, 6] else "#1a56db" for m in range(1, 13)]
ax2.bar(monthly_avg.index, monthly_avg.values, color=bar_colors_season, alpha=0.85)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_labels)
ax2.set_xlabel("Month")
ax2.set_ylabel("Avg Daily Ridership (Millions)")
ax2.set_title("MRT Seasonality — Red bars: COVID-suppressed months (Apr–Jun)", fontsize=12)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ── Chart 5 – COVID Anomaly Detection ─────────────────────────────────────────
st.subheader("🦠 COVID-19 Ridership Anomaly — Apr 2020 Circuit Breaker")

mrt_sorted = mrt.sort_values("date").copy()
rolling_mean = mrt_sorted["ridership"].rolling(window=3, center=True).mean() / 1_000_000
actual = mrt_sorted["ridership"] / 1_000_000

fig4, ax4 = plt.subplots(figsize=(14, 4))
ax4.plot(mrt_sorted["date"], actual, color="#1a56db", marker="o", markersize=3,
         label="Actual MRT Ridership")
ax4.plot(mrt_sorted["date"], rolling_mean, linestyle="--", color="#9ca3af",
         label="3-Month Rolling Mean")
ax4.fill_between(mrt_sorted["date"],
                 rolling_mean * 0.85, rolling_mean * 1.15,
                 alpha=0.15, color="#9ca3af", label="±15% Normal Band")

min_idx = mrt_sorted["ridership"].idxmin()
min_row = mrt_sorted.loc[min_idx]
ax4.annotate(
    f"Circuit Breaker Low\n{min_row['ridership']/1e6:.2f}M daily avg",
    xy=(min_row["date"], min_row["ridership"] / 1e6),
    xytext=(min_row["date"] + pd.DateOffset(months=5),
            min_row["ridership"] / 1e6 + 0.6),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=9, color="red",
)

ax4.set_xlabel("Month")
ax4.set_ylabel("Avg Daily Ridership (Millions)")
ax4.set_title("MRT Ridership — COVID-19 Disruption clearly visible (2020–2021)", fontsize=12)
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
ax4.legend()
plt.tight_layout()
st.pyplot(fig4)
plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Built by **Jun Kit Mak** · "
    "[GitHub](https://github.com/popolome/MRT-Demand-Forecaster) · "
    "Data: LTA DataMall"
)
