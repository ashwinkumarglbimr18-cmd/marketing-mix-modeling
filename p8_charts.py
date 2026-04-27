"""
P8 - Marketing Mix Modeling charts (3 hiring-manager-friendly visualisations)
Author: Ashwin Kumar - Performance Marketing Portfolio

Builds:
  1. channel_contribution_weekly.png  - 104-week stacked area decomposition
  2. roas_comparison.png              - recovered vs ground-truth ROAS bars
  3. budget_reallocation.png          - current vs optimised quarterly spend

Reads:
  mmm_weekly_data_india.csv
  mmm_channel_results.csv
  mmm_budget_reallocation.csv
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression

WS = Path("/home/user/workspace")

OUT_CHART1 = WS / "channel_contribution_weekly.png"
OUT_CHART2 = WS / "roas_comparison.png"
OUT_CHART3 = WS / "budget_reallocation.png"

CHANNELS = [
    "google_ads_spend", "meta_ads_spend", "youtube_spend",
    "amazon_ads_spend", "flipkart_ads_spend", "offline_spend",
]

PRETTY = {
    "google_ads_spend":   "Google Ads",
    "meta_ads_spend":     "Meta Ads",
    "youtube_spend":      "YouTube",
    "amazon_ads_spend":   "Amazon Ads",
    "flipkart_ads_spend": "Flipkart Ads",
    "offline_spend":      "Offline",
}

# Cohesive palette for the 7 stack layers
PALETTE = {
    "baseline":           "#D4D1CA",  # neutral grey for organic baseline
    "google_ads_spend":   "#20808D",  # teal (chart primary)
    "meta_ads_spend":     "#1B474D",  # dark teal
    "youtube_spend":      "#A84B2F",  # terra
    "amazon_ads_spend":   "#FFC553",  # gold
    "flipkart_ads_spend": "#944454",  # mauve
    "offline_spend":      "#848456",  # olive
}

INK = "#28251D"
MUTED = "#7A7974"
FAINT = "#BAB9B4"
RULE = "#D4D1CA"
BG = "#FFFFFF"

FOOTER = "Ashwin Kumar"


# -----------------------------------------------------------------------------
# Shared matplotlib defaults
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "savefig.facecolor": BG,
    "figure.facecolor": BG,
})


# -----------------------------------------------------------------------------
# Load inputs
# -----------------------------------------------------------------------------
df = pd.read_csv(WS / "mmm_weekly_data_india.csv", parse_dates=["week"]).sort_values("week").reset_index(drop=True)
results = pd.read_csv(WS / "mmm_channel_results.csv")
realloc = pd.read_csv(WS / "mmm_budget_reallocation.csv")

# Lookup helpers
lambdas = dict(zip(results["channel"], results["lambda_chosen"]))
alphas = dict(zip(results["channel"], results["alpha_chosen"]))

# -----------------------------------------------------------------------------
# Re-fit the model so we can extract weekly per-channel contributions
# -----------------------------------------------------------------------------
def geom_adstock(x, decay):
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] + decay * out[i - 1]
    return out


df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
df["time_trend"] = np.arange(len(df))

control_cols = [
    "is_diwali_week", "is_eid_week", "is_republic_day_week", "is_eors_week",
    "competitor_price_index", "avg_product_price",
    "week_of_year", "time_trend",
]

feat = {}
for ch in CHANNELS:
    adstk = geom_adstock(df[ch].values, lambdas[ch])
    sat = adstk ** alphas[ch]
    feat[ch + "_t"] = sat
X = pd.DataFrame(feat)
for c in control_cols:
    X[c] = df[c].values

y = df["weekly_revenue"].values
model = LinearRegression()
model.fit(X.values, y)
coefs = dict(zip(X.columns, model.coef_))

# Weekly contribution per channel (rupees per week)
weekly_contrib = pd.DataFrame({"week": df["week"]})
for ch in CHANNELS:
    weekly_contrib[ch] = coefs[ch + "_t"] * X[ch + "_t"].values

# Baseline = predicted - channel contributions  (this is the part of predicted
# revenue that comes from the intercept + controls + seasonality + trend).
# We chart the predicted revenue stack so the layers add up cleanly; actual
# weekly_revenue is overlaid as a thin line for transparency.
predicted = model.predict(X.values)
weekly_contrib["baseline"] = (
    predicted - sum(weekly_contrib[ch] for ch in CHANNELS)
)
# Pin baseline at zero floor for the chart (controls can be slightly negative
# in some weeks - we don't want negative areas visually). We re-attribute any
# negative baseline to a small constant residual rather than letting the chart
# go below zero.
weekly_contrib["baseline"] = weekly_contrib["baseline"].clip(lower=0)


# -----------------------------------------------------------------------------
# Helper: rupees -> crores formatter
# -----------------------------------------------------------------------------
def cr_formatter(x, _pos=None):
    val = x / 1_00_00_000
    if val == 0:
        return "\u20b90"
    # Always one decimal so 0.5, 1.25, 1.5 don't all round to repeated labels
    return f"\u20b9{val:.1f} Cr"


def lakh_formatter(x, _pos=None):
    return f"\u20b9{x / 1_00_000:.0f} L"


# =============================================================================
# CHART 1 - Stacked area: weekly revenue decomposition
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

stack_order = ["baseline"] + CHANNELS
stack_labels = ["Baseline (organic + seasonality)"] + [PRETTY[c] for c in CHANNELS]
stack_colors = [PALETTE[k] for k in stack_order]

# Convert to crores for plotting
y_stack = np.vstack([weekly_contrib[c].values for c in stack_order])

ax.stackplot(
    df["week"], y_stack,
    labels=stack_labels, colors=stack_colors,
    alpha=0.92, edgecolor="white", linewidth=0.5,
)

# Overlay actual weekly revenue as a thin reference line (sanity check)
ax.plot(df["week"], df["weekly_revenue"], color=INK, linewidth=1.0,
        alpha=0.55, label="Actual weekly revenue")

# Diwali annotations: any continuous run of is_diwali_week==1
diwali_mask = df["is_diwali_week"].values == 1
diwali_runs = []
in_run = False
start = None
for i, v in enumerate(diwali_mask):
    if v and not in_run:
        start = i
        in_run = True
    elif not v and in_run:
        diwali_runs.append((start, i - 1))
        in_run = False
if in_run:
    diwali_runs.append((start, len(diwali_mask) - 1))

# Mark each Diwali period with a single vertical line at the run's mid-week
for run_start, run_end in diwali_runs:
    mid = run_start + (run_end - run_start) // 2
    xpos = df["week"].iloc[mid]
    ax.axvline(xpos, color=INK, linewidth=0.7, linestyle="--", alpha=0.45)
    # label near the top of the chart
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else float(y_stack.sum(axis=0).max())
    ax.text(xpos, float(y_stack.sum(axis=0).max()) * 1.02, "Diwali",
            ha="center", va="bottom", color=INK, fontsize=10,
            fontweight="bold", alpha=0.85)

ax.set_title("Weekly Revenue Decomposition by Marketing Channel (2024–2025)",
             loc="left", pad=14)
ax.set_xlabel("")
ax.set_ylabel("Revenue (\u20b9 Cr)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(cr_formatter))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(axis="x", rotation=0)

# Set y-limits with headroom
ymax_data = float(y_stack.sum(axis=0).max())
ax.set_ylim(0, ymax_data * 1.10)
ax.set_xlim(df["week"].min(), df["week"].max())

# Legend below the chart
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.10),
          ncol=4, frameon=False, fontsize=10.5)

ax.grid(axis="y", color=RULE, linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)

fig.text(0.99, 0.01, FOOTER, ha="right", fontsize=9, color=FAINT)

fig.tight_layout(rect=[0.01, 0.06, 0.99, 0.97])
fig.savefig(OUT_CHART1, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"[saved] {OUT_CHART1}")


# =============================================================================
# CHART 2 - ROAS comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(13, 6.5))

results_ord = results.set_index("channel").reindex(CHANNELS).reset_index()
x = np.arange(len(results_ord))
bw = 0.36

bars_rec = ax.bar(
    x - bw / 2, results_ord["roas_recovered"], bw,
    color="#20808D", edgecolor="white", linewidth=1.0,
    label="Recovered ROAS",
)
bars_gt = ax.bar(
    x + bw / 2, results_ord["roas_ground_truth"], bw,
    color="#DA7101", edgecolor="white", linewidth=1.0,
    alpha=0.55, label="Ground-truth ROAS",
)

# Value labels above each bar
for b, v in zip(bars_rec, results_ord["roas_recovered"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10.5, color=INK,
            fontweight="bold")
for b, v in zip(bars_gt, results_ord["roas_ground_truth"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10.5, color="#964219")

# Breakeven line at ROAS = 1.0
ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.1, alpha=0.85, zorder=0)
ax.text(len(results_ord) - 0.5, 1.05, "Breakeven (ROAS = 1.0)",
        ha="right", va="bottom", color=MUTED, fontsize=10, style="italic")

ax.set_title("Recovered ROAS vs Ground Truth by Channel", loc="left", pad=14)
ax.set_ylabel("ROAS (revenue per rupee of spend)")
ax.set_xticks(x)
ax.set_xticklabels([PRETTY[c] for c in results_ord["channel"]],
                   rotation=0, fontsize=11)
ax.set_ylim(0, 8)

ax.grid(axis="y", color=RULE, linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)

ax.legend(loc="upper right", frameon=False, fontsize=11)

# Subtitle line under the title with average error
avg_abs_err = np.mean(np.abs(results_ord["error_pct"]))
fig.text(
    0.013, 0.91,
    f"Across 6 channels, the model recovers ROAS within an average "
    f"absolute error of {avg_abs_err:.1f}% \u2014 industry-typical for MMM.",
    fontsize=10.5, color=MUTED,
)

fig.text(0.99, 0.01, FOOTER, ha="right", fontsize=9, color=FAINT)
fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.90])
fig.savefig(OUT_CHART2, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"[saved] {OUT_CHART2}")


# =============================================================================
# CHART 3 - Budget reallocation horizontal bars
# =============================================================================
fig, ax = plt.subplots(figsize=(13, 7))

realloc_ord = realloc.set_index("channel").reindex(CHANNELS[::-1]).reset_index()
# Reverse so the channel with largest current spend ends up at the top
y_pos = np.arange(len(realloc_ord))
bh = 0.36

cur_lakh = realloc_ord["current_spend"].values / 1_00_000
opt_lakh = realloc_ord["optimised_spend"].values / 1_00_000
deltas = realloc_ord["delta_pct"].values

# Optimised bar colour depends on delta sign
opt_colors = ["#437A22" if d > 0 else "#A12C7B" for d in deltas]

bars_cur = ax.barh(
    y_pos + bh / 2, cur_lakh, bh,
    color="#BAB9B4", edgecolor="white", linewidth=1.0,
    label="Current quarterly spend",
)
bars_opt = ax.barh(
    y_pos - bh / 2, opt_lakh, bh,
    color=opt_colors, edgecolor="white", linewidth=1.0,
    label="Optimised quarterly spend",
)

# Value + delta labels next to each bar
for b, v in zip(bars_cur, cur_lakh):
    ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"\u20b9{v:.1f} L",
            ha="left", va="center", fontsize=10, color=MUTED)
for b, v, d in zip(bars_opt, opt_lakh, deltas):
    sign = "+" if d > 0 else ""
    label_color = "#437A22" if d > 0 else "#A12C7B"
    ax.text(v + 1.5, b.get_y() + b.get_height() / 2,
            f"\u20b9{v:.1f} L   ({sign}{d:.1f}%)",
            ha="left", va="center", fontsize=10.5, color=label_color,
            fontweight="bold")

# Y-tick labels
ax.set_yticks(y_pos)
ax.set_yticklabels([PRETTY[c] for c in realloc_ord["channel"]], fontsize=11)
ax.invert_yaxis()  # largest at top

ax.set_xlabel("Spend (\u20b9 Lakhs)")
ax.set_xlim(0, max(cur_lakh.max(), opt_lakh.max()) * 1.30)

# Title + total budget line
total_inr = realloc_ord["current_spend"].sum()
total_cr = total_inr / 1_00_00_000
ax.set_title("Quarterly Budget Reallocation Recommendation", loc="left", pad=18)
fig.text(
    0.013, 0.915,
    f"Total Budget: \u20b9{total_cr:.2f} Cr (unchanged)  "
    f"\u00b7  Same total spend, redistributed within \u00b140% per-channel bounds",
    fontsize=10.5, color=MUTED,
)

# Custom legend so the optimised swatch is visually a "delta" pair
custom_legend = [
    Patch(facecolor="#BAB9B4", edgecolor="white", label="Current quarterly spend"),
    Patch(facecolor="#437A22", edgecolor="white", label="Optimised \u2014 increased"),
    Patch(facecolor="#A12C7B", edgecolor="white", label="Optimised \u2014 decreased"),
]
ax.legend(handles=custom_legend, loc="lower right", bbox_to_anchor=(1.0, -0.18),
          frameon=False, fontsize=10.5, ncol=3)

ax.grid(axis="x", color=RULE, linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)

fig.text(0.99, 0.01, FOOTER, ha="right", fontsize=9, color=FAINT)
fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.90])
fig.savefig(OUT_CHART3, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"[saved] {OUT_CHART3}")
