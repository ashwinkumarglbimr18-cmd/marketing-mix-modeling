# Marketing Mix Modeling — Quantifying Channel ROAS and Recommending a Quarterly Budget Reallocation

## Overview

This is the capstone project of my eight-project performance marketing portfolio. I built a Marketing Mix Model (MMM) on 104 weeks of India-market e-commerce data, recovered the true ROAS of six paid channels, and produced a quarterly budget reallocation that holds total spend fixed while raising predicted revenue.

MMM is the only analytical framework that answers a board-level budget question: *"if I shift ₹X from channel A to channel B, what happens to revenue?"* Last-click attribution credits the final touchpoint, multi-touch attribution distributes credit across the user journey, and incrementality experiments measure lift on a single channel — but none of them tell you how to redistribute a fixed media budget across the entire mix. MMM does.

This project closes the analytical loop on the rest of my portfolio:

- **P3 — Multi-Touch Attribution**: credits touchpoints within a customer journey
- **P4 — A/B Testing & ML Predictions**: measures lift and predicts conversion
- **P5 — Funnel Analytics**: tracks where users drop off
- **P6 — Cohort Retention**: measures the *quality* of acquired users over time
- **P7 — RFM Segmentation**: groups customers by recency, frequency, monetary value
- **P8 — Marketing Mix Modeling**: reallocates the media budget itself

## Dataset

- **104 weeks** of weekly data, January 2024 – December 2025
- **Geography**: India market, INR currency
- **6 paid channels**: Google Ads, Meta Ads, YouTube, Amazon Ads, Flipkart Ads, Offline
- **Festival controls**: Diwali, Eid, Republic Day, End-of-Reason-Sale (EORS)
- **Macro controls**: competitor price index, average product price
- **Trend & seasonality**: weekly trend term, sin/cos seasonality terms

### Why synthetic data

Real MMM datasets are proprietary — brands almost never release weekly spend × revenue data with ground-truth channel contributions, because doing so would expose competitive strategy. Meta's open-source Robyn library ships its own simulated dataset for the same reason. I modelled my dataset on Robyn's structure and embedded **known ground-truth ROAS values** into the data generation, so the model's recovery accuracy can be measured directly. This is the only honest way to validate an MMM pipeline outside of a production deployment.

## Methodology

| Step | What I did |
|---|---|
| 1. Adstock | Applied geometric decay per channel, grid-searched λ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7} |
| 2. Saturation | Applied Hill-style power saturation per channel, grid-searched α ∈ {0.5, 0.6, 0.7, 0.8} |
| 3. Regression | OLS: `weekly_revenue ~ adstocked-saturated channel spends + festival controls + competitor price + product price + trend + seasonality` |
| 4. Contribution | Per-channel contribution = coefficient × adstocked-saturated spend, summed across all 104 weeks |
| 5. ROAS | Recovered ROAS = total channel contribution ÷ total raw channel spend |
| 6. Reallocation | Same total quarterly spend, redistributed within ±40% per-channel bounds toward channels with highest marginal ROAS at current saturation |

## Key Findings

- **R² = 0.9920 (Adjusted R² = 0.9908)** across 104 weeks — the model explains 99.2% of weekly revenue variance
- **Channel ROAS recovered within 18.0% average absolute error** of ground truth — industry-typical for MMM, in line with published Robyn benchmarks
- **Rank-order recovery is correct**: Amazon > Flipkart > Meta > Google > YouTube > Offline matches the ground-truth ordering
- **Diwali multiplier of ~1.8×** is clearly visible in the weekly decomposition — confirms that festival budget front-loading is working
- **Reallocation recommendation**: shift **~₹28 lakh/quarter** out of Google Ads, YouTube, and Offline into Meta, Amazon, and Flipkart, holding the **₹3.13 Cr** total budget constant. Predicted uplift: **+₹56 lakh/quarter (+3.1%)**.

## Strategic Recommendations

- **Meta Ads (+39%)**: recovered ROAS 3.62 — the most under-invested channel in the mix. Currently at ₹72L/quarter while Google Ads consumes ₹136L for a lower ROAS of 3.15.
- **Amazon Ads (+33%) and Flipkart Ads (+29.5%)**: the highest recovered ROAS in the mix (7.34 and 5.05) but the lowest spend share. Marketplace ads are the biggest missed opportunity in the current plan.
- **YouTube (−39%) and Offline (−22%)**: recovered ROAS of 2.43 and 1.10 respectively. Both below the digital median. I recommend reducing — not zeroing out — since both contribute upper-funnel reach that this single-equation MMM cannot fully capture.
- **Google Ads (−18%)**: still the largest channel after reallocation, but clearly at saturation. Marginal returns favour shifting incremental rupees elsewhere.

## Files

| File | What it is |
|---|---|
| `README.md` | This file |
| `mmm_report.pdf` | 2-page hiring-manager summary |
| `mmm_weekly_data_india.csv` | 104-week synthetic dataset |
| `mmm_channel_results.csv` | Per-channel contribution, recovered ROAS, ground-truth ROAS, error % |
| `mmm_budget_reallocation.csv` | Current vs optimised quarterly spend per channel |
| `mmm_model_diagnostics.txt` | R², adjusted R², chosen λ and α per channel |
| `p8_methodology.txt` | Plain-text methodology and limitations |
| `p8_data_generation_notes.txt` | How the synthetic dataset was built (ground-truth ROAS etc.) |
| `channel_contribution_weekly.png` | 104-week stacked-area decomposition |
| `roas_comparison.png` | Recovered vs ground-truth ROAS |
| `budget_reallocation.png` | Current vs optimised quarterly spend |
| `p8_mmm_modeling.py` | Reproducible modeling script |
| `p8_charts.py` | Reproducible charting script |

## Tools

Python, pandas, numpy, scikit-learn, statsmodels, matplotlib, reportlab

## How to Reproduce

1. Run `python p8_mmm_modeling.py` — generates the dataset, fits the model, saves all CSVs and the diagnostics file
2. Run `python p8_charts.py` — generates the three chart PNGs
3. Open `mmm_report.pdf` for the 2-page summary

## Limitations

- **Synthetic dataset**: a real deployment should use 2+ years of actual weekly spend and revenue data. Synthetic data is appropriate for showcasing methodology but cannot be used to make production budget decisions.
- **Classical OLS** was chosen for interpretability and rupee-level coefficient readability. A Bayesian MMM (PyMC, Robyn) would add posterior uncertainty intervals and is the correct tool for production use.
- **MMM is correlational, not causal**. A geo-holdout test or a conversion-lift experiment on the recommended reallocation should run before any full rollout — even when the model fits well, structural correlations between spend and demand can mislead.

## Contact

- **Email**: [ashwinkumarglbimr18@gmail.com](mailto:ashwinkumarglbimr18@gmail.com)
- **LinkedIn**: [https://www.linkedin.com/in/ashwin-kumar-180816174/](https://www.linkedin.com/in/ashwin-kumar-180816174/)


