# ASENXO MSME Project Completion Risk Management Dashboard

**Objective:** a centralized risk management dashboard that indicates MSMEs
at risk of project incompletion, giving administrators evidence-based
results \u2014 powered by Logistic Regression \u2014 to support decision-making.

Workflow: **Monitor \u2192 Predict \u2192 Explain Risk \u2192 Prioritize \u2192 Intervene \u2192 Monitor**

## Files

- `app.py` \u2014 the Streamlit app
- `requirements.txt` \u2014 Python dependencies
- `MSME_filled.xlsx` \u2014 one row per project (must stay next to `app.py`)
- `logistic_regression_model.pkl` \u2014 the fitted model (must stay next to `app.py`)

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy to Streamlit Community Cloud

1. Push this folder (including the `.xlsx` and `.pkl` files) to a **public
   GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and point
   it at the repo with `app.py` as the main file.
3. Streamlit Cloud installs `requirements.txt` automatically \u2014 no other
   configuration is needed.

## About this version of the data & model

This dashboard was rebuilt around a new dataset and a new model, and both are
structurally different from the earlier panel-data version:

- **One row per project** (321 rows), not a monthly panel. There is no
  `Project_ID` column in the source file, so the app builds a surrogate key
  from `Beneficiary_Name + Year` (verified unique in this file).
- **11 model predictors**, not 3: 7 numeric (Debt-service coverage,
  Penetration Rate, net_profit_margin, Current capacity utilization rate,
  Total Employment, Total Assets, Years in operation) and 4 categorical,
  one-hot encoded (Province, Sector, type_of_ownership, size_of_enterprise).
  Province and Sector are now genuine model inputs, not just filters.
- **No time dimension per project.** Because there's no monthly history,
  month-over-month trend charts and "declining probability" / "persistent
  risk" flags (which need a time series) aren't possible here. They were
  replaced with dataset-appropriate alternatives:
  - *Project Risk Profile* shows a feature-contribution chart (each
    predictor's push toward "Completed" vs "Not Completed") instead of a
    time trend.
  - *Early Warning* flags current High/Critical risk, standing well below
    the project's sector average, and a beneficiary having another project
    on record marked "Not Completed" (a real, disclosed historical signal,
    not a model input).
- **Data quality note:** 5 of 321 rows contain a Province/Sector/ownership
  value the model wasn't trained on (e.g. spelling variants such as
  "Horticulture and Agriculture" vs. "Horticulture & Agriculture", or a
  seemingly misentered ownership value). The model's OneHotEncoder treats
  these as "no signal" for that field rather than erroring or guessing \u2014
  the app surfaces exactly which rows and fields are affected in an
  expander at the top of the page, so nothing is silently degraded.
- `Project_Cost` is not in this dataset; the at-risk projects table shows
  `Total Assets` in its place, labeled accordingly.

## Notes

- The dataset is synthetic and intended for demonstration only (shown via a
  banner in the app itself).
- Risk tiers (Critical / High / Medium / Low) and their percentage
  boundaries are configurable in the sidebar and are explicitly labeled a
  Proposed / System Risk Classification, not an official DOST standard.
- "Why flagged" text and the contribution chart are generated directly from
  the fitted logistic regression's own coefficients (transformed feature
  value \u00d7 coefficient) \u2014 no SHAP or extra dependencies needed for a
  linear model.
- The Intervention Monitoring log is editable during a session
  (`st.session_state`) and can be exported as CSV; it is not written back to
  any database.
