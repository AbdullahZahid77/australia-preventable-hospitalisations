# Indigenous Australian Preventable Hospitalisations

### Statistical Analysis of Geographic, Socioeconomic & Clinical Drivers · 2012–2018

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-OLS-4B8BBE?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A data-driven investigation into why preventable hospitalisation rates are **up to 29× higher** in remote Australia than in major cities — and which conditions drive the greatest burden for Indigenous Australians.

**[Launch Interactive Dashboard](https://australia-preventable-hospitalisations-abdullah-zahid.streamlit.app/)** — live ML predictions, geographic analysis, and condition explorer, no install required.

---

## Overview

Potentially Preventable Hospitalisations (PPH) are admissions for conditions where timely, appropriate primary care could have prevented or reduced the need for hospital treatment. For Indigenous Australians, the gap between remote and metropolitan PPH rates is one of the most pronounced health inequities in the developed world.

This project applies statistical and machine learning methods to AIHW national datasets across **31 Primary Health Networks** and **32 clinical conditions** over a six-year period (2012–2018).

**Three research questions drive the analysis:**

1. What geographic and socioeconomic factors explain variation in PPH rates across Australia's 31 PHNs?
2. Which conditions drive the greatest hospitalisation burden, and has this changed across 2012–2018?
3. Can we accurately predict average length of stay for PPH admissions based on condition, category, and patient characteristics?

---

## Results at a Glance

### Model 1 — OLS Regression (PHN-Level Access Model)

| Metric                 | Value                                   |
| ---------------------- | --------------------------------------- |
| R²                     | **0.739**                               |
| Adjusted R²            | 0.709                                   |
| F-statistic            | 25.42 (p < 0.001)                       |
| Remoteness coefficient | **+2.82** per 100 residents (p = 0.011) |
| SEIFA coefficient      | 0.49 (p = 0.507, not significant)       |

> Each step further from a major city adds approximately **2.82 preventable hospitalisations per 100 residents** — holding socioeconomic status and health check coverage constant.

### Model 2 — Random Forest Regressor (Condition-Level LOS Prediction)

| Model             |   Test R² |      Test MAE |     Test RMSE |
| ----------------- | --------: | ------------: | ------------: |
| Linear Regression |     0.108 |     1.64 days |     2.21 days |
| Ridge Regression  |     0.108 |     1.65 days |     2.21 days |
| **Random Forest** | **0.952** | **0.34 days** | **0.51 days** |
| Gradient Boosting |     0.940 |     0.39 days |     0.57 days |

> The Random Forest model predicts average length of stay to within **8 hours** on the held-out test set. PPH condition alone accounts for ~72% of predictive power — the specific diagnosis is by far the strongest driver of resource consumption.

### Key Findings

- **Geography dominates:** Remoteness explains more variance in PPH rates than socioeconomic disadvantage once both are controlled for
- **Chronic burden is disproportionate:** Chronic conditions account for 42% of admissions but 56% of total bed days
- **Severity diverges from volume:** Gangrene (~15 days avg LOS) and nutritional deficiencies (~14 days) are the most resource-intensive conditions despite low admission counts
- **No improvement over time:** Total PPH bed day burden remained broadly stable across the full 2012–2018 period — no condition showed a sustained downward trend
- **Linear models fail:** Random Forest (R² = 0.952) vastly outperforms linear regression (R² = 0.108) — the LOS–condition relationship is strongly non-linear

---

## Repository Structure

```
australia-preventable-hospitalisations/
│
├── australia_pph_analysis.ipynb        # Single consolidated analysis notebook
├── app.py                              # Streamlit interactive dashboard
├── requirements.txt                    # Python dependencies
├── .streamlit/
│   └── config.toml                     # Dashboard theme
│
├── pre-processed-datasets/
│   ├── A1.csv                          # PHN-level PPH rates and health checks
│   ├── remoteness.csv                  # PPH rates by ASGS remoteness category
│   └── socio-economic-area.csv         # PPH rates by SEIFA quintile
│
├── processed-datasets/
│   ├── master_data_frame.csv           # PHN-level merged dataset (31 rows)
│   └── indigenous_pre_data_frame.csv   # Cleaned Indigenous PPH dataset (576 rows)
│
└── README.md
```

---

## Dataset

| File                            | Source               | Description                                                     |  Rows |
| ------------------------------- | -------------------- | --------------------------------------------------------------- | ----: |
| `A1.csv`                        | AIHW HPF-51 Table A1 | PHN-level PPH rates, Indigenous health checks, follow-up visits |    31 |
| `remoteness.csv`                | AIHW HPF-50          | PPH rates by ASGS remoteness area, condition, sex, year         | 2,880 |
| `socio-economic-area.csv`       | AIHW HPF-50          | PPH rates by SEIFA socioeconomic quintile                       |   960 |
| `indigenous_pre_data_frame.csv` | AIHW (processed)     | Indigenous PPH by condition × sex × year · 2012–2018            |   576 |

Source: [Australian Institute of Health and Welfare — Potentially Preventable Hospitalisations (HPF 50–51)](https://www.aihw.gov.au/reports/primary-health-care/potentially-preventable-hospitalisations)

---

## Methods

### Data Processing

- PHN-level master dataset constructed by merging three sources: AIHW A1 table, remoteness summary (aggregated from condition-level data), and SEIFA quintile mapping
- Indigenous dataset cleaned to remove AIHW roll-up rows and separate sex-aggregate rows from individual records to prevent triple-counting
- AIHW privacy suppression (76 nulls in clinical metrics) handled via `.dropna()` at analysis time — no imputation applied to suppressed cells

### Exploratory Analysis

- 11 publication-quality visualisations covering distribution, socioeconomic gradient, remoteness boxplots, top conditions by volume and severity, category burden split, sex comparison, and temporal trends
- Risk score composite index constructed to rank PHNs by combined geographic and socioeconomic pressure

### Model 1 — OLS Multiple Regression

- Predictors ordinal-encoded: remoteness rank (1–5), SEIFA rank (1–5), health checks per 1,000 Indigenous residents
- Multicollinearity assessed via VIF (remoteness VIF = 13.1, health checks VIF = 17.7) — acknowledged as a structural feature of MBS Item 715 policy targeting, not removed
- Residual diagnostics: homoscedasticity and Q-Q plots

### Model 2 — Ensemble Models

- Dataset filtered to 264 rows (sex-disaggregated, non-aggregate conditions with observed LOS)
- Train / validation / test split: 68% / 12% / 20% (stratified by random seed)
- Four models benchmarked; Random Forest selected on validation R²
- Feature importance via mean decrease in impurity; learning curve via 5-fold cross-validation

---

## Setup

### Interactive Dashboard (no install)

The easiest way to explore the project is the live app:

**[https://australia-preventable-hospitalisations-abdullah-zahid.streamlit.app/](https://australia-preventable-hospitalisations-abdullah-zahid.streamlit.app/)**

### Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/indigenous-pph-australia.git
cd indigenous-pph-australia

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py

# Or open the analysis notebook
jupyter notebook australia_pph_analysis.ipynb
```

The notebook and dashboard both use relative paths — no configuration required after cloning.

---

## Tech Stack

| Category              | Tools                                                |
| --------------------- | ---------------------------------------------------- |
| Data manipulation     | pandas, NumPy                                        |
| Visualisation         | Matplotlib, Seaborn, Plotly                          |
| Statistical modelling | statsmodels (OLS, VIF)                               |
| Machine learning      | scikit-learn (RandomForest, GradientBoosting, Ridge) |
| Dashboard             | Streamlit (deployed on Streamlit Community Cloud)    |
| Environment           | Python 3.10, Jupyter Notebook                        |

---

## Limitations

| Limitation                                 | Impact on Results                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| n = 31 PHNs — small regression sample      | Wide confidence intervals; NT as sole Very Remote observation is high-leverage |
| AIHW privacy suppression (76 nulls in LOS) | Model trained on 264 of 576 potential rows                                     |
| PHN → remoteness/SEIFA is a manual mapping | Classification uncertainty for ambiguous regional PHNs                         |
| Cross-sectional PHN structure              | Cannot track individual PHN performance change over time                       |

---

## Author

**Abdullah Zahid**

Data Scientist · University of Technology Sydney

[LinkedIn](https://www.linkedin.com/in/abdullahzahid77/) · [GitHub](https://github.com/AbdullahZahid77)

---

_Data sourced from the Australian Institute of Health and Welfare (AIHW), publicly available at [aihw.gov.au](https://www.aihw.gov.au). Research conducted as part of the Statistical Thinking for Data Science subject at UTS._
