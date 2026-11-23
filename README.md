# Lidl Candy Case 

What drives customer sentiment for candy, and should a new store-brand product
be **cookie-based (chocolate) or gummy (fruity)?**

This repo answers both using the [FiveThirtyEight candy-power-ranking dataset](https://github.com/fivethirtyeight/data/tree/master/candy-power-ranking)
(85 candies). The business target is `winpercent` — the share of head-to-head
taste matchups a candy wins (0–100).

## The one-line answer

**Launch a chocolate-based bar (cookie/wafer + nut), not a fruity gummy.**
Chocolate is by far the strongest sentiment driver; fruity and hard candies
score lowest.

## Hard constraints honoured

- **Python only.**
- **Modelling uses ONLY numpy + scipy.** The Random Forest (decision trees +
  bootstrap bagging + random feature subsampling + permutation importance +
  OOB R²) is implemented from scratch in `src/random_forest.py`, and an
  **OLS linear regression** (closed-form β, with hand-computed standard errors,
  t-statistics and p-values via `scipy.stats.t`) in `src/ols.py`. No
  scikit-learn / statsmodels / xgboost anywhere.
- **Random Forest with hyperparameter tuning** via grid search + from-scratch
  K-fold cross-validation, plus an interpretable OLS baseline on the same split.
- pandas / matplotlib / seaborn are used only for data loading and plots.
- **Reproducible:** every seed is fixed at 42; deps pinned in
  `requirements.txt`; one command runs everything.

## Reproduce

```bash
# from the repo root
pip install -r requirements.txt
python3 src/analysis.py
```

That single command loads the data, runs EDA + figures, tunes the Random Forest
with 5-fold CV, fits the best model, evaluates on a held-out test set against a
mean-predictor baseline, computes permutation importance, **fits the OLS
baseline on the same seed-42 split and writes `results/ols_coefficients.csv`
(feature, coef, std_err, t, p_value, sig)**, and writes all other outputs. It
takes a few minutes (the forest and CV are pure numpy).

## Layout

```
data/        candy-data.csv (input, 85 candies)
src/
  random_forest.py   from-scratch RandomForestRegressor + DecisionTreeRegressor
  ols.py             from-scratch OLSRegression (coefs, SE, t, p, R2, adj R2)
  analysis.py        end-to-end pipeline (EDA, tuning, RF + OLS eval, recommendation)
figures/     all generated plots (.png), incl. ols_coefficients, rf_vs_ols_comparison
results/     metrics.json, feature_importance.csv, cv_results.csv, ols_coefficients.csv
```

## Findings summary

| Metric | Value |
|---|---|
| Test R² (Random Forest) | **0.57** |
| Test RMSE (Random Forest) | 10.39 win-points |
| Test R² (OLS) | 0.56 |
| Test RMSE (OLS) | 10.52 win-points |
| OLS adjusted R² (train) | 0.41 |
| Mean-baseline RMSE | 15.98 |
| OOB R² | 0.33 |
| Best CV R² | 0.27 |

RF and OLS land in a near dead heat on held-out accuracy. OLS gives signed,
testable per-trait effects: only `chocolate` (+22.8, p<0.001) and `fruity`
(+10.6, p<0.05) are significant; both models put chocolate first by a wide
margin.

**Top features by permutation importance:** `chocolate` (0.33), `pricepercent`
(0.09), `peanutyalmondy` (0.07), `bar` (0.05), `crispedricewafer` (0.03).

**Win-rate lift (present − absent):** chocolate +18.8, crispedricewafer +17.3,
peanutyalmondy +16.0, bar +14.6 … fruity −11.2, hard −11.9.

The chocolate/cookie/bar profile averages **62.8%** win rate vs **43.8%** for
the fruity/gummy profile.


## Notes

- Only 85 rows — treat exact numbers as directional, not precise. The analysis
  uses OOB and CV honestly and the write-up states the limits.
- `winpercent` is survey taste-preference, not sales; pair with a margin model
  before a real launch.
