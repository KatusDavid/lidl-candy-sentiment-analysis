"""End-to-end Lidl candy case pipeline.

Loads the FiveThirtyEight candy data, runs EDA and figures, tunes a
from-scratch Random Forest with K-fold cross-validation, fits the best model,
reports test metrics against a mean-predictor baseline, computes permutation
feature importance, and derives a concrete product recommendation.

Modelling uses ONLY numpy/scipy (via src/random_forest.py). pandas, matplotlib
and seaborn are used for data loading and plotting only.

Run:
    python3 src/analysis.py
from the project root. All random seeds are fixed at 42.
"""

from __future__ import annotations

import json
import os
import itertools

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns

from random_forest import RandomForestRegressor, r2_score, rmse
from ols import OLSRegression

SEED = 42

# Project paths (resolved relative to this file so cwd does not matter).
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(ROOT, "data", "candy-data.csv")
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")

BINARY_FEATURES = [
    "chocolate",
    "fruity",
    "caramel",
    "peanutyalmondy",
    "nougat",
    "crispedricewafer",
    "hard",
    "bar",
    "pluribus",
]
CONT_FEATURES = ["sugarpercent", "pricepercent"]
FEATURES = BINARY_FEATURES + CONT_FEATURES
TARGET = "winpercent"


# ----------------------------------------------------------------------------
# Data + EDA
# ----------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def eda_figures(df):
    """Produce the required EDA figures and return a small summary dict."""
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. winpercent distribution
    plt.figure(figsize=(7, 4.5))
    sns.histplot(df[TARGET], bins=15, kde=True, color="#4C72B0")
    plt.axvline(df[TARGET].mean(), color="red", linestyle="--",
                label=f"mean = {df[TARGET].mean():.1f}")
    plt.xlabel("winpercent (head-to-head win rate, %)")
    plt.ylabel("count")
    plt.title("Distribution of winpercent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "winpercent_distribution.png"), dpi=130)
    plt.close()

    # 2. mean winpercent by binary feature (present vs absent)
    means_present = {f: df.loc[df[f] == 1, TARGET].mean() for f in BINARY_FEATURES}
    means_absent = {f: df.loc[df[f] == 0, TARGET].mean() for f in BINARY_FEATURES}
    order = sorted(BINARY_FEATURES, key=lambda f: means_present[f], reverse=True)
    x = np.arange(len(order))
    w = 0.4
    plt.figure(figsize=(9, 5))
    plt.bar(x - w / 2, [means_present[f] for f in order], width=w,
            label="feature present", color="#55A868")
    plt.bar(x + w / 2, [means_absent[f] for f in order], width=w,
            label="feature absent", color="#C44E52")
    plt.xticks(x, order, rotation=45, ha="right")
    plt.ylabel("mean winpercent")
    plt.title("Mean winpercent by binary feature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_vs_winpercent.png"), dpi=130)
    plt.close()

    # 3. correlation of every feature with winpercent
    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    plt.figure(figsize=(8, 5))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in corr.values]
    plt.barh(corr.index, corr.values, color=colors)
    plt.xlabel("Pearson correlation with winpercent")
    plt.title("Feature correlation with winpercent")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_with_winpercent.png"), dpi=130)
    plt.close()

    summary = {
        "n_rows": int(len(df)),
        "winpercent_mean": float(df[TARGET].mean()),
        "winpercent_std": float(df[TARGET].std()),
        "winpercent_min": float(df[TARGET].min()),
        "winpercent_max": float(df[TARGET].max()),
        "mean_winpercent_present": {f: float(means_present[f]) for f in BINARY_FEATURES},
        "mean_winpercent_absent": {f: float(means_absent[f]) for f in BINARY_FEATURES},
        "correlation_with_winpercent": {k: float(v) for k, v in corr.items()},
    }
    return summary


# ----------------------------------------------------------------------------
# Split + cross-validation (implemented from scratch with numpy)
# ----------------------------------------------------------------------------
def train_test_split(X, y, test_frac=0.2, seed=SEED):
    """Seeded random split into train/test arrays."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


def kfold_indices(n, k, seed=SEED):
    """Yield (train_idx, val_idx) for K folds over ``n`` samples."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


def cross_val_r2(params, X, y, k=5, seed=SEED):
    """Mean K-fold validation R^2 and RMSE for one hyperparameter set."""
    r2s, rmses = [], []
    for train_idx, val_idx in kfold_indices(X.shape[0], k, seed=seed):
        model = RandomForestRegressor(random_state=seed, **params)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        r2s.append(r2_score(y[val_idx], pred))
        rmses.append(rmse(y[val_idx], pred))
    return float(np.mean(r2s)), float(np.mean(rmses)), float(np.std(r2s))


def grid_search(X, y, grid, k=5, seed=SEED):
    """Exhaustive grid search with K-fold CV. Returns results list + best."""
    keys = list(grid.keys())
    results = []
    best = None
    for combo in itertools.product(*(grid[k_] for k_ in keys)):
        params = dict(zip(keys, combo))
        mean_r2, mean_rmse, std_r2 = cross_val_r2(params, X, y, k=k, seed=seed)
        row = dict(params)
        row.update({"cv_r2_mean": mean_r2, "cv_rmse_mean": mean_rmse,
                    "cv_r2_std": std_r2})
        results.append(row)
        if best is None or mean_r2 > best["cv_r2_mean"]:
            best = row
    return results, best


# ----------------------------------------------------------------------------
# Figures that need the fitted model
# ----------------------------------------------------------------------------
def plot_permutation_importance(names, imp_mean, imp_std):
    order = np.argsort(imp_mean)
    plt.figure(figsize=(8, 5.5))
    plt.barh(np.array(names)[order], imp_mean[order],
             xerr=imp_std[order], color="#4C72B0", ecolor="gray")
    plt.xlabel("Permutation importance (R^2 drop when shuffled)")
    plt.title("Permutation feature importance (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "permutation_importance.png"), dpi=130)
    plt.close()


def plot_pred_vs_actual(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, color="#55A868", s=50, edgecolor="k", alpha=0.8)
    lo = min(y_true.min(), y_pred.min()) - 2
    hi = max(y_true.max(), y_pred.max()) + 2
    plt.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
    plt.xlabel("actual winpercent")
    plt.ylabel("predicted winpercent")
    plt.title("Predicted vs actual (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "predicted_vs_actual.png"), dpi=130)
    plt.close()


def plot_cv_vs_hyperparam(cv_results):
    """CV R^2 vs n_estimators, one line per max_depth (other params at best)."""
    df = pd.DataFrame(cv_results)
    plt.figure(figsize=(8, 5))
    for depth in sorted(df["max_depth"].unique(), key=lambda d: (d is None, d)):
        sub = df[df["max_depth"] == depth]
        # Average over the other varying params for a clean trend line.
        agg = sub.groupby("n_estimators")["cv_r2_mean"].mean().reset_index()
        label = f"max_depth={depth}"
        plt.plot(agg["n_estimators"], agg["cv_r2_mean"], marker="o", label=label)
    plt.xlabel("n_estimators")
    plt.ylabel("mean CV R^2")
    plt.title("Cross-validation R^2 vs hyperparameters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cv_score_vs_hyperparam.png"), dpi=130)
    plt.close()


def plot_ols_coefficients(summary_rows):
    """Coefficient plot with +/-1.96*SE error bars; intercept excluded."""
    rows = [r for r in summary_rows if r["feature"] != "intercept"]
    rows = sorted(rows, key=lambda r: r["coef"])
    names = [r["feature"] for r in rows]
    coefs = np.array([r["coef"] for r in rows])
    errs = np.array([1.96 * r["std_err"] for r in rows])  # ~95% CI
    sig = [r["p_value"] < 0.05 for r in rows]
    colors = ["#4C72B0" if s else "#B0B0B0" for s in sig]
    plt.figure(figsize=(8, 5.5))
    plt.errorbar(coefs, np.arange(len(names)), xerr=errs, fmt="none",
                 ecolor="gray", capsize=3, zorder=1)
    plt.scatter(coefs, np.arange(len(names)), c=colors, s=55, zorder=2,
                edgecolor="k")
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.yticks(np.arange(len(names)), names)
    plt.xlabel("OLS coefficient (change in winpercent; 95% CI)")
    plt.title("OLS coefficients (blue = significant at p<0.05)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ols_coefficients.png"), dpi=130)
    plt.close()


def plot_rf_vs_ols(rf_r2, rf_rmse, ols_r2, ols_rmse, base_rmse):
    """Side-by-side RF vs OLS (vs baseline) on test R^2 and RMSE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.bar(["Random Forest", "OLS"], [rf_r2, ols_r2],
            color=["#4C72B0", "#55A868"])
    ax1.set_ylabel("test R^2")
    ax1.set_title("Test R^2: RF vs OLS")
    for i, v in enumerate([rf_r2, ols_r2]):
        ax1.text(i, v, f"{v:.3f}", ha="center",
                 va="bottom" if v >= 0 else "top")

    ax2.bar(["Random Forest", "OLS", "Baseline"],
            [rf_rmse, ols_rmse, base_rmse],
            color=["#4C72B0", "#55A868", "#C44E52"])
    ax2.set_ylabel("test RMSE (win-points)")
    ax2.set_title("Test RMSE: RF vs OLS vs baseline")
    for i, v in enumerate([rf_rmse, ols_rmse, base_rmse]):
        ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "rf_vs_ols_comparison.png"), dpi=130)
    plt.close()


# ----------------------------------------------------------------------------
# Recommendation
# ----------------------------------------------------------------------------
def build_recommendation(df, imp_df, eda_summary):
    """Derive a concrete product recommendation from importances + data."""
    top = imp_df.sort_values("importance_mean", ascending=False)
    top_feats = top.head(3)["feature"].tolist()

    present = eda_summary["mean_winpercent_present"]
    absent = eda_summary["mean_winpercent_absent"]

    def lift(f):
        return present[f] - absent[f]

    # Which binary features give the biggest positive win-rate lift.
    lifts = {f: lift(f) for f in BINARY_FEATURES}
    choc_lift = lifts["chocolate"]
    fruity_lift = lifts["fruity"]
    bar_lift = lifts["bar"]

    # Cookie-based (chocolate + crispedricewafer/bar) vs gummy (fruity + pluribus).
    cookie_profile = ["chocolate", "crispedricewafer", "bar"]
    gummy_profile = ["fruity", "pluribus", "hard"]
    cookie_score = float(np.mean([present[f] for f in cookie_profile]))
    gummy_score = float(np.mean([present[f] for f in gummy_profile]))

    headline = (
        "Launch a chocolate-based bar (cookie/wafer style), not a fruity gummy."
        if choc_lift > fruity_lift
        else "Launch a fruity gummy, not a chocolate bar."
    )

    rec = {
        "headline": headline,
        "top_features_by_permutation_importance": top_feats,
        "winrate_lift": {f: float(lifts[f]) for f in BINARY_FEATURES},
        "chocolate_lift": float(choc_lift),
        "fruity_lift": float(fruity_lift),
        "bar_lift": float(bar_lift),
        "cookie_profile_mean_winpercent": cookie_score,
        "gummy_profile_mean_winpercent": gummy_score,
    }
    return rec


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} candies, {len(FEATURES)} features.")

    # --- EDA ---
    eda_summary = eda_figures(df)

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    # --- seeded train/test split ---
    X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split(X, y, 0.2, SEED)
    print(f"Train {X_tr.shape[0]} / Test {X_te.shape[0]}")

    # --- hyperparameter tuning: grid search + 5-fold CV ---
    # Grid kept small and shallow on purpose: only 85 rows total.
    grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "min_samples_leaf": [2, 3],
        "max_features": ["third", "sqrt"],
    }
    cv_results, best = grid_search(X_tr, y_tr, grid, k=5, seed=SEED)
    best_params = {k: best[k] for k in grid.keys()}
    print(f"Best params: {best_params}  (CV R^2={best['cv_r2_mean']:.3f})")

    cv_df = pd.DataFrame(cv_results).sort_values("cv_r2_mean", ascending=False)
    cv_df.to_csv(os.path.join(RES_DIR, "cv_results.csv"), index=False)

    # --- fit best model on full training set ---
    model = RandomForestRegressor(random_state=SEED, **best_params)
    model.fit(X_tr, y_tr)

    # --- test metrics + baseline (mean predictor) ---
    y_pred = model.predict(X_te)
    test_r2 = r2_score(y_te, y_pred)
    test_rmse = rmse(y_te, y_pred)

    baseline_pred = np.full_like(y_te, fill_value=float(np.mean(y_tr)))
    baseline_rmse = rmse(y_te, baseline_pred)
    baseline_r2 = r2_score(y_te, baseline_pred)  # ~0 or negative by construction

    print(f"Test  R^2={test_r2:.3f}  RMSE={test_rmse:.3f}")
    print(f"Base  R^2={baseline_r2:.3f}  RMSE={baseline_rmse:.3f}")
    print(f"OOB   R^2={model.oob_r2_:.3f}")

    # --- permutation importance (on the test set) ---
    imp_mean, imp_std = model.permutation_importance(
        X_te, y_te, n_repeats=30, random_state=SEED
    )
    imp_df = pd.DataFrame(
        {"feature": FEATURES, "importance_mean": imp_mean, "importance_std": imp_std}
    ).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(os.path.join(RES_DIR, "feature_importance.csv"), index=False)

    # --- model-dependent figures ---
    plot_permutation_importance(FEATURES, imp_mean, imp_std)
    plot_pred_vs_actual(y_te, y_pred)
    plot_cv_vs_hyperparam(cv_results)

    # --- OLS linear baseline (same seed=42 train/test split) ---
    # Interpretable linear model: each coefficient is the signed, testable
    # effect of a trait on winpercent. numpy/scipy only (see src/ols.py).
    ols = OLSRegression(feature_names=FEATURES).fit(X_tr, y_tr)
    ols_pred = ols.predict(X_te)
    ols_test_r2 = r2_score(y_te, ols_pred)
    ols_test_rmse = rmse(y_te, ols_pred)
    ols_rows = ols.summary_table()
    print(f"OLS   R^2={ols_test_r2:.3f}  RMSE={ols_test_rmse:.3f}  "
          f"adj_R^2(train)={ols.adj_r_squared:.3f}")
    print("Model test comparison:  RF R^2={:.3f}/RMSE={:.2f}   "
          "OLS R^2={:.3f}/RMSE={:.2f}   baseline RMSE={:.2f}".format(
              test_r2, test_rmse, ols_test_r2, ols_test_rmse, baseline_rmse))

    # OLS coefficient table -> CSV (acceptance artifact), excluding intercept
    # row? Keep intercept too for completeness; feature column marks it.
    ols_df = pd.DataFrame(
        [{"feature": r["feature"], "coef": r["coef"], "std_err": r["std_err"],
          "t": r["t"], "p_value": r["p_value"], "sig": r["sig"]}
         for r in ols_rows]
    )
    ols_df.to_csv(os.path.join(RES_DIR, "ols_coefficients.csv"), index=False)

    ols_significant = [r["feature"] for r in ols_rows
                       if r["feature"] != "intercept" and r["p_value"] < 0.05]

    # OLS figures
    plot_ols_coefficients(ols_rows)
    plot_rf_vs_ols(test_r2, test_rmse, ols_test_r2, ols_test_rmse, baseline_rmse)

    # --- recommendation ---
    rec = build_recommendation(df, imp_df, eda_summary)
    print("Recommendation:", rec["headline"])

    # --- metrics.json ---
    metrics = {
        "seed": SEED,
        "n_rows": int(len(df)),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "best_params": best_params,
        "cv_r2_best": float(best["cv_r2_mean"]),
        "cv_rmse_best": float(best["cv_rmse_mean"]),
        "test_r2": float(test_r2),
        "test_rmse": float(test_rmse),
        "oob_r2": float(model.oob_r2_),
        "baseline_r2": float(baseline_r2),
        "baseline_rmse": float(baseline_rmse),
        "rmse_improvement_vs_baseline": float(baseline_rmse - test_rmse),
        "ols_test_r2": float(ols_test_r2),
        "ols_test_rmse": float(ols_test_rmse),
        "ols_adj_r2": float(ols.adj_r_squared),
        "ols_significant_coefficients": ols_significant,
        "top3_features": imp_df.head(3)["feature"].tolist(),
        "recommendation": rec,
        "eda_summary": eda_summary,
    }
    with open(os.path.join(RES_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Wrote results/metrics.json, results/feature_importance.csv, "
          "results/cv_results.csv, results/ols_coefficients.csv and figures/.")
    return metrics


if __name__ == "__main__":
    main()
