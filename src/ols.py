"""From-scratch Ordinary Least Squares (OLS) linear regression.

Uses only numpy and scipy (scipy.stats for the t-distribution), honouring the
same brief as the Random Forest: no scikit-learn, statsmodels or xgboost.

The model fits

    y = X beta + e,   beta_hat = (X^T X)^-1 X^T y

solved numerically with ``numpy.linalg.lstsq`` (stable, handles rank
deficiency gracefully). Classical inference is computed by hand:

    sigma^2 = RSS / (n - p)                 residual variance (p incl. intercept)
    Var(beta_hat) = sigma^2 (X^T X)^-1      coefficient variance-covariance
    SE(beta_j)    = sqrt(diag(Var))         standard errors
    t_j           = beta_j / SE(beta_j)     t-statistic
    p_j           = 2 * P(T_{n-p} > |t_j|)  two-sided p-value (scipy.stats.t)

Feature scale
-------------
The design matrix uses the features on their natural scale: the 9 recipe flags
are binary (0/1) and ``sugarpercent`` / ``pricepercent`` are already percentile
ranks in [0, 1]. An intercept column of ones is prepended. On this scale each
coefficient reads directly as the change in ``winpercent`` (in win-points) for a
one-unit change in the feature: for a binary flag that is "present vs absent",
and for the two continuous ranks it is "bottom to top of the rank range". No
standardization is applied, so the signs and magnitudes are directly
interpretable in the units of the target.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class OLSRegression:
    """Ordinary least squares with classical (t-based) inference.

    Parameters
    ----------
    feature_names : list of str or None
        Names for the non-intercept columns, used in the summary table. If
        ``None``, generic ``x0, x1, ...`` names are used.
    add_intercept : bool
        Prepend a column of ones (default True).

    Attributes (set after ``fit``)
    -------------------------------
    coefficients : np.ndarray, shape (p,)
        Estimated coefficients, intercept first when ``add_intercept``.
    standard_errors, t_values, p_values : np.ndarray, shape (p,)
        Classical standard errors, t-statistics and two-sided p-values.
    r_squared, adj_r_squared : float
        Coefficient of determination and its adjusted form.
    sigma2_ : float
        Residual variance estimate ``RSS / (n - p)``.
    names_ : list of str
        Column names aligned with ``coefficients`` (``"intercept"`` first).
    """

    def __init__(self, feature_names=None, add_intercept=True):
        self.feature_names = feature_names
        self.add_intercept = add_intercept
        self.coefficients = None
        self.standard_errors = None
        self.t_values = None
        self.p_values = None
        self.r_squared = None
        self.adj_r_squared = None
        self.sigma2_ = None
        self.names_ = None
        self._n = None
        self._p = None

    def _design(self, X):
        """Return the design matrix, prepending an intercept column if asked."""
        X = np.asarray(X, dtype=float)
        if self.add_intercept:
            ones = np.ones((X.shape[0], 1), dtype=float)
            return np.hstack([ones, X])
        return X

    def fit(self, X, y):
        """Fit the model and compute coefficients, inference and R^2.

        Parameters
        ----------
        X : array-like, shape (n, k)
            Feature matrix (without intercept).
        y : array-like, shape (n,)
            Target vector.
        """
        Xd = self._design(X)
        y = np.asarray(y, dtype=float)
        n, p = Xd.shape
        self._n, self._p = n, p

        # Closed-form OLS via least squares (numerically stable).
        beta, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)

        resid = y - Xd @ beta
        rss = float(resid @ resid)
        dof = n - p  # residual degrees of freedom
        sigma2 = rss / dof if dof > 0 else np.nan

        # Variance-covariance of beta: sigma^2 (X^T X)^-1. Use pinv for safety
        # against near-singular X^T X (collinear features).
        xtx_inv = np.linalg.pinv(Xd.T @ Xd)
        var_beta = sigma2 * np.diag(xtx_inv)
        se = np.sqrt(np.clip(var_beta, 0.0, None))

        with np.errstate(divide="ignore", invalid="ignore"):
            t_vals = np.where(se > 0, beta / se, np.nan)
        # Two-sided p-values from Student's t with (n - p) dof.
        p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=dof)

        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
        # Adjusted R^2 uses k = number of predictors excluding the intercept.
        k = p - 1 if self.add_intercept else p
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if (n - k - 1) > 0 else np.nan

        # Column names aligned with beta.
        if self.feature_names is not None:
            base = list(self.feature_names)
        else:
            base = [f"x{i}" for i in range(Xd.shape[1] - (1 if self.add_intercept else 0))]
        self.names_ = (["intercept"] + base) if self.add_intercept else base

        self.coefficients = beta
        self.standard_errors = se
        self.t_values = t_vals
        self.p_values = p_vals
        self.sigma2_ = sigma2
        self.r_squared = float(r2)
        self.adj_r_squared = float(adj_r2)
        return self

    def predict(self, X):
        """Predict the target for rows of ``X`` (without intercept column)."""
        Xd = self._design(X)
        return Xd @ self.coefficients

    @staticmethod
    def _stars(p):
        """Significance stars: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1."""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        if p < 0.1:
            return "."
        return ""

    def summary_table(self):
        """Return a tidy list of dict rows: feature, coef, std_err, t, p, sig.

        Ordered as the coefficients are (intercept first when present).
        """
        rows = []
        for name, b, se, t, p in zip(
            self.names_, self.coefficients, self.standard_errors,
            self.t_values, self.p_values
        ):
            rows.append({
                "feature": name,
                "coef": float(b),
                "std_err": float(se),
                "t": float(t),
                "p_value": float(p),
                "sig": self._stars(float(p)),
            })
        return rows

    def summary_str(self):
        """Human-readable summary string (for printing to the console)."""
        lines = [
            f"OLS results  (n={self._n}, predictors={self._p - (1 if self.add_intercept else 0)})",
            f"R^2 = {self.r_squared:.3f}   adj R^2 = {self.adj_r_squared:.3f}   "
            f"sigma = {np.sqrt(self.sigma2_):.3f}",
            f"{'feature':<18}{'coef':>10}{'std_err':>10}{'t':>8}{'p':>10}  sig",
        ]
        for r in self.summary_table():
            lines.append(
                f"{r['feature']:<18}{r['coef']:>10.3f}{r['std_err']:>10.3f}"
                f"{r['t']:>8.2f}{r['p_value']:>10.4f}  {r['sig']}"
            )
        return "\n".join(lines)
