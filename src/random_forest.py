"""From-scratch Random Forest regressor using only numpy and scipy.

Implements a CART-style DecisionTreeRegressor with variance-reduction (MSE)
splits, and a RandomForestRegressor built from bootstrap bagging plus random
feature subsampling at each split. Also provides out-of-bag (OOB) R^2 and
permutation feature importance.

No scikit-learn, statsmodels or any other ML library is used -- only numpy
(and scipy is available for utilities, though the core math here is numpy).
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Decision tree
# ----------------------------------------------------------------------------
class _Node:
    """A single node in a regression tree.

    A leaf stores ``value`` (the mean target of the samples that reached it).
    An internal node stores the split ``feature`` index, the ``threshold``
    (go left when ``x[feature] <= threshold``), and ``left`` / ``right`` child
    nodes.
    """

    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeRegressor:
    """CART regression tree with MSE (variance-reduction) splits.

    Parameters
    ----------
    max_depth : int or None
        Maximum tree depth. ``None`` means grow until other stop rules fire.
    min_samples_split : int
        Minimum samples required to attempt a split.
    min_samples_leaf : int
        Minimum samples required in each child for a split to be accepted.
    max_features : int or None
        Number of features considered at each split (random subset). ``None``
        uses all features.
    random_state : int or None
        Seed for the per-tree feature-subsampling RNG.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root_ = None
        self.n_features_ = None

    # -- fitting -------------------------------------------------------------
    def fit(self, X, y):
        """Build the tree from training data ``X`` (n, p) and target ``y`` (n,)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.root_ = self._grow(X, y, depth=0)
        return self

    def _grow(self, X, y, depth):
        n_samples = X.shape[0]
        node_value = float(np.mean(y))

        # Stop rules: depth, too few samples, or pure node.
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or np.all(y == y[0])
        ):
            return _Node(value=node_value)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return _Node(value=node_value)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature=feature, threshold=threshold, left=left, right=right)

    def _best_split(self, X, y):
        """Find the (feature, threshold) that most reduces weighted child MSE.

        Equivalent to maximising variance reduction. Returns ``(None, None)``
        when no admissible split improves on the parent.
        """
        n_samples, n_features = X.shape
        parent_sse = self._sse(y)  # sum of squared errors of the parent node

        # Random feature subset (random_forest subsampling happens here).
        if self.max_features is not None and self.max_features < n_features:
            feat_indices = self._rng.choice(
                n_features, size=self.max_features, replace=False
            )
        else:
            feat_indices = np.arange(n_features)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feat in feat_indices:
            values = X[:, feat]
            # Candidate thresholds = midpoints between consecutive unique values.
            uniq = np.unique(values)
            if uniq.size < 2:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            for thr in thresholds:
                left_mask = values <= thr
                n_left = int(np.count_nonzero(left_mask))
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                sse_left = self._sse(y[left_mask])
                sse_right = self._sse(y[~left_mask])
                gain = parent_sse - (sse_left + sse_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feat)
                    best_threshold = float(thr)

        return best_feature, best_threshold

    @staticmethod
    def _sse(y):
        """Sum of squared errors about the mean = variance * n."""
        if y.size == 0:
            return 0.0
        return float(np.sum((y - np.mean(y)) ** 2))

    # -- prediction ----------------------------------------------------------
    def predict(self, X):
        """Predict targets for rows of ``X`` (n, p)."""
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root_) for row in X])

    def _predict_row(self, row, node):
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


# ----------------------------------------------------------------------------
# Random forest
# ----------------------------------------------------------------------------
class RandomForestRegressor:
    """Bootstrap-bagged ensemble of regression trees (from scratch).

    Each tree is trained on a bootstrap sample of the rows and considers a
    random subset of ``max_features`` features at every split. Predictions
    average the per-tree predictions. Provides out-of-bag (OOB) R^2 and
    permutation feature importance.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    max_depth : int or None
        Maximum depth of each tree.
    min_samples_split : int
        Minimum samples to attempt a split.
    min_samples_leaf : int
        Minimum samples in each child of a split.
    max_features : int, float, str or None
        Features per split. ``int`` = that many; ``float`` in (0,1] = that
        fraction of p; ``"sqrt"`` = round(sqrt(p)); ``"third"`` = round(p/3)
        (a common regression default); ``None`` = all features.
    random_state : int or None
        Master seed; each tree gets a derived, reproducible seed.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="third",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self.oob_r2_ = None
        self.n_features_ = None

    def _resolve_max_features(self, n_features):
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(round(np.sqrt(n_features))))
            if mf == "third":
                return max(1, int(round(n_features / 3.0)))
            raise ValueError(f"unknown max_features string: {mf!r}")
        if isinstance(mf, float):
            return max(1, int(round(mf * n_features)))
        return max(1, min(int(mf), n_features))

    def fit(self, X, y):
        """Fit the forest and compute the OOB R^2."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        max_feat = self._resolve_max_features(n_features)

        rng = np.random.default_rng(self.random_state)
        # Per-tree seeds, derived once so the whole fit is reproducible.
        tree_seeds = rng.integers(0, 2**32 - 1, size=self.n_estimators)

        self.trees_ = []
        # OOB bookkeeping: sum of predictions and counts per sample.
        oob_pred_sum = np.zeros(n_samples)
        oob_pred_cnt = np.zeros(n_samples)

        for t in range(self.n_estimators):
            tree_rng = np.random.default_rng(int(tree_seeds[t]))
            # Bootstrap sample (sample n rows with replacement).
            boot_idx = tree_rng.integers(0, n_samples, size=n_samples)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[np.unique(boot_idx)] = False

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_feat,
                random_state=int(tree_seeds[t]),
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees_.append(tree)

            # Accumulate OOB predictions for samples not in this bootstrap.
            if np.any(oob_mask):
                preds = tree.predict(X[oob_mask])
                oob_pred_sum[oob_mask] += preds
                oob_pred_cnt[oob_mask] += 1

        # OOB R^2 over samples that were OOB for at least one tree.
        have_oob = oob_pred_cnt > 0
        if np.any(have_oob):
            oob_pred = oob_pred_sum[have_oob] / oob_pred_cnt[have_oob]
            self.oob_r2_ = r2_score(y[have_oob], oob_pred)
        else:
            self.oob_r2_ = np.nan
        return self

    def predict(self, X):
        """Average predictions across all trees."""
        X = np.asarray(X, dtype=float)
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += tree.predict(X)
        return preds / len(self.trees_)

    def permutation_importance(self, X, y, n_repeats=10, random_state=None, scorer=None):
        """Permutation feature importance.

        Shuffle each feature column in turn, measure the drop in score, and
        average over ``n_repeats`` shuffles. Positive importance = the feature
        matters (shuffling it hurts the score).

        Parameters
        ----------
        X, y : arrays
            Data to measure importance on (typically a held-out set).
        n_repeats : int
            Number of shuffles per feature.
        random_state : int or None
            Seed for the shuffling RNG.
        scorer : callable(y_true, y_pred) -> float or None
            Score to use; defaults to R^2. Higher is better.

        Returns
        -------
        importances_mean, importances_std : np.ndarray, np.ndarray
            Mean and std of the score drop per feature.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if scorer is None:
            scorer = r2_score
        rng = np.random.default_rng(random_state)

        baseline = scorer(y, self.predict(X))
        n_features = X.shape[1]
        importances = np.zeros((n_features, n_repeats))

        for feat in range(n_features):
            for rep in range(n_repeats):
                X_perm = X.copy()
                rng.shuffle(X_perm[:, feat])
                score = scorer(y, self.predict(X_perm))
                importances[feat, rep] = baseline - score

        return importances.mean(axis=1), importances.std(axis=1)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def r2_score(y_true, y_pred):
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
