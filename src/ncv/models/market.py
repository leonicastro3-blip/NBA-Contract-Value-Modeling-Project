"""Market-value model: what the open market pays for a player's track record.

Trained only on veteran (non-rookie-scale) contracts, because rookie-scale
salaries are set by the CBA rather than negotiated. Applying the model to
rookie-scale players then estimates what they would cost on the open market.

Validation is rolling-origin: to score season T, train on seasons < T only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ncv.config import SETTINGS
from ncv.data.panel import FEATURES

TARGET = "pct_cap"

# Basketball sense: more production / reputation never lowers market value.
MONOTONE_UP = {
    "l1_vorp", "l1_ws", "l1_mp", "l1_pts_per_game", "wt_bpm", "l1_all_star",
    "career_all_stars_prior", "l2_vorp", "l1_mp_per_game",
}
BASELINE_SIMPLE = ["l1_vorp", "wt_bpm", "l1_mp", "age"]


def _monotone(features: list[str]) -> tuple[int, ...]:
    return tuple(1 if f in MONOTONE_UP else 0 for f in features)


def make_xgb(n_estimators: int | None = None, **overrides) -> xgb.XGBRegressor:
    params = dict(SETTINGS.xgb_params)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    params.update(overrides)
    return xgb.XGBRegressor(objective="reg:squarederror", random_state=SETTINGS.random_state,
                            monotone_constraints=_monotone(FEATURES), **params)


def fit_xgb(train: pd.DataFrame) -> xgb.XGBRegressor:
    """Pick tree count by early stopping on the latest training season, then refit on all."""
    last = train["season"].max()
    inner_tr, inner_va = train[train["season"] < last], train[train["season"] == last]
    probe = make_xgb(early_stopping_rounds=50)
    probe.fit(inner_tr[FEATURES], inner_tr[TARGET],
              eval_set=[(inner_va[FEATURES], inner_va[TARGET])], verbose=False)
    n_trees = max(100, int(probe.best_iteration) + 1)
    model = make_xgb(n_estimators=n_trees)
    model.fit(train[FEATURES], train[TARGET])
    return model


def fit_ridge(train: pd.DataFrame, cols: list[str]):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      RidgeCV(alphas=np.logspace(-2, 3, 20)))
    m.fit(train[cols], train[TARGET])
    return m


def market_training_rows(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[(~panel["is_rookie_scale"]) & panel[TARGET].notna()]


@dataclass
class BacktestResult:
    predictions: pd.DataFrame   # out-of-sample predictions for every player in each test season
    metrics: pd.DataFrame       # per-season metrics by model (veterans only)


def backtest(panel: pd.DataFrame, seasons: tuple[int, ...] = SETTINGS.backtest_seasons) -> BacktestResult:
    vets = market_training_rows(panel)
    preds, rows = [], []
    for T in seasons:
        train = vets[vets["season"] < T]
        test_all = panel[(panel["season"] == T)].copy()
        models = {
            "xgboost": fit_xgb(train),
            "ridge_all_features": fit_ridge(train, FEATURES),
            "ridge_simple": fit_ridge(train, BASELINE_SIMPLE),
        }
        test_all["market_value"] = np.clip(models["xgboost"].predict(test_all[FEATURES]), 0, None)
        preds.append(test_all)

        test = test_all[(~test_all["is_rookie_scale"]) & test_all[TARGET].notna()]
        for name, m in models.items():
            cols = BASELINE_SIMPLE if name == "ridge_simple" else FEATURES
            p = np.clip(m.predict(test[cols]), 0, None)
            rows.append({
                "season": T, "model": name, "n": len(test),
                "mae_pct_cap": mean_absolute_error(test[TARGET], p),
                "mae_dollars": mean_absolute_error(test[TARGET] * test["salary_cap"],
                                                   p * test["salary_cap"]),
                "r2": r2_score(test[TARGET], p),
                "spearman": spearmanr(test[TARGET], p).statistic,
            })
    return BacktestResult(pd.concat(preds, ignore_index=True), pd.DataFrame(rows))


def fit_final(panel: pd.DataFrame, through_season: int) -> xgb.XGBRegressor:
    vets = market_training_rows(panel)
    return fit_xgb(vets[vets["season"] <= through_season])


def shap_values(model: xgb.XGBRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP contributions from XGBoost (last column is the bias)."""
    contrib = model.get_booster().predict(xgb.DMatrix(X[FEATURES]), pred_contribs=True)
    return pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=X.index)


def top_drivers(shap_df: pd.DataFrame, k: int = 3) -> pd.Series:
    """For each player, the k features pushing market value up or down the most."""
    from ncv.data.panel import label

    def fmt(row: pd.Series) -> str:
        top = row.abs().sort_values(ascending=False).head(k).index
        return "; ".join(f"{label(f)} ({'+' if row[f] >= 0 else '−'}{abs(row[f]) * 100:.1f}% cap)"
                         for f in top)
    return shap_df.apply(fmt, axis=1)
