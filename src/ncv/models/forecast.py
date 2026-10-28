"""Probabilistic forecast of next-season wins above replacement (WAR).

Multi-quantile XGBoost produces P10 / P50 / P90 WAR for season T from data
through T-1. Availability risk (games missed) is part of the target, so the
interval reflects both performance and injury uncertainty.

Benchmarked against:
  * naive: last season's WAR carried forward
  * Marcel-style: 5/4 weighting of the last two seasons, 20% regression to
    the league mean, standard Marcel age adjustment (Tango, 2004)

Known limitation: players absent from the NBA in season T (retired, waived,
or out all year) have no WAR_T and are excluded from training, which biases
forecasts slightly upward for players at high risk of missing a full season.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from ncv.config import SETTINGS, VORP_TO_WINS
from ncv.data.panel import FEATURES

Q = SETTINGS.forecast_quantiles
QCOLS = [f"war_p{int(q * 100)}" for q in Q]


def _fit_quantile(train: pd.DataFrame) -> xgb.XGBRegressor:
    params = dict(SETTINGS.xgb_params)
    params.update(n_estimators=500, learning_rate=0.03, max_depth=4)
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(Q),
                         random_state=SETTINGS.random_state, **params)
    m.fit(train[FEATURES], train["war_T"])
    return m


def predict_quantiles(model: xgb.XGBRegressor, X: pd.DataFrame) -> pd.DataFrame:
    raw = model.predict(X[FEATURES])
    raw = np.sort(raw, axis=1)  # enforce non-crossing quantiles
    return pd.DataFrame(raw, columns=QCOLS, index=X.index)


def marcel(X: pd.DataFrame, league_mean: float) -> pd.Series:
    w1 = X["l1_vorp"] * VORP_TO_WINS
    w2 = X["l2_vorp"] * VORP_TO_WINS
    weighted = np.where(w2.notna(), (5 * w1 + 4 * w2.fillna(0)) / 9, w1)
    regressed = 0.8 * weighted + 0.2 * league_mean
    age = X["age"]
    age_adj = np.where(age < 29, 1 + (29 - age) * 0.006, 1 + (29 - age) * 0.003)
    return pd.Series(regressed * age_adj, index=X.index)


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


@dataclass
class ForecastBacktest:
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def forecast_rows(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["war_T"].notna()]


def backtest(panel: pd.DataFrame, seasons: tuple[int, ...] = SETTINGS.backtest_seasons) -> ForecastBacktest:
    rows_all = forecast_rows(panel)
    preds, rows = [], []
    for T in seasons:
        train, test = rows_all[rows_all["season"] < T], rows_all[rows_all["season"] == T].copy()
        m = _fit_quantile(train)
        qs = predict_quantiles(m, test)
        test = test.join(qs)
        test["war_marcel"] = marcel(test, train["war_T"].mean())
        test["war_naive"] = test["l1_vorp"] * VORP_TO_WINS
        preds.append(test)
        y = test["war_T"].to_numpy()
        lo, hi = test[QCOLS[0]], test[QCOLS[-1]]
        rows.append({
            "season": T, "n": len(test),
            "mae_xgb_p50": float(np.mean(np.abs(y - test[QCOLS[1]]))),
            "mae_marcel": float(np.mean(np.abs(y - test["war_marcel"]))),
            "mae_naive": float(np.mean(np.abs(y - test["war_naive"]))),
            "coverage_p10_p90": float(np.mean((y >= lo) & (y <= hi))),
            "below_p10": float(np.mean(y < lo)),
            "above_p90": float(np.mean(y > hi)),
            "pinball_mean": float(np.mean([pinball(y, test[c].to_numpy(), q) for c, q in zip(QCOLS, Q)])),
        })
    return ForecastBacktest(pd.concat(preds, ignore_index=True), pd.DataFrame(rows))


def fit_final(panel: pd.DataFrame, through_season: int) -> xgb.XGBRegressor:
    rows = forecast_rows(panel)
    return _fit_quantile(rows[rows["season"] <= through_season])
