"""End-to-end pipeline: data -> models -> validation -> boards -> figures -> report.

Usage:  python -m ncv.pipeline
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd

from ncv.analysis import roster
from ncv.config import PROCESSED_DIR, SALARY_CAP, SETTINGS, TABLE_DIR, season_label
from ncv.data import panel, sources
from ncv.data.cleaning import name_key
from ncv.models import forecast, market
from ncv.models.archetypes import ArchetypeModel, pricing_by_archetype
from ncv.models.production import league_win_prices, production_value
from ncv.report import write_report
from ncv.viz import plots

log = logging.getLogger("ncv")

CURRENT = SETTINGS.last_salary_season      # last completed season with salaries (2025-26)
NEXT = CURRENT + 1                         # season being projected (2026-27)
FOCUS_TEAMS = ("LAL",)                     # teams that get a dedicated report section


def run(download: bool = True) -> dict:
    t0 = time.time()
    if download:
        from importlib import import_module
        import sys
        from ncv.config import ROOT
        sys.path.insert(0, str(ROOT / "scripts"))
        import_module("download_data").download()

    log.info("Building season table")
    st, join_diag = panel.load_season_table()
    arche = ArchetypeModel().fit(st)
    st["archetype"] = arche.assign(st)

    feats = panel.attach_targets(panel.build_features(st, list(range(2010, NEXT + 1))), st)
    prices = league_win_prices(st[st["season"] >= SETTINGS.first_stat_season])

    log.info("Backtesting market-value model")
    mkt_bt = market.backtest(feats)
    log.info("Backtesting forecast model")
    fc_bt = forecast.backtest(feats)

    # Current-season board: out-of-sample market values + per-player SHAP drivers
    cur_rows = mkt_bt.predictions[mkt_bt.predictions["season"] == CURRENT]
    model_prev = market.fit_final(feats, through_season=CURRENT - 1)
    drivers = market.top_drivers(market.shap_values(model_prev, cur_rows))
    board = roster.current_board(mkt_bt.predictions, st, prices, CURRENT, drivers)

    teams = roster.team_summary(board, sources.load_team_records().query("season == @CURRENT")
                                .drop(columns="season"))

    # Archetype pricing on all out-of-sample veteran seasons
    oof = mkt_bt.predictions
    vets = oof[(~oof["is_rookie_scale"]) & oof["war_T"].notna() & (oof["mp_T"] >= SETTINGS.min_minutes_value)]
    vets = vets.merge(st[["player_id", "season", "archetype"]], on=["player_id", "season"])
    vets["production_value"] = production_value(vets["war_T"], vets["season"], prices)
    arche_table = pricing_by_archetype(vets.dropna(subset=["archetype"]))

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Fitting final models through %s and projecting %s", CURRENT, NEXT)
    nxt = feats[feats["season"] == NEXT]
    mkt_final = market.fit_final(feats, through_season=CURRENT)
    fc_final = forecast.fit_final(feats, through_season=CURRENT)
    shap_next = market.shap_values(mkt_final, nxt)
    contracts = sources.load_next_season_contracts()
    nxt = nxt.assign(drivers=market.top_drivers(shap_next).to_numpy())
    proj = roster.projection_board(nxt, forecast.predict_quantiles(fc_final, nxt),
                                   mkt_final.predict(nxt[panel.FEATURES]), prices, st, NEXT, contracts)
    proj = proj.merge(nxt[["player_id", "drivers"]], on="player_id", how="left")

    # Team views: current rosters from the contracts snapshot, unless a roster file overrides
    team_views = {}
    for team in FOCUS_TEAMS:
        path = roster.ROSTER_DIR / f"{team}_{NEXT}.csv"
        if path.exists():
            names = roster.read_roster(path)
        elif contracts is not None:
            names = contracts.loc[contracts["team"] == team, "player"].tolist()
        else:
            names = proj.loc[proj["team"] == team, "player"].tolist()
        keys = {name_key(n): n for n in names}
        view = proj[proj["player"].map(name_key).isin(keys)].copy()
        missing = [n for k, n in keys.items() if k not in set(view["player"].map(name_key))]
        team_views[team] = {"projection": view, "no_history": missing}
    if contracts is not None:
        contracts.to_parquet(PROCESSED_DIR / f"contracts_{NEXT}.parquet", index=False)

    # Save tables (CSV for humans, parquet for the dashboard)
    outputs = {
        f"value_board_{CURRENT}": board, f"projections_{NEXT}": proj, f"teams_{CURRENT}": teams,
        "archetype_pricing": arche_table, "market_backtest": mkt_bt.metrics,
        "forecast_backtest": fc_bt.metrics, "win_prices": prices,
        "archetype_centroids": arche.centroids().reset_index(names="archetype"),
    }
    for name, df in outputs.items():
        df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    history = st[st["season"] >= SETTINGS.first_stat_season][
        ["player", "player_id", "season", "team", "age", "pos", "archetype", "salary", "pct_cap",
         "war", "vorp", "bpm", "mp", "g", "team_games", "games_pct", "injury_shortened",
         "healthy_war"]].copy()
    history["production_value"] = production_value(history["war"], history["season"], prices)
    history["healthy_value"] = production_value(history["healthy_war"], history["season"], prices)
    history.to_parquet(PROCESSED_DIR / "player_history.parquet", index=False)
    pd.Series(market.shap_values(mkt_final, nxt).abs().mean()).rename("mean_abs_shap") \
        .to_frame().to_parquet(PROCESSED_DIR / "shap_importance.parquet")
    for team, v in team_views.items():
        v["projection"].to_csv(TABLE_DIR / f"projection_{team}_{NEXT}.csv", index=False)

    log.info("Rendering figures")
    figs = {
        "market_backtest": plots.market_backtest(mkt_bt.metrics),
        "forecast_backtest": plots.forecast_backtest(fc_bt.metrics),
        "shap": plots.shap_importance(shap_next),
        "scatter": plots.value_scatter(board, CURRENT),
        "leaders": plots.surplus_leaders(board, CURRENT),
        "teams": plots.team_efficiency(teams, CURRENT),
        "archetypes": plots.archetype_pricing(arche_table),
    }
    for team, v in team_views.items():
        if len(v["projection"]):
            figs[f"projection_{team}"] = plots.team_projection(v["projection"], team, NEXT)

    summary = summarize(mkt_bt, fc_bt, join_diag, prices, teams)
    (TABLE_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, board, proj, teams, arche_table, team_views, figs, CURRENT, NEXT)
    log.info("Done in %.0fs", time.time() - t0)
    return summary


def summarize(mkt_bt, fc_bt, join_diag, prices, teams) -> dict:
    mm = mkt_bt.metrics.groupby("model")[["mae_pct_cap", "mae_dollars", "r2", "spearman"]].mean()
    fm = fc_bt.metrics.mean(numeric_only=True)
    p = prices.set_index("season")
    return {
        "seasons_backtested": [season_label(s) for s in SETTINGS.backtest_seasons],
        "salary_match_rate": round(join_diag["match_rate_rotation"], 4),
        "market": {m: {k: round(float(v), 4) for k, v in row.items()} for m, row in mm.iterrows()},
        "forecast": {k: round(float(v), 4) for k, v in fm.items() if k not in ("season", "n")},
        "forecast_beats_marcel_every_season": bool(
            (fc_bt.metrics["mae_xgb_p50"] < fc_bt.metrics["mae_marcel"]).all()),
        "usd_per_war_current": round(float(p.loc[CURRENT, "pct_cap_per_war"] * SALARY_CAP[CURRENT]), 0),
        "team_surplus_win_corr": round(float(teams[["realized_surplus", "w"]].corr().iloc[0, 1]), 3),
        "team_war_win_corr": round(float(teams[["war", "w"]].corr().iloc[0, 1]), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    print(json.dumps(run(download=not args.skip_download), indent=2))


if __name__ == "__main__":
    main()
