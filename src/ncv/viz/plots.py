"""Static figures (PNG) for the README and report. Colorblind-safe palette."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ncv.config import FIG_DIR, SALARY_CAP, SETTINGS, season_label  # noqa: E402
from ncv.data.panel import label  # noqa: E402

INK, MUTED, GRID = "#1F2933", "#7B8794", "#E4E7EB"
BLUE, ORANGE, PURPLE, GOLD = "#2F6DB5", "#D9822B", "#552583", "#E3A82B"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK,
})


def _save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def market_backtest(metrics: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = {"xgboost": ("XGBoost (monotonic)", BLUE), "ridge_all_features": ("Ridge, all features", ORANGE),
             "ridge_simple": ("Ridge, 4 stats", MUTED)}
    for key, (lab, col) in names.items():
        d = metrics[metrics["model"] == key]
        ax.plot([season_label(s) for s in d["season"]], d["mae_dollars"] / 1e6, marker="o", color=col,
                label=lab, lw=2 if key == "xgboost" else 1.4)
    ax.set_ylabel("Mean absolute error ($M)")
    ax.set_title("Market-value model: out-of-sample error by season")
    ax.legend(frameon=False)
    return _save(fig, "market_backtest.png")


def forecast_backtest(metrics: pd.DataFrame) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = [season_label(s) for s in metrics["season"]]
    a1.plot(x, metrics["mae_xgb_p50"], marker="o", color=BLUE, lw=2, label="Quantile XGBoost (P50)")
    a1.plot(x, metrics["mae_marcel"], marker="o", color=ORANGE, label="Marcel-style")
    a1.plot(x, metrics["mae_naive"], marker="o", color=MUTED, label="Last season")
    a1.set_ylabel("Mean absolute error (wins)")
    a1.set_title("Next-season WAR forecast vs baselines")
    a1.legend(frameon=False)
    a2.bar(x, metrics["coverage_p10_p90"] * 100, color=BLUE, alpha=0.85)
    a2.axhline(80, color=INK, ls="--", lw=1)
    a2.text(len(x) - 0.5, 81, "target 80%", ha="right", color=INK, fontsize=9)
    a2.set_ylim(50, 100)
    a2.set_ylabel("Actual outcomes inside P10–P90 (%)")
    a2.set_title("Interval calibration")
    for a in (a1, a2):
        a.tick_params(axis="x", rotation=45)
    return _save(fig, "forecast_backtest.png")


def shap_importance(shap_df: pd.DataFrame, top: int = 15) -> str:
    imp = shap_df.abs().mean().sort_values(ascending=False).head(top)[::-1] * 100
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh([label(f) for f in imp.index], imp.values, color=BLUE)
    ax.set_xlabel("Mean |SHAP| (percentage points of cap)")
    ax.set_title("What drives market value")
    ax.grid(axis="y", visible=False)
    return _save(fig, "shap_importance.png")


def value_scatter(board: pd.DataFrame, season: int, n_label: int = 6) -> str:
    cap = SALARY_CAP[season] / 1e6
    b = board[board["salary"].notna() & (board["mp"] >= SETTINGS.min_minutes_value)]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for ct, col in [("Veteran", BLUE), ("Rookie scale", ORANGE)]:
        d = b[b["contract_type"] == ct]
        ax.scatter(d["market_value"] * cap, d["pct_cap"] * cap, s=18, alpha=0.6, color=col, label=ct)
    m = max(b["market_value"].max(), b["pct_cap"].max()) * cap
    ax.plot([0, m], [0, m], color=INK, ls="--", lw=1)
    ax.text(m * 0.98, m * 0.93, "paid at market", ha="right", color=INK, fontsize=9, rotation=40)
    for _, r in pd.concat([b.nlargest(n_label, "market_surplus"), b.nsmallest(n_label, "market_surplus")]).iterrows():
        ax.annotate(r["player"], (r["market_value"] * cap, r["pct_cap"] * cap), fontsize=7.5,
                    xytext=(4, 2), textcoords="offset points", color=INK)
    ax.set_xlabel("Market value ($M, out-of-sample)")
    ax.set_ylabel("Actual salary ($M)")
    ax.set_title(f"{season_label(season)}: salary vs market value (below the line = bargain)")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, f"value_scatter_{season}.png")


def surplus_leaders(board: pd.DataFrame, season: int, n: int = 12) -> str:
    cap = SALARY_CAP[season] / 1e6
    b = board[board["salary"].notna() & (board["mp"] >= SETTINGS.min_minutes_value)]
    d = pd.concat([b.nlargest(n, "realized_surplus"), b.nsmallest(n, "realized_surplus")])
    d = d.sort_values("realized_surplus")
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = np.where(d["realized_surplus"] > 0, BLUE, ORANGE)
    ax.barh(d["player"] + " (" + d["team"] + ")", d["realized_surplus"] * cap, color=colors)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Realized surplus ($M): value of wins produced minus salary")
    ax.set_title(f"{season_label(season)}: biggest realized surpluses and shortfalls")
    ax.grid(axis="y", visible=False)
    return _save(fig, f"surplus_leaders_{season}.png")


def team_efficiency(teams: pd.DataFrame, season: int, highlight: str = "LAL") -> str:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    col = np.where(teams["team"] == highlight, PURPLE, BLUE)
    ax.scatter(teams["payroll_tracked"] / 1e6, teams["w"], c=col, s=50)
    for _, r in teams.iterrows():
        ax.annotate(r["team"], (r["payroll_tracked"] / 1e6, r["w"]), fontsize=8, xytext=(4, 2),
                    textcoords="offset points", color=PURPLE if r["team"] == highlight else INK,
                    fontweight="bold" if r["team"] == highlight else "normal")
    rho = teams[["realized_surplus", "w"]].corr().iloc[0, 1]
    ax.set_xlabel("Payroll of tracked players ($M)")
    ax.set_ylabel("Wins")
    ax.set_title(f"{season_label(season)}: payroll vs wins (corr. of team surplus with wins = {rho:.2f})")
    return _save(fig, f"team_efficiency_{season}.png")


def archetype_pricing(table: pd.DataFrame) -> str:
    t = table.sort_values("mean_gap_pct_cap")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(t))
    colors = np.where(t["mean_gap_pct_cap"] > 0, BLUE, ORANGE)
    ax.barh(y, t["mean_gap_pct_cap"] * 100, color=colors, alpha=0.9)
    ax.errorbar(t["mean_gap_pct_cap"] * 100, y,
                xerr=[(t["mean_gap_pct_cap"] - t["ci_low"]) * 100, (t["ci_high"] - t["mean_gap_pct_cap"]) * 100],
                fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.set_yticks(y, t["archetype"])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Realized production value − market price (% of cap, 90% CI)")
    ax.set_title("Which roles does the market under- or over-price?")
    ax.grid(axis="y", visible=False)
    return _save(fig, "archetype_pricing.png")


def team_projection(proj: pd.DataFrame, team: str, season: int) -> str:
    d = proj.sort_values("war_p50")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(d) + 1)))
    y = np.arange(len(d))
    ax.hlines(y, d["war_p10"], d["war_p90"], color=PURPLE, lw=5, alpha=0.35)
    ax.scatter(d["war_p50"], y, color=PURPLE, zorder=3, s=40)
    ax.set_yticks(y, d["player"])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Projected wins above replacement (dot = P50, bar = P10–P90)")
    ax.set_title(f"{team} {season_label(season)} projections")
    ax.grid(axis="y", visible=False)
    return _save(fig, f"projection_{team}_{season}.png")
