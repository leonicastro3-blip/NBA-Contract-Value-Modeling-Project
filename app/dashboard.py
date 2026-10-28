"""Contract value dashboard.  Run:  streamlit run app/dashboard.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ncv.config import PROCESSED_DIR, SALARY_CAP, SETTINGS, season_label  # noqa: E402
from ncv.data.cleaning import name_key  # noqa: E402
from ncv.data.panel import label  # noqa: E402

CUR, NXT = SETTINGS.last_salary_season, SETTINGS.last_salary_season + 1
BLUE, ORANGE, PURPLE, MUTED = "#2F6DB5", "#D9822B", "#552583", "#7B8794"
ROSTER_DIR = ROOT / "data" / "rosters"

st.set_page_config(page_title="Contract value", layout="wide")


@st.cache_data
def load() -> dict[str, pd.DataFrame]:
    names = [f"value_board_{CUR}", f"projections_{NXT}", f"teams_{CUR}", "archetype_pricing",
             "market_backtest", "forecast_backtest", "player_history", "shap_importance"]
    return {n: pd.read_parquet(PROCESSED_DIR / f"{n}.parquet") for n in names}


if not (PROCESSED_DIR / f"projections_{NXT}.parquet").exists():
    st.error("No model outputs found. Run `python -m ncv.pipeline` from the project root, then reload.")
    st.stop()

d = load()
board, proj, teams = d[f"value_board_{CUR}"], d[f"projections_{NXT}"], d[f"teams_{CUR}"]
ncap, cap = SALARY_CAP[NXT], SALARY_CAP[CUR]


def money(x) -> str:
    if pd.isna(x):
        return "—"
    return f"{'−' if x < 0 else ''}${abs(x) / 1e6:,.1f}M"


# ---------------- sidebar ----------------
team_list = sorted(proj["team"].dropna().unique())
st.sidebar.markdown("### Team")
team = st.sidebar.selectbox("Team", team_list, index=team_list.index("LAL") if "LAL" in team_list else 0,
                            label_visibility="collapsed")
roster_file = ROSTER_DIR / f"{team}_{NXT}.csv"
use_file = False
if roster_file.exists():
    use_file = st.sidebar.toggle(f"Use {season_label(NXT)} roster file", value=True,
                                 help=f"Players listed in data/rosters/{roster_file.name}. "
                                      f"Off = {season_label(CUR)} end-of-season roster.")
st.sidebar.caption(f"Data through {season_label(CUR)}. Projections are for {season_label(NXT)}. "
                   f"Dollar figures use the {season_label(NXT)} cap of {money(ncap)}.")

tab_roster, tab_targets, tab_board, tab_player, tab_model = st.tabs(
    ["Roster outlook", "Target finder", f"{season_label(CUR)} value board", "Player history", "How the models work"])

# ---------------- roster outlook ----------------
with tab_roster:
    contracts_path = PROCESSED_DIR / f"contracts_{NXT}.parquet"
    if use_file:
        names = pd.read_csv(roster_file)["player"].tolist()
    elif contracts_path.exists():
        c = pd.read_parquet(contracts_path)
        names = c.loc[c["team"] == team, "player"].tolist()
    else:
        names = proj.loc[proj["team"] == team, "player"].tolist()
    keys = {name_key(n) for n in names}
    view = proj[proj["player"].map(name_key).isin(keys)].copy()
    missing = [n for n in names if name_key(n) not in set(view["player"].map(name_key))]
    view = view.sort_values("war_p50", ascending=False)

    st.markdown(f"## {team}: projected {season_label(NXT)} production")
    c1, c2, c3 = st.columns(3)
    c1.metric("Projected WAR (sum of P50)", f"{view['war_p50'].sum():.1f}")
    c2.metric("Projected value (P50)", money(view["value_p50"].sum() * ncap))
    c3.metric("Salary of projected players", money((view["pct_cap"] * ncap).sum()))
    if missing:
        st.caption(f"Too few NBA minutes for a projection (rookies, two-way and deep-bench "
                   f"players): {', '.join(missing)}.")
    st.caption("Summing player medians ignores lineup fit and minutes competition; "
               "read the total as a rough capacity check, not a win projection.")

    v = view.sort_values("war_p50")
    fig = go.Figure()
    for _, r in v.iterrows():
        fig.add_trace(go.Scatter(x=[r["war_p10"], r["war_p90"]], y=[r["player"]] * 2, mode="lines",
                                 line=dict(color=PURPLE, width=8), opacity=0.35, showlegend=False,
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=v["war_p50"], y=v["player"], mode="markers",
                             marker=dict(color=PURPLE, size=11), showlegend=False,
                             customdata=v[["war_p10", "war_p90"]],
                             hovertemplate="%{y}<br>P50 %{x:.1f} WAR<br>P10 %{customdata[0]:.1f}, "
                                           "P90 %{customdata[1]:.1f}<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(height=max(320, 42 * len(v) + 80), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Wins above replacement (dot = P50, bar = P10 to P90)")
    st.plotly_chart(fig, width="stretch", theme="streamlit")

    st.dataframe(pd.DataFrame({
        "Player": view["player"], "Age": view["age"].astype(int), "Archetype": view["archetype"],
        "WAR P10": view["war_p10"].round(1), "WAR P50": view["war_p50"].round(1),
        "WAR P90": view["war_p90"].round(1), "Value P50": (view["value_p50"] * ncap).map(money),
        "Market value": (view["market_value"] * ncap).map(money),
        "Salary": (view["pct_cap"] * ncap).map(money),
        "Surplus P50": (view["surplus_p50"] * ncap).map(money), "Cost basis": view["cost_basis"],
        "Why the market values him": view["drivers"],
    }), hide_index=True, width="stretch")
    st.caption(f"Rosters and salaries: Spotrac {season_label(NXT)} cash tables. Add "
               f"data/rosters/contracts_{NXT}.csv to override any salary.")

# ---------------- target finder ----------------
with tab_targets:
    st.markdown("## Find players worth more than they cost")
    a, b, c = st.columns(3)
    lo, hi = a.slider("Salary band ($M)", 0, 70, (5, 25), help="Useful for salary matching in trades.")
    pos = b.multiselect("Positions", ["PG", "SG", "SF", "PF", "C"], default=[])
    max_age = c.slider("Maximum age", 19, 40, 31)
    safe = st.checkbox("Only players with positive surplus even in the P10 (downside) case", value=False)
    t = proj[proj["surplus_p50"].notna() & (proj["team"] != team)
             & ~proj["team"].isin(["Unsigned", "Dead money only"])].copy()
    dollars = t["pct_cap"] * ncap
    t = t[dollars.between(lo * 1e6, hi * 1e6) & (t["age"] <= max_age)]
    if pos:
        t = t[t["pos"].isin(pos)]
    if safe:
        t = t[t["surplus_p10"] > 0]
    t = t.sort_values("surplus_p50", ascending=False).head(40)
    if t.empty:
        st.info("No players match. Widen the salary band or raise the age limit.")
    else:
        st.dataframe(pd.DataFrame({
            "Player": t["player"], "Team": t["team"], "Pos": t["pos"], "Age": t["age"].astype(int),
            "Archetype": t["archetype"], "Cost": (t["pct_cap"] * ncap).map(money),
            "Value P50": (t["value_p50"] * ncap).map(money),
            "Surplus P10": (t["surplus_p10"] * ncap).map(money),
            "Surplus P50": (t["surplus_p50"] * ncap).map(money),
            "Contract": t["contract_type"],
        }), hide_index=True, width="stretch")
        st.caption("Rookie-scale players are cheap by rule; their surplus is real but their "
                   "acquisition cost in a trade is not captured here.")

# ---------------- value board ----------------
with tab_board:
    st.markdown(f"## Who delivered value in {season_label(CUR)}")
    a, b = st.columns(2)
    which = a.radio("Compare salary with", ["Realized value (wins produced)", "Market value (track record)"],
                    horizontal=False)
    ct = b.multiselect("Contract type", ["Veteran", "Rookie scale"], default=["Veteran", "Rookie scale"])
    healthy = a.checkbox("Value injury-shortened players at healthy pace", value=False,
                         help="Adds the wins a player would have produced over games he missed.")
    hide_inj = b.checkbox("Hide injury-shortened seasons (played < half his team's games)", value=False)
    bb = board[board["salary"].notna() & (board["mp"] >= SETTINGS.min_minutes_value) & board["contract_type"].isin(ct)].copy()
    if hide_inj:
        bb = bb[bb["injury_shortened"] != 1]
    if which.startswith("Realized"):
        col, surplus = ("healthy_value", "healthy_surplus") if healthy else ("production_value", "realized_surplus")
    else:
        col, surplus = "market_value", "market_surplus"
    sel = bb["team"] == team  # 2025-26 team
    fig = go.Figure()
    for mask, colr, name in [(~sel, MUTED, "League"), (sel, PURPLE, team)]:
        s = bb[mask]
        fig.add_trace(go.Scatter(x=s[col] * cap / 1e6, y=s["salary"] / 1e6, mode="markers", name=name,
                                 marker=dict(color=colr, size=8 if name == team else 6,
                                             opacity=0.9 if name == team else 0.45),
                                 text=s["player"], hovertemplate="%{text}<br>value $%{x:.1f}M"
                                                                 "<br>salary $%{y:.1f}M<extra></extra>"))
    m = max(bb[col].max() * cap, bb["salary"].max()) / 1e6
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", line=dict(color=MUTED, dash="dash"),
                             name="Paid at value", hoverinfo="skip"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=f"{which.split(' (')[0]} ($M)", yaxis_title="Salary ($M)")
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    st.caption("Below the dashed line: the player was worth more than his salary. Realized value "
               "counts availability, so injury-shortened seasons score low.")
    st.dataframe(pd.DataFrame({
        "Player": bb["player"], "Team": bb["team"], "Contract": bb["contract_type"],
        "Salary": bb["salary"].map(money), "Market value": (bb["market_value"] * cap).map(money),
        "Realized value": (bb["production_value"] * cap).map(money), "WAR": bb["war"].round(1),
        "Games": bb["g"].astype(int).astype(str) + "/" + bb["team_games"].astype(int).astype(str)
                 + bb["injury_shortened"].map({1: " ⚕", 0: ""}),
        "Availability cost": (bb["availability_cost"] * cap).map(money),
        "Surplus": (bb[surplus] * cap).map(money), "_s": bb[surplus],
    }).sort_values("_s", ascending=False).drop(columns="_s"), hide_index=True, width="stretch")

# ---------------- player history ----------------
with tab_player:
    h = d["player_history"]
    players = sorted(h.loc[h["season"] >= 2016, "player"].unique())
    default = players.index("Austin Reaves") if "Austin Reaves" in players else 0
    who = st.selectbox("Player", players, index=default)
    ph = h[(h["player"] == who)].sort_values("season")
    ph = ph.assign(label=ph["season"].map(season_label), cap=ph["season"].map(SALARY_CAP))
    st.markdown(f"## {who}: salary vs value produced")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ph["label"], y=ph["salary"] / 1e6, name="Salary", mode="lines+markers",
                             line=dict(color=ORANGE, width=3)))
    fig.add_trace(go.Scatter(x=ph["label"], y=ph["production_value"] * ph["cap"] / 1e6,
                             name="Value of wins produced", mode="lines+markers",
                             line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=ph["label"], y=ph["healthy_value"] * ph["cap"] / 1e6,
                             name="Value at healthy pace", mode="lines",
                             line=dict(color=BLUE, width=1.5, dash="dash")))
    nxt = proj[proj["player"] == who]
    if len(nxt):
        r = nxt.iloc[0]
        fig.add_trace(go.Scatter(x=[season_label(NXT)] * 3,
                                 y=[r["value_p10"] * ncap / 1e6, r["value_p50"] * ncap / 1e6,
                                    r["value_p90"] * ncap / 1e6],
                                 mode="lines+markers", name=f"{season_label(NXT)} projection (P10–P90)",
                                 line=dict(color=BLUE, dash="dot"), marker=dict(size=[6, 11, 6])))
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="$M (that season's cap)")
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    st.dataframe(pd.DataFrame({"Season": ph["label"], "Team": ph["team"], "Age": ph["age"],
                               "Archetype": ph["archetype"], "Minutes": ph["mp"],
                               "Games": ph["g"].astype(int).astype(str) + "/" + ph["team_games"].astype(int).astype(str)
                                        + ph["injury_shortened"].map({1: " ⚕", 0: ""}),
                               "WAR": ph["war"].round(1), "Salary": ph["salary"].map(money),
                               "Value produced": (ph["production_value"] * ph["cap"]).map(money)}),
                 hide_index=True, width="stretch")

# ---------------- model card ----------------
with tab_model:
    st.markdown("## How the models work and how well they hold up")
    st.markdown(
        "- **Market value** — gradient-boosted trees trained on veteran contracts only, using stats "
        "from the two prior seasons. Predicts what the open market pays for a track record.\n"
        "- **Realized value** — wins above replacement (VORP × 2.7) priced at that season's league "
        "cost of a win.\n"
        "- **Projections** — quantile gradient boosting for next-season WAR at P10, P50, P90.\n\n"
        "Every season below was scored by a model trained only on earlier seasons.")
    mb = d["market_backtest"]
    agg = mb.groupby("model")[["mae_dollars", "r2", "spearman"]].mean().sort_values("mae_dollars")
    st.dataframe(pd.DataFrame({"Market model": agg.index, "Avg. error": agg["mae_dollars"].map(money),
                               "R²": agg["r2"].round(3), "Rank corr.": agg["spearman"].round(3)}),
                 hide_index=True)
    fb = d["forecast_backtest"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast error (wins)", f"{fb['mae_xgb_p50'].mean():.2f}",
              f"{fb['mae_xgb_p50'].mean() - fb['mae_marcel'].mean():+.2f} vs Marcel", delta_color="inverse")
    c2.metric("P10–P90 coverage", f"{fb['coverage_p10_p90'].mean():.0%}", "target 80%", delta_color="off")
    c3.metric("Seasons beating Marcel", f"{(fb['mae_xgb_p50'] < fb['mae_marcel']).sum()} of {len(fb)}")
    shap = d["shap_importance"]["mean_abs_shap"].sort_values(ascending=False).head(10)[::-1]
    fig = go.Figure(go.Bar(x=shap.values * 100, y=[label(i) for i in shap.index], orientation="h",
                           marker_color=BLUE))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      title="What drives market value (mean |SHAP|, % of cap)")
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    ap = d["archetype_pricing"]
    st.markdown("**Which roles the market under- or over-prices** (realized value minus market price)")
    st.dataframe(pd.DataFrame({"Archetype": ap["archetype"], "Player-seasons": ap["n"],
                               "Mean gap": (ap["mean_gap_pct_cap"] * ncap).map(money),
                               "90% CI low": (ap["ci_low"] * ncap).map(money),
                               "90% CI high": (ap["ci_high"] * ncap).map(money),
                               "Examples": ap["example_players_2026"]}), hide_index=True,
                 width="stretch")
    st.caption("Realized value is box-score based, which tends to credit on-ball and rim roles more "
               "than off-ball shooting and defense. Treat role-level gaps as hypotheses to test with "
               "tracking or on/off data.")
