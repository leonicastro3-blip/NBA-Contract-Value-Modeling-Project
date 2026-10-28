import numpy as np
import pandas as pd
import pytest

from ncv.analysis.roster import find_targets
from ncv.data.cleaning import collapse_multi_team, name_key
from ncv.data.panel import FEATURES, attach_targets, build_features, is_rookie_scale
from ncv.models import forecast, market
from ncv.models.production import league_win_prices, production_value


# ---------- cleaning ----------
@pytest.mark.parametrize("raw,expected", [
    ("Nikola Jokić", "nikola jokic"),
    ("Jaren Jackson Jr.", "jaren jackson"),
    ("GG Jackson II", "gregory jackson"),          # verified alias
    ("D'Angelo Russell", "dangelo russell"),
    ("  Luka   Dončić ", "luka doncic"),
])
def test_name_key(raw, expected):
    assert name_key(raw) == expected


def test_collapse_multi_team_keeps_total_row_and_final_team():
    df = pd.DataFrame({
        "player_id": ["a", "a", "a", "b"], "season": [2020] * 4,
        "team": ["2TM", "NOP", "SAC", "LAL"], "g": [82, 57, 25, 70],
    })
    out = collapse_multi_team(df).set_index("player_id")
    assert len(out) == 2
    assert out.loc["a", "g"] == 82          # season total, not a single stint
    assert out.loc["a", "team"] == "SAC"    # final team of the season
    assert out.loc["a", "n_teams"] == 2


# ---------- panel ----------
def test_rookie_scale_rule():
    r = pd.Series([1, 1, 2, np.nan])
    e = pd.Series([4, 5, 1, 1])
    assert is_rookie_scale(r, e).tolist() == [True, False, False, False]


def test_features_never_use_target_season(season_table):
    """Corrupting every season-T value must leave season-T features unchanged."""
    T = 2016
    base = build_features(season_table, [T]).set_index("player_id")[FEATURES]
    poisoned = season_table.copy()
    num = poisoned.select_dtypes("number").columns.difference(["season", "age", "experience",
                                                               "draft_round", "draft_pick"])
    poisoned.loc[poisoned["season"] >= T, num] = 9999.0
    after = build_features(poisoned, [T]).set_index("player_id")[FEATURES]
    pd.testing.assert_frame_equal(base.sort_index(), after.sort_index())


def test_injury_return_rows_kept(season_table):
    st = season_table.copy()
    pid = st["player_id"].iloc[0]
    st.loc[(st["player_id"] == pid) & (st["season"] == 2014), "mp"] = 2000
    st.loc[(st["player_id"] == pid) & (st["season"] == 2015), "mp"] = 100   # injured year
    f = build_features(st, [2016])
    assert pid in set(f["player_id"])


def test_age_and_experience_advance_one_year(season_table):
    f = build_features(season_table, [2015]).set_index("player_id")
    prev = season_table[season_table["season"] == 2014].set_index("player_id")
    common = f.index.intersection(prev.index)
    assert (f.loc[common, "age"] == prev.loc[common, "age"] + 1).all()


# ---------- market model ----------
def _panel(st):
    return attach_targets(build_features(st, list(range(2011, 2019))), st)


def test_market_backtest_has_no_lookahead(season_table):
    """Changing salaries in the test season or later must not change its predictions."""
    p = _panel(season_table)
    a = market.backtest(p, seasons=(2017,)).predictions.set_index("player_id")["market_value"]
    p2 = p.copy()
    p2.loc[p2["season"] >= 2017, "pct_cap"] *= 3
    b = market.backtest(p2, seasons=(2017,)).predictions.set_index("player_id")["market_value"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_market_excludes_rookie_scale_from_training(season_table):
    rows = market.market_training_rows(_panel(season_table))
    assert not rows["is_rookie_scale"].any()


# ---------- forecast ----------
def test_quantiles_do_not_cross(season_table):
    p = _panel(season_table)
    m = forecast.fit_final(p, through_season=2017)
    q = forecast.predict_quantiles(m, p[p["season"] == 2018])
    assert (q["war_p10"] <= q["war_p50"]).all() and (q["war_p50"] <= q["war_p90"]).all()


def test_marcel_age_adjustment_direction():
    X = pd.DataFrame({"l1_vorp": [1.0, 1.0], "l2_vorp": [1.0, 1.0], "age": [23, 35]})
    m = forecast.marcel(X, league_mean=2.7)
    assert m.iloc[0] > m.iloc[1]   # young players project to improve, old to decline


def test_pinball_loss_zero_when_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert forecast.pinball(y, y, 0.5) == 0


# ---------- production value ----------
def test_production_value_identity(season_table):
    st = season_table[season_table["season"] >= 2010]
    prices = league_win_prices(st)
    s = pd.Series([2015, 2015])
    v = production_value(pd.Series([0.0, 5.0]), s, prices)
    rep = prices.set_index("season").loc[2015]
    assert v.iloc[0] == pytest.approx(rep["rep_pct_cap"])
    assert v.iloc[1] == pytest.approx(rep["rep_pct_cap"] + 5 * rep["pct_cap_per_war"])
    assert production_value(pd.Series([-50.0]), pd.Series([2015]), prices).iloc[0] == 0


# ---------- roster tools ----------
def test_find_targets_filters():
    proj = pd.DataFrame({
        "player": list("abcd"), "team": ["LAL", "BOS", "BOS", "NYK"], "pos": ["C", "C", "PG", "C"],
        "age": [25, 26, 24, 33], "pct_cap": [0.1, 0.1, 0.1, 0.1], "salary_cap": [1e8] * 4,
        "surplus_p10": [0.05, 0.02, 0.03, -0.01], "surplus_p50": [0.1, 0.08, 0.09, 0.05],
    })
    out = find_targets(proj, min_salary=5e6, max_salary=15e6, positions=["C"], max_age=30,
                       require_positive_p10=True, exclude_team="LAL")
    assert out["player"].tolist() == ["b"]


# ---------------- availability ----------------
def _avail_frame():
    from ncv.data.panel import add_availability
    rows = []
    for pid, g, mp, season, war, tg in [
        ("full", 82, 2400, 2024, 3.0, 82), ("full", 82, 2400, 2025, 3.0, 82),
        ("hurt", 82, 2600, 2024, 4.0, 82), ("hurt", 30, 900, 2025, 1.5, 82),
        ("bench", 30, 300, 2025, 0.2, 82),                    # low-minute: not flagged
        ("short", 36, 1000, 2021, 2.0, 72),                   # 36/72 = 50%: not flagged (< is strict)
        ("short2", 70, 2100, 2021, 2.0, 72),
        ("neg", 40, 1000, 2025, -1.0, 82),                    # negative rate floors at 0
    ]:
        rows.append(dict(player_id=pid, season=season, g=g, mp=mp, mp_per_game=mp / g, war=war,
                         team_games=tg))
    return add_availability(pd.DataFrame(rows))


def test_full_season_has_zero_availability_cost():
    a = _avail_frame()
    r = a[(a.player_id == "full") & (a.season == 2025)].iloc[0]
    assert r.games_missed == 0 and r.healthy_war == pytest.approx(r.war)


def test_injury_flag_uses_team_games_and_rotation_status():
    a = _avail_frame().set_index(["player_id", "season"])
    assert a.loc[("hurt", 2025), "injury_shortened"] == 1
    assert a.loc[("bench", 2025), "injury_shortened"] == 0      # not a rotation player
    assert a.loc[("short", 2021), "injury_shortened"] == 0      # shortened season, exactly half
    assert a.loc[("short2", 2021), "games_pct"] == pytest.approx(70 / 72)


def test_healthy_war_never_below_realized_war():
    a = _avail_frame()
    assert (a["healthy_war"] >= a["war"] - 1e-9).all()
    n = a[a.player_id == "neg"].iloc[0]
    assert n.healthy_war == pytest.approx(n.war)               # negative-rate player: no gain


def test_healthy_war_shrinks_toward_prior_rate():
    a = _avail_frame().set_index(["player_id", "season"])
    r = a.loc[("hurt", 2025)]
    raw = 1.5 / 30
    prior = 4.0 / 82
    assert min(raw, prior) <= r.healthy_rate <= max(raw, prior)
    assert r.healthy_war > r.war
