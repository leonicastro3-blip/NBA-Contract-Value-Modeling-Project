"""Player archetypes from statistical role profiles, and how the market prices them.

KMeans on standardized rate stats (usage, creation, rebounding, defense
events, shot profile, efficiency). Roles are a continuum, so cluster
separation is modest (silhouette ~0.12); the clusters are used for
interpretation, not as hard labels.

Cluster names are assigned by matching each centroid to a prototype profile
with the Hungarian algorithm, so names stay stable if the model is refit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ncv.config import SETTINGS

ROLE_STATS = ["usg_percent", "ast_percent", "orb_percent", "drb_percent", "stl_percent",
              "blk_percent", "x3p_ar", "f_tr", "ts_percent", "tov_percent"]

# Expected direction of each stat (in ROLE_STATS order) for each archetype.
PROTOTYPES = {
    "Primary creator":          [2, 2, -0.5, 0, 0.3, -0.4, 0, 0.7, 0.4, 0.2],
    "Secondary playmaker":      [0, 1.3, -0.6, -0.6, 0.8, -0.6, 0, -0.3, -0.8, 1.5],
    "Point-of-attack defender": [-0.7, -0.4, 0, -0.3, 1.4, 0, 0.2, -0.3, -0.4, 0],
    "Spot-up shooter":          [-0.8, -0.6, -0.6, -0.6, -0.4, -0.5, 1.2, -0.9, 0.1, -0.8],
    "Scoring guard":            [0.4, 0.2, -0.7, -0.7, -0.2, -0.6, 0.2, -0.3, -0.4, -0.3],
    "Two-way forward":          [0, -0.5, 0.2, 0.5, -0.5, 0.2, 0.2, 0, 0.2, -0.5],
    "Rim-running big":          [-0.7, -0.7, 2, 1.3, -0.4, 1.8, -1.5, 1.6, 1.4, 0.6],
    "Interior hub big":         [0.2, -0.4, 1.3, 1.3, -0.3, 0.9, -1.3, 0.4, 0, 0.2],
}


class ArchetypeModel:
    def __init__(self, k: int = SETTINGS.n_archetypes, first_season: int = 2014,
                 min_minutes: int = SETTINGS.min_minutes_value):
        self.k, self.first_season, self.min_minutes = k, first_season, min_minutes
        self.scaler = StandardScaler()
        self.km = KMeans(k, n_init=20, random_state=SETTINGS.random_state)
        self.names: dict[int, str] = {}

    def fit(self, season_table: pd.DataFrame) -> "ArchetypeModel":
        x = season_table[(season_table["season"] >= self.first_season)
                         & (season_table["mp"] >= self.min_minutes)].dropna(subset=ROLE_STATS)
        z = self.scaler.fit_transform(x[ROLE_STATS])
        self.km.fit(z)
        protos = np.array(list(PROTOTYPES.values()))
        cen = self.km.cluster_centers_
        sim = (cen / np.linalg.norm(cen, axis=1, keepdims=True)) @ \
              (protos / np.linalg.norm(protos, axis=1, keepdims=True)).T
        rows, cols = linear_sum_assignment(-sim)
        names = list(PROTOTYPES)
        self.names = {int(r): names[c] for r, c in zip(rows, cols)}
        return self

    def assign(self, df: pd.DataFrame) -> pd.Series:
        ok = df[ROLE_STATS].notna().all(axis=1)
        out = pd.Series(pd.NA, index=df.index, dtype="object")
        labels = self.km.predict(self.scaler.transform(df.loc[ok, ROLE_STATS]))
        out.loc[ok] = [self.names[int(c)] for c in labels]
        return out

    def centroids(self) -> pd.DataFrame:
        c = pd.DataFrame(self.km.cluster_centers_, columns=ROLE_STATS)
        c.index = [self.names[i] for i in range(self.k)]
        return c


def pricing_by_archetype(df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Mean gap between realized production value and ex-ante market value.

    Positive gap = the market's pricing of this role's track record
    under-anticipates the production it delivers. Bootstrap 90% CIs.
    """
    rng = np.random.default_rng(SETTINGS.random_state)
    out = []
    for name, g in df.groupby("archetype"):
        gap = (g["production_value"] - g["market_value"]).to_numpy()
        boots = rng.choice(gap, size=(n_boot, len(gap)), replace=True).mean(axis=1)
        out.append({
            "archetype": name, "n": len(gap),
            "mean_gap_pct_cap": gap.mean(),
            "ci_low": np.quantile(boots, 0.05), "ci_high": np.quantile(boots, 0.95),
            "median_market_value": g["market_value"].median(),
            "median_production_value": g["production_value"].median(),
            "example_players_2026": ", ".join(
                g[g["season"] == g["season"].max()].nlargest(4, "mp_T")["player"]),
        })
    return pd.DataFrame(out).sort_values("mean_gap_pct_cap", ascending=False).reset_index(drop=True)
