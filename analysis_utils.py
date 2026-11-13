"""Pure analysis helpers shared by the app and regression checks."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def standardize_metrics(data, features):
    numeric = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median()).fillna(0)
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] == 0:
        return pd.DataFrame(index=data.index)
    return pd.DataFrame(StandardScaler().fit_transform(numeric), index=data.index, columns=numeric.columns)


def get_player_history(data, name):
    return data.loc[data["선수명"] == name].sort_values(["연도", "팀명"]).copy()


def recommend_players(data, player, limit=5):
    candidates = data.loc[
        (data["선수명"] != player["선수명"])
        & (data["연도"] == player["연도"])
        & (data["POSITION_GROUP"] == player["POSITION_GROUP"])
        & (data["포지션"] == player["포지션"])
    ].copy()
    features = [c for c in data if c.startswith("ML_") and pd.notna(player.get(c))]
    if candidates.empty or not features:
        return pd.DataFrame()
    target = player[features].to_numpy(dtype=float).reshape(1, -1)
    vectors = candidates[features].to_numpy(dtype=float)
    valid = np.isfinite(vectors).all(axis=1) & (np.linalg.norm(vectors, axis=1) > 1e-12)
    if not np.isfinite(target).all() or np.linalg.norm(target) <= 1e-12:
        return pd.DataFrame()
    candidates = candidates.loc[valid].copy()
    if candidates.empty:
        return candidates
    candidates["코사인 유사도"] = cosine_similarity(target, vectors[valid])[0]
    candidates["_same_cluster"] = candidates["CLUSTER"] == player["CLUSTER"]
    return candidates.sort_values(["_same_cluster", "코사인 유사도", "SCOUTING_SCORE"], ascending=False).head(limit).drop(columns="_same_cluster")
