"""Entirely fictional, reproducible demonstration data; no real player records."""
import numpy as np
import pandas as pd


def make_sample_data():
    rng = np.random.default_rng(42)
    rows = []
    for year in range(2022, 2026):
        for i in range(36):
            minutes = int(rng.integers(100, 2900))
            row = {"연도": str(year), "팀명": f"가상구단 {i % 3 + 1}",
                   "선수명": f"가상선수 {i + 1:02d}", "포지션": ["FW", "MF", "DF"][i // 12],
                   "출전시간": minutes}
            for metric, mean in {"Goal_Qty": 0.25, "SHOT": 2.5, "Assist_Qty": 0.15,
                                 "DRIBBLE": 3, "DRIBBLE_SUCCESS": 1.5, "PASS": 40,
                                 "PASS_ACC": 30, "INTERCEPT": 2, "TACKLE_SUCCESS": 2,
                                 "AIR_CHALG_WON": 2, "BLOCKING_BALL": 1, "BALL_MISS": 2}.items():
                total = int(rng.poisson(mean * minutes / 90))
                row[metric] = total
                row["Per_" + metric] = total * 90 / minutes
            rows.append(row)
    return pd.DataFrame(rows)
