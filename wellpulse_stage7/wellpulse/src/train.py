"""Train final model, out-of-fold residuals, anomaly flags; save artifacts for the dashboard."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.pipeline import (FEATURES_V2, TARGET, add_history_features,
                          clean_for_soft_sensor, load_daily, metrics, time_split)


def lowo_mean_rmse(data):
    rmses = []
    for w in data["well"].unique():
        tr, te = data[data["well"] != w], data[data["well"] == w]
        m = HistGradientBoostingRegressor(random_state=42).fit(tr[FEATURES_V2], tr[TARGET])
        rmses.append(metrics(te[TARGET], m.predict(te[FEATURES_V2]))["RMSE"])
    return float(np.mean(rmses))


def main():
    raw = load_daily()
    data = add_history_features(raw, clean_for_soft_sensor(raw))
    data = data.sort_values(["well", "date"]).reset_index(drop=True)

    rank = data.groupby("well").cumcount() / data.groupby("well")["date"].transform("size")
    data["fold"] = np.floor(rank * 5).astype(int).clip(upper=4)
    data["pred_oof"] = np.nan
    for k in range(5):
        tr_k, te_k = data[data["fold"] != k], data[data["fold"] == k]
        m = HistGradientBoostingRegressor(random_state=42).fit(tr_k[FEATURES_V2], tr_k[TARGET])
        data.loc[te_k.index, "pred_oof"] = m.predict(te_k[FEATURES_V2])

    data["residual"] = data[TARGET] - data["pred_oof"]
    med = float(data["residual"].median())
    sigma = float(1.4826 * (data["residual"] - med).abs().median())
    data["is_anomaly"] = (data["residual"] - med).abs() > 3 * sigma

    final = HistGradientBoostingRegressor(random_state=42).fit(data[FEATURES_V2], data[TARGET])
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES_V2, "target": TARGET}, "models/hgb_v2.joblib")
    data.to_csv("models/app_data.csv", index=False)   # contains Volve values: keep out of git

    tr, te = time_split(data)
    honest = HistGradientBoostingRegressor(random_state=42).fit(tr[FEATURES_V2], tr[TARGET])
    report = {
        "time_split": {k: round(v, 2) for k, v in metrics(te[TARGET], honest.predict(te[FEATURES_V2])).items()},
        "lowo_mean_rmse": round(lowo_mean_rmse(data), 1),
        "blocked_cv_oof": {k: round(v, 2) for k, v in metrics(data[TARGET], data["pred_oof"]).items()},
        "anomaly_sigma_bar": round(sigma, 2),
        "anomaly_median_residual": round(med, 2),
        "rows": int(len(data)),
    }
    Path("models/metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
