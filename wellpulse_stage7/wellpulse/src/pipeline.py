"""Volve soft sensor: loading, cleaning, features, split, metrics."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/Volve_production_data.xlsx")
SHEET = "Daily Production Data"
TARGET = "AVG_DOWNHOLE_PRESSURE"
FEATURES = [
    "AVG_WHP_P", "AVG_WHT_P", "AVG_CHOKE_SIZE_P",
    "ON_STREAM_HRS", "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL",
]
FEATURES_V2 = FEATURES + ["days_on_prod", "cum_oil", "cum_water", "water_cut"]


def load_daily(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET)
    df = df.rename(columns={"WELL_BORE_CODE": "well", "DATEPRD": "date"})
    return df.sort_values(["well", "date"]).reset_index(drop=True)


def clean_for_soft_sensor(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["ON_STREAM_HRS"] > 0]
    out = out[out[TARGET] > 0]
    out = out[out["BORE_WAT_VOL"].fillna(0) >= 0]
    out = out.dropna(subset=FEATURES + [TARGET])
    return out[["well", "date", TARGET] + FEATURES].reset_index(drop=True)


def add_history_features(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    r = raw.sort_values(["well", "date"]).copy()
    r["oil0"] = r["BORE_OIL_VOL"].fillna(0).clip(lower=0)
    r["wat0"] = r["BORE_WAT_VOL"].fillna(0).clip(lower=0)
    r["cum_oil"] = r.groupby("well")["oil0"].cumsum()
    r["cum_water"] = r.groupby("well")["wat0"].cumsum()
    first_day = r[r["ON_STREAM_HRS"] > 0].groupby("well")["date"].min()
    r["days_on_prod"] = (r["date"] - r["well"].map(first_day)).dt.days
    out = clean.merge(r[["well", "date", "cum_oil", "cum_water", "days_on_prod"]],
                      on=["well", "date"], how="left")
    liquid = out["BORE_OIL_VOL"] + out["BORE_WAT_VOL"]
    out["water_cut"] = (out["BORE_WAT_VOL"] / liquid.where(liquid > 0)).fillna(0)
    return out


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values(["well", "date"])
    rank = df.groupby("well").cumcount() / df.groupby("well")["date"].transform("size")
    is_test = rank >= 1 - test_frac
    return df[~is_test], df[is_test]
