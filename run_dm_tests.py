from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from epftoolbox.evaluation import DM

HOUR_COLS = [f"y_h{i:02d}" for i in range(1, 25)]
MARKETS = ["DE_LU", "FR", "ES"]
BASE_DIR = Path(r"C:\Users\Manuel Hilgers\OneDrive\Desktop\Uni\Projektarbeit")

FORECAST_PATHS = {
    "LEAR": {
        "DE_LU": BASE_DIR / r"Modellcodes Python\LEAR\results_lear\DE_LU_lear_forecasts.csv",
        "FR": BASE_DIR / r"Modellcodes Python\LEAR\results_lear\FR_lear_forecasts.csv",
        "ES": BASE_DIR / r"Modellcodes Python\LEAR\results_lear\ES_lear_forecasts.csv",
    },
    "DNN": {
        "DE_LU": BASE_DIR / r"Modellcodes Python\DNN\forecast_files\DNN_forecast_nl2_datDE_LU_YT2_SFH1_CW4_DE_LU_final.csv",
        "FR": BASE_DIR / r"Modellcodes Python\DNN\forecast_files\DNN_forecast_nl2_datFR_YT2_SFH1_CW4_FR_final.csv",
        "ES": BASE_DIR / r"Modellcodes Python\DNN\forecast_files\DNN_forecast_nl2_datES_YT2_SFH1_CW4_ES_final.csv",
    },
    "RF": {
        "DE_LU": BASE_DIR / r"Modellcodes Python\RF\outputs\RF_rolling_forecast_DE_LU_final.csv",
        "FR": BASE_DIR / r"Modellcodes Python\RF\outputs\RF_rolling_forecast_FR_final.csv",
        "ES": BASE_DIR / r"Modellcodes Python\RF\outputs\RF_rolling_forecast_ES_final.csv",
    },
    "SVR": {
        "DE_LU": BASE_DIR / r"Modellcodes Python\SVR\outputs\SVR_rolling_forecast_DE_LU_final.csv",
        "FR": BASE_DIR / r"Modellcodes Python\SVR\outputs\SVR_rolling_forecast_FR_final.csv",
        "ES": BASE_DIR / r"Modellcodes Python\SVR\outputs\SVR_rolling_forecast_ES_final.csv",
    },
}

TEST_PATHS = {
    "DE_LU": BASE_DIR / r"data_splits_v2\DE_LU_test.csv",
    "FR": BASE_DIR / r"data_splits_v2\FR_test.csv",
    "ES": BASE_DIR / r"data_splits_v2\ES_test.csv",
}


def normalize_forecast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = df.columns.tolist()

    if "forecast_day" in df.columns:
        out = df.copy()
        out["forecast_day"] = pd.to_datetime(out["forecast_day"])
        dnn_style = False
    elif "Date" in df.columns:
        out = df.copy()
        out["forecast_day"] = pd.to_datetime(out["Date"])
        dnn_style = "h0" in cols
    else:
        raise ValueError(f"Keine Datums-Spalte gefunden in {path.name}")

    rename = {}
    for col in cols:
        if col.startswith("y_h"):
            rename[col] = col
        elif col.startswith("h") and col[1:].isdigit():
            num = int(col[1:])
            if dnn_style:
                rename[col] = f"y_h{num + 1:02d}"   # h0..h23
            else:
                rename[col] = f"y_h{num:02d}"       # h01..h24

    out = out.rename(columns=rename)

    needed = ["forecast_day"] + HOUR_COLS
    missing = [col for col in needed if col not in out.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in {path.name}: {missing}")

    return out[needed].sort_values("forecast_day").reset_index(drop=True)


def load_actual_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["forecast_day"] = df["Date"].dt.floor("D")
    df["hour_idx"] = df["Date"].dt.hour + 1

    piv = df.pivot(index="forecast_day", columns="hour_idx", values="Price").sort_index()

    for h in range(1, 25):
        if h not in piv.columns:
            piv[h] = np.nan

    piv = piv.reindex(sorted(piv.columns), axis=1)
    piv.columns = [f"y_h{hour:02d}" for hour in piv.columns]
    piv = piv[HOUR_COLS]

    return piv.reset_index()


def align_market(market: str):
    actual = load_actual_daily(TEST_PATHS[market])

    forecasts = {
        model: normalize_forecast(FORECAST_PATHS[model][market])
        for model in ["LEAR", "DNN", "RF", "SVR"]
    }

    common_dates = set(actual["forecast_day"])
    for df in forecasts.values():
        common_dates &= set(df["forecast_day"])

    common_dates = sorted(common_dates)
    common_idx = pd.DatetimeIndex(common_dates)

    actual_eval = (
        actual[actual["forecast_day"].isin(common_idx)]
        .copy()
        .sort_values("forecast_day")
        .reset_index(drop=True)
    )

    forecast_eval = {}
    for model, df in forecasts.items():
        forecast_eval[model] = (
            df[df["forecast_day"].isin(common_idx)]
            .copy()
            .sort_values("forecast_day")
            .reset_index(drop=True)
        )

    return actual_eval, forecast_eval


rows = []

for market in MARKETS:
    actual_eval, fc = align_market(market)

    p_real = actual_eval[HOUR_COLS].to_numpy()

    comparisons = [
        ("DNN", "LEAR"),
        ("RF", "LEAR"),
        ("SVR", "LEAR"),
    ]

    for model_a, model_b in comparisons:
        # WICHTIG:
        # DM prüft einseitig, ob p_pred_2 signifikant besser ist als p_pred_1
        p_pred_1 = fc[model_a][HOUR_COLS].to_numpy()
        p_pred_2 = fc[model_b][HOUR_COLS].to_numpy()

        p_value = DM(
            p_real=p_real,
            p_pred_1=p_pred_1,
            p_pred_2=p_pred_2,
            norm=1,
            version="multivariate",
        )

        rows.append(
            {
                "market": market,
                "comparison": f"{model_b} vs. {model_a}",
                "tested_better_model": model_b,
                "reference_model": model_a,
                "p_value": float(p_value),
                "significant_at_5pct": bool(float(p_value) < 0.05),
                "start_date": actual_eval["forecast_day"].min().date().isoformat(),
                "end_date": actual_eval["forecast_day"].max().date().isoformat(),
                "n_days": len(actual_eval),
            }
        )

out = pd.DataFrame(rows).sort_values(["market", "comparison"]).reset_index(drop=True)

out_path = BASE_DIR / r"Modellcodes Python\evaluation_outputs\dm_test_results_multivariate_norm1.csv"
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print(f"\nGespeichert unter: {out_path}")