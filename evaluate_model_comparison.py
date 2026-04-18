from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HOUR_COLS = [f"y_h{i:02d}" for i in range(1, 25)]
MODELS = ["LEAR", "DNN", "RF", "SVR"]
MARKETS = ["DE_LU", "FR", "ES"]

# =========================================
# FESTE DATEIPFADE
# =========================================

PROJECT_DIR = Path(r"C:\Users\Manuel Hilgers\OneDrive\Desktop\Uni\Projektarbeit")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "evaluation_outputs"

FORECAST_PATHS = {
    "LEAR": {
        "DE_LU": PROJECT_DIR / r"Modellcodes Python\LEAR\results_lear\DE_LU_lear_forecasts.csv",
        "FR": PROJECT_DIR / r"Modellcodes Python\LEAR\results_lear\FR_lear_forecasts.csv",
        "ES": PROJECT_DIR / r"Modellcodes Python\LEAR\results_lear\ES_lear_forecasts.csv",
    },
    "DNN": {
        "DE_LU": PROJECT_DIR / r"Modellcodes Python\DNN\forecast_files\DNN_forecast_nl2_datDE_LU_YT2_SFH1_CW4_DE_LU_final.csv",
        "FR": PROJECT_DIR / r"Modellcodes Python\DNN\forecast_files\DNN_forecast_nl2_datFR_YT2_SFH1_CW4_FR_final.csv",
        "ES": PROJECT_DIR / r"Modellcodes Python\DNN\forecast_files\DNN_forecast_nl2_datES_YT2_SFH1_CW4_ES_final.csv",
    },
    "RF": {
        "DE_LU": PROJECT_DIR / r"Modellcodes Python\RF\outputs\RF_rolling_forecast_DE_LU_final.csv",
        "FR": PROJECT_DIR / r"Modellcodes Python\RF\outputs\RF_rolling_forecast_FR_final.csv",
        "ES": PROJECT_DIR / r"Modellcodes Python\RF\outputs\RF_rolling_forecast_ES_final.csv",
    },
    "SVR": {
        "DE_LU": PROJECT_DIR / r"Modellcodes Python\SVR\outputs\SVR_rolling_forecast_DE_LU_final.csv",
        "FR": PROJECT_DIR / r"Modellcodes Python\SVR\outputs\SVR_rolling_forecast_FR_final.csv",
        "ES": PROJECT_DIR / r"Modellcodes Python\SVR\outputs\SVR_rolling_forecast_ES_final.csv",
    },
}

# WICHTIG: hier bewusst data_splits_v2 und NICHT data_clean
TEST_PATHS = {
    "DE_LU": PROJECT_DIR / r"data_splits_v2\DE_LU_test.csv",
    "FR": PROJECT_DIR / r"data_splits_v2\FR_test.csv",
    "ES": PROJECT_DIR / r"data_splits_v2\ES_test.csv",
}

# =========================================
# OPTIONALE FAPS-FARBEN
# =========================================

COLORS = {
    "ACTUAL": "#222222",   
    "LEAR": "#00549F",     
    "DNN": "#57ABDB",      
    "RF": "#7A1FA2",       
    "SVR": "#009B77",      
}


def ensure_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")


def normalize_forecast(path: Path) -> pd.DataFrame:
    ensure_exists(path)
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
                rename[col] = f"y_h{num + 1:02d}"   # h0..h23 -> y_h01..y_h24
            else:
                rename[col] = f"y_h{num:02d}"       # h01..h24 -> y_h01..y_h24

    out = out.rename(columns=rename)

    needed = ["forecast_day"] + HOUR_COLS
    missing = [col for col in needed if col not in out.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in {path.name}: {missing}")

    return out[needed].sort_values("forecast_day").reset_index(drop=True)


def load_actual_daily_from_hourly(path: Path) -> pd.DataFrame:
    ensure_exists(path)
    df = pd.read_csv(path)

    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError(f"Erwarte Spalten Date und Price in {path.name}")

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


def mae_arr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_arr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape_arr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)
    return float(np.mean(ratio) * 100.0)


def mape_arr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.abs(y_true)
    valid = den != 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0)


def flatten_daily_matrix(df: pd.DataFrame) -> pd.DataFrame:
    long_rows = []
    for _, row in df.iterrows():
        day = row["forecast_day"]
        for h in range(1, 25):
            ts = pd.Timestamp(day) + pd.Timedelta(hours=h - 1)
            long_rows.append({"DateTime": ts, "Price": row[f"y_h{h:02d}"]})
    return pd.DataFrame(long_rows)


def prepare_market(market: str):
    actual_full = load_actual_daily_from_hourly(TEST_PATHS[market])

    forecast_by_model = {
        model: normalize_forecast(FORECAST_PATHS[model][market])
        for model in MODELS
    }

    # gemeinsamer Zeitraum
    common_dates = set(actual_full["forecast_day"])
    for df in forecast_by_model.values():
        common_dates &= set(df["forecast_day"])

    # naive Wochenreferenz
    naive_base = actual_full[["forecast_day"] + HOUR_COLS].copy()
    naive_shift = naive_base.copy()
    naive_shift["forecast_day"] = naive_shift["forecast_day"] + pd.Timedelta(days=7)

    common_dates &= set(naive_shift["forecast_day"])
    common_dates = sorted(common_dates)
    common_idx = pd.DatetimeIndex(common_dates)

    actual_eval = (
        actual_full[actual_full["forecast_day"].isin(common_idx)]
        .copy()
        .sort_values("forecast_day")
        .reset_index(drop=True)
    )

    naive_shift = (
        naive_shift[naive_shift["forecast_day"].isin(common_idx)]
        .copy()
        .sort_values("forecast_day")
        .reset_index(drop=True)
        .rename(columns={col: f"{col}_naive" for col in HOUR_COLS})
    )

    metrics_rows = []
    monthly_rows = []
    aligned_forecasts = {}

    for model, forecast_df in forecast_by_model.items():
        fc = (
            forecast_df[forecast_df["forecast_day"].isin(common_idx)]
            .copy()
            .sort_values("forecast_day")
            .reset_index(drop=True)
        )
        aligned_forecasts[model] = fc

        merged = (
            actual_eval[["forecast_day"] + HOUR_COLS]
            .merge(fc, on="forecast_day", suffixes=("_true", "_pred"))
            .merge(naive_shift, on="forecast_day", how="inner")
        )

        y_true = merged[[f"{col}_true" for col in HOUR_COLS]].to_numpy()
        y_pred = merged[[f"{col}_pred" for col in HOUR_COLS]].to_numpy()
        naive = merged[[f"{col}_naive" for col in HOUR_COLS]].to_numpy()

        model_mae = mae_arr(y_true, y_pred)
        naive_mae = mae_arr(y_true, naive)

        metrics_rows.append(
            {
                "market": market,
                "model": model,
                "start_date": merged["forecast_day"].min().date().isoformat(),
                "end_date": merged["forecast_day"].max().date().isoformat(),
                "n_days": len(merged),
                "rMAE": model_mae / naive_mae,
                "MAE": model_mae,
                "RMSE": rmse_arr(y_true, y_pred),
                "sMAPE": smape_arr(y_true, y_pred),
                "MAPE": mape_arr(y_true, y_pred),
            }
        )

        months = merged["forecast_day"].dt.to_period("M").astype(str)
        for month in pd.unique(months):
            idx = np.where(months == month)[0]
            yt = y_true[idx]
            yp = y_pred[idx]
            nv = naive[idx]

            monthly_rows.append(
                {
                    "market": market,
                    "model": model,
                    "month": month,
                    "rMAE": mae_arr(yt, yp) / mae_arr(yt, nv),
                    "MAE": mae_arr(yt, yp),
                    "RMSE": rmse_arr(yt, yp),
                    "sMAPE": smape_arr(yt, yp),
                    "MAPE": mape_arr(yt, yp),
                }
            )

    return actual_eval, aligned_forecasts, pd.DataFrame(metrics_rows), pd.DataFrame(monthly_rows)


def make_market_timeseries_plot(
    out_dir: Path,
    market: str,
    actual_eval: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    days: int = 14,
):
    actual_window = actual_eval.head(days)
    actual_long = flatten_daily_matrix(actual_window)

    plt.figure(figsize=(16, 6))
    plt.plot(
        actual_long["DateTime"],
        actual_long["Price"],
        label="Ist-Wert",
        linewidth=2.4,
        color=COLORS["ACTUAL"],
    )

    for model in MODELS:
        fc_window = forecasts[model].head(days)
        fc_long = flatten_daily_matrix(fc_window)
        plt.plot(
            fc_long["DateTime"],
            fc_long["Price"],
            label=model,
            linewidth=1.7,
            color=COLORS[model],
        )

    plt.title(f"Modellvergleich: Ist-Wert vs. Prognosen ({market})")
    plt.xlabel("Zeit")
    plt.ylabel("Preis")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"{market}_timeseries_model_comparison.png", dpi=150)
    plt.close()


def make_monthly_plot(out_dir: Path, monthly_df: pd.DataFrame, market: str, metric: str):
    sub = monthly_df[monthly_df["market"] == market].copy()

    plt.figure(figsize=(14, 5))
    for model in MODELS:
        model_sub = sub[sub["model"] == model].copy()
        plt.plot(
            model_sub["month"],
            model_sub[metric],
            marker="o",
            label=model,
            color=COLORS[model],
        )

    plt.title(f"Monatlicher {metric}-Vergleich ({market})")
    plt.xlabel("Monat")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{market}_monthly_{metric.lower()}_comparison.png", dpi=150)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_monthly = []

    print("Verwendete Dateien:")
    for market in MARKETS:
        print(f"\n{market}")
        print(f"  TEST : {TEST_PATHS[market]}")
        for model in MODELS:
            print(f"  {model:4}: {FORECAST_PATHS[model][market]}")

    for market in MARKETS:
        actual_eval, forecasts, metrics_df, monthly_df = prepare_market(market)

        all_metrics.append(metrics_df)
        all_monthly.append(monthly_df)

        make_market_timeseries_plot(OUT_DIR, market, actual_eval, forecasts, days=14)
        make_monthly_plot(OUT_DIR, monthly_df, market, "MAE")
        make_monthly_plot(OUT_DIR, monthly_df, market, "rMAE")

    metrics = pd.concat(all_metrics, ignore_index=True)
    monthly = pd.concat(all_monthly, ignore_index=True)

    metrics["rank_rMAE_market"] = metrics.groupby("market")["rMAE"].rank(method="dense")
    metrics["rank_MAE_market"] = metrics.groupby("market")["MAE"].rank(method="dense")

    metrics = metrics.sort_values(["market", "rMAE", "MAE"]).reset_index(drop=True)
    monthly = monthly.sort_values(["market", "month", "model"]).reset_index(drop=True)

    metrics.to_csv(OUT_DIR / "model_comparison_metrics_common_horizon.csv", index=False)
    monthly.to_csv(OUT_DIR / "model_comparison_monthly_metrics_common_horizon.csv", index=False)

    print("\nDone.")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HOUR_COLS = [f"y_h{i:02d}" for i in range(1, 25)]
MODELS = ["LEAR", "DNN", "RF", "SVR"]
MARKETS = ["DE_LU", "FR", "ES"]

# =========================================
# FESTE DATEIPFADE
# =========================================

PROJECT_DIR = Path(r"C:\Users\Manuel Hilgers\OneDrive\Desktop\Uni\Projektarbeit")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "evaluation_outputs"

FORECAST_PATHS = {
    "LEAR": {
        "DE_LU": PROJECT_DIR / r"Modellcodes Python\LEAR\results_lear\DE_LU_lear_forecasts.csv",
        "FR": PROJECT_DIR / r"Modellcodes Python\LEAR\results_lear\FR_lear_forecasts.csv",
        "ES": PROJECT_DIR / r"Modellcodes Python\LEAR\results_lear\ES_lear_forecasts.csv",
    },
    "DNN": {
        "DE_LU": PROJECT_DIR / r"Modellcodes Python\DNN\forecast_files\DNN_forecast_nl2_datDE_LU_YT2_SFH1_CW4_DE_LU_final.csv",
        "FR": PROJECT_DIR / r"Modellcodes Python\DNN\forecast_files\DNN_forecast_nl2_datFR_YT2_SFH1_CW4_FR_final.csv",
        "ES": PROJECT_DIR / r"Modellcodes Python\DNN\forecast_files\DNN_forecast_nl2_datES_YT2_SFH1_CW4_ES_final.csv",
    },
    "RF": {
        "DE_LU": PROJECT_DIR / r"Modellcodes Python\RF\outputs\RF_rolling_forecast_DE_LU_final.csv",
        "FR": PROJECT_DIR / r"Modellcodes Python\RF\outputs\RF_rolling_forecast_FR_final.csv",
        "ES": PROJECT_DIR / r"Modellcodes Python\RF\outputs\RF_rolling_forecast_ES_final.csv",
    },
    "SVR": {
        "DE_LU": PROJECT_DIR / r"Modellcodes Python\SVR\outputs\SVR_rolling_forecast_DE_LU_final.csv",
        "FR": PROJECT_DIR / r"Modellcodes Python\SVR\outputs\SVR_rolling_forecast_FR_final.csv",
        "ES": PROJECT_DIR / r"Modellcodes Python\SVR\outputs\SVR_rolling_forecast_ES_final.csv",
    },
}

# WICHTIG: hier bewusst data_splits_v2 und NICHT data_clean
TEST_PATHS = {
    "DE_LU": PROJECT_DIR / r"data_splits_v2\DE_LU_test.csv",
    "FR": PROJECT_DIR / r"data_splits_v2\FR_test.csv",
    "ES": PROJECT_DIR / r"data_splits_v2\ES_test.csv",
}

# =========================================
# OPTIONALE FAPS-FARBEN
# =========================================

COLORS = {
    "ACTUAL": "#222222",   
    "LEAR": "#00549F",     
    "DNN": "#57ABDB",      
    "RF": "#7A1FA2",       
    "SVR": "#009B77",      
}


def ensure_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")


def normalize_forecast(path: Path) -> pd.DataFrame:
    ensure_exists(path)
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
                rename[col] = f"y_h{num + 1:02d}"   # h0..h23 -> y_h01..y_h24
            else:
                rename[col] = f"y_h{num:02d}"       # h01..h24 -> y_h01..y_h24

    out = out.rename(columns=rename)

    needed = ["forecast_day"] + HOUR_COLS
    missing = [col for col in needed if col not in out.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in {path.name}: {missing}")

    return out[needed].sort_values("forecast_day").reset_index(drop=True)


def load_actual_daily_from_hourly(path: Path) -> pd.DataFrame:
    ensure_exists(path)
    df = pd.read_csv(path)

    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError(f"Erwarte Spalten Date und Price in {path.name}")

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


def mae_arr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_arr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape_arr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)
    return float(np.mean(ratio) * 100.0)


def mape_arr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.abs(y_true)
    valid = den != 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0)


def flatten_daily_matrix(df: pd.DataFrame) -> pd.DataFrame:
    long_rows = []
    for _, row in df.iterrows():
        day = row["forecast_day"]
        for h in range(1, 25):
            ts = pd.Timestamp(day) + pd.Timedelta(hours=h - 1)
            long_rows.append({"DateTime": ts, "Price": row[f"y_h{h:02d}"]})
    return pd.DataFrame(long_rows)


def prepare_market(market: str):
    actual_full = load_actual_daily_from_hourly(TEST_PATHS[market])

    forecast_by_model = {
        model: normalize_forecast(FORECAST_PATHS[model][market])
        for model in MODELS
    }

    # gemeinsamer Zeitraum
    common_dates = set(actual_full["forecast_day"])
    for df in forecast_by_model.values():
        common_dates &= set(df["forecast_day"])

    # naive Wochenreferenz
    naive_base = actual_full[["forecast_day"] + HOUR_COLS].copy()
    naive_shift = naive_base.copy()
    naive_shift["forecast_day"] = naive_shift["forecast_day"] + pd.Timedelta(days=7)

    common_dates &= set(naive_shift["forecast_day"])
    common_dates = sorted(common_dates)
    common_idx = pd.DatetimeIndex(common_dates)

    actual_eval = (
        actual_full[actual_full["forecast_day"].isin(common_idx)]
        .copy()
        .sort_values("forecast_day")
        .reset_index(drop=True)
    )

    naive_shift = (
        naive_shift[naive_shift["forecast_day"].isin(common_idx)]
        .copy()
        .sort_values("forecast_day")
        .reset_index(drop=True)
        .rename(columns={col: f"{col}_naive" for col in HOUR_COLS})
    )

    metrics_rows = []
    monthly_rows = []
    aligned_forecasts = {}

    for model, forecast_df in forecast_by_model.items():
        fc = (
            forecast_df[forecast_df["forecast_day"].isin(common_idx)]
            .copy()
            .sort_values("forecast_day")
            .reset_index(drop=True)
        )
        aligned_forecasts[model] = fc

        merged = (
            actual_eval[["forecast_day"] + HOUR_COLS]
            .merge(fc, on="forecast_day", suffixes=("_true", "_pred"))
            .merge(naive_shift, on="forecast_day", how="inner")
        )

        y_true = merged[[f"{col}_true" for col in HOUR_COLS]].to_numpy()
        y_pred = merged[[f"{col}_pred" for col in HOUR_COLS]].to_numpy()
        naive = merged[[f"{col}_naive" for col in HOUR_COLS]].to_numpy()

        model_mae = mae_arr(y_true, y_pred)
        naive_mae = mae_arr(y_true, naive)

        metrics_rows.append(
            {
                "market": market,
                "model": model,
                "start_date": merged["forecast_day"].min().date().isoformat(),
                "end_date": merged["forecast_day"].max().date().isoformat(),
                "n_days": len(merged),
                "rMAE": model_mae / naive_mae,
                "MAE": model_mae,
                "RMSE": rmse_arr(y_true, y_pred),
                "sMAPE": smape_arr(y_true, y_pred),
                "MAPE": mape_arr(y_true, y_pred),
            }
        )

        months = merged["forecast_day"].dt.to_period("M").astype(str)
        for month in pd.unique(months):
            idx = np.where(months == month)[0]
            yt = y_true[idx]
            yp = y_pred[idx]
            nv = naive[idx]

            monthly_rows.append(
                {
                    "market": market,
                    "model": model,
                    "month": month,
                    "rMAE": mae_arr(yt, yp) / mae_arr(yt, nv),
                    "MAE": mae_arr(yt, yp),
                    "RMSE": rmse_arr(yt, yp),
                    "sMAPE": smape_arr(yt, yp),
                    "MAPE": mape_arr(yt, yp),
                }
            )

    return actual_eval, aligned_forecasts, pd.DataFrame(metrics_rows), pd.DataFrame(monthly_rows)


def make_market_timeseries_plot(
    out_dir: Path,
    market: str,
    actual_eval: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    days: int = 14,
):
    actual_window = actual_eval.head(days)
    actual_long = flatten_daily_matrix(actual_window)

    plt.figure(figsize=(16, 6))
    plt.plot(
        actual_long["DateTime"],
        actual_long["Price"],
        label="Ist-Wert",
        linewidth=2.4,
        color=COLORS["ACTUAL"],
    )

    for model in MODELS:
        fc_window = forecasts[model].head(days)
        fc_long = flatten_daily_matrix(fc_window)
        plt.plot(
            fc_long["DateTime"],
            fc_long["Price"],
            label=model,
            linewidth=1.7,
            color=COLORS[model],
        )

    plt.title(f"Modellvergleich: Ist-Wert vs. Prognosen ({market})")
    plt.xlabel("Zeit")
    plt.ylabel("Preis")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"{market}_timeseries_model_comparison.png", dpi=150)
    plt.close()


def make_monthly_plot(out_dir: Path, monthly_df: pd.DataFrame, market: str, metric: str):
    sub = monthly_df[monthly_df["market"] == market].copy()

    plt.figure(figsize=(14, 5))
    for model in MODELS:
        model_sub = sub[sub["model"] == model].copy()
        plt.plot(
            model_sub["month"],
            model_sub[metric],
            marker="o",
            label=model,
            color=COLORS[model],
        )

    plt.title(f"Monatlicher {metric}-Vergleich ({market})")
    plt.xlabel("Monat")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{market}_monthly_{metric.lower()}_comparison.png", dpi=150)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_monthly = []

    print("Verwendete Dateien:")
    for market in MARKETS:
        print(f"\n{market}")
        print(f"  TEST : {TEST_PATHS[market]}")
        for model in MODELS:
            print(f"  {model:4}: {FORECAST_PATHS[model][market]}")

    for market in MARKETS:
        actual_eval, forecasts, metrics_df, monthly_df = prepare_market(market)

        all_metrics.append(metrics_df)
        all_monthly.append(monthly_df)

        make_market_timeseries_plot(OUT_DIR, market, actual_eval, forecasts, days=14)
        make_monthly_plot(OUT_DIR, monthly_df, market, "MAE")
        make_monthly_plot(OUT_DIR, monthly_df, market, "rMAE")

    metrics = pd.concat(all_metrics, ignore_index=True)
    monthly = pd.concat(all_monthly, ignore_index=True)

    metrics["rank_rMAE_market"] = metrics.groupby("market")["rMAE"].rank(method="dense")
    metrics["rank_MAE_market"] = metrics.groupby("market")["MAE"].rank(method="dense")

    metrics = metrics.sort_values(["market", "rMAE", "MAE"]).reset_index(drop=True)
    monthly = monthly.sort_values(["market", "month", "model"]).reset_index(drop=True)

    metrics.to_csv(OUT_DIR / "model_comparison_metrics_common_horizon.csv", index=False)
    monthly.to_csv(OUT_DIR / "model_comparison_monthly_metrics_common_horizon.csv", index=False)

    print("\nDone.")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()