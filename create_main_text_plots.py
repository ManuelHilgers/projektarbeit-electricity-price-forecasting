import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# =========================
# Einstellungen
# =========================
PROJECT_DIR = Path(r"C:\Users\Manuel Hilgers\OneDrive\Desktop\Uni\Projektarbeit")
CODE_DIR = PROJECT_DIR / "Modellcodes Python"
OUTPUT_DIR = CODE_DIR / "evaluation_outputs_faps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOUR_COLS = [f"h{i}" for i in range(24)]

MARKETS = ["DE_LU", "ES", "FR"]

TEST_FILES = {
    "DE_LU": PROJECT_DIR / "data_splits_v2" / "DE_LU_test.csv",
    "ES": PROJECT_DIR / "data_splits_v2" / "ES_test.csv",
    "FR": PROJECT_DIR / "data_splits_v2" / "FR_test.csv",
}

FORECAST_FILES = {
    "LEAR": {
        "DE_LU": CODE_DIR / "LEAR" / "results_lear" / "DE_LU_lear_forecasts.csv",
        "ES": CODE_DIR / "LEAR" / "results_lear" / "ES_lear_forecasts.csv",
        "FR": CODE_DIR / "LEAR" / "results_lear" / "FR_lear_forecasts.csv",
    },
    "DNN": {
        "DE_LU": CODE_DIR / "DNN" / "forecast_files" / "DNN_forecast_nl2_datDE_LU_YT2_SFH1_CW4_DE_LU_final.csv",
        "ES": CODE_DIR / "DNN" / "forecast_files" / "DNN_forecast_nl2_datES_YT2_SFH1_CW4_ES_final.csv",
        "FR": CODE_DIR / "DNN" / "forecast_files" / "DNN_forecast_nl2_datFR_YT2_SFH1_CW4_FR_final.csv",
    },
    "RF": {
        "DE_LU": CODE_DIR / "RF" / "outputs" / "RF_rolling_forecast_DE_LU_final.csv",
        "ES": CODE_DIR / "RF" / "outputs" / "RF_rolling_forecast_ES_final.csv",
        "FR": CODE_DIR / "RF" / "outputs" / "RF_rolling_forecast_FR_final.csv",
    },
    "SVR": {
        "DE_LU": CODE_DIR / "SVR" / "outputs" / "SVR_rolling_forecast_DE_LU_final.csv",
        "ES": CODE_DIR / "SVR" / "outputs" / "SVR_rolling_forecast_ES_final.csv",
        "FR": CODE_DIR / "SVR" / "outputs" / "SVR_rolling_forecast_FR_final.csv",
    },
}

COMMON_START = pd.Timestamp("2024-01-11")
COMMON_END = pd.Timestamp("2025-12-31")

WINDOW_START = pd.Timestamp("2025-01-09 00:00:00")
WINDOW_END = pd.Timestamp("2025-01-22 23:00:00")

# =========================
# FAPS-Farben
# =========================
COLORS = {
    "actual": (220/255, 30/255, 38/255),        # Sonderfa. Echtrot
    "LEAR": (0/255, 47/255, 108/255),           # FAU-Blau
    "DNN": (52/255, 103/255, 125/255),          # Hausfarbe Türkis
    "RF": (101/255, 141/255, 103/255),          # Hausfarbe Grün
    "SVR": (149/255, 162/255, 171/255),         # Grau 1
}

# =========================
# Hilfsfunktionen
# =========================
def read_test_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def test_long_to_daily_wide(df_test: pd.DataFrame) -> pd.DataFrame:
    df = df_test.copy()
    df["forecast_day"] = df["Date"].dt.floor("D")
    df["hour"] = df["Date"].dt.hour
    wide = df.pivot(index="forecast_day", columns="hour", values="Price").sort_index()
    wide.columns = [f"h{i}" for i in wide.columns]
    wide = wide[HOUR_COLS].reset_index()
    return wide

def normalize_hour_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    protected_cols = set()
    for c in ["forecast_day", "Date"]:
        if c in df.columns:
            protected_cols.add(c)

    hour_candidates = []
    extracted = []

    for col in df.columns:
        if col in protected_cols:
            continue

        col_str = str(col).strip().lower()

        # Fall 1: 0,1,2,...,23 oder 1,...,24
        if re.fullmatch(r"\d+", col_str):
            num = int(col_str)
            hour_candidates.append(col)
            extracted.append((col, num))
            continue

        # Fall 2: h0 ... h23 oder h1 ... h24
        if re.fullmatch(r"h\d+", col_str):
            num = int(re.findall(r"\d+", col_str)[0])
            hour_candidates.append(col)
            extracted.append((col, num))
            continue

        # Fall 3: y_h01 ... y_h24
        if re.fullmatch(r"y_h\d{2}", col_str):
            num = int(re.findall(r"\d+", col_str)[0])
            hour_candidates.append(col)
            extracted.append((col, num))
            continue

    if len(hour_candidates) < 24:
        raise ValueError(
            f"Es konnten nicht genügend Stunden-Spalten erkannt werden. "
            f"Gefundene Kandidaten: {hour_candidates}\n"
            f"Verfügbare Spalten: {list(df.columns)}"
        )

    nums = sorted(num for _, num in extracted)

    rename_map = {}

    # 0..23
    if nums == list(range(24)):
        for col, num in extracted:
            rename_map[col] = f"h{num}"

    # 1..24
    elif nums == list(range(1, 25)):
        for col, num in extracted:
            rename_map[col] = f"h{num - 1}"

    else:
        raise ValueError(
            f"Stunden-Spalten konnten nicht eindeutig auf 24 Stunden gemappt werden. "
            f"Extrahierte Stunden: {nums}\n"
            f"Spalten: {list(df.columns)}"
        )

    df = df.rename(columns=rename_map)
    return df

def read_forecast_file(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if model == "DNN":
        if "Date" not in df.columns:
            raise KeyError(f"DNN-Datei {path} enthält keine 'Date'-Spalte.")
        df["forecast_day"] = pd.to_datetime(df["Date"])
        df = df.drop(columns=["Date"], errors="ignore")

    elif "forecast_day" in df.columns:
        df["forecast_day"] = pd.to_datetime(df["forecast_day"])

    elif "Date" in df.columns:
        df["forecast_day"] = pd.to_datetime(df["Date"])
        df = df.drop(columns=["Date"], errors="ignore")

    else:
        raise KeyError(
            f"Datei {path} enthält weder 'forecast_day' noch 'Date'. "
            f"Verfügbare Spalten: {list(df.columns)}"
        )

    df = normalize_hour_columns(df)

    missing = [c for c in HOUR_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Nach der Umbenennung fehlen Stunden-Spalten in {path}: {missing}\n"
            f"Verfügbare Spalten: {list(df.columns)}"
        )

    return df[["forecast_day"] + HOUR_COLS].sort_values("forecast_day").reset_index(drop=True)

def daily_ecdf(values):
    values = np.sort(np.asarray(values))
    y = np.arange(1, len(values) + 1) / len(values)
    return values, y

def prepare_market_data(market: str):
    # Komplette Testdaten laden
    test_df = read_test_file(TEST_FILES[market])
    test_wide_full = test_long_to_daily_wide(test_df)

    # Naive Referenz aus vollem Zeitraum
    naive_lookup = test_wide_full.set_index("forecast_day")

    # Erst danach auf gemeinsamen Zeitraum schneiden
    test_wide_eval = test_wide_full[
        (test_wide_full["forecast_day"] >= COMMON_START) &
        (test_wide_full["forecast_day"] <= COMMON_END)
    ].copy()

    forecasts = {}
    daily_mae = {}
    daily_rmae = {}

    for model in ["LEAR", "DNN", "RF", "SVR"]:
        fc = read_forecast_file(FORECAST_FILES[model][market], model)

        merged = test_wide_eval.merge(
            fc, on="forecast_day", suffixes=("_actual", "_forecast")
        ).copy()

        actual_arr = merged[[f"h{i}_actual" for i in range(24)]].to_numpy()
        forecast_arr = merged[[f"h{i}_forecast" for i in range(24)]].to_numpy()

        naive_this = []
        for day in merged["forecast_day"]:
            ref_day = day - pd.Timedelta(days=7)
            if ref_day not in naive_lookup.index:
                raise KeyError(
                    f"Keine naive Referenz für {day} (Referenztag {ref_day}) gefunden."
                )
            naive_this.append(naive_lookup.loc[ref_day, HOUR_COLS].to_numpy())
        naive_this = np.vstack(naive_this)

        abs_err = np.abs(actual_arr - forecast_arr)
        abs_err_naive = np.abs(actual_arr - naive_this)

        daily_mae_model = abs_err.mean(axis=1)
        daily_mae_naive = abs_err_naive.mean(axis=1)
        daily_rmae_model = daily_mae_model / daily_mae_naive

        forecasts[model] = merged[["forecast_day"] + [f"h{i}_forecast" for i in range(24)]].copy()
        daily_mae[model] = pd.Series(daily_mae_model, index=merged["forecast_day"])
        daily_rmae[model] = pd.Series(daily_rmae_model, index=merged["forecast_day"])

    # Zwei-Wochen-Ausschnitt
    actual_long = test_df[
        (test_df["Date"] >= WINDOW_START) & (test_df["Date"] <= WINDOW_END)
    ].copy()

    forecast_long = {}
    for model in ["LEAR", "DNN", "RF", "SVR"]:
        fc = forecasts[model].copy()
        fc = fc[
            (fc["forecast_day"] >= WINDOW_START.floor("D")) &
            (fc["forecast_day"] <= WINDOW_END.floor("D"))
        ].copy()

        fc_long = fc.melt(
            id_vars="forecast_day",
            value_vars=[f"h{i}_forecast" for i in range(24)],
            var_name="hour",
            value_name="forecast"
        )
        fc_long["hour"] = fc_long["hour"].str.extract(r"(\d+)").astype(int)
        fc_long["Date"] = fc_long["forecast_day"] + pd.to_timedelta(fc_long["hour"], unit="h")
        fc_long = fc_long[
            (fc_long["Date"] >= WINDOW_START) & (fc_long["Date"] <= WINDOW_END)
        ].copy()

        forecast_long[model] = fc_long.sort_values("Date")

    return actual_long, forecast_long, daily_mae, daily_rmae

# =========================
# Plots
# =========================
def plot_two_week_window(market, actual_long, forecast_long):
    plt.figure(figsize=(13, 5))
    plt.plot(
        actual_long["Date"],
        actual_long["Price"],
        color=COLORS["actual"],
        linewidth=2.2,
        label="Ist-Wert"
    )

    for model in ["LEAR", "DNN", "RF", "SVR"]:
        plt.plot(
            forecast_long[model]["Date"],
            forecast_long[model]["forecast"],
            color=COLORS[model],
            linewidth=1.3,
            label=model
        )

    plt.title(f"Zwei-Wochen-Ausschnitt: Ist-Wert und Prognosen ({market})")
    plt.xlabel("Zeit")
    plt.ylabel("Preis")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{market}_two_week_window_faps.png", dpi=300)
    plt.close()

def plot_ecdf(daily_metric_dict, market, metric_name, xlabel, filename):
    plt.figure(figsize=(8, 5))

    for model in ["LEAR", "DNN", "RF", "SVR"]:
        x, y = daily_ecdf(daily_metric_dict[model].dropna())
        plt.step(x, y, where="post", color=COLORS[model], linewidth=2, label=model)

    if metric_name == "rMAE":
        plt.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="rMAE = 1")

    plt.title(f"ECDF der täglichen {metric_name}-Werte ({market})")
    plt.xlabel(xlabel)
    plt.ylabel("Kumulativer Anteil der Tage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

# =========================
# Hauptteil
# =========================
def main():
    for market in MARKETS:
        print(f"Bearbeite {market} ...")
        actual_long, forecast_long, daily_mae, daily_rmae = prepare_market_data(market)

        plot_two_week_window(market, actual_long, forecast_long)

        plot_ecdf(
            daily_metric_dict=daily_mae,
            market=market,
            metric_name="MAE",
            xlabel="Täglicher MAE",
            filename=f"{market}_ecdf_daily_mae_faps.png"
        )

        plot_ecdf(
            daily_metric_dict=daily_rmae,
            market=market,
            metric_name="rMAE",
            xlabel="Täglicher rMAE",
            filename=f"{market}_ecdf_daily_rmae_faps.png"
        )

    print("\nFertig. Grafiken gespeichert unter:")
    print(OUTPUT_DIR)

if __name__ == "__main__":
    main()