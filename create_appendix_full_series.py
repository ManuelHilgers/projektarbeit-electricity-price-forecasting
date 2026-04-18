import pandas as pd
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

MARKETS = ["DE_LU", "ES", "FR"]
HOUR_COLS = [f"h{i}" for i in range(24)]

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
COMMON_END = pd.Timestamp("2025-12-31 23:00:00")

COLORS = {
    "actual": (220/255, 30/255, 38/255),        # Sonderfa. Echtrot
    "LEAR": (0/255, 47/255, 108/255),           # FAU-Blau
    "DNN": (52/255, 103/255, 125/255),          # Hausfarbe Türkis
    "RF": (101/255, 141/255, 103/255),          # Hausfarbe Grün
    "SVR": (149/255, 162/255, 171/255),         # Grau 1
}

def read_test(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

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

        if re.fullmatch(r"\d+", col_str):
            num = int(col_str)
            hour_candidates.append(col)
            extracted.append((col, num))
            continue

        if re.fullmatch(r"h\d+", col_str):
            num = int(re.findall(r"\d+", col_str)[0])
            hour_candidates.append(col)
            extracted.append((col, num))
            continue

        if re.fullmatch(r"y_h\d{2}", col_str):
            num = int(re.findall(r"\d+", col_str)[0])
            hour_candidates.append(col)
            extracted.append((col, num))
            continue

    if len(hour_candidates) < 24:
        raise ValueError(
            f"Zu wenige Stunden-Spalten erkannt. "
            f"Kandidaten: {hour_candidates}\n"
            f"Spalten: {list(df.columns)}"
        )

    nums = sorted(num for _, num in extracted)

    rename_map = {}

    if nums == list(range(24)):
        for col, num in extracted:
            rename_map[col] = f"h{num}"
    elif nums == list(range(1, 25)):
        for col, num in extracted:
            rename_map[col] = f"h{num - 1}"
    else:
        raise ValueError(
            f"Stunden-Spalten konnten nicht eindeutig gemappt werden. "
            f"Extrahierte Stunden: {nums}\n"
            f"Spalten: {list(df.columns)}"
        )

    return df.rename(columns=rename_map)

def read_forecast(path, model):
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

    return df[["forecast_day"] + HOUR_COLS].sort_values("forecast_day")

def forecast_to_long(df):
    long_df = df.melt(
        id_vars="forecast_day",
        value_vars=HOUR_COLS,
        var_name="hour",
        value_name="forecast"
    )
    long_df["hour"] = long_df["hour"].str.extract(r"(\d+)").astype(int)
    long_df["Date"] = long_df["forecast_day"] + pd.to_timedelta(long_df["hour"], unit="h")
    return long_df[["Date", "forecast"]].sort_values("Date")

def main():
    for market in MARKETS:
        print(f"Bearbeite {market} ...")
        test_df = read_test(TEST_FILES[market])
        test_df = test_df[
            (test_df["Date"] >= COMMON_START) &
            (test_df["Date"] <= COMMON_END)
        ].copy()

        plt.figure(figsize=(14, 5))
        plt.plot(
            test_df["Date"],
            test_df["Price"],
            color=COLORS["actual"],
            linewidth=1.2,
            label="Ist-Wert"
        )

        for model in ["LEAR", "DNN", "RF", "SVR"]:
            fc = read_forecast(FORECAST_FILES[model][market], model)
            fc = fc[
                (fc["forecast_day"] >= COMMON_START.floor("D")) &
                (fc["forecast_day"] <= COMMON_END.floor("D"))
            ].copy()

            fc_long = forecast_to_long(fc)
            fc_long = fc_long[
                (fc_long["Date"] >= COMMON_START) &
                (fc_long["Date"] <= COMMON_END)
            ].copy()

            plt.plot(
                fc_long["Date"],
                fc_long["forecast"],
                color=COLORS[model],
                linewidth=0.9,
                alpha=0.9,
                label=model
            )

        plt.title(f"Gesamtverlauf: Ist-Wert und Prognosen ({market})")
        plt.xlabel("Zeit")
        plt.ylabel("Preis")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{market}_full_series_faps.png", dpi=300)
        plt.close()

    print("\nFertig. Anhang-Grafiken gespeichert unter:")
    print(OUTPUT_DIR)

if __name__ == "__main__":
    main()