import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Einstellungen
# ============================================================

BASE_DIR = r"C:\Users\Manuel Hilgers\OneDrive\Desktop\Uni\Projektarbeit"
CODE_DIR = os.path.join(BASE_DIR, "Modellcodes Python")
OUTPUT_DIR = os.path.join(CODE_DIR, "evaluation_outputs_faps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MARKETS = ["DE_LU", "ES", "FR"]
MODELS = ["LEAR", "DNN", "RF", "SVR"]
HOUR_COLS = [f"h{i}" for i in range(24)]

# FAPS-Farben
COLORS = {
    "LEAR": (4/255, 30/255, 66/255),      # FAU-Blau dunkel
    "DNN":  (52/255, 103/255, 125/255),   # Hausfarbe Türkis
    "RF":   (101/255, 141/255, 103/255),  # Hausfarbe Grün
    "SVR":  (149/255, 162/255, 171/255),  # Grau 1
}

# Dateien
TEST_FILES = {
    "DE_LU": os.path.join(BASE_DIR, "data_splits_v2", "DE_LU_test.csv"),
    "ES":    os.path.join(BASE_DIR, "data_splits_v2", "ES_test.csv"),
    "FR":    os.path.join(BASE_DIR, "data_splits_v2", "FR_test.csv"),
}

FORECAST_FILES = {
    "LEAR": {
        "DE_LU": os.path.join(CODE_DIR, "LEAR", "results_lear", "DE_LU_lear_forecasts.csv"),
        "ES":    os.path.join(CODE_DIR, "LEAR", "results_lear", "ES_lear_forecasts.csv"),
        "FR":    os.path.join(CODE_DIR, "LEAR", "results_lear", "FR_lear_forecasts.csv"),
    },
    "DNN": {
        "DE_LU": os.path.join(CODE_DIR, "DNN", "forecast_files", "DNN_forecast_nl2_datDE_LU_YT2_SFH1_CW4_DE_LU_final.csv"),
        "ES":    os.path.join(CODE_DIR, "DNN", "forecast_files", "DNN_forecast_nl2_datES_YT2_SFH1_CW4_ES_final.csv"),
        "FR":    os.path.join(CODE_DIR, "DNN", "forecast_files", "DNN_forecast_nl2_datFR_YT2_SFH1_CW4_FR_final.csv"),
    },
    "RF": {
        "DE_LU": os.path.join(CODE_DIR, "RF", "outputs", "RF_rolling_forecast_DE_LU_final.csv"),
        "ES":    os.path.join(CODE_DIR, "RF", "outputs", "RF_rolling_forecast_ES_final.csv"),
        "FR":    os.path.join(CODE_DIR, "RF", "outputs", "RF_rolling_forecast_FR_final.csv"),
    },
    "SVR": {
        "DE_LU": os.path.join(CODE_DIR, "SVR", "outputs", "SVR_rolling_forecast_DE_LU_final.csv"),
        "ES":    os.path.join(CODE_DIR, "SVR", "outputs", "SVR_rolling_forecast_ES_final.csv"),
        "FR":    os.path.join(CODE_DIR, "SVR", "outputs", "SVR_rolling_forecast_FR_final.csv"),
    },
}


# ============================================================
# Hilfsfunktionen
# ============================================================

def normalize_hour_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vereinheitlicht Stunden-Spalten auf h0 ... h23.
    Unterstützte Varianten:
    - h0 ... h23
    - h00 ... h23
    - 0 ... 23
    - 1 ... 24
    - y_h01 ... y_h24
    """
    df = df.copy()

    protected_cols = {"forecast_day", "Date"}
    rename_map = {}
    extracted = []

    for col in df.columns:
        if col in protected_cols:
            continue

        col_str = str(col).strip().lower()

        # Fall 1: 0 ... 23
        if re.fullmatch(r"\d+", col_str):
            num = int(col_str)
            extracted.append((col, num))
            continue

        # Fall 2: h0 ... h23 oder h00 ... h23
        m = re.fullmatch(r"h(\d{1,2})", col_str)
        if m:
            num = int(m.group(1))
            extracted.append((col, num))
            continue

        # Fall 3: y_h01 ... y_h24
        m = re.fullmatch(r"y_h(\d{2})", col_str)
        if m:
            num = int(m.group(1))
            extracted.append((col, num))
            continue

    if len(extracted) < 24:
        raise ValueError(
            "Es konnten nicht genügend Stunden-Spalten erkannt werden.\n"
            f"Gefundene Kandidaten: {[c for c, _ in extracted]}\n"
            f"Verfügbare Spalten: {list(df.columns)}"
        )

    nums = sorted(num for _, num in extracted)

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
            "Stunden-Spalten konnten nicht eindeutig auf 24 Stunden gemappt werden.\n"
            f"Extrahierte Stunden: {nums}\n"
            f"Spalten: {list(df.columns)}"
        )

    df = df.rename(columns=rename_map)

    missing = [c for c in HOUR_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Nach der Umbenennung fehlen Stunden-Spalten: {missing}\n"
            f"Verfügbare Spalten: {list(df.columns)}"
        )

    return df


def read_actual_data(path: str) -> pd.DataFrame:
    """
    Liest Testdaten ein und formt sie in tägliche 24h-Matrizen um.
    Die Stunde wird direkt aus der Date-Spalte abgeleitet.
    """
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    value_col = None
    for candidate in ["Price", "price", "value", "y", "SpotPrice"]:
        if candidate in df.columns:
            value_col = candidate
            break

    if value_col is None:
        raise KeyError(
            f"Keine Preis-Spalte gefunden. Verfügbare Spalten: {list(df.columns)}"
        )

    df["forecast_day"] = df["Date"].dt.floor("D")
    df["h"] = df["Date"].dt.hour

    actual = df.pivot(index="forecast_day", columns="h", values=value_col).reset_index()
    actual = actual.rename(columns={i: f"h{i}" for i in range(24)})
    actual = actual.sort_values("forecast_day").reset_index(drop=True)

    missing = [c for c in HOUR_COLS if c not in actual.columns]
    if missing:
        raise KeyError(
            f"In den Testdaten fehlen Stunden-Spalten nach dem Pivot: {missing}"
        )

    return actual


def read_forecast_file(path: str, model: str) -> pd.DataFrame:
    """
    Liest Prognosedateien ein und vereinheitlicht Datums- und Stunden-Spalten.
    """
    df = pd.read_csv(path)

    if "forecast_day" in df.columns:
        df["forecast_day"] = pd.to_datetime(df["forecast_day"])
    elif "Date" in df.columns:
        df["forecast_day"] = pd.to_datetime(df["Date"])
        df = df.drop(columns=["Date"], errors="ignore")
    else:
        raise KeyError(
            f"Keine Datums-Spalte in {path} gefunden. "
            f"Verfügbare Spalten: {list(df.columns)}"
        )

    df = normalize_hour_columns(df)
    return df[["forecast_day"] + HOUR_COLS].sort_values("forecast_day").reset_index(drop=True)


def build_naive_reference(actual_full: pd.DataFrame, eval_days: pd.Series) -> pd.DataFrame:
    """
    Erzeugt die naive Referenz auf Basis des gleichen Wochentags der Vorwoche.
    """
    lookup = actual_full.set_index("forecast_day")
    rows = []

    for day in eval_days:
        ref_day = day - pd.Timedelta(days=7)
        if ref_day not in lookup.index:
            raise KeyError(
                f"Keine naive Referenz für {day.date()} "
                f"(Referenztag {ref_day.date()}) gefunden."
            )

        row = lookup.loc[ref_day]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        out = {"forecast_day": day}
        for h in range(24):
            out[f"h{h}"] = row[f"h{h}"]
        rows.append(out)

    return pd.DataFrame(rows)


def prepare_market_data(market: str) -> pd.DataFrame:
    """
    Berechnet tägliche MAE- und rMAE-Werte für alle Modelle
    auf dem gemeinsamen Vergleichshorizont.
    """
    actual_full = read_actual_data(TEST_FILES[market])

    forecast_dict = {}
    start_dates = []
    end_dates = []

    for model in MODELS:
        fc = read_forecast_file(FORECAST_FILES[model][market], model)
        forecast_dict[model] = fc
        start_dates.append(fc["forecast_day"].min())
        end_dates.append(fc["forecast_day"].max())

    common_start = max(start_dates)
    common_end = min(end_dates)

    actual_eval = actual_full[
        (actual_full["forecast_day"] >= common_start) &
        (actual_full["forecast_day"] <= common_end)
    ].copy()

    naive_eval = build_naive_reference(actual_full, actual_eval["forecast_day"])

    daily_metrics = []

    for model in MODELS:
        fc = forecast_dict[model]
        fc = fc[
            (fc["forecast_day"] >= common_start) &
            (fc["forecast_day"] <= common_end)
        ].copy()

        merged = actual_eval.merge(fc, on="forecast_day", suffixes=("_actual", "_forecast"))
        merged = merged.merge(naive_eval, on="forecast_day", suffixes=("", "_naive"))

        for _, row in merged.iterrows():
            actual_vals = np.array([row[f"h{i}_actual"] for i in range(24)], dtype=float)
            forecast_vals = np.array([row[f"h{i}_forecast"] for i in range(24)], dtype=float)
            naive_vals = np.array([row[f"h{i}"] for i in range(24)], dtype=float)

            mae = np.mean(np.abs(actual_vals - forecast_vals))
            mae_naive = np.mean(np.abs(actual_vals - naive_vals))
            rmae = np.nan if mae_naive == 0 else mae / mae_naive

            daily_metrics.append({
                "market": market,
                "model": model,
                "forecast_day": row["forecast_day"],
                "MAE": mae,
                "rMAE": rmae
            })

    return pd.DataFrame(daily_metrics)


def compute_class_shares(series: pd.Series, bins, labels):
    """
    Berechnet den prozentualen Anteil der Werte in den angegebenen Klassen.
    """
    cats = pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=True)
    shares = cats.value_counts(normalize=True).sort_index() * 100
    return shares.reindex(labels, fill_value=0)


def plot_distribution_bars(market: str, df_market: pd.DataFrame):
    """
    Erstellt pro Markt eine Abbildung mit zwei Balkendiagrammen:
    links MAE, rechts rMAE.
    """
    mae_bins = [-np.inf, 5, 10, 15, 20, 30, np.inf]
    mae_labels = ["≤ 5", "5–10", "10–15", "15–20", "20–30", "> 30"]

    rmae_bins = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
    rmae_labels = ["≤ 0,5", "0,5–1,0", "1,0–1,5", "1,5–2,0", "> 2,0"]

    mae_shares = {}
    rmae_shares = {}

    for model in MODELS:
        df_model = df_market[df_market["model"] == model].copy()
        mae_shares[model] = compute_class_shares(df_model["MAE"], mae_bins, mae_labels)
        rmae_shares[model] = compute_class_shares(df_model["rMAE"], rmae_bins, rmae_labels)

    mae_df = pd.DataFrame(mae_shares)
    rmae_df = pd.DataFrame(rmae_shares)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    width = 0.2

    # -------- links: MAE --------
    x_mae = np.arange(len(mae_labels))
    for i, model in enumerate(MODELS):
        axes[0].bar(
            x_mae + (i - 1.5) * width,
            mae_df[model].values,
            width=width,
            label=model,
            color=COLORS[model]
        )

    axes[0].set_xticks(x_mae)
    axes[0].set_xticklabels(mae_labels)
    axes[0].set_xlabel("Täglicher MAE")
    axes[0].set_ylabel("Anteil der Tage in %")
    axes[0].set_title(f"Verteilung des täglichen MAE ({market})")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_axisbelow(True)

    # -------- rechts: rMAE --------
    x_rmae = np.arange(len(rmae_labels))
    for i, model in enumerate(MODELS):
        axes[1].bar(
            x_rmae + (i - 1.5) * width,
            rmae_df[model].values,
            width=width,
            label=model,
            color=COLORS[model]
        )

    axes[1].set_xticks(x_rmae)
    axes[1].set_xticklabels(rmae_labels)
    axes[1].set_xlabel("Täglicher rMAE")
    axes[1].set_ylabel("Anteil der Tage in %")
    axes[1].set_title(f"Verteilung des täglichen rMAE ({market})")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)

    outpath = os.path.join(OUTPUT_DIR, f"{market}_distribution_bars_mae_rmae_faps.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Gespeichert: {outpath}")


# ============================================================
# Hauptteil
# ============================================================

def main():
    for market in MARKETS:
        print(f"Bearbeite {market} ...")
        df_market = prepare_market_data(market)
        plot_distribution_bars(market, df_market)

    print("\nFertig. Grafiken gespeichert unter:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()