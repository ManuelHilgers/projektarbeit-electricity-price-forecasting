from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(r"C:\Users\Manuel Hilgers\OneDrive\Desktop\Uni\Projektarbeit")
MARKETS = ["DE_LU", "FR", "ES"]

INPUT_DIR = BASE_DIR / "results_lear_comparison"
OUTPUT_DIR = BASE_DIR / "results_lear_plots"

# Zeitfenster für den Zeitreihenplot
PLOT_START = "2024-01-04"
PLOT_END = "2024-01-31"


# =========================
# FAPS / FAU colors
# =========================
def rgb(r, g, b):
    return (r / 255, g / 255, b / 255)


COLOR_ACTUAL = rgb(4, 30, 66)        # FAU-Blau dunkel
COLOR_FORECAST = rgb(151, 193, 57)   # FAPS-Grün
COLOR_MAE = rgb(106, 138, 34)        # FAPS-Grün dunkel
COLOR_RMSE = rgb(245, 130, 31)       # Sonderfa. Orange
COLOR_REF = rgb(149, 162, 171)       # Grau 1
COLOR_GRID = rgb(209, 217, 222)      # Grau 3


def hourly_long_format(df_daily: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df_daily.iterrows():
        day = pd.Timestamp(row["forecast_day"])

        for h in range(24):
            timestamp = day + pd.Timedelta(hours=h)
            rows.append({
                "timestamp": timestamp,
                "forecast": float(row[f"h{h+1:02d}"]),
                "actual": float(row[f"actual_h{h+1:02d}"]),
            })

    df_hourly = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df_hourly["error"] = df_hourly["forecast"] - df_hourly["actual"]
    df_hourly["abs_error"] = np.abs(df_hourly["error"])
    df_hourly["squared_error"] = df_hourly["error"] ** 2
    df_hourly["month"] = df_hourly["timestamp"].dt.to_period("M").astype(str)

    return df_hourly


def make_pretty_axes():
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.35, color=COLOR_GRID)
    plt.gca().set_axisbelow(True)


def plot_timeseries_window(df_hourly: pd.DataFrame, market: str) -> None:
    start = pd.Timestamp(PLOT_START)
    end = pd.Timestamp(PLOT_END) + pd.Timedelta(hours=23)

    df_plot = df_hourly[(df_hourly["timestamp"] >= start) & (df_hourly["timestamp"] <= end)].copy()

    plt.figure(figsize=(14, 6))
    plt.plot(
        df_plot["timestamp"],
        df_plot["actual"],
        label="Tatsächlicher Wert",
        linewidth=1.8,
        color=COLOR_ACTUAL
    )
    plt.plot(
        df_plot["timestamp"],
        df_plot["forecast"],
        label="Prognose",
        linewidth=1.5,
        color=COLOR_FORECAST
    )

    plt.xlabel("Zeit")
    plt.ylabel("Preis")
    plt.title(f"LEAR: Ist-Wert und Prognose ({market})")
    make_pretty_axes()
    plt.legend(frameon=True, facecolor="white", edgecolor=COLOR_GRID)
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{market}_lear_timeseries_window.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved timeseries plot: {out_path}")


def plot_scatter(df_hourly: pd.DataFrame, market: str) -> None:
    y_true = df_hourly["actual"].to_numpy()
    y_pred = df_hourly["forecast"].to_numpy()

    lower_q = min(np.quantile(y_true, 0.001), np.quantile(y_pred, 0.001))
    upper_q = max(np.quantile(y_true, 0.999), np.quantile(y_pred, 0.999))

    padding = 0.05 * (upper_q - lower_q)
    min_val = lower_q - padding
    max_val = upper_q + padding

    plt.figure(figsize=(8, 8))
    plt.scatter(
        y_true,
        y_pred,
        s=10,
        alpha=0.14,
        color=COLOR_FORECAST,
        edgecolors="none"
    )
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linewidth=2.2,
        color=COLOR_REF
    )

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.xlabel("Tatsächlicher Preis")
    plt.ylabel("Prognostizierter Preis")
    plt.title(f"LEAR: Prognose vs. Ist-Wert ({market})")
    make_pretty_axes()
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{market}_lear_scatter.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved scatter plot: {out_path}")


def plot_monthly_errors(df_hourly: pd.DataFrame, market: str) -> None:
    monthly = (
        df_hourly.groupby("month")
        .agg(
            MAE=("abs_error", "mean"),
            RMSE=("squared_error", lambda x: np.sqrt(np.mean(x))),
        )
        .reset_index()
    )

    plt.figure(figsize=(14, 6))
    plt.plot(
        monthly["month"],
        monthly["MAE"],
        marker="o",
        markersize=4,
        linewidth=2,
        color=COLOR_MAE,
        label="MAE"
    )
    plt.plot(
        monthly["month"],
        monthly["RMSE"],
        marker="o",
        markersize=4,
        linewidth=2,
        color=COLOR_RMSE,
        label="RMSE"
    )

    plt.xlabel("Monat")
    plt.ylabel("Fehler")
    plt.title(f"LEAR: Monatliche Fehlerkennzahlen ({market})")
    plt.xticks(rotation=45)
    make_pretty_axes()
    plt.legend(frameon=True, facecolor="white", edgecolor=COLOR_GRID)
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{market}_lear_monthly_errors.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    monthly_out = OUTPUT_DIR / f"{market}_lear_monthly_errors.csv"
    monthly.to_csv(monthly_out, index=False)

    print(f"[OK] Saved monthly error plot: {out_path}")
    print(f"[OK] Saved monthly error table: {monthly_out}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for market in MARKETS:
        file_path = INPUT_DIR / f"{market}_lear_forecast_vs_actual.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing input file: {file_path}")

        df_daily = pd.read_csv(file_path, parse_dates=["forecast_day"])
        df_hourly = hourly_long_format(df_daily)

        print(f"\n=== {market} ===")
        print("Input shape (daily) :", df_daily.shape)
        print("Hourly shape        :", df_hourly.shape)

        plot_timeseries_window(df_hourly, market)
        plot_scatter(df_hourly, market)
        plot_monthly_errors(df_hourly, market)

    print("\nAll LEAR plots created successfully.")


if __name__ == "__main__":
    main()