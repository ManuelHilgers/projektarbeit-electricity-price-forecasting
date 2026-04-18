from pathlib import Path
import pandas as pd

from epftoolbox.models import LEAR


# =========================
# Config
# =========================
BASE_DIR = Path(r"C:\Users\Manuel Hilgers\OneDrive\Desktop\Uni\Projektarbeit")

MARKETS = ["DE_LU", "FR", "ES"]
OUTPUT_DIR = BASE_DIR / "results_lear"

CALIBRATION_WINDOW = 1092  # 3 Jahre à 364 Tage


def run_market(market: str) -> None:
    data_file = BASE_DIR / "data_clean" / f"{market}_clean_benchmark.csv"
    split_file = BASE_DIR / "data_splits_v2" / f"{market}_test.csv"

    # Vollständige Benchmark-Daten laden
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df[["Price", "Exogenous 1", "Exogenous 2"]].copy().astype(float)

    # Test-Split laden, um Start/Ende der Testtage zu bestimmen
    df_test = pd.read_csv(split_file, index_col=0, parse_dates=True)
    df_test.index.name = "Date"

    test_start = df_test.index.min().normalize()
    test_end = df_test.index.max().normalize()

    print(f"\n=== Running LEAR for {market} ===")
    print("Full data shape :", df.shape)
    print("Test data shape :", df_test.shape)
    print("Test period     :", test_start.date(), "->", test_end.date())
    print("Data file       :", data_file)
    print("Split file      :", split_file)

    model = LEAR(calibration_window=CALIBRATION_WINDOW)

    forecast_rows = []

    current_day = test_start
    while current_day <= test_end:
        forecast = model.recalibrate_and_forecast_next_day(
            df=df,
            calibration_window=CALIBRATION_WINDOW,
            next_day_date=current_day,
        )

        forecast_24 = forecast.flatten()

        if forecast_24.shape[0] != 24:
            raise ValueError(
                f"[{market}] Expected 24 hourly values for {current_day.date()}, got {forecast_24.shape}"
            )

        row = {"forecast_day": current_day.date()}
        for h in range(24):
            row[f"h{h+1:02d}"] = float(forecast_24[h])

        forecast_rows.append(row)

        if len(forecast_rows) % 30 == 0:
            print(f"[OK] {market}: processed {len(forecast_rows)} forecast days... latest: {current_day.date()}")

        current_day += pd.Timedelta(days=1)

    forecast_df = pd.DataFrame(forecast_rows)

    out_path = OUTPUT_DIR / f"{market}_lear_forecasts.csv"
    forecast_df.to_csv(out_path, index=False)

    print(f"\n[{market}] Run completed.")
    print("Forecast rows   :", forecast_df.shape[0])
    print("Forecast cols   :", forecast_df.shape[1])
    print("Saved file      :", out_path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for market in MARKETS:
        run_market(market)

    print("\n=== All LEAR runs completed successfully. ===")


if __name__ == "__main__":
    main()