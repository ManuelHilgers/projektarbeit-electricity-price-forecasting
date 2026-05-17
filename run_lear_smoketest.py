from pathlib import Path
import pandas as pd

from epftoolbox.models import LEAR


# -------------------------
# Config
# -------------------------
DATA_FILE = Path(r"C:\Users\manue\OneDrive\Desktop\Uni\Projektarbeit\data_clean\DE_LU_clean_benchmark.csv")

# train_full endet bei dir am 2024-01-03 23:00
# daher testen wir als nächsten Tag: 2024-01-04
NEXT_DAY = pd.Timestamp("2024-01-04 00:00:00")

CALIBRATION_WINDOW = 1092  # 3 Jahre à 364 Tage


def main() -> None:
    # Daten laden
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    df.index.name = "Date"

    # nur die benötigten Spalten
    df = df[["Price", "Exogenous 1", "Exogenous 2"]].copy().astype(float)

    print("Data shape:", df.shape)
    print("Date range :", df.index.min(), "->", df.index.max())
    print("Testing next-day forecast for:", NEXT_DAY.date())

    # Modell initialisieren
    model = LEAR(calibration_window=CALIBRATION_WINDOW)

    # Ein-Tages-Prognose
    forecast = model.recalibrate_and_forecast_next_day(
        df=df,
        calibration_window=CALIBRATION_WINDOW,
        next_day_date=NEXT_DAY,
    )

    print("\nForecast type :", type(forecast))
    print("Forecast shape:", getattr(forecast, "shape", "no shape"))
    print("Forecast values:", forecast)

    # Plausibilitätscheck
    if getattr(forecast, "shape", None) != (1, 24):
        raise ValueError(
            f"Expected forecast shape (1, 24), got {getattr(forecast, 'shape', None)}"
        )

    forecast_24 = forecast.flatten()
    print("Flattened shape:", forecast_24.shape)
    print("First values   :", forecast_24[:5])

    print("\n[OK] LEAR smoketest successful. 1 forecast day with 24 hourly values returned.")


if __name__ == "__main__":
    main()