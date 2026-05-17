from pathlib import Path
import pandas as pd


BASE_DIR = Path(r"C:\Users\Manuel Hilgers\OneDrive\Desktop\Uni\Projektarbeit")
MARKETS = ["DE_LU", "FR", "ES"]

FORECAST_DIR = BASE_DIR / "results_lear"
TEST_DIR = BASE_DIR / "data_splits_v2"
OUTPUT_DIR = BASE_DIR / "results_lear_comparison"


def build_actual_daily_matrix(df_test: pd.DataFrame) -> pd.DataFrame:
    df = df_test.copy()
    df = df.sort_index()

    rows = []

    unique_days = pd.Index(df.index.normalize().unique())

    for day in unique_days:
        block = df.loc[day: day + pd.Timedelta(hours=23)]

        if len(block) != 24:
            continue

        row = {"forecast_day": day.date()}
        for h in range(24):
            row[f"actual_h{h+1:02d}"] = float(block["Price"].iloc[h])

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for market in MARKETS:
        forecast_file = FORECAST_DIR / f"{market}_lear_forecasts.csv"
        test_file = TEST_DIR / f"{market}_test.csv"

        if not forecast_file.exists():
            raise FileNotFoundError(f"Missing forecast file: {forecast_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Missing test file: {test_file}")

        forecast_df = pd.read_csv(forecast_file, parse_dates=["forecast_day"])
        test_df = pd.read_csv(test_file, index_col=0, parse_dates=True)

        actual_df = build_actual_daily_matrix(test_df)
        actual_df["forecast_day"] = pd.to_datetime(actual_df["forecast_day"])

        merged = forecast_df.merge(actual_df, on="forecast_day", how="inner")

        out_file = OUTPUT_DIR / f"{market}_lear_forecast_vs_actual.csv"
        merged.to_csv(out_file, index=False)

        print(f"\n=== {market} ===")
        print("Forecast file :", forecast_file)
        print("Test file     :", test_file)
        print("Merged shape  :", merged.shape)
        print("Saved to      :", out_file)

    print("\nAll LEAR comparison files created successfully.")


if __name__ == "__main__":
    main()