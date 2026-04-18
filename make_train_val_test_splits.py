from pathlib import Path
import pandas as pd

INPUT_DIR = Path(r"C:\Users\manue\OneDrive\Desktop\Uni\Projektarbeit\data_clean")
OUTPUT_DIR = Path(r"C:\Users\manue\OneDrive\Desktop\Uni\Projektarbeit\data_splits_v2")

TEST_YEARS = 2
VAL_YEARS = 1
HOURS_PER_YEAR = 364 * 24


def split_train_val_test(df: pd.DataFrame, test_years: int = 2, val_years: int = 1):
    test_hours = test_years * HOURS_PER_YEAR
    val_hours = val_years * HOURS_PER_YEAR

    if len(df) <= test_hours + val_hours:
        raise ValueError(
            f"Dataset too short. len={len(df)}, "
            f"required>{test_hours + val_hours}"
        )

    df_test = df.iloc[-test_hours:].copy()
    df_dev = df.iloc[:-test_hours].copy()          # everything before test
    df_val = df_dev.iloc[-val_hours:].copy()       # last part of dev
    df_train_core = df_dev.iloc[:-val_hours].copy()
    df_train_full = df_dev.copy()                  # train_core + val

    return df_train_core, df_val, df_train_full, df_test


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(INPUT_DIR.glob("*_clean_benchmark.csv"))
    if not files:
        raise FileNotFoundError(f"No *_clean_benchmark.csv files found in {INPUT_DIR}")

    for path in files:
        market = path.name.replace("_clean_benchmark.csv", "")
        print(f"\nProcessing {market}...")

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "Date"

        df_train_core, df_val, df_train_full, df_test = split_train_val_test(df)

        df_train_core.to_csv(OUTPUT_DIR / f"{market}_train_core.csv")
        df_val.to_csv(OUTPUT_DIR / f"{market}_val.csv")
        df_train_full.to_csv(OUTPUT_DIR / f"{market}_train_full.csv")
        df_test.to_csv(OUTPUT_DIR / f"{market}_test.csv")

        print(f"[OK] {market}_train_core.csv -> {df_train_core.shape}")
        print(f"[OK] {market}_val.csv        -> {df_val.shape}")
        print(f"[OK] {market}_train_full.csv -> {df_train_full.shape}")
        print(f"[OK] {market}_test.csv       -> {df_test.shape}")

        print("Date ranges:")
        print(f"  train_core: {df_train_core.index.min()} -> {df_train_core.index.max()}")
        print(f"  val       : {df_val.index.min()} -> {df_val.index.max()}")
        print(f"  train_full: {df_train_full.index.min()} -> {df_train_full.index.max()}")
        print(f"  test      : {df_test.index.min()} -> {df_test.index.max()}")

    print(f"\nDone. Files saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()