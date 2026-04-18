from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


# =========================
# Config
# =========================

INPUT_DIR = Path(r"C:\Users\manue\OneDrive\Desktop\Uni\Projektarbeit\data_splits_v2")
OUTPUT_DIR = Path(r"C:\Users\manue\OneDrive\Desktop\Uni\Projektarbeit\data_features_rf_svr")

MARKETS = ["DE_LU", "ES", "FR"]

REQUIRED_COLUMNS = ["Price", "Exogenous 1", "Exogenous 2"]


# =========================
# Helpers
# =========================

def load_split_csv(path: Path) -> pd.DataFrame:
    """Load one split CSV and ensure expected structure."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    df = df[REQUIRED_COLUMNS].copy().astype(float)
    df = df.sort_index()

    if df.index.has_duplicates:
        raise ValueError(f"Duplicate timestamps found in {path.name}")

    return df


def get_day_block(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Return 24-hour block for one day."""
    start = pd.Timestamp(day).normalize()
    end = start + pd.Timedelta(hours=23)
    return df.loc[start:end]


def weekday_one_hot(day: pd.Timestamp) -> dict:
    """Create one-hot weekday encoding."""
    weekday = pd.Timestamp(day).weekday()  # Monday = 0
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {f"dow_{name}": int(i == weekday) for i, name in enumerate(names)}


def build_day_sample(df: pd.DataFrame, day: pd.Timestamp) -> Tuple[dict, dict] | None:
    """
    Build one supervised sample for forecast day d.
    Returns:
        X_row: feature dict
        Y_row: target dict
    or None if required history is incomplete.
    """
    d = pd.Timestamp(day).normalize()

    required_days = [
        d,
        d - pd.Timedelta(days=1),
        d - pd.Timedelta(days=2),
        d - pd.Timedelta(days=3),
        d - pd.Timedelta(days=7),
    ]

    blocks: dict[pd.Timestamp, pd.DataFrame] = {}
    for rd in required_days:
        block = get_day_block(df, rd)
        if len(block) != 24:
            return None
        blocks[rd] = block

    x_row: dict[str, float | int | pd.Timestamp] = {"forecast_day": d}
    y_row: dict[str, float | pd.Timestamp] = {"forecast_day": d}

    # -------------------------
    # 1) Price lags: d-1, d-2, d-3, d-7
    # -------------------------
    for lag in [1, 2, 3, 7]:
        block = blocks[d - pd.Timedelta(days=lag)]
        for h in range(24):
            x_row[f"price_dminus{lag}_h{h+1:02d}"] = float(block["Price"].iloc[h])

    # -------------------------
    # 2) Current-day exogenous forecasts: d
    # -------------------------
    block_d = blocks[d]
    for h in range(24):
        x_row[f"x1_d_h{h+1:02d}"] = float(block_d["Exogenous 1"].iloc[h])
        x_row[f"x2_d_h{h+1:02d}"] = float(block_d["Exogenous 2"].iloc[h])

    # -------------------------
    # 3) Historical exogenous values: d-1 and d-7
    # -------------------------
    for lag in [1, 7]:
        block = blocks[d - pd.Timedelta(days=lag)]
        for h in range(24):
            x_row[f"x1_dminus{lag}_h{h+1:02d}"] = float(block["Exogenous 1"].iloc[h])
            x_row[f"x2_dminus{lag}_h{h+1:02d}"] = float(block["Exogenous 2"].iloc[h])

    # -------------------------
    # 4) Weekday information
    # -------------------------
    x_row.update(weekday_one_hot(d))

    # -------------------------
    # 5) Target: 24 prices of day d
    # -------------------------
    for h in range(24):
        y_row[f"y_h{h+1:02d}"] = float(block_d["Price"].iloc[h])

    return x_row, y_row


def build_supervised_from_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert one split DataFrame into:
      X: rows = forecast days, cols = engineered features
      Y: rows = forecast days, cols = 24 hourly targets
    """
    unique_days = pd.Index(df.index.normalize().unique())

    x_rows = []
    y_rows = []

    for day in unique_days:
        sample = build_day_sample(df, day)
        if sample is None:
            continue

        x_row, y_row = sample
        x_rows.append(x_row)
        y_rows.append(y_row)

    if not x_rows or not y_rows:
        raise ValueError("No supervised samples could be generated.")

    X = pd.DataFrame(x_rows).set_index("forecast_day").sort_index()
    Y = pd.DataFrame(y_rows).set_index("forecast_day").sort_index()

    return X, Y


def save_xy(X: pd.DataFrame, Y: pd.DataFrame, market: str, split_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_path = OUTPUT_DIR / f"{market}_X_{split_name}.csv"
    y_path = OUTPUT_DIR / f"{market}_Y_{split_name}.csv"

    X.to_csv(x_path)
    Y.to_csv(y_path)

    print(f"[OK] Saved {x_path.name} -> {X.shape}")
    print(f"[OK] Saved {y_path.name} -> {Y.shape}")


# =========================
# Main
# =========================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_names = ["train_core", "val", "train_full", "test"]

    for market in MARKETS:
        print(f"\n=== Processing market: {market} ===")

        for split_name in split_names:
            path = INPUT_DIR / f"{market}_{split_name}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing split file: {path}")

            df = load_split_csv(path)
            X, Y = build_supervised_from_split(df)

            save_xy(X, Y, market, split_name)

            print(
                f"Date range ({split_name}): "
                f"{X.index.min().date()} -> {X.index.max().date()}"
            )

    print(f"\nDone. Feature files saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()