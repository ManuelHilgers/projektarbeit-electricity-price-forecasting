from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


FILE_RE = re.compile(
    r"^(?P<market>[A-Z_]+)_(?P<kind>price|loadfc|windsolarfc)_(?P<year>\d{4})\.csv$"
)


@dataclass(frozen=True)
class Config:
    raw_dir: Path
    clean_dir: Path
    start_utc: str
    end_utc: str
    years_test: int
    write_train_test: bool


def read_one_series_csv(path: Path, value_name: str) -> pd.Series:
    """
    Reads one of your raw CSVs and returns a Series with a UTC-aware DatetimeIndex.
    Supports two layouts:
      (A) columns: Date, <value>
      (B) first column is datetime index + one value column
    """
    df = pd.read_csv(path)

    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        value_cols = [c for c in df.columns if c != "Date"]
        if not value_cols:
            raise ValueError(f"No value column found in {path}")
        val = pd.to_numeric(df[value_cols[0]], errors="coerce")
        s = pd.Series(val.values, index=dt, name=value_name)
    else:
        df2 = pd.read_csv(path, index_col=0, parse_dates=True)
        idx = pd.to_datetime(df2.index, errors="coerce", utc=True)
        val = pd.to_numeric(df2.iloc[:, 0], errors="coerce")
        s = pd.Series(val.values, index=idx, name=value_name)

    s = s[~s.index.isna()]
    s = s[~s.index.duplicated(keep="first")]
    s = s.sort_index()
    return s


def parse_files(raw_dir: Path) -> Dict[Tuple[str, str, str], Path]:
    """
    Maps (market, kind, year) -> file path for files like:
      DE_LU_price_2020.csv
      FR_loadfc_2021.csv
      ES_windsolarfc_2022.csv
    """
    mapping: Dict[Tuple[str, str, str], Path] = {}
    for f in raw_dir.glob("*.csv"):
        m = FILE_RE.match(f.name)
        if not m:
            continue
        market = m.group("market")
        kind = m.group("kind")
        year = m.group("year")
        mapping[(market, kind, year)] = f
    return mapping


def build_market_dataframe(mapping: Dict[Tuple[str, str, str], Path], market: str) -> pd.DataFrame:
    """
    For each year: join price + load forecast + wind/solar forecast on Date.
    Then concat all years into one market DataFrame.
    """
    years = sorted({year for (m, _, year) in mapping.keys() if m == market})
    if not years:
        raise FileNotFoundError(f"No files found for market {market}")

    parts: List[pd.DataFrame] = []
    for y in years:
        p_price = mapping.get((market, "price", y))
        p_load = mapping.get((market, "loadfc", y))
        p_ws = mapping.get((market, "windsolarfc", y))

        if not (p_price and p_load and p_ws):
            missing = [k for k, pth in [("price", p_price), ("loadfc", p_load), ("windsolarfc", p_ws)] if pth is None]
            raise FileNotFoundError(f"Missing raw files for {market} {y}: {missing}")

        s_price = read_one_series_csv(p_price, "Price")
        s_load = read_one_series_csv(p_load, "Load_Forecast")
        s_ws = read_one_series_csv(p_ws, "WindSolar_Forecast")

        df_y = pd.concat([s_price, s_load, s_ws], axis=1)
        df_y = df_y[~df_y.index.duplicated(keep="first")].sort_index()
        parts.append(df_y)

    df = pd.concat(parts, axis=0)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.index.name = "Date"
    return df


def enforce_hourly_utc(df: pd.DataFrame) -> pd.DataFrame:
    start = df.index.min()
    end = df.index.max()
    full_idx = pd.date_range(start, end, freq="h", tz="UTC")
    df = df.reindex(full_idx)
    df.index.name = "Date"
    return df


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Load_Forecast", "WindSolar_Forecast"]:
        df[col] = df[col].ffill().bfill()
    df["Price"] = df["Price"].interpolate(method="time", limit_direction="both")
    df = df.dropna(how="all")
    return df


def to_lago_benchmark(df_utc: pd.DataFrame) -> pd.DataFrame:
    """
    Final benchmark:
      - UTC tz-aware index -> drop tz -> naive datetime64[ns] (like Lago)
      - columns: Price, Exogenous 1, Exogenous 2
    """
    df = df_utc.copy()

    # ensure UTC tz-aware
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # drop tz -> naive timestamps
    df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    df = df.rename(columns={
        "Load_Forecast": "Exogenous 1",
        "WindSolar_Forecast": "Exogenous 2",
    })

    needed = ["Price", "Exogenous 1", "Exogenous 2"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after renaming: {missing}. Available: {list(df.columns)}")

    return df[needed].astype(float)


def split_like_lago(df: pd.DataFrame, years_test: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_hours = years_test * 364 * 24
    if len(df) <= test_hours:
        raise ValueError(f"Dataset too short for years_test={years_test}. len={len(df)} < {test_hours}")
    return df.iloc[:-test_hours].copy(), df.iloc[-test_hours:].copy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean your local ENTSO-E raw CSVs into Lago-compatible benchmark CSVs.")
    ap.add_argument("--raw-dir", default="data_raw")
    ap.add_argument("--clean-dir", default="data_clean")
    ap.add_argument("--start", default="2020-01-01 00:00")
    ap.add_argument("--end", default="2025-12-31 23:00")
    ap.add_argument("--years-test", type=int, default=1)
    ap.add_argument("--write-train-test", action="store_true")
    args = ap.parse_args()

    cfg = Config(
        raw_dir=Path(args.raw_dir),
        clean_dir=Path(args.clean_dir),
        start_utc=args.start,
        end_utc=args.end,
        years_test=args.years_test,
        write_train_test=args.write_train_test,
    )

    cfg.clean_dir.mkdir(parents=True, exist_ok=True)

    mapping = parse_files(cfg.raw_dir)
    if not mapping:
        raise FileNotFoundError(
            f"No matching raw files found in {cfg.raw_dir}. Expected names like DE_LU_price_2020.csv"
        )

    markets = sorted({m for (m, _, _) in mapping.keys()})
    print("Found markets:", markets)

    start = pd.Timestamp(cfg.start_utc, tz="UTC")
    end = pd.Timestamp(cfg.end_utc, tz="UTC")

    for market in markets:
        df0 = build_market_dataframe(mapping, market)
        df0 = df0.loc[start:end].copy()

        df1 = enforce_hourly_utc(df0)
        df2 = clean_missing(df1)
        df_bench = to_lago_benchmark(df2)

        out_bench = cfg.clean_dir / f"{market}_clean_benchmark.csv"
        df_bench.to_csv(out_bench)
        print(f"[OK] Saved benchmark clean: {out_bench}")

        if cfg.write_train_test:
            df_train, df_test = split_like_lago(df_bench, cfg.years_test)
            df_train.to_csv(cfg.clean_dir / f"{market}_train.csv")
            df_test.to_csv(cfg.clean_dir / f"{market}_test.csv")
            print(f"[OK] Saved train/test for {market}")

    print(f"Done. Clean files in: {cfg.clean_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())