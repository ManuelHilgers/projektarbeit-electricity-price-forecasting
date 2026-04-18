from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from entsoe import EntsoePandasClient


# ----------------------------
# Naming convention (matches your data_raw folder)
# ----------------------------
# Example: DE_LU_price_2020.csv, FR_loadfc_2021.csv, ES_windsolarfc_2025.csv
RAW_FILENAME = "{market}_{kind}_{year}.csv"


@dataclass(frozen=True)
class PipelineConfig:
    api_key: str
    raw_dir: Path
    clean_dir: Path
    start: str
    end: str
    years_test: int
    write_train_test: bool
    zones: Dict[str, str]  # market -> bidding zone code


# ----------------------------
# Helper: yearly chunks
# ----------------------------

def year_ranges(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    ranges = []
    cur = pd.Timestamp(year=start.year, month=1, day=1, tz=start.tz)
    if cur < start:
        cur = start
    while cur <= end:
        year_end = pd.Timestamp(year=cur.year, month=12, day=31, hour=23, tz=cur.tz)
        chunk_end = min(year_end, end)
        ranges.append((cur, chunk_end))
        cur = pd.Timestamp(year=cur.year + 1, month=1, day=1, tz=start.tz)
    return ranges


def safe_to_csv(obj: pd.Series | pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj.to_csv(path)


# ----------------------------
# Download from ENTSO-E (raw archive)
# ----------------------------

def download_series_year(
    client: EntsoePandasClient,
    zone_code: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    kind: str,
) -> pd.Series:
    """
    kind in {"price", "loadfc", "windsolarfc"}
    Returns a Series with UTC tz-aware index.
    """
    if kind == "price":
        s = client.query_day_ahead_prices(zone_code, start=start_utc, end=end_utc)
        s.name = "Price"
    elif kind == "loadfc":
        s = client.query_load_forecast(zone_code, start=start_utc, end=end_utc)
        s.name = "Load_Forecast"
    elif kind == "windsolarfc":
        s = client.query_wind_and_solar_forecast(zone_code, start=start_utc, end=end_utc)
        s.name = "WindSolar_Forecast"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]

    # Ensure UTC tz-aware
    if getattr(s.index, "tz", None) is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")

    return s


def download_raw_if_missing(cfg: PipelineConfig) -> None:
    """
    Downloads raw yearly CSVs in your naming convention:
      {market}_price_{year}.csv
      {market}_loadfc_{year}.csv
      {market}_windsolarfc_{year}.csv
    Skips existing files.
    """
    client = EntsoePandasClient(api_key=cfg.api_key)

    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")

    for market, zone_code in cfg.zones.items():
        for (y_start, y_end) in year_ranges(start, end):
            year = y_start.year
            for kind in ["price", "loadfc", "windsolarfc"]:
                out_path = cfg.raw_dir / RAW_FILENAME.format(market=market, kind=kind, year=year)

                if out_path.exists():
                    print(f"[SKIP] Raw exists: {out_path}")
                    continue

                print(f"[DL] {market} {kind} {year}: {y_start} -> {y_end}")
                s = download_series_year(client, zone_code, y_start, y_end, kind)
                safe_to_csv(s, out_path)
                print(f"[OK] Saved raw: {out_path}")


# ----------------------------
# Cleaning (same logic as your working 01_prepare_clean_from_raw_only.py)
# ----------------------------

FILE_RE = re.compile(
    r"^(?P<market>[A-Z_]+)_(?P<kind>price|loadfc|windsolarfc)_(?P<year>\d{4})\.csv$"
)


def read_one_series_csv(path: Path, value_name: str) -> pd.Series:
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
    mapping: Dict[Tuple[str, str, str], Path] = {}
    for f in raw_dir.glob("*.csv"):
        m = FILE_RE.match(f.name)
        if not m:
            continue
        mapping[(m.group("market"), m.group("kind"), m.group("year"))] = f
    return mapping


def build_market_dataframe(mapping: Dict[Tuple[str, str, str], Path], market: str) -> pd.DataFrame:
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
    df["Load_Forecast"] = df["Load_Forecast"].ffill().bfill()
    df["WindSolar_Forecast"] = df["WindSolar_Forecast"].ffill().bfill()
    df["Price"] = df["Price"].interpolate(method="time", limit_direction="both")
    df = df.dropna(how="all")
    return df


def to_lago_benchmark(df_utc: pd.DataFrame) -> pd.DataFrame:
    df = df_utc.copy()

    # ensure UTC tz-aware
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # drop tz -> naive
    df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    df = df.rename(columns={
        "Load_Forecast": "Exogenous 1",
        "WindSolar_Forecast": "Exogenous 2",
    })

    needed = ["Price", "Exogenous 1", "Exogenous 2"]
    return df[needed].astype(float)


def split_like_lago(df: pd.DataFrame, years_test: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_hours = years_test * 364 * 24
    if len(df) <= test_hours:
        raise ValueError(f"Dataset too short for years_test={years_test}.")
    return df.iloc[:-test_hours].copy(), df.iloc[-test_hours:].copy()


def clean_from_raw_dir(cfg: PipelineConfig) -> None:
    cfg.clean_dir.mkdir(parents=True, exist_ok=True)

    mapping = parse_files(cfg.raw_dir)
    if not mapping:
        raise FileNotFoundError(
            f"No matching raw files found in {cfg.raw_dir}. Expected names like DE_LU_price_2020.csv"
        )

    markets = sorted({m for (m, _, _) in mapping.keys()})
    print("Found markets:", markets)

    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")

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


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv: List[str]) -> PipelineConfig:
    p = argparse.ArgumentParser(description="ENTSO-E download -> raw archive -> clean Lago-compatible benchmark CSVs")

    # API key: allow env var fallback to make it nicer
    env_key = os.getenv("ENTSOE_API_KEY")
    p.add_argument("--api-key", default=env_key, help="ENTSO-E API key (or set ENTSOE_API_KEY env var).")

    p.add_argument("--raw-dir", default="data_raw", help="Folder to store raw yearly CSVs")
    p.add_argument("--clean-dir", default="data_clean", help="Folder to store clean benchmark CSVs")
    p.add_argument("--start", default="2020-01-01 00:00", help="Start (UTC)")
    p.add_argument("--end", default="2025-12-31 23:00", help="End (UTC)")
    p.add_argument("--years-test", type=int, default=1, help="Test split size in Lago years (364 days)")
    p.add_argument("--write-train-test", action="store_true", help="Also write train/test CSVs")

    # bidding zones
    p.add_argument("--zone-de-lu", default="10Y1001A1001A82H", help="ENTSO-E bidding zone code for DE-LU")
    p.add_argument("--zone-fr", default="10YFR-RTE------C", help="ENTSO-E bidding zone code for FR")
    p.add_argument("--zone-es", default="10YES-REE------0", help="ENTSO-E bidding zone code for ES")

    a = p.parse_args(argv)

    if not a.api_key:
        raise SystemExit(
            "ERROR: No ENTSO-E API key provided.\n"
            "Provide it via --api-key YOUR_KEY or set env var ENTSOE_API_KEY."
        )

    zones = {"DE_LU": a.zone_de_lu, "FR": a.zone_fr, "ES": a.zone_es}

    return PipelineConfig(
        api_key=a.api_key,
        raw_dir=Path(a.raw_dir),
        clean_dir=Path(a.clean_dir),
        start=a.start,
        end=a.end,
        years_test=a.years_test,
        write_train_test=a.write_train_test,
        zones=zones,
    )


def main(argv: List[str]) -> int:
    cfg = parse_args(argv)

    # Step 1: Download raw if missing (skip existing files)
    download_raw_if_missing(cfg)

    # Step 2: Clean from raw directory to benchmark CSVs
    clean_from_raw_dir(cfg)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))