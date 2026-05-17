from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def read_csv_clean(path: Path) -> pd.DataFrame:
    """Load CSV and drop unnamed index columns if present."""
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def load_xy(data_dir: Path, market: str, split: str):
    """
    Load one split pair:
      X: <MARKET>_X_<split>.csv
      Y: <MARKET>_Y_<split>.csv
    """
    x_path = data_dir / f"{market}_X_{split}.csv"
    y_path = data_dir / f"{market}_Y_{split}.csv"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")

    x_df = read_csv_clean(x_path)
    y_df = read_csv_clean(y_path)

    if "forecast_day" not in x_df.columns:
        raise ValueError(f"'forecast_day' missing in {x_path.name}")
    if "forecast_day" not in y_df.columns:
        raise ValueError(f"'forecast_day' missing in {y_path.name}")

    forecast_day = y_df["forecast_day"].copy()

    x = x_df.drop(columns=["forecast_day"]).copy()
    y = y_df.drop(columns=["forecast_day"]).copy()

    # force numeric
    x = x.apply(pd.to_numeric, errors="raise")
    y = y.apply(pd.to_numeric, errors="raise")

    return x, y, forecast_day, list(y.columns)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric MAPE in percent.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )
    return float(np.mean(ratio) * 100.0)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error in percent.
    Zero targets are excluded to avoid division by zero.
    """
    denom = np.abs(y_true)
    valid = denom != 0

    if not np.any(valid):
        return float("nan")

    ratio = np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])
    return float(np.mean(ratio) * 100.0)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    smape_value = smape(y_true, y_pred)
    mape_value = mape(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape_value,
        "MAPE": mape_value,
    }


def run_smoke_test(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_val: pd.DataFrame,
):
    """
    Very small quick run to verify everything works.
    """
    x_train_small = x_train.iloc[:120].copy()
    y_train_small = y_train.iloc[:120].copy()
    x_val_small = x_val.iloc[:30].copy()
    y_val_small = y_val.iloc[:30].copy()

    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train_small, y_train_small)
    pred = model.predict(x_val_small)

    metrics = evaluate_metrics(y_val_small.to_numpy(), pred)

    print("\n=== SMOKE TEST FINISHED ===")
    print(f"Rows train_small: {len(x_train_small)}")
    print(f"Rows val_small  : {len(x_val_small)}")
    print(f"MAE   : {metrics['MAE']:.4f}")
    print(f"RMSE  : {metrics['RMSE']:.4f}")
    print(f"sMAPE : {metrics['sMAPE']:.2f}%")
    print(f"MAPE  : {metrics['MAPE']:.2f}%")

    return model


def tune_rf(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> dict:
    """
    Manual grid search on validation MAE.
    """
    param_grid = {
        "n_estimators": [300, 600],
        "max_depth": [20, 40, None],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.3],
    }

    best_params = None
    best_mae = float("inf")

    print("\n=== TUNING RF ON train_core / val ===")

    for n_estimators, max_depth, min_samples_leaf, max_features in product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["min_samples_leaf"],
        param_grid["max_features"],
    ):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": 42,
            "n_jobs": -1,
        }

        model = RandomForestRegressor(**params)
        model.fit(x_train, y_train)
        pred_val = model.predict(x_val)
        mae_val = mean_absolute_error(y_val, pred_val)

        print(f"Params={params} -> val MAE={mae_val:.4f}")

        if mae_val < best_mae:
            best_mae = float(mae_val)
            best_params = params

    print("\nBest validation params:")
    print(best_params)
    print(f"Best validation MAE: {best_mae:.4f}")

    return best_params


def save_outputs(
    out_dir: Path,
    market: str,
    forecast_day: pd.Series,
    y_pred: np.ndarray,
    target_columns: list[str],
    metrics: dict,
    best_params: dict,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast_df = pd.DataFrame(y_pred, columns=target_columns)
    forecast_df.insert(0, "forecast_day", forecast_day.reset_index(drop=True))

    forecast_path = out_dir / f"RF_forecast_{market}_final.csv"
    metrics_path = out_dir / f"RF_metrics_{market}_final.json"

    forecast_df.to_csv(forecast_path, index=False)

    payload = {
        "market": market,
        "metrics": metrics,
        "best_params": best_params,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved files:")
    print(forecast_path)
    print(metrics_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder with X/Y CSV files")
    parser.add_argument("--market", required=True, help="Market prefix, e.g. DE_LU")
    parser.add_argument(
        "--mode",
        choices=["smoke", "final"],
        default="smoke",
        help="smoke = quick test, final = full tuning + final forecast",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output folder. Default: <script_folder>/outputs",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else script_dir / "outputs"

    market = args.market

    print(f"\n=== RF RUN FOR {market} ===")
    print(f"Data dir : {data_dir}")
    print(f"Mode     : {args.mode}")
    print(f"Out dir  : {out_dir}")

    # load splits
    x_train_core, y_train_core, _, _ = load_xy(data_dir, market, "train_core")
    x_val, y_val, _, _ = load_xy(data_dir, market, "val")
    x_train_full, y_train_full, _, _ = load_xy(data_dir, market, "train_full")
    x_test, y_test, forecast_day_test, target_columns = load_xy(data_dir, market, "test")

    print("\nLoaded shapes:")
    print(f"train_core X: {x_train_core.shape}, Y: {y_train_core.shape}")
    print(f"val        X: {x_val.shape}, Y: {y_val.shape}")
    print(f"train_full X: {x_train_full.shape}, Y: {y_train_full.shape}")
    print(f"test       X: {x_test.shape}, Y: {y_test.shape}")

    if args.mode == "smoke":
        run_smoke_test(x_train_core, y_train_core, x_val, y_val)
        return

    # full tuning on train_core / val
    best_params = tune_rf(x_train_core, y_train_core, x_val, y_val)

    # final fit on train_full
    print("\n=== FIT FINAL RF ON train_full ===")
    final_model = RandomForestRegressor(**best_params)
    final_model.fit(x_train_full, y_train_full)

    # final forecast on test
    print("=== PREDICT ON TEST ===")
    y_pred_test = final_model.predict(x_test)

    metrics = evaluate_metrics(y_test.to_numpy(), y_pred_test)

    print("\n=== FINAL TEST METRICS ===")
    print(f"MAE   : {metrics['MAE']:.6f}")
    print(f"RMSE  : {metrics['RMSE']:.6f}")
    print(f"sMAPE : {metrics['sMAPE']:.2f}%")
    print(f"MAPE  : {metrics['MAPE']:.2f}%")

    save_outputs(
        out_dir=out_dir,
        market=market,
        forecast_day=forecast_day_test,
        y_pred=y_pred_test,
        target_columns=target_columns,
        metrics=metrics,
        best_params=best_params,
    )


if __name__ == "__main__":
    main()