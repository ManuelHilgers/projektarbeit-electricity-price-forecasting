from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def load_xy(data_dir: Path, market: str, split: str):
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

    x = x.apply(pd.to_numeric, errors="raise")
    y = y.apply(pd.to_numeric, errors="raise")

    return x, y, forecast_day, list(y.columns)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def tune_rf(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> dict:
    """
    Ex-ante hyperparameter tuning on train_core / val.
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


def rolling_recalibration_forecast(
    x_train_full: pd.DataFrame,
    y_train_full: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    forecast_day_test: pd.Series,
    best_params: dict,
    calibration_window_days: int,
    n_test_days: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Daily recalibration:
    For each test day i, fit on the previous calibration_window_days
    available rows and predict only day i.
    """
    if n_test_days is None:
        n_test_days = len(x_test)

    n_test_days = min(n_test_days, len(x_test))

    x_hist = pd.concat([x_train_full, x_test], axis=0, ignore_index=True)
    y_hist = pd.concat([y_train_full, y_test], axis=0, ignore_index=True)

    train_full_len = len(x_train_full)
    preds = []

    print("\n=== DAILY RECALIBRATION FORECAST ===")
    print(f"Calibration window (days): {calibration_window_days}")
    print(f"Number of test days      : {n_test_days}")

    start_time = time.time()

    for i in range(n_test_days):
        current_global_idx = train_full_len + i
        train_end = current_global_idx
        train_start = max(0, train_end - calibration_window_days)

        x_recal = x_hist.iloc[train_start:train_end].copy()
        y_recal = y_hist.iloc[train_start:train_end].copy()
        x_next = x_test.iloc[[i]].copy()

        model = RandomForestRegressor(**best_params)
        model.fit(x_recal, y_recal)
        pred_next = model.predict(x_next)[0]
        preds.append(pred_next)

        if (i + 1) % 25 == 0 or i == 0 or (i + 1) == n_test_days:
            elapsed = time.time() - start_time
            print(
                f"Processed {i + 1}/{n_test_days} test days "
                f"(elapsed: {elapsed/60:.2f} min)"
            )

    pred_array = np.array(preds)
    y_true_eval = y_test.iloc[:n_test_days].to_numpy()
    forecast_days_eval = forecast_day_test.iloc[:n_test_days].reset_index(drop=True)

    return pred_array, y_true_eval, forecast_days_eval


def save_outputs(
    out_dir: Path,
    market: str,
    mode: str,
    forecast_day: pd.Series,
    y_pred: np.ndarray,
    target_columns: list[str],
    metrics: dict,
    best_params: dict,
    calibration_window_days: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast_df = pd.DataFrame(y_pred, columns=target_columns)
    forecast_df.insert(0, "forecast_day", forecast_day.reset_index(drop=True))

    suffix = "smoke" if mode == "smoke" else "final"

    forecast_path = out_dir / f"RF_rolling_forecast_{market}_{suffix}.csv"
    metrics_path = out_dir / f"RF_rolling_metrics_{market}_{suffix}.json"

    forecast_df.to_csv(forecast_path, index=False)

    payload = {
        "market": market,
        "mode": mode,
        "metrics": metrics,
        "best_params": best_params,
        "calibration_window_days": calibration_window_days,
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
        help="smoke = first 14 test days, final = full test period",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output folder. Default: <script_folder>/outputs",
    )
    parser.add_argument(
        "--smoke-days",
        type=int,
        default=14,
        help="How many test days to evaluate in smoke mode",
    )
    parser.add_argument(
        "--calibration-window-days",
        type=int,
        default=None,
        help="Rolling calibration window. Default: len(train_full)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else script_dir / "outputs"
    market = args.market

    print(f"\n=== RF ROLLING RUN FOR {market} ===")
    print(f"Data dir : {data_dir}")
    print(f"Mode     : {args.mode}")
    print(f"Out dir  : {out_dir}")

    x_train_core, y_train_core, _, _ = load_xy(data_dir, market, "train_core")
    x_val, y_val, _, _ = load_xy(data_dir, market, "val")
    x_train_full, y_train_full, _, _ = load_xy(data_dir, market, "train_full")
    x_test, y_test, forecast_day_test, target_columns = load_xy(data_dir, market, "test")

    print("\nLoaded shapes:")
    print(f"train_core X: {x_train_core.shape}, Y: {y_train_core.shape}")
    print(f"val        X: {x_val.shape}, Y: {y_val.shape}")
    print(f"train_full X: {x_train_full.shape}, Y: {y_train_full.shape}")
    print(f"test       X: {x_test.shape}, Y: {y_test.shape}")

    calibration_window_days = (
        args.calibration_window_days
        if args.calibration_window_days is not None
        else len(x_train_full)
    )

    best_params = tune_rf(x_train_core, y_train_core, x_val, y_val)

    n_eval_days = args.smoke_days if args.mode == "smoke" else len(x_test)

    y_pred, y_true_eval, forecast_days_eval = rolling_recalibration_forecast(
        x_train_full=x_train_full,
        y_train_full=y_train_full,
        x_test=x_test,
        y_test=y_test,
        forecast_day_test=forecast_day_test,
        best_params=best_params,
        calibration_window_days=calibration_window_days,
        n_test_days=n_eval_days,
    )

    metrics = evaluate_metrics(y_true_eval, y_pred)

    print("\n=== FINAL METRICS FOR CURRENT RUN ===")
    print(f"MAE   : {metrics['MAE']:.6f}")
    print(f"RMSE  : {metrics['RMSE']:.6f}")
    print(f"sMAPE : {metrics['sMAPE']:.2f}%")
    print(f"MAPE  : {metrics['MAPE']:.2f}%")

    save_outputs(
        out_dir=out_dir,
        market=market,
        mode=args.mode,
        forecast_day=forecast_days_eval,
        y_pred=y_pred,
        target_columns=target_columns,
        metrics=metrics,
        best_params=best_params,
        calibration_window_days=calibration_window_days,
    )


if __name__ == "__main__":
    main()