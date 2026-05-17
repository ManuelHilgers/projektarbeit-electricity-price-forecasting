from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(r"C:\Users\Manuel Hilgers\OneDrive\Desktop\Uni\Projektarbeit")
MARKETS = ["DE_LU", "FR", "ES"]

INPUT_DIR = BASE_DIR / "results_lear_comparison"
OUTPUT_DIR = BASE_DIR / "results_lear_metrics"


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for market in MARKETS:
        file_path = INPUT_DIR / f"{market}_lear_forecast_vs_actual.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing comparison file: {file_path}")

        df = pd.read_csv(file_path)

        forecast_cols = [f"h{h:02d}" for h in range(1, 25)]
        actual_cols = [f"actual_h{h:02d}" for h in range(1, 25)]

        y_pred = df[forecast_cols].to_numpy(dtype=float).reshape(-1)
        y_true = df[actual_cols].to_numpy(dtype=float).reshape(-1)

        market_mae = mae(y_true, y_pred)
        market_rmse = rmse(y_true, y_pred)
        market_mape = mape(y_true, y_pred)
        market_smape = smape(y_true, y_pred)

        summary_rows.append({
            "market": market,
            "MAE": market_mae,
            "RMSE": market_rmse,
            "MAPE_percent": market_mape,
            "sMAPE_percent": market_smape,
            "n_days": df.shape[0],
            "n_hours": len(y_true),
        })

        print(f"\n=== {market} ===")
        print(f"MAE           : {market_mae:.4f}")
        print(f"RMSE          : {market_rmse:.4f}")
        print(f"MAPE (%)      : {market_mape:.4f}")
        print(f"sMAPE (%)     : {market_smape:.4f}")
        print(f"Days evaluated: {df.shape[0]}")
        print(f"Hours total   : {len(y_true)}")

    summary_df = pd.DataFrame(summary_rows)
    out_file = OUTPUT_DIR / "lear_metrics_summary.csv"
    summary_df.to_csv(out_file, index=False)

    print("\nSaved summary to:")
    print(out_file)


if __name__ == "__main__":
    main()