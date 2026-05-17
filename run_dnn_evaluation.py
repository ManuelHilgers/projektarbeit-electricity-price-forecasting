from pathlib import Path

from epftoolbox.models import evaluate_dnn_in_test_dataset


# =========================
# Config
# =========================
BASE_DIR = Path(r"C:\Users\Manuel Hilgers\OneDrive\Desktop\Uni\Projektarbeit\Modellcodes Python\DNN")

DATASETS_DIR = BASE_DIR / "datasets"
EXPERIMENTAL_DIR = BASE_DIR / "experimental_files"
FORECAST_DIR = BASE_DIR / "forecast_files"

# DE_LU und FR ist bereits fertig
MARKETS = ["ES"]

NLAYERS = 2
YEARS_TEST = 2
CALIBRATION_WINDOW = 4
SHUFFLE_TRAIN = 1
DATA_AUGMENTATION = 0
NEW_RECALIBRATION = True


def run_market(market: str) -> None:
    experiment_id = f"{market}_final"

    print(f"\n=== Running DNN evaluation for {market} ===")
    print("Dataset folder       :", DATASETS_DIR)
    print("Hyperparameter folder:", EXPERIMENTAL_DIR)
    print("Forecast folder      :", FORECAST_DIR)
    print("Dataset              :", market)
    print("Experiment ID        :", experiment_id)

    forecast = evaluate_dnn_in_test_dataset(
        experiment_id=experiment_id,
        path_datasets_folder=str(DATASETS_DIR),
        path_hyperparameter_folder=str(EXPERIMENTAL_DIR),
        path_recalibration_folder=str(FORECAST_DIR),
        nlayers=NLAYERS,
        dataset=market,
        years_test=YEARS_TEST,
        shuffle_train=SHUFFLE_TRAIN,
        data_augmentation=DATA_AUGMENTATION,
        calibration_window=CALIBRATION_WINDOW,
        new_recalibration=NEW_RECALIBRATION,
        begin_test_date=None,
        end_test_date=None,
    )

    print(f"\n[OK] DNN evaluation completed for {market}.")
    print("Forecast shape:", forecast.shape)


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)

    for market in MARKETS:
        run_market(market)

    print("\n=== All DNN evaluation runs completed successfully. ===")


if __name__ == "__main__":
    main()