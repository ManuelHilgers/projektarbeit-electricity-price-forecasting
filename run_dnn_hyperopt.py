from pathlib import Path

from epftoolbox.models import hyperparameter_optimizer


# =========================
# Config
# =========================
BASE_DIR = Path(r"C:\Users\Manuel Hilgers\OneDrive\Desktop\Uni\Projektarbeit\Modellcodes Python\DNN")

DATASETS_DIR = BASE_DIR / "datasets"
EXPERIMENTAL_DIR = BASE_DIR / "experimental_files"

MARKETS = ["DE_LU", "FR", "ES"]

NLAYERS = 2
YEARS_TEST = 2
CALIBRATION_WINDOW = 4
SHUFFLE_TRAIN = 1
DATA_AUGMENTATION = 0

MAX_EVALS = 25
NEW_HYPEROPT = 1


def run_market(market: str) -> None:
    experiment_id = f"{market}_final"

    print(f"\n=== Running DNN hyperopt for {market} ===")
    print("Dataset folder      :", DATASETS_DIR)
    print("Experimental folder :", EXPERIMENTAL_DIR)
    print("Dataset             :", market)
    print("Experiment ID       :", experiment_id)
    print("Max evals           :", MAX_EVALS)

    hyperparameter_optimizer(
        path_datasets_folder=str(DATASETS_DIR),
        path_hyperparameters_folder=str(EXPERIMENTAL_DIR),
        new_hyperopt=NEW_HYPEROPT,
        max_evals=MAX_EVALS,
        nlayers=NLAYERS,
        dataset=market,
        years_test=YEARS_TEST,
        calibration_window=CALIBRATION_WINDOW,
        shuffle_train=SHUFFLE_TRAIN,
        data_augmentation=DATA_AUGMENTATION,
        experiment_id=experiment_id,
        begin_test_date=None,
        end_test_date=None,
    )

    print(f"\n[OK] Hyperopt completed for {market}.")


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTAL_DIR.mkdir(parents=True, exist_ok=True)

    for market in MARKETS:
        run_market(market)

    print("\n=== All DNN hyperopt runs completed successfully. ===")


if __name__ == "__main__":
    main()