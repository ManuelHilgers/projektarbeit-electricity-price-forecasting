from pathlib import Path

from epftoolbox.models import hyperparameter_optimizer


# =========================
# Config
# =========================
BASE_DIR = Path(r"C:\Users\Manuel Hilgers\OneDrive\Desktop\Uni\Projektarbeit\Modellcodes Python\DNN")

DATASETS_DIR = BASE_DIR / "datasets"
EXPERIMENTAL_DIR = BASE_DIR / "experimental_files"

DATASET = "DE_LU"   # custom dataset name -> expects DE_LU.csv in DATASETS_DIR
NLAYERS = 2
YEARS_TEST = 2
CALIBRATION_WINDOW = 4   # in years for DNN hyperopt/evaluation
SHUFFLE_TRAIN = 1
DATA_AUGMENTATION = 0
EXPERIMENT_ID = "DE_LU_test_smoke"
MAX_EVALS = 5
NEW_HYPEROPT = 1


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset folder      :", DATASETS_DIR)
    print("Experimental folder :", EXPERIMENTAL_DIR)
    print("Dataset             :", DATASET)
    print("Experiment ID       :", EXPERIMENT_ID)

    hyperparameter_optimizer(
        path_datasets_folder=str(DATASETS_DIR),
        path_hyperparameters_folder=str(EXPERIMENTAL_DIR),
        new_hyperopt=NEW_HYPEROPT,
        max_evals=MAX_EVALS,
        nlayers=NLAYERS,
        dataset=DATASET,
        years_test=YEARS_TEST,
        calibration_window=CALIBRATION_WINDOW,
        shuffle_train=SHUFFLE_TRAIN,
        data_augmentation=DATA_AUGMENTATION,
        experiment_id=EXPERIMENT_ID,
        begin_test_date=None,
        end_test_date=None,
    )

    print("\n[OK] DNN hyperopt smoketest completed.")


if __name__ == "__main__":
    main()