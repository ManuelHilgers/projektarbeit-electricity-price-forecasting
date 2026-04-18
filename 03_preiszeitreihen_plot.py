from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt


def rgb255(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


# Lehrstuhl-/FAU-Farben
COLORS: Dict[str, tuple[float, float, float]] = {
    "DE_LU": rgb255(106, 138, 34),   # FAPS-Grün dunkel
    "FR":    rgb255(197, 222, 137),  # FAPS-Grün hell
    "ES":    rgb255(63, 95, 68),     # Grün dunkel
}


LABELS_DE: Dict[str, str] = {
    "DE_LU": "Deutschland/Luxemburg (DE-LU)",
    "FR": "Frankreich (FR)",
    "ES": "Spanien (ES)",
}


def load_benchmark(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Erstellt Zeitreihenplots der Day-Ahead-Strompreise für DE_LU, FR und ES.")
    ap.add_argument("--in-dir", default="data_clean", help="Ordner mit den *_clean_benchmark.csv Dateien")
    ap.add_argument("--out", default="preiszeitreihen_maerkte.png", help="Name der Ausgabedatei")
    ap.add_argument("--clip-upper", type=float, default=None,
                    help="Optional: obere y-Achsen-Grenze für bessere Lesbarkeit, z. B. 1000")
    ap.add_argument("--clip-lower", type=float, default=None,
                    help="Optional: untere y-Achsen-Grenze für bessere Lesbarkeit, z. B. -500")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)

    files = {
        "DE_LU": in_dir / "DE_LU_clean_benchmark.csv",
        "FR": in_dir / "FR_clean_benchmark.csv",
        "ES": in_dir / "ES_clean_benchmark.csv",
    }

    for market, f in files.items():
        if not f.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {f}")

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(11, 8), sharex=True)

    order = ["DE_LU", "FR", "ES"]

    for ax, market in zip(axes, order):
        df = load_benchmark(files[market])

        ax.plot(df.index, df["Price"], linewidth=0.6, color=COLORS[market])

        ax.set_ylabel("Preis\n(EUR/MWh)")
        ax.set_title(LABELS_DE[market], fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        if args.clip_lower is not None or args.clip_upper is not None:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(
                args.clip_lower if args.clip_lower is not None else ymin,
                args.clip_upper if args.clip_upper is not None else ymax
            )

    axes[-1].set_xlabel("Zeit")

    fig.suptitle("Zeitliche Entwicklung der Day-Ahead-Strompreise (2020–2025)", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    fig.savefig(args.out, dpi=300)
    plt.close(fig)

    print(f"Gespeichert: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())