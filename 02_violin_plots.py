from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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


def load_benchmark(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def trim_series_by_quantiles(s: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    return s[(s >= lower) & (s <= upper)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Erstellt Violin-Plots der Day-Ahead-Strompreise für DE_LU, FR und ES.")
    ap.add_argument("--in-dir", default="data_clean", help="Ordner mit den *_clean_benchmark.csv Dateien")
    ap.add_argument("--out", default="violin_preise_ohne_extremwerte.png", help="Name der Ausgabedatei")
    ap.add_argument("--lower-q", type=float, default=0.01,
                    help="Unteres Quantil für das Entfernen von Extremwerten, z. B. 0.01")
    ap.add_argument("--upper-q", type=float, default=0.99,
                    help="Oberes Quantil für das Entfernen von Extremwerten, z. B. 0.99")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("*_clean_benchmark.csv"))
    if not files:
        raise FileNotFoundError(f"Keine *_clean_benchmark.csv Dateien in {in_dir} gefunden.")

    data: List[Tuple[str, pd.Series]] = []
    for f in files:
        market = f.name.replace("_clean_benchmark.csv", "")
        df = load_benchmark(f)

        s = df["Price"].dropna()
        s_trimmed = trim_series_by_quantiles(s, args.lower_q, args.upper_q)

        data.append((market, s_trimmed))

    order = ["DE_LU", "FR", "ES"]
    data = sorted(data, key=lambda x: order.index(x[0]) if x[0] in order else 999)

    labels = [m for m, _ in data]
    values = [s.values for _, s in data]

    fig, ax = plt.subplots(figsize=(9, 6))

    parts = ax.violinplot(values, showmedians=True, showextrema=True)

    for i, body in enumerate(parts["bodies"]):
        mkt = labels[i]
        color = COLORS.get(mkt, rgb255(176, 188, 196))
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.85)

    # Median und Linien etwas deutlicher
    for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.0)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title("Verteilung der Day-Ahead-Strompreise (2020–2025)")
    ax.set_xlabel("Markt")
    ax.set_ylabel("Preis (EUR/MWh)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    plt.close(fig)

    print(f"Gespeichert: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())