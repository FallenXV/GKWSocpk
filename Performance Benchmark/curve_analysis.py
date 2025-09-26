#!/usr/bin/env python3
"""
curve_analysis.py

This standalone script reads a CSV file produced by either
`cpu_curve_parser.py` or `gpu_curve_parser.py`, computes basic
statistics on the board power, performance score and efficiency for
each model, and generates several visualizations:

* Average efficiency bar chart
* Efficiency vs Score line plot
* Efficiency vs Board Power line plot

Colors are assigned ONCE (by alphanumeric model order, case-insensitive)
and reused across every plot so each model keeps the same color everywhere.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple, Dict, List

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ----------------------------
# Data loading / normalization
# ----------------------------

def load_and_normalize(csv_path: str) -> Tuple[pd.DataFrame, str]:
    """Load a CSV and normalize column names.

    Detects CPU/GPU data by column names and renames to common names:
    ['Model', 'Board_Power_W', 'Score', 'Efficiency'].
    If Efficiency is missing, computes Score / Board_Power_W (safe for 0).

    Returns
    -------
    (df, model_label) where model_label is 'CPU' or 'GPU'
    """
    df = pd.read_csv(csv_path)

    # Determine which label column exists
    if 'CPU' in df.columns:
        model_label = 'CPU'
        score_label = 'GB6_Multi_Score'
    elif 'GPU' in df.columns:
        model_label = 'GPU'
        score_label = 'GPU_Score'
    else:
        raise ValueError("CSV file must contain either a 'CPU' or 'GPU' column.")

    # Basic column presence checks
    if score_label not in df.columns:
        raise ValueError(f"Missing expected score column '{score_label}'.")
    if 'Board_Power_W' not in df.columns:
        raise ValueError("Missing expected 'Board_Power_W' column.")

    # Rename columns for uniform processing
    df = df.rename(columns={model_label: 'Model', score_label: 'Score'})

    # Ensure efficiency column exists (avoid div-by-zero)
    if 'Efficiency' not in df.columns:
        power = df['Board_Power_W'].replace(0, np.nan)
        df['Efficiency'] = df['Score'] / power
        # Optional: if you prefer 0 instead of NaN when power is 0, uncomment:
        # df['Efficiency'] = df['Efficiency'].fillna(0)

    return df[['Model', 'Board_Power_W', 'Score', 'Efficiency']], model_label


# ----------------------------
# Stats
# ----------------------------

def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean board power, score and efficiency per model."""
    summary = df.groupby('Model', dropna=True).agg(
        Avg_Power_W=('Board_Power_W', 'mean'),
        Avg_Score=('Score', 'mean'),
        Avg_Efficiency=('Efficiency', 'mean'),
    ).reset_index()
    return summary


# ----------------------------
# Color handling (stable & shared)
# ----------------------------

def build_color_map(models: List[str]) -> Dict[str, tuple]:
    """Deterministically assign colors by alphanumeric model order (case-insensitive).

    Strategy:
      1) Use tab20, tab20b, tab20c (20*3 = 60 distinct qualitative colors).
      2) If there are more models than 60, top up with evenly spaced hues from hsv.
    """
    models_sorted = sorted(models, key=lambda s: (s or "").lower())

    # Build large qualitative palette
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = cm.get_cmap(cmap_name, 20)
        palette.extend([cmap(i) for i in range(cmap.N)])

    # Top up with hsv if needed
    if len(models_sorted) > len(palette):
        need = len(models_sorted) - len(palette)
        hsv = cm.get_cmap("hsv")
        # Use 0..need-1 over need to space hues; skip the very last 1.0 to avoid repeat of 0.0
        palette.extend([hsv(i / max(need, 1)) for i in range(need)])

    return {m: palette[i] for i, m in enumerate(models_sorted)}


# ----------------------------
# Plots (use shared colors)
# ----------------------------

def plot_efficiency_bar(summary: pd.DataFrame, colors: Dict[str, tuple], save: bool = False) -> None:
    """Bar chart of average efficiency for each model, colored consistently."""
    if summary.empty:
        return
    ordered = summary.sort_values('Avg_Efficiency', ascending=False)
    bar_colors = [colors.get(m, (0.5, 0.5, 0.5, 1.0)) for m in ordered['Model']]

    plt.figure()
    plt.bar(ordered['Model'], ordered['Avg_Efficiency'], color=bar_colors)
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel('Average Efficiency (score/W)')
    plt.title('Average Efficiency per Model')
    plt.tight_layout()
    if save:
        plt.savefig('efficiency_bar.png', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_efficiency_vs_score(df: pd.DataFrame, colors: Dict[str, tuple], save: bool = False) -> None:
    """Line plot: efficiency vs score for each model, colored consistently."""
    if df.empty:
        return
    models = sorted(df['Model'].dropna().unique(), key=lambda s: s.lower())

    plt.figure()
    for model in models:
        sub = df[df['Model'] == model].sort_values('Score')
        if sub.empty:
            continue
        plt.plot(
            sub['Score'],
            sub['Efficiency'],
            label=model,
            color=colors.get(model, None),
        )
    plt.xlabel('Score')
    plt.ylabel('Efficiency (score/W)')
    plt.title('Efficiency vs Score')
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    if save:
        plt.savefig('efficiency_vs_score.png', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_efficiency_vs_power(df: pd.DataFrame, colors: Dict[str, tuple], save: bool = False) -> None:
    """Line plot: efficiency vs board power for each model, colored consistently."""
    if df.empty:
        return
    models = sorted(df['Model'].dropna().unique(), key=lambda s: s.lower())

    plt.figure()
    for model in models:
        sub = df[df['Model'] == model].sort_values('Board_Power_W')
        if sub.empty:
            continue
        plt.plot(
            sub['Board_Power_W'],
            sub['Efficiency'],
            label=model,
            color=colors.get(model, None),
        )
    plt.xlabel('Board Power (W)')
    plt.ylabel('Efficiency (score/W)')
    plt.title('Efficiency vs Board Power')
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    if save:
        plt.savefig('efficiency_vs_power.png', dpi=300)
        plt.close()
    else:
        plt.show()


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Analyse curve CSV and plot efficiency statistics')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the CSV file produced by cpu_curve_parser or gpu_curve_parser')
    parser.add_argument('--save', action='store_true',
                        help='Save plots as PNG files instead of displaying them')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file {args.input} does not exist.")
        return

    try:
        df, model_label = load_and_normalize(args.input)
    except Exception as exc:
        print(f"Failed to load data: {exc}")
        return

    # Build a shared, stable color map (by alphanumeric model order)
    models_all = list(sorted(df['Model'].dropna().unique(), key=lambda s: s.lower()))
    color_map = build_color_map(models_all)

    # Compute and print summary statistics
    summary = compute_statistics(df)
    print(summary)

    # Plot visualizations using the same color map
    plot_efficiency_bar(summary, colors=color_map, save=args.save)
    plot_efficiency_vs_score(df, colors=color_map, save=args.save)
    plot_efficiency_vs_power(df, colors=color_map, save=args.save)


if __name__ == '__main__':
    main()
