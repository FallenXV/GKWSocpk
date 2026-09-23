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
import sys
from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from cpu_benchmarks import CPU_BENCHMARKS, CORE_COLUMNS, cpu_profile_label


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
        matches = [(name, benchmark) for name, benchmark in CPU_BENCHMARKS.items()
                   if benchmark.score_column in df.columns]
        if len(matches) != 1:
            raise ValueError("CPU CSV must contain exactly one recognized benchmark score column.")
        name, benchmark = matches[0]
        score_label = benchmark.score_column
        if "Benchmark" in df and not df["Benchmark"].dropna().eq(name).all():
            raise ValueError("Benchmark metadata does not match the score column.")
        if benchmark.single_core:
            if not {"Core", "Core_Group"}.issubset(df.columns):
                raise ValueError("SPEC CSV requires Core and Core_Group columns.")
            for column in CORE_COLUMNS:
                if column not in df:
                    df[column] = ""
                df[column] = df[column].fillna("").astype(str).str.strip()
            df["CPU"] = [cpu_profile_label(*values) for values in
                         df[["CPU", *CORE_COLUMNS]].itertuples(index=False, name=None)]
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
    """Integrate piecewise-linear score curves over their shared power interval.

    Score/W is integrated analytically on each linear score segment, so adding
    collinear samples does not change the mean or weight one region more heavily.
    """
    curves = []
    for name, rows in df.groupby('Model', dropna=False):
        points = rows[['Board_Power_W', 'Score']].drop_duplicates().sort_values('Board_Power_W')
        values = points.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Power and score must be finite and positive.')
        if points.Board_Power_W.duplicated().any():
            raise ValueError(f'{name}: conflicting scores at duplicate power.')
        if len(points) < 2:
            raise ValueError(f'{name}: at least two distinct powers are required.')
        curves.append((name, values[:, 0], values[:, 1]))
    if not curves:
        raise ValueError('No curves to analyse.')
    low = max(x[0] for _, x, _ in curves)
    high = min(x[-1] for _, x, _ in curves)
    if low >= high:
        raise ValueError('No shared power interval across the selected profiles.')
    summary = []
    for name, powers, scores in curves:
        x = np.unique(np.r_[low, powers[(powers > low) & (powers < high)], high])
        y = np.interp(x, powers, scores)
        widths = np.diff(x)
        slopes = np.diff(y) / widths
        intercepts = y[:-1] - slopes * x[:-1]
        efficiency_integral = np.sum(slopes * widths + intercepts * np.log(x[1:] / x[:-1]))
        summary.append(dict(Model=name, Power_Min_W=low, Power_Max_W=high,
                            Avg_Power_W=(low + high) / 2,
                            Avg_Score=np.sum((y[:-1] + y[1:]) * widths / 2) / (high - low),
                            Avg_Efficiency=efficiency_integral / (high - low)))
    return pd.DataFrame(summary)


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
        cmap = plt.get_cmap(cmap_name, 20)
        palette.extend([cmap(i) for i in range(cmap.N)])

    # Top up with hsv if needed
    if len(models_sorted) > len(palette):
        need = len(models_sorted) - len(palette)
        hsv = plt.get_cmap("hsv")
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
    plt.ylabel('Power-interval mean efficiency (score/W)')
    plt.title(f'Shared power interval: {summary.Power_Min_W.iloc[0]:.3f}–{summary.Power_Max_W.iloc[0]:.3f} W')
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
        raise SystemExit(1)

    try:
        df, model_label = load_and_normalize(args.input)
    except Exception as exc:
        print(f"Failed to load data: {exc}")
        raise SystemExit(1)

    # Build a shared, stable color map (by alphanumeric model order)
    models_all = list(sorted(df['Model'].dropna().unique(), key=lambda s: s.lower()))
    color_map = build_color_map(models_all)

    # Compute and print summary statistics
    try:
        summary = compute_statistics(df)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("Power-weighted means over a shared interval; piecewise-linear estimates, no extrapolation.")
    print(summary.to_string(index=False))
    if args.save:
        from socpk_client import new_snapshot
        with new_snapshot("curve_summary.csv") as output:
            summary.to_csv(output, index=False)
        print(f"Wrote summary: {output.name}")

    # Plot visualizations using the same color map
    plot_efficiency_bar(summary, colors=color_map, save=args.save)
    plot_efficiency_vs_score(df, colors=color_map, save=args.save)
    plot_efficiency_vs_power(df, colors=color_map, save=args.save)


if __name__ == '__main__':
    main()
