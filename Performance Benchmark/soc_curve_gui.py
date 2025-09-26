#!/usr/bin/env python3
"""
soc_curve_gui.py

Interactive GUI to visualize SoC efficiency curves from CSVs produced by
cpu_curve_parser.py or gpu_curve_parser.py.

Improvements:
- Deterministic, distinct colors shared across all plots:
  * Colors assigned ONCE by alphanumeric SoC name (case-insensitive).
  * Palette chain: tab20 + tab20b + tab20c (60 colors), then hsv top-up.
- Clean shutdown when window closes.
"""

from __future__ import annotations

import argparse
import os
import tkinter as tk
from typing import Tuple, Dict, List

import matplotlib
matplotlib.use('TkAgg')  # Use the Tk backend for embedding in Tkinter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd  # type: ignore


# ----------------------------
# Data loading / normalization
# ----------------------------

def load_csv(csv_path: str) -> Tuple[pd.DataFrame, str, str, str, str]:
    """Load a CPU/GPU curve CSV and identify key columns.
    Returns (df, model_col, score_col, power_col, eff_col).
    """
    df = pd.read_csv(csv_path)

    # Identify whether this is a CPU or GPU CSV by column names
    if 'CPU' in df.columns and 'GB6_Multi_Score' in df.columns:
        model_col = 'CPU'
        score_col = 'GB6_Multi_Score'
        power_col = 'Board_Power_W'
        eff_col = 'Efficiency'
    elif 'GPU' in df.columns and 'GPU_Score' in df.columns:
        model_col = 'GPU'
        score_col = 'GPU_Score'
        power_col = 'Board_Power_W'
        eff_col = 'Efficiency'
    else:
        raise ValueError(
            "CSV format not recognized. Expected columns for CPU ('CPU', 'GB6_Multi_Score') or "
            "GPU ('GPU', 'GPU_Score')."
        )

    # Compute efficiency if missing
    if eff_col not in df.columns:
        df[eff_col] = df[score_col] / df[power_col].replace(0, pd.NA)

    return df, model_col, score_col, power_col, eff_col


# ----------------------------
# Color handling (stable & distinct)
# ----------------------------

def build_color_map(models: List[str]) -> Dict[str, tuple]:
    """Assign distinct colors deterministically by alphanumeric model name.

    Strategy:
      1) Use qualitative palettes tab20, tab20b, tab20c (60 distinct).
      2) If more needed, top up with evenly spaced hues from hsv.
    """
    models_sorted = sorted(models, key=lambda s: (s or "").lower())

    # Build large qualitative palette
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = cm.get_cmap(cmap_name, 20)
        palette.extend([cmap(i) for i in range(cmap.N)])

    # Top up with hsv if still short
    if len(models_sorted) > len(palette):
        need = len(models_sorted) - len(palette)
        hsv = cm.get_cmap("hsv")
        palette.extend([hsv(i / max(need, 1)) for i in range(need)])

    return {m: palette[i] for i, m in enumerate(models_sorted)}


# ----------------------------
# GUI
# ----------------------------

class CurveGUI:
    """A Tkinter-based GUI for plotting SoC efficiency curves."""

    def __init__(self, csv_path: str) -> None:
        # Load data and columns
        self.df, self.model_col, self.score_col, self.power_col, self.eff_col = load_csv(csv_path)
        self.models = sorted(self.df[self.model_col].dropna().unique(), key=lambda s: s.lower())

        # Deterministic, distinct colors by model name
        self.model_colors: Dict[str, tuple] = build_color_map(self.models)

        # Create main window
        self.root = tk.Tk()
        self.root.title("SoC Efficiency Curve Plotter")

        # Proper cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Frames
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Controls + Figure
        self._init_controls()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._draw_initial()
        self.root.mainloop()

    def on_close(self):
        """Handle window close properly."""
        self.root.quit()
        self.root.destroy()

    def _init_controls(self) -> None:
        """Initialize controls: label and listbox for model selection."""
        label = tk.Label(self.control_frame, text="Select SoCs:")
        label.pack(anchor='nw', pady=(0, 2))

        # Multi-select listbox
        self.listbox = tk.Listbox(
            self.control_frame,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            height=min(len(self.models), 20),
        )
        for model in self.models:
            self.listbox.insert(tk.END, model)
        self.listbox.pack(fill=tk.Y, expand=True)

        # Bind selection event
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

    def _draw_initial(self) -> None:
        """Draw empty plots with axis labels."""
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title("Efficiency vs Score")
        self.ax1.set_xlabel("Score")
        self.ax1.set_ylabel("Efficiency (score/W)")
        self.ax2.set_title("Efficiency vs Board Power")
        self.ax2.set_xlabel("Board Power (W)")
        self.ax2.set_ylabel("Efficiency (score/W)")
        self.canvas.draw()

    def on_select(self, event) -> None:
        """Handle selection changes in the listbox and update plots."""
        selected_indices = self.listbox.curselection()
        selected_models = [self.listbox.get(i) for i in selected_indices]
        self.update_plots(selected_models)

    def update_plots(self, selected: list) -> None:
        """Clear and redraw the plots for the selected models."""
        self.ax1.cla()
        self.ax2.cla()

        # If no models selected, draw empty axes and return
        if not selected:
            self._draw_initial()
            return

        # Plot lines for each selected model
        for model in selected:
            sub = self.df[self.df[self.model_col] == model]
            if sub.empty:
                continue

            # Sort for smooth lines
            sub_score_sorted = sub.sort_values(self.score_col)
            sub_power_sorted = sub.sort_values(self.power_col)
            color = self.model_colors.get(model, None)

            # Slightly thicker lines for visibility when many SOCs overlap
            lw = 2.0

            # Efficiency vs Score
            self.ax1.plot(
                sub_score_sorted[self.score_col],
                sub_score_sorted[self.eff_col],
                label=model,
                color=color,
                linewidth=lw,
            )

            # Efficiency vs Power
            self.ax2.plot(
                sub_power_sorted[self.power_col],
                sub_power_sorted[self.eff_col],
                label=model,
                color=color,
                linewidth=lw,
            )

        # Labels and legends
        self.ax1.set_title("Efficiency vs Score")
        self.ax1.set_xlabel("Score")
        self.ax1.set_ylabel("Efficiency (score/W)")
        self.ax1.legend(fontsize=8, ncol=2)

        self.ax2.set_title("Efficiency vs Board Power")
        self.ax2.set_xlabel("Board Power (W)")
        self.ax2.set_ylabel("Efficiency (score/W)")
        self.ax2.legend(fontsize=8, ncol=2)

        # Autoscale
        self.ax1.relim(); self.ax1.autoscale_view()
        self.ax2.relim(); self.ax2.autoscale_view()
        self.canvas.draw()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive SoC curve plotting GUI")
    parser.add_argument('--input', type=str, required=True, help="Path to the CSV file containing SoC curves")
    args = parser.parse_args()
    if not os.path.isfile(args.input):
        raise SystemExit(f"CSV file '{args.input}' does not exist.")
    CurveGUI(args.input)


if __name__ == '__main__':
    main()
