#!/usr/bin/env python3
"""Polished comparison dashboard for every SoCPK CSV in a project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cpu_benchmarks import CPU_BENCHMARKS, CORE_COLUMNS, cpu_profile_label

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Segoe UI",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


APP_BG = "#0b1020"
PANEL = "#121a2d"
PANEL_2 = "#172138"
PLOT_BG = "#10182a"
TEXT = "#f3f6ff"
MUTED = "#8e9bb6"
GRID = "#293552"
ACCENT = "#7c6cff"
CYAN = "#38d6d0"
GREEN = "#63d69d"
ORANGE = "#ffb15c"
RED = "#ff718b"

PALETTE = (
    "#8b7cff",
    "#35d0c8",
    "#ffad5a",
    "#ff6f91",
    "#57a8ff",
    "#8bd45a",
    "#e17bff",
    "#ffd166",
    "#45b7d1",
    "#f28482",
    "#84dcc6",
    "#a9def9",
)


@dataclass(frozen=True)
class DatasetDefinition:
    key: str
    title: str
    kicker: str
    id_column: str
    required: frozenset[str]
    numeric: tuple[str, ...]
    dedupe: tuple[str, ...]
    views: tuple[str, ...]
    score_column: str = ""
    score_label: str = ""
    score_decimals: int = 0
    tab_label: str = ""


@dataclass(frozen=True)
class HoverPoint:
    x: float
    y: float
    label: str
    details: str
    color: str


@dataclass(frozen=True)
class BatteryCapacityMeasurement:
    advertised_mah: int
    measured_mah: int

    @property
    def shortfall_mah(self) -> int:
        return self.advertised_mah - self.measured_mah

    @property
    def shortfall_pct(self) -> float:
        return self.shortfall_mah / self.advertised_mah * 100.0


_BATTERY_BRAND_ALIASES = {
    "苹果": "apple",
    "三星": "samsung",
    "谷歌": "google",
    "华为": "huawei",
    "一加": "oneplus",
    "真我": "realme",
    "红魔": "redmagic",
    "努比亚": "nubia",
    "小米": "xiaomi",
    "红米": "redmi",
    "荣耀": "honor",
}


def _battery_key(brand: str, model: str) -> tuple[str, str]:
    normalized_brand = _BATTERY_BRAND_ALIASES.get(brand.strip().casefold(), brand.strip().casefold())
    normalized_model = (
        model.strip()
        .casefold()
        .replace("至尊版", "supreme edition")
        .replace("+", " plus ")
    )
    normalized_model = "".join(character for character in normalized_model if character.isalnum())
    return normalized_brand, normalized_model


_GEEKERWAN_CAPACITY_ROWS = (
    ("Samsung", "S26 Ultra", 5000, 4775),
    ("Google", "Pixel 10 Pro XL", 5200, 5134),
    ("Apple", "iPhone 17 Pro Max", 4823, 4718),
    ("Huawei", "Mate 80 Pro", 5750, 5552),
    ("Huawei", "Mate 70 Pro+", 5700, 5431),
    ("OPPO", "Find X9", 7025, 6873),
    ("OnePlus", "Ace 6 Supreme Edition", 8600, 8202),
    ("OnePlus", "Ace 6", 7800, 7472),
    ("OnePlus", "15", 7300, 7084),
    ("OnePlus", "13", 6000, 5252),
    ("Realme", "GT8 Pro", 7000, 6736),
    ("Realme", "GT8", 7000, 6598),
    ("vivo", "X300s", 7100, 6520),
    ("vivo", "X300", 6040, 5588),
    ("iQOO", "15T", 8000, 7368),
    ("iQOO", "15", 7000, 6155),
    ("iQOO", "13", 6150, 5452),
    ("RedMagic", "11 Pro", 8000, 7243),
    ("Nubia", "Z80 Ultra", 7200, 6276),
    ("Xiaomi", "17 Pro Max", 7500, 6581),
    ("Xiaomi", "17", 7000, 6120),
    ("Xiaomi", "15", 5400, 4789),
    ("Redmi", "K90 Max", 8550, 8002),
    ("Redmi", "K90", 7100, 6572),
    ("Redmi", "K80 Pro", 6000, 5565),
    ("Honor", "WIN", 10000, 8568),
    ("Honor", "GT Pro", 7200, 6151),
)

GEEKERWAN_CAPACITY_MEASUREMENTS = {
    _battery_key(brand, model): BatteryCapacityMeasurement(advertised, measured)
    for brand, model, advertised, measured in _GEEKERWAN_CAPACITY_ROWS
}


def add_geekerwan_capacity_overlay(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach static measured-capacity points without changing pulled SoCPK values."""
    frame = frame.copy()
    measurements = [
        GEEKERWAN_CAPACITY_MEASUREMENTS.get(_battery_key(str(brand), str(model)))
        for brand, model in zip(frame["brand"], frame["model"])
    ]
    advertised = pd.Series(
        [measurement.advertised_mah if measurement else np.nan for measurement in measurements],
        index=frame.index,
        dtype=float,
    )
    measured = pd.Series(
        [measurement.measured_mah if measurement else np.nan for measurement in measurements],
        index=frame.index,
        dtype=float,
    )
    ratio = measured / advertised
    frame["geekerwanAdvertisedMah"] = advertised
    frame["geekerwanMeasuredMah"] = measured
    frame["geekerwanShortfallMah"] = advertised - measured
    frame["geekerwanShortfallPct"] = (advertised - measured) / advertised * 100.0
    frame["geekerwanCapacityWh"] = frame["capacityWh"] * ratio
    frame["geekerwanAvgPowerW"] = frame["avgPowerW"] * ratio
    frame["geekerwanMinPerWh"] = frame["minPerWh"] / ratio
    return frame


DATASET_DEFINITIONS = {
    **{
        benchmark.dataset_key: DatasetDefinition(
            key=benchmark.dataset_key,
            title="Single-core efficiency" if benchmark.single_core else "Multi-core efficiency",
            kicker=benchmark.title,
            id_column="CPU",
            required=frozenset({"CPU", "Board_Power_W", benchmark.score_column}
                               | ({"Core", "Core_Group"} if benchmark.single_core else set())),
            numeric=("Board_Power_W", benchmark.score_column, "Efficiency"),
            dedupe=("CPU", *CORE_COLUMNS, "Board_Power_W", benchmark.score_column)
                   if benchmark.single_core else ("CPU", "Board_Power_W", benchmark.score_column),
            views=("Efficiency curve", "Performance curve", "Efficiency vs score"),
            score_column=benchmark.score_column,
            score_label=(f"SPEC 2026 {benchmark.suite} score" if benchmark.single_core
                         else benchmark.title + " score"),
            score_decimals=3 if benchmark.single_core else 0,
            tab_label={"GB6": "GB6 MULTI", "GB7": "GB7 MULTI",
                       "SPEC2026_INT": "SPEC26 INT", "SPEC2026_FP": "SPEC26 FP"}[name],
        )
        for name, benchmark in CPU_BENCHMARKS.items()
    },
    "GPU": DatasetDefinition(
        key="GPU",
        title="Mobile GPU performance",
        kicker="3DMark Steel Nomad Light",
        score_column="GPU_Score", score_label="Steel Nomad Light score",
        id_column="GPU",
        required=frozenset({"GPU", "Board_Power_W", "GPU_Score"}),
        numeric=("Board_Power_W", "GPU_Score", "Efficiency"),
        dedupe=("GPU", "Board_Power_W", "GPU_Score"),
        views=("Performance curve", "Efficiency curve", "Efficiency vs score"),
    ),
    "Laptop GPU": DatasetDefinition(
        key="Laptop GPU",
        title="Laptop GPU performance",
        kicker="3DMark Time Spy Graphics",
        score_column="GPU_Score", score_label="Time Spy graphics score",
        id_column="GPU",
        required=frozenset({"GPU", "Board_Power_W", "GPU_Score"}),
        numeric=("Board_Power_W", "GPU_Score", "Efficiency"),
        dedupe=("GPU", "Board_Power_W", "GPU_Score"),
        views=("Performance curve", "Efficiency curve", "Efficiency vs score"),
    ),
    "Battery": DatasetDefinition(
        key="Battery",
        title="Battery endurance",
        kicker="Device runtime and efficiency",
        id_column="model",
        required=frozenset({"brand", "model", "minutes", "capacityWh", "avgPowerW"}),
        numeric=("minutes", "hours", "capacityWh", "avgPowerW", "minPerWh"),
        dedupe=("brand", "model", "os", "minutes", "capacityWh"),
        views=("Runtime vs capacity", "Energy efficiency", "Average power draw"),
    ),
}

CPU_DATASETS = frozenset(benchmark.dataset_key for benchmark in CPU_BENCHMARKS.values())
CURVE_DATASETS = CPU_DATASETS | {"GPU", "Laptop GPU"}


def classify_columns(columns: Iterable[str]) -> str | None:
    """Return the recognized dataset kind for a CSV schema."""
    column_set = frozenset(columns)
    if sum(benchmark.score_column in column_set for benchmark in CPU_BENCHMARKS.values()) > 1:
        raise ValueError("CPU benchmarks must be stored in separate CSVs.")
    for key in (*[b.dataset_key for b in CPU_BENCHMARKS.values()], "GPU", "Battery"):
        definition = DATASET_DEFINITIONS[key]
        if definition.required.issubset(column_set):
            return key
    return None


def classify_frame(frame: pd.DataFrame, path: Path) -> str | None:
    """Classify a CSV, separating incompatible mobile and laptop GPU scores."""
    kind = classify_columns(frame.columns)
    if kind != "GPU":
        return kind

    source_name = str(path).casefold().replace("-", "_")
    if "laptop_gpu" in source_name or "laptopgpu" in source_name:
        return "Laptop GPU"

    descriptor_columns = [
        column
        for column in ("Platform", "platform", "Benchmark", "benchmark")
        if column in frame
    ]
    descriptors = " ".join(
        frame[column].dropna().astype(str).str.casefold().str.cat(sep=" ")
        for column in descriptor_columns
    )
    if "laptop" in descriptors or "time spy" in descriptors:
        return "Laptop GPU"

    power = pd.to_numeric(frame.get("Board_Power_W"), errors="coerce")
    score = pd.to_numeric(frame.get("GPU_Score"), errors="coerce")
    if power.max(skipna=True) > 30 or score.max(skipna=True) > 5000:
        return "Laptop GPU"
    return "GPU"


def discover_csv_files(root: Path) -> list[Path]:
    """Find project CSVs, excluding virtual environments and tool metadata."""
    ignored = {".git", ".venv", "venv", "__pycache__", ".idea", ".pytest_cache"}
    files: list[Path] = []
    for path in root.rglob("*.csv"):
        if any(part in ignored for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return sorted(files, key=lambda item: str(item).casefold())


def _source_label(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _prepare_frame(
    frame: pd.DataFrame,
    kind: str,
    source: Path,
    root: Path,
) -> pd.DataFrame:
    definition = DATASET_DEFINITIONS[kind]
    frame = frame.copy()
    for column in definition.numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if kind in CURVE_DATASETS:
        score_column = definition.score_column
        if "Efficiency" not in frame:
            frame["Efficiency"] = frame[score_column] / frame["Board_Power_W"]
        else:
            missing = frame["Efficiency"].isna()
            frame.loc[missing, "Efficiency"] = (
                frame.loc[missing, score_column] / frame.loc[missing, "Board_Power_W"]
            )
        frame = frame[
            frame[definition.id_column].notna()
            & frame["Board_Power_W"].gt(0)
            & frame[score_column].notna()
        ]
        frame["__label"] = frame[definition.id_column].astype(str).str.strip()
        if kind in CPU_DATASETS:
            benchmark_name, benchmark = next((name, item) for name, item in CPU_BENCHMARKS.items()
                                             if item.dataset_key == kind)
            if "Benchmark" in frame and not frame["Benchmark"].dropna().eq(benchmark_name).all():
                raise ValueError("Benchmark metadata does not match the score column.")
            if benchmark.single_core:
                for column in CORE_COLUMNS:
                    if column not in frame:
                        frame[column] = ""
                    frame[column] = frame[column].fillna("").astype(str).str.strip()
                if frame["Core"].eq("").any() or frame["Core_Group"].eq("").any():
                    raise ValueError("SPEC profiles require both Core and Core_Group.")
                frame["__label"] = [cpu_profile_label(*values) for values in
                                     frame[["CPU", *CORE_COLUMNS]].itertuples(index=False, name=None)]

    else:
        if "hours" not in frame:
            frame["hours"] = frame["minutes"] / 60.0
        if "minPerWh" not in frame:
            frame["minPerWh"] = frame["minutes"] / frame["capacityWh"]
        frame = frame[
            frame["model"].notna()
            & frame["minutes"].gt(0)
            & frame["capacityWh"].gt(0)
        ]
        brand = frame["brand"].fillna("").astype(str).str.strip()
        model = frame["model"].astype(str).str.strip()
        os_name = frame.get("os", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
        base = (brand + " · " + model).str.strip(" ·")
        repeated = base.duplicated(keep=False)
        frame["__label"] = base
        frame.loc[repeated & os_name.ne(""), "__label"] = (
            base[repeated & os_name.ne("")] + " — " + os_name[repeated & os_name.ne("")]
        )
        frame = add_geekerwan_capacity_overlay(frame)

    frame["__source"] = _source_label(source, root)
    return frame


def load_collections(
    root: Path,
    paths: Iterable[Path] | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load and merge every recognized CSV, returning warnings separately."""
    grouped: dict[str, list[pd.DataFrame]] = {key: [] for key in DATASET_DEFINITIONS}
    warnings: list[str] = []
    candidates = list(paths) if paths is not None else discover_csv_files(root)

    for path in candidates:
        path = Path(path)
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            warnings.append(f"{path.name}: {exc}")
            continue
        try:
            kind = classify_frame(frame, path)
            if kind is None:
                continue
            grouped[kind].append(_prepare_frame(frame, kind, path, root))
        except Exception as exc:
            warnings.append(f"{path.name}: {exc}")

    collections: dict[str, pd.DataFrame] = {}
    for kind, frames in grouped.items():
        if not frames:
            continue
        definition = DATASET_DEFINITIONS[kind]
        merged = pd.concat(frames, ignore_index=True)
        subset = [column for column in definition.dedupe if column in merged]
        merged = merged.drop_duplicates(subset=subset, keep="first")
        merged = merged.sort_values(["__label"], kind="stable").reset_index(drop=True)
        collections[kind] = merged
    return collections, warnings


def collection_summary(collections: dict[str, pd.DataFrame]) -> dict[str, tuple[int, int]]:
    """Return profile and point counts; kept independent for headless tests."""
    return {
        kind: (int(frame["__label"].nunique()), len(frame))
        for kind, frame in collections.items()
    }


class ComparisonDashboard:
    def __init__(
        self,
        project_root: Path,
        csv_paths: list[Path] | None = None,
        initial_dataset: str | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.csv_paths = csv_paths
        self.collections: dict[str, pd.DataFrame] = {}
        self.load_warnings: list[str] = []
        self.selected: dict[str, set[str]] = {}
        self.visible_labels: list[str] = []
        self.dataset_key = ""
        self.nav_buttons: dict[str, ttk.Button] = {}
        self.line_artists = []
        self.ranking_values = pd.Series(dtype=float)
        self.ranking_metric_label = ""
        self.ranking_higher_is_better = True
        self.ranking_offset = 0
        self.ranking_page_size = 9
        self.hover_points: list[HoverPoint] = []
        self.hover_marker = None
        self.hover_annotation = None
        self.hover_vertical = None
        self.hover_horizontal = None

        self.root = tk.Tk()
        self.root.title("SoCPK Comparison Lab")
        self.root.geometry("1460x880")
        self.root.minsize(1080, 680)
        self.root.configure(bg=APP_BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._configure_styles()
        self._build_layout()
        self.reload_data(initial_dataset=initial_dataset)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            ".",
            background=APP_BG,
            foreground=TEXT,
            fieldbackground=PANEL_2,
            bordercolor=GRID,
            lightcolor=GRID,
            darkcolor=GRID,
            font=("Segoe UI", 10),
        )
        style.configure(
            "TButton",
            padding=(14, 9),
            background=PANEL_2,
            foreground=TEXT,
            borderwidth=0,
            focuscolor=PANEL_2,
            font=("Segoe UI Semibold", 10),
        )
        style.map("TButton", background=[("active", "#22304e")])
        style.configure(
            "Compact.TButton",
            width=6,
            padding=(6, 8),
            background=PANEL_2,
            foreground=TEXT,
            borderwidth=0,
            font=("Segoe UI Semibold", 9),
        )
        style.map("Compact.TButton", background=[("active", "#22304e")])
        style.configure(
            "Nav.TButton",
            padding=(20, 11),
            background=PANEL,
            foreground=MUTED,
            font=("Segoe UI Semibold", 10),
        )
        style.map("Nav.TButton", background=[("active", PANEL_2)], foreground=[("active", TEXT)])
        style.configure(
            "ActiveNav.TButton",
            padding=(20, 11),
            background=ACCENT,
            foreground="#ffffff",
            font=("Segoe UI Semibold", 10),
        )
        style.map("ActiveNav.TButton", background=[("active", "#9387ff")])
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#ffffff",
            font=("Segoe UI Semibold", 10),
        )
        style.map("Accent.TButton", background=[("active", "#9387ff")])
        style.configure(
            "TEntry",
            padding=(12, 10),
            fieldbackground=PANEL_2,
            foreground=TEXT,
            insertcolor=TEXT,
            borderwidth=1,
        )
        style.configure(
            "TCombobox",
            padding=(10, 8),
            fieldbackground=PANEL_2,
            background=PANEL_2,
            foreground=TEXT,
            arrowcolor=MUTED,
            borderwidth=0,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PANEL_2)],
            selectbackground=[("readonly", PANEL_2)],
            selectforeground=[("readonly", TEXT)],
        )

    def _build_layout(self) -> None:
        shell = tk.Frame(self.root, bg=APP_BG)
        shell.pack(fill=tk.BOTH, expand=True, padx=24, pady=(20, 22))

        header = tk.Frame(shell, bg=APP_BG)
        header.pack(fill=tk.X, pady=(0, 16))

        brand = tk.Frame(header, bg=APP_BG)
        brand.pack(side=tk.LEFT)
        tk.Label(
            brand,
            text="SoCPK",
            bg=APP_BG,
            fg=TEXT,
            font=("Segoe UI Semibold", 24),
        ).pack(side=tk.LEFT)
        tk.Label(
            brand,
            text="COMPARISON LAB",
            bg=ACCENT,
            fg="#ffffff",
            font=("Segoe UI Semibold", 8),
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=(10, 0), pady=(7, 0))

        actions = tk.Frame(header, bg=APP_BG)
        actions.pack(side=tk.RIGHT)
        ttk.Button(actions, text="Reload data", command=self.reload_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            actions,
            text="Export chart",
            style="Accent.TButton",
            command=self.export_chart,
        ).pack(side=tk.LEFT, padx=(5, 0))

        nav_row = tk.Frame(shell, bg=APP_BG)
        nav_row.pack(fill=tk.X, pady=(0, 14))
        nav = tk.Frame(nav_row, bg=PANEL, padx=4, pady=4)
        nav.pack(side=tk.LEFT)
        for index, (key, definition) in enumerate(DATASET_DEFINITIONS.items()):
            button = ttk.Button(
                nav,
                text=definition.tab_label or key.upper(),
                style="Nav.TButton",
                command=lambda chosen=key: self.set_dataset(chosen),
            )
            button.grid(row=index // 4, column=index % 4, sticky="ew", padx=1, pady=1)
            self.nav_buttons[key] = button
        self.source_note = tk.Label(
            nav_row,
            text="",
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 9),
        )
        self.source_note.pack(side=tk.RIGHT)

        self.hero = tk.Frame(shell, bg=APP_BG)
        self.hero.pack(fill=tk.X, pady=(0, 14))
        hero_text = tk.Frame(self.hero, bg=APP_BG)
        hero_text.pack(side=tk.LEFT)
        self.kicker_label = tk.Label(
            hero_text,
            text="",
            bg=APP_BG,
            fg=CYAN,
            font=("Segoe UI Semibold", 10),
        )
        self.kicker_label.pack(anchor="w")
        self.title_label = tk.Label(
            hero_text,
            text="",
            bg=APP_BG,
            fg=TEXT,
            font=("Segoe UI Semibold", 28),
        )
        self.title_label.pack(anchor="w", pady=(2, 0))

        self.stats_frame = tk.Frame(self.hero, bg=APP_BG)
        self.stats_frame.pack(side=tk.RIGHT)
        self.stat_values: dict[str, tk.Label] = {}
        for key, label, color in (
            ("profiles", "PROFILES", ACCENT),
            ("points", "DATA POINTS", CYAN),
            ("leader", "CURRENT LEADER", GREEN),
        ):
            self._make_stat_card(self.stats_frame, key, label, color)

        body = tk.Frame(shell, bg=APP_BG)
        body.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(body, bg=PANEL, width=285, padx=18, pady=18)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="COMPARE",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w")
        self.view_var = tk.StringVar()
        self.view_combo = ttk.Combobox(
            sidebar,
            textvariable=self.view_var,
            state="readonly",
            font=("Segoe UI", 10),
        )
        self.view_combo.pack(fill=tk.X, pady=(7, 16))
        self.view_combo.bind("<<ComboboxSelected>>", lambda _event: self.draw_charts())

        tk.Label(
            sidebar,
            text="SEARCH PROFILES",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w")
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(sidebar, textvariable=self.search_var)
        self.search_entry.pack(fill=tk.X, pady=(7, 12))
        self.search_var.trace_add("write", lambda *_args: self.refresh_profile_list())

        list_shell = tk.Frame(sidebar, bg=PANEL_2, highlightthickness=1, highlightbackground=GRID)
        list_shell.pack(fill=tk.BOTH, expand=True)
        self.profile_list = tk.Listbox(
            list_shell,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            activestyle="none",
            bg=PANEL_2,
            fg=TEXT,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            highlightthickness=0,
            borderwidth=0,
            font=("Segoe UI", 10),
            relief=tk.FLAT,
        )
        scrollbar = ttk.Scrollbar(list_shell, command=self.profile_list.yview)
        horizontal = ttk.Scrollbar(list_shell, orient=tk.HORIZONTAL, command=self.profile_list.xview)
        self.profile_list.configure(yscrollcommand=scrollbar.set, xscrollcommand=horizontal.set)
        list_shell.rowconfigure(0, weight=1)
        list_shell.columnconfigure(0, weight=1)
        self.profile_list.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.profile_list.bind("<<ListboxSelect>>", self.on_profile_select)

        quick = tk.Frame(sidebar, bg=PANEL)
        quick.pack(fill=tk.X, pady=(0, 8), before=list_shell)
        ttk.Button(
            quick,
            text="Top 5",
            style="Compact.TButton",
            command=self.select_top_five,
        ).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4)
        )
        ttk.Button(
            quick,
            text="All shown",
            style="Compact.TButton",
            command=self.select_visible,
        ).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        ttk.Button(
            quick,
            text="Clear",
            style="Compact.TButton",
            command=self.clear_selection,
        ).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0)
        )
        self.selection_note = tk.Label(
            sidebar,
            text="",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
            wraplength=245,
            justify=tk.LEFT,
        )
        self.selection_note.pack(anchor="w", pady=(0, 8), before=list_shell)

        chart_panel = tk.Frame(body, bg=PANEL, padx=12, pady=12)
        chart_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(11.5, 6.2), dpi=100, facecolor=PANEL)
        grid = self.figure.add_gridspec(
            1, 2, width_ratios=(2.15, 1), left=0.09, right=0.96, top=0.86, bottom=0.16, wspace=0.6
        )
        self.chart_grid = grid
        self.main_axis = self.figure.add_subplot(grid[0, 0])
        self.rank_axis = self.figure.add_subplot(grid[0, 1])
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_panel)
        self.canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("pick_event", self.on_chart_pick)
        self.canvas.mpl_connect("scroll_event", self.on_ranking_scroll)
        self.canvas.mpl_connect("motion_notify_event", self.on_chart_motion)
        self.canvas.mpl_connect("resize_event", self._on_chart_resize)

        toolbar_frame = tk.Frame(chart_panel, bg=PANEL)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, before=self.canvas.get_tk_widget())
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.configure(background=PANEL)
        for child in self.toolbar.winfo_children():
            try:
                child.configure(background=PANEL)
            except tk.TclError:
                pass
        self.toolbar.pack(side=tk.RIGHT)
        self.chart_note = tk.Label(
            toolbar_frame,
            text="",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        )
        self.chart_note.pack(side=tk.LEFT, padx=8)

    def _make_stat_card(self, parent: tk.Frame, key: str, label: str, color: str) -> None:
        card = tk.Frame(parent, bg=PANEL, padx=16, pady=11, highlightthickness=1, highlightbackground=GRID)
        card.pack(side=tk.LEFT, padx=(9, 0))
        tk.Frame(card, bg=color, width=3, height=38).pack(side=tk.LEFT, padx=(0, 11))
        content = tk.Frame(card, bg=PANEL)
        content.pack(side=tk.LEFT)
        tk.Label(
            content,
            text=label,
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 8),
        ).pack(anchor="w")
        value = tk.Label(
            content,
            text="—",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 13),
        )
        value.pack(anchor="w")
        self.stat_values[key] = value

    def reload_data(self, initial_dataset: str | None = None) -> None:
        collections, warnings = load_collections(self.project_root, self.csv_paths)
        self.collections = collections
        self.load_warnings = warnings
        self.selected = {
            key: self.selected.get(key, set()) & set(frame["__label"].unique())
            for key, frame in collections.items()
        }
        for button in self.nav_buttons.values():
            button.state(["!disabled"])

        target = initial_dataset if initial_dataset in DATASET_DEFINITIONS else self.dataset_key
        if target not in DATASET_DEFINITIONS:
            target = next(
                (key for key in DATASET_DEFINITIONS if key in collections),
                next(iter(DATASET_DEFINITIONS)),
            )
        self.set_dataset(target, choose_defaults=not bool(self.selected.get(target)))

    def set_dataset(self, key: str, choose_defaults: bool = True) -> None:
        if key not in DATASET_DEFINITIONS:
            return
        self.dataset_key = key
        definition = DATASET_DEFINITIONS[key]
        for name, button in self.nav_buttons.items():
            button.configure(style="ActiveNav.TButton" if name == key else "Nav.TButton")
        self.kicker_label.configure(text=definition.kicker.upper())
        self.title_label.configure(text=definition.title)
        self.view_combo.configure(values=definition.views)
        self.view_var.set(definition.views[0])
        self.search_var.set("")
        self.selected.setdefault(key, set())
        if key not in self.collections:
            self.visible_labels = []
            self.profile_list.delete(0, tk.END)
            self._draw_unavailable_dataset(definition)
            return
        if choose_defaults and not self.selected.get(key):
            self.selected[key] = set(self._ranked_labels(key)[:5])
        self.refresh_profile_list()
        self.draw_charts()

    def _ranked_labels(self, key: str) -> list[str]:
        frame = self.collections[key]
        if key in CURVE_DATASETS:
            metric = "Efficiency" if key in CPU_DATASETS else DATASET_DEFINITIONS[key].score_column
            ranking = (
                frame.groupby("__label", sort=False)[metric]
                .max()
                .sort_values(ascending=False)
            )
        else:
            ranking = frame.groupby("__label", sort=False)["minPerWh"].max().sort_values(ascending=False)
        return list(ranking.index)

    def refresh_profile_list(self) -> None:
        if self.dataset_key not in self.collections:
            self.visible_labels = []
            self.profile_list.delete(0, tk.END)
            self._update_selection_note()
            return
        needle = self.search_var.get().strip().casefold()
        labels = sorted(
            self.collections[self.dataset_key]["__label"].dropna().unique(),
            key=str.casefold,
        )
        self.visible_labels = [label for label in labels if needle in label.casefold()]
        chosen = self.selected.setdefault(self.dataset_key, set())
        self.profile_list.delete(0, tk.END)
        for index, label in enumerate(self.visible_labels):
            self.profile_list.insert(tk.END, label)
            if label in chosen:
                self.profile_list.selection_set(index)
        self._update_selection_note()

    def on_profile_select(self, _event=None) -> None:
        chosen_visible = {self.visible_labels[index] for index in self.profile_list.curselection()}
        visible_set = set(self.visible_labels)
        self.selected[self.dataset_key] = (
            self.selected.get(self.dataset_key, set()) - visible_set
        ) | chosen_visible
        self.draw_charts()

    def select_visible(self) -> None:
        self.selected.setdefault(self.dataset_key, set()).update(self.visible_labels)
        self.refresh_profile_list()
        self.draw_charts()

    def select_top_five(self) -> None:
        if self.dataset_key not in self.collections:
            return
        self.selected[self.dataset_key] = set(self._ranked_labels(self.dataset_key)[:5])
        self.refresh_profile_list()
        self.draw_charts()

    def clear_selection(self) -> None:
        self.selected[self.dataset_key] = set()
        self.refresh_profile_list()
        self.draw_charts()

    def _selected_frame(self) -> pd.DataFrame:
        frame = self.collections[self.dataset_key]
        labels = self.selected.get(self.dataset_key, set())
        return frame[frame["__label"].isin(labels)].copy()

    def _style_axis(self, axis, title: str, subtitle: str = "") -> None:
        axis.clear()
        axis.set_facecolor(PLOT_BG)
        axis.set_title(title, loc="left", color=TEXT, fontsize=14, fontweight="bold", pad=18)
        if subtitle:
            axis.text(
                0,
                1.015,
                subtitle,
                transform=axis.transAxes,
                color=MUTED,
                fontsize=9,
                va="bottom",
            )
        axis.tick_params(colors=MUTED, labelsize=9, length=0, pad=7)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.grid(True, color=GRID, linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        axis.xaxis.label.set_color(MUTED)
        axis.yaxis.label.set_color(MUTED)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=7))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=7))

    def _on_chart_resize(self, event) -> None:
        # Reserve pixels for titles and units, even when the window is short.
        height = max(event.height, 200)
        self.chart_grid.update(top=1 - 58 / height, bottom=72 / height)
        self.canvas.draw_idle()

    def draw_charts(self) -> None:
        if self.dataset_key not in self.collections:
            return
        selected_frame = self._selected_frame()
        self.line_artists = []
        self.hover_points = []
        self.hover_marker = None
        self.hover_annotation = None
        self.hover_vertical = None
        self.hover_horizontal = None
        if selected_frame.empty:
            self._draw_empty()
        elif self.dataset_key in CURVE_DATASETS:
            self._draw_curve_charts(selected_frame)
        else:
            self._draw_battery_charts(selected_frame)
        if self.hover_points:
            self._install_hover_artists()
        self._update_stats(selected_frame)
        self._update_selection_note()
        self.canvas.draw_idle()

    def _draw_empty(self) -> None:
        self.ranking_values = pd.Series(dtype=float)
        self.ranking_offset = 0
        for axis, title in (
            (self.main_axis, "Choose profiles to compare"),
            (self.rank_axis, "Ranking"),
        ):
            self._style_axis(axis, title)
            axis.text(
                0.5,
                0.48,
                "Select one or more profiles from the left panel",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=10,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        self.chart_note.configure(text="No profiles selected")

    def _draw_unavailable_dataset(self, definition: DatasetDefinition) -> None:
        self.ranking_values = pd.Series(dtype=float)
        self.ranking_offset = 0
        commands = {
            **{benchmark.dataset_key: (
                "snapshots/" + benchmark.filename,
                f'python "Performance Benchmark\\cpu_curve_parser.py" --benchmark {name}',
            ) for name, benchmark in CPU_BENCHMARKS.items()},
            "GPU": (
                "snapshots/gpu_snl_curves.csv",
                'python "Performance Benchmark\\gpu_curve_parser.py"',
            ),
            "Laptop GPU": (
                "snapshots/laptop_gpu_curves.csv",
                'python "Performance Benchmark\\laptop_gpu_curve_parser.py"',
            ),
            "Battery": (
                "snapshots/battery_results.csv",
                "python Battery\\battery_parser.py",
            ),
        }
        filename, command = commands[definition.key]
        self._style_axis(
            self.main_axis,
            f"No {definition.title.lower()} data yet",
            f"Expected {filename}",
        )
        self.main_axis.text(
            0.5,
            0.56,
            "Generate the dataset, then choose Reload data",
            transform=self.main_axis.transAxes,
            ha="center",
            va="center",
            color=TEXT,
            fontsize=12,
            fontweight="bold",
        )
        self.main_axis.text(
            0.5,
            0.44,
            command,
            transform=self.main_axis.transAxes,
            ha="center",
            va="center",
            color=CYAN,
            fontsize=10,
            fontfamily="monospace",
            bbox={
                "boxstyle": "round,pad=0.7",
                "facecolor": APP_BG,
                "edgecolor": GRID,
            },
        )
        self.main_axis.set_xticks([])
        self.main_axis.set_yticks([])

        self._style_axis(self.rank_axis, "Selected ranking", "waiting for data")
        self.rank_axis.text(
            0.5,
            0.5,
            "Nothing to rank yet",
            transform=self.rank_axis.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        self.rank_axis.set_xticks([])
        self.rank_axis.set_yticks([])

        self.stat_values["profiles"].configure(text="0 / 0")
        self.stat_values["points"].configure(text="0")
        self.stat_values["leader"].configure(text="—")
        self.selection_note.configure(text="0 selected · 0 shown · 0 available")
        self.source_note.configure(text="No matching CSV found")
        self.chart_note.configure(text=f"Waiting for {filename}")
        self.canvas.draw_idle()

    def _draw_curve_charts(self, frame: pd.DataFrame) -> None:
        kind = self.dataset_key
        view = self.view_var.get()
        definition = DATASET_DEFINITIONS[kind]
        score_column = definition.score_column
        score_label = definition.score_label
        decimals = definition.score_decimals
        if view == "Performance curve":
            x_column, y_column = "Board_Power_W", score_column
            x_label, y_label = "Board power (W)", score_label
            chart_title = "Performance scaling"
            rank_column, rank_label = score_column, "Peak score"
        elif view == "Efficiency vs score":
            x_column, y_column = score_column, "Efficiency"
            x_label, y_label = score_label, "Score per watt"
            chart_title = "Efficiency across performance"
            rank_column, rank_label = "Efficiency", "Peak score/W"
        else:
            x_column, y_column = "Board_Power_W", "Efficiency"
            x_label, y_label = "Board power (W)", "Score per watt"
            chart_title = "Efficiency curve"
            rank_column, rank_label = "Efficiency", "Peak score/W"

        self._style_axis(
            self.main_axis,
            chart_title,
            f"{len(self.selected[kind])} profiles · move near a point to inspect",
        )
        self.main_axis.set_xlabel(x_label, labelpad=10)
        self.main_axis.set_ylabel(y_label, labelpad=10)

        labels = sorted(self.selected[kind], key=str.casefold)
        for index, label in enumerate(labels):
            subset = frame[frame["__label"] == label].sort_values(x_column)
            color = PALETTE[index % len(PALETTE)]
            for source_index, (_source, source_frame) in enumerate(
                subset.groupby("__source", sort=False)
            ):
                source_frame = source_frame.sort_values(x_column)
                has_curve = len(source_frame) >= 4
                legend_label = label if source_index == 0 else "_nolegend_"
                if has_curve:
                    (line,) = self.main_axis.plot(
                        source_frame[x_column],
                        source_frame[y_column],
                        color=color,
                        linewidth=2.4,
                        alpha=0.94 if source_index == 0 else 0.5,
                        solid_capstyle="round",
                        label=legend_label,
                        picker=5,
                    )
                    line._socpk_label = label
                    self.line_artists.append(line)

                points = self.main_axis.scatter(
                    source_frame[x_column],
                    source_frame[y_column],
                    s=15 if has_curve else 64,
                    color=color,
                    alpha=0.38 if has_curve else 0.95,
                    edgecolors="none" if has_curve else TEXT,
                    linewidths=0 if has_curve else 0.9,
                    zorder=3 if has_curve else 5,
                    label="_nolegend_" if has_curve else legend_label,
                    picker=True,
                )
                points._socpk_label = label
                self.line_artists.append(points)

                for _, row in source_frame.iterrows():
                    x_value = float(row[x_column])
                    y_value = float(row[y_column])
                    if not np.isfinite(x_value) or not np.isfinite(y_value):
                        continue
                    self.hover_points.append(
                        HoverPoint(
                            x=x_value,
                            y=y_value,
                            label=label,
                            details=(
                                f"{label}\n"
                                f"{score_label}: {float(row[score_column]):,.{decimals}f}\n"
                                f"Board power: {float(row['Board_Power_W']):.2f} W\n"
                                f"Efficiency: {float(row['Efficiency']):,.{decimals}f} score/W"
                            ),
                            color=color,
                        )
                    )

        legend = self.main_axis.legend(
            loc="best",
            frameon=True,
            facecolor=PANEL_2,
            edgecolor=GRID,
            labelcolor=TEXT,
            fontsize=8,
            ncols=2 if len(labels) > 5 else 1,
        )
        legend.get_frame().set_alpha(0.92)

        ranking = (
            frame.groupby("__label", sort=False)[rank_column]
            .max()
        )
        self._draw_ranking(ranking, rank_label, higher_is_better=True)
        self.chart_note.configure(text=f"{len(frame):,} curve points shown")

    def _draw_battery_charts(self, frame: pd.DataFrame) -> None:
        view = self.view_var.get()
        summary = (
            frame.groupby("__label", as_index=False)
            .agg(
                minutes=("minutes", "mean"),
                hours=("hours", "mean"),
                capacityWh=("capacityWh", "mean"),
                avgPowerW=("avgPowerW", "mean"),
                minPerWh=("minPerWh", "mean"),
                geekerwanAdvertisedMah=("geekerwanAdvertisedMah", "mean"),
                geekerwanMeasuredMah=("geekerwanMeasuredMah", "mean"),
                geekerwanShortfallMah=("geekerwanShortfallMah", "mean"),
                geekerwanShortfallPct=("geekerwanShortfallPct", "mean"),
                geekerwanCapacityWh=("geekerwanCapacityWh", "mean"),
                geekerwanAvgPowerW=("geekerwanAvgPowerW", "mean"),
                geekerwanMinPerWh=("geekerwanMinPerWh", "mean"),
            )
        )
        labels = sorted(summary["__label"], key=str.casefold)
        color_map = {label: PALETTE[index % len(PALETTE)] for index, label in enumerate(labels)}

        if view == "Energy efficiency":
            x_column, y_column = "avgPowerW", "minPerWh"
            x_label, y_label = "Average power draw (W)", "Minutes per Wh"
            title = "Efficiency sweet spot"
            rank_column, rank_label, higher = "minPerWh", "Minutes per Wh", True
        elif view == "Average power draw":
            x_column, y_column = "capacityWh", "avgPowerW"
            x_label, y_label = "Battery capacity (Wh)", "Average power draw (W)"
            title = "Power draw by battery size"
            rank_column, rank_label, higher = "avgPowerW", "Average power (W)", False
        else:
            x_column, y_column = "capacityWh", "hours"
            x_label, y_label = "Battery capacity (Wh)", "Runtime (hours)"
            title = "Endurance landscape"
            rank_column, rank_label, higher = "hours", "Runtime (hours)", True

        measured_columns = {
            "capacityWh": "geekerwanCapacityWh",
            "avgPowerW": "geekerwanAvgPowerW",
            "minPerWh": "geekerwanMinPerWh",
            "hours": "hours",
        }
        measured_x_column = measured_columns[x_column]
        measured_y_column = measured_columns[y_column]
        measured_count = int(summary["geekerwanMeasuredMah"].notna().sum())

        show_point_labels = len(summary) <= 18
        interaction_note = (
            "labels shown · move near a point for details"
            if show_point_labels
            else "move near a point to identify it"
        )
        self._style_axis(
            self.main_axis,
            title,
            f"{len(summary)} device profiles · {interaction_note}",
        )
        self.main_axis.set_xlabel(x_label, labelpad=10)
        self.main_axis.set_ylabel(y_label, labelpad=10)
        for _, row in summary.iterrows():
            label = row["__label"]
            color = color_map[label]
            points = self.main_axis.scatter(
                row[x_column],
                row[y_column],
                s=74,
                color=color,
                edgecolor=PLOT_BG,
                linewidth=1.5,
                zorder=3,
                picker=True,
            )
            points._socpk_label = label
            self.line_artists.append(points)
            x_value = float(row[x_column])
            y_value = float(row[y_column])
            if np.isfinite(x_value) and np.isfinite(y_value):
                self.hover_points.append(
                    HoverPoint(
                        x=x_value,
                        y=y_value,
                        label=label,
                        details=(
                            f"{label}\n"
                            f"Runtime: {float(row['hours']):.2f} h "
                            f"({float(row['minutes']):.0f} min)\n"
                            f"Capacity: {float(row['capacityWh']):.2f} Wh\n"
                            f"Average power: {float(row['avgPowerW']):.2f} W\n"
                            f"Efficiency: {float(row['minPerWh']):.2f} min/Wh"
                        ),
                        color=color,
                    )
                )
            if pd.notna(row["geekerwanMeasuredMah"]):
                measured_x = float(row[measured_x_column])
                measured_y = float(row[measured_y_column])
                self.main_axis.plot(
                    [x_value, measured_x],
                    [y_value, measured_y],
                    color=color,
                    linestyle=(0, (2, 2)),
                    linewidth=1.2,
                    alpha=0.7,
                    zorder=2,
                )
                measured_points = self.main_axis.scatter(
                    measured_x,
                    measured_y,
                    s=88,
                    marker="D",
                    facecolor=PLOT_BG,
                    edgecolor=color,
                    linewidth=2.0,
                    zorder=4,
                    picker=True,
                )
                measured_label = f"{label} · Geekerwan measured"
                measured_points._socpk_label = measured_label
                self.line_artists.append(measured_points)
                self.hover_points.append(
                    HoverPoint(
                        x=measured_x,
                        y=measured_y,
                        label=measured_label,
                        details=(
                            f"{label}\n"
                            "Geekerwan measured usable capacity\n"
                            f"Measured: {float(row['geekerwanMeasuredMah']):,.0f} mAh "
                            f"of {float(row['geekerwanAdvertisedMah']):,.0f} mAh\n"
                            f"Locked/lost: {float(row['geekerwanShortfallMah']):,.0f} mAh "
                            f"({float(row['geekerwanShortfallPct']):.2f}%)\n"
                            f"Adjusted capacity: {float(row['geekerwanCapacityWh']):.2f} Wh\n"
                            f"Adjusted average power: {float(row['geekerwanAvgPowerW']):.2f} W\n"
                            f"Adjusted efficiency: {float(row['geekerwanMinPerWh']):.2f} min/Wh"
                        ),
                        color=color,
                    )
                )
            if show_point_labels:
                self.main_axis.annotate(
                    self._short_label(label, 24),
                    (row[x_column], row[y_column]),
                    xytext=(7, 7),
                    textcoords="offset points",
                    color=TEXT,
                    fontsize=8,
                    alpha=0.9,
                )
        if measured_count:
            legend = self.main_axis.legend(
                handles=(
                    Line2D(
                        [], [], marker="o", linestyle="none", markersize=7,
                        markerfacecolor=MUTED, markeredgecolor=PLOT_BG,
                        label="SoCPK / advertised capacity",
                    ),
                    Line2D(
                        [], [], marker="D", linestyle="none", markersize=7,
                        markerfacecolor=PLOT_BG, markeredgecolor=MUTED,
                        markeredgewidth=1.8, label="Geekerwan measured capacity",
                    ),
                ),
                loc="best",
                frameon=True,
                facecolor=PANEL_2,
                edgecolor=GRID,
                labelcolor=TEXT,
                fontsize=8,
            )
            legend.get_frame().set_alpha(0.92)
        ranking = summary.set_index("__label")[rank_column]
        self._draw_ranking(ranking, rank_label, higher_is_better=higher)
        self.chart_note.configure(
            text=(
                f"{len(frame):,} battery tests shown · "
                f"{measured_count} Geekerwan measured overlays"
            )
        )

    def _draw_ranking(
        self,
        values: pd.Series,
        metric_label: str,
        higher_is_better: bool,
    ) -> None:
        self.ranking_values = values.dropna().sort_values(
            ascending=not higher_is_better
        )
        self.ranking_metric_label = metric_label
        self.ranking_higher_is_better = higher_is_better
        self.ranking_offset = 0
        self._render_ranking()

    def _render_ranking(self) -> None:
        total = len(self.ranking_values)
        page_size = min(self.ranking_page_size, total)
        maximum_offset = max(0, total - page_size)
        self.ranking_offset = min(max(0, self.ranking_offset), maximum_offset)
        start = self.ranking_offset
        end = min(total, start + page_size)
        visible = self.ranking_values.iloc[start:end]
        # barh draws the final item at the top, so reverse the best-first page.
        values = visible.iloc[::-1]

        higher_is_better = self.ranking_higher_is_better
        direction = "higher is better" if higher_is_better else "lower is better"
        range_note = f"{start + 1}–{end} of {total}"
        if total > page_size:
            range_note += " · scroll to browse"
        self._style_axis(
            self.rank_axis,
            "Ranking",
            f"{direction} · {range_note}",
        )
        short_labels = [self._ranking_label(label) for label in values.index]
        selected_labels = sorted(
            self.selected.get(self.dataset_key, set()),
            key=str.casefold,
        )
        color_map = {
            label: PALETTE[index % len(PALETTE)]
            for index, label in enumerate(selected_labels)
        }
        colors = [color_map.get(label, ACCENT) for label in values.index]
        row_positions = list(range(len(values)))
        bars = self.rank_axis.barh(
            row_positions,
            values.values,
            color=colors,
            height=0.58,
            alpha=0.9,
        )
        self.rank_axis.set_yticks(row_positions, labels=short_labels)
        self.rank_axis.set_xlabel(self.ranking_metric_label, labelpad=10)
        self.rank_axis.grid(axis="y", visible=False)
        maximum = float(self.ranking_values.max()) if total else 0.0
        span = max(maximum, 1.0)
        self.rank_axis.set_xlim(left=0.0, right=maximum + span * 0.24)
        for bar, value in zip(bars, values.values):
            self.rank_axis.text(
                value + span * 0.025,
                bar.get_y() + bar.get_height() / 2,
                (f"{float(value):.3f}" if DATASET_DEFINITIONS[self.dataset_key].score_decimals
                 else self._format_number(float(value))),
                va="center",
                color=TEXT,
                fontsize=8,
                fontweight="bold",
            )

        if total > page_size:
            track_top, track_bottom = 0.94, 0.06
            track_height = track_top - track_bottom
            thumb_height = track_height * page_size / total
            progress = start / maximum_offset if maximum_offset else 0.0
            thumb_top = track_top - progress * (track_height - thumb_height)
            thumb_bottom = thumb_top - thumb_height
            self.rank_axis.plot(
                [1.018, 1.018],
                [track_bottom, track_top],
                transform=self.rank_axis.transAxes,
                color=GRID,
                linewidth=4,
                solid_capstyle="round",
                clip_on=False,
            )
            self.rank_axis.plot(
                [1.018, 1.018],
                [thumb_bottom, thumb_top],
                transform=self.rank_axis.transAxes,
                color=ACCENT,
                linewidth=4,
                solid_capstyle="round",
                clip_on=False,
            )

    def on_ranking_scroll(self, event) -> None:
        if event.inaxes is not self.rank_axis:
            return
        total = len(self.ranking_values)
        if total <= self.ranking_page_size:
            return
        step = getattr(event, "step", 0)
        if not step:
            step = 1 if getattr(event, "button", None) == "up" else -1
        old_offset = self.ranking_offset
        self.ranking_offset -= int(step * 3)
        self.ranking_offset = min(
            max(0, self.ranking_offset),
            total - self.ranking_page_size,
        )
        if self.ranking_offset != old_offset:
            self._render_ranking()
            self.canvas.draw_idle()

    def _update_stats(self, frame: pd.DataFrame) -> None:
        all_frame = self.collections[self.dataset_key]
        selected_count = len(self.selected.get(self.dataset_key, set()))
        self.stat_values["profiles"].configure(
            text=f"{selected_count} / {all_frame['__label'].nunique()}"
        )
        self.stat_values["points"].configure(text=f"{len(frame):,}")
        if frame.empty:
            leader = "—"
        elif self.dataset_key in CURVE_DATASETS:
            metric = (DATASET_DEFINITIONS[self.dataset_key].score_column
                      if self.view_var.get() == "Performance curve" else "Efficiency")
            leader = frame.groupby("__label")[metric].max().idxmax()
        else:
            metric = {
                "Runtime vs capacity": "hours",
                "Energy efficiency": "minPerWh",
                "Average power draw": "avgPowerW",
            }[self.view_var.get()]
            grouped = frame.groupby("__label")[metric].mean()
            leader = grouped.idxmin() if metric == "avgPowerW" else grouped.idxmax()
        leader_text = (self._ranking_label(str(leader)) if self.dataset_key in {"SPEC INT", "SPEC FP"}
                       else self._short_label(str(leader), 25))
        self.stat_values["leader"].configure(text=leader_text)

        source_count = all_frame["__source"].nunique()
        warning_note = f" · {len(self.load_warnings)} warning(s)" if self.load_warnings else ""
        self.source_note.configure(
            text=f"{source_count} source file{'s' if source_count != 1 else ''}{warning_note}"
        )

    def _update_selection_note(self) -> None:
        if not self.dataset_key:
            return
        selected_count = len(self.selected.get(self.dataset_key, set()))
        shown = len(self.visible_labels)
        total = (
            self.collections[self.dataset_key]["__label"].nunique()
            if self.dataset_key in self.collections
            else 0
        )
        self.selection_note.configure(
            text=f"{selected_count} selected · {shown} shown · {total} available"
        )

    def _install_hover_artists(self) -> None:
        # Crosshair helpers use data coordinates. Preserve limits so their
        # temporary zero positions never pull an otherwise positive chart to 0.
        data_xlim = self.main_axis.get_xlim()
        data_ylim = self.main_axis.get_ylim()
        (self.hover_marker,) = self.main_axis.plot(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=10,
            markerfacecolor=ACCENT,
            markeredgecolor="#ffffff",
            markeredgewidth=1.5,
            zorder=20,
            visible=False,
            label="_nolegend_",
        )
        self.hover_vertical = self.main_axis.axvline(
            0,
            color=MUTED,
            linewidth=0.8,
            linestyle=(0, (3, 4)),
            alpha=0.55,
            zorder=8,
            visible=False,
        )
        self.hover_horizontal = self.main_axis.axhline(
            0,
            color=MUTED,
            linewidth=0.8,
            linestyle=(0, (3, 4)),
            alpha=0.55,
            zorder=8,
            visible=False,
        )
        self.hover_annotation = self.main_axis.annotate(
            "",
            xy=(0, 0),
            xytext=(18, 18),
            textcoords="offset points",
            color=TEXT,
            fontsize=9,
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.7",
                "facecolor": APP_BG,
                "edgecolor": ACCENT,
                "linewidth": 1.2,
                "alpha": 0.97,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": MUTED,
                "linewidth": 0.8,
            },
            zorder=30,
            annotation_clip=False,
            visible=False,
        )
        self.main_axis.set_xlim(data_xlim)
        self.main_axis.set_ylim(data_ylim)

    def on_chart_motion(self, event) -> None:
        if (
            not self.hover_points
            or self.hover_marker is None
            or event.inaxes is not self.main_axis
            or event.x is None
            or event.y is None
            or bool(self.toolbar.mode)
        ):
            self._hide_hover()
            return

        data_positions = np.array(
            [(point.x, point.y) for point in self.hover_points],
            dtype=float,
        )
        display_positions = self.main_axis.transData.transform(data_positions)
        distances = np.square(display_positions[:, 0] - event.x) + np.square(
            display_positions[:, 1] - event.y
        )
        nearest_index = int(np.argmin(distances))
        if float(distances[nearest_index]) > 24.0**2:
            self._hide_hover()
            return

        point = self.hover_points[nearest_index]
        self.hover_marker.set_data([point.x], [point.y])
        self.hover_marker.set_markerfacecolor(point.color)
        self.hover_vertical.set_xdata([point.x, point.x])
        self.hover_horizontal.set_ydata([point.y, point.y])

        axis_box = self.main_axis.bbox
        place_left = event.x > axis_box.x0 + axis_box.width * 0.68
        place_below = event.y > axis_box.y0 + axis_box.height * 0.72
        x_offset = -18 if place_left else 18
        y_offset = -18 if place_below else 18
        self.hover_annotation.xy = (point.x, point.y)
        self.hover_annotation.set_position((x_offset, y_offset))
        self.hover_annotation.set_ha("right" if place_left else "left")
        self.hover_annotation.set_va("top" if place_below else "bottom")
        self.hover_annotation.set_text(point.details)
        self.hover_annotation.get_bbox_patch().set_edgecolor(point.color)

        self.hover_marker.set_visible(True)
        self.hover_vertical.set_visible(True)
        self.hover_horizontal.set_visible(True)
        self.hover_annotation.set_visible(True)
        self.canvas.draw_idle()

    def _hide_hover(self) -> None:
        artists = (
            self.hover_marker,
            self.hover_vertical,
            self.hover_horizontal,
            self.hover_annotation,
        )
        if not any(artist is not None and artist.get_visible() for artist in artists):
            return
        for artist in artists:
            if artist is not None:
                artist.set_visible(False)
        self.canvas.draw_idle()

    def on_chart_pick(self, event) -> None:
        label = getattr(event.artist, "_socpk_label", None)
        if label:
            self.chart_note.configure(text=f"Selected chart profile: {label}")

    def export_chart(self) -> None:
        default_name = f"socpk-{self.dataset_key.lower()}-comparison.png"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export comparison chart",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=(("PNG image", "*.png"), ("SVG image", "*.svg"), ("PDF", "*.pdf")),
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=180, facecolor=PANEL, bbox_inches="tight")
            self.chart_note.configure(text=f"Exported {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self.root)

    def _ranking_label(self, label: str) -> str:
        if self.dataset_key in {"SPEC INT", "SPEC FP"} and " — " in label:
            chip, core = label.split(" — ", 1)
            return self._short_label(chip, 18) + "\n" + self._short_label(core, 20)
        return self._short_label(label, 22)

    @staticmethod
    def _short_label(label: str, length: int) -> str:
        return label if len(label) <= length else label[: length - 1].rstrip() + "…"

    @staticmethod
    def _format_number(value: float) -> str:
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 100:
            return f"{value:.0f}"
        return f"{value:.2f}"

    def close(self) -> None:
        self.root.quit()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare every collected SoCPK CSV in one GUI")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project folder to scan for CSV files",
    )
    parser.add_argument(
        "--csv",
        "--input",
        dest="csv_paths",
        type=Path,
        action="append",
        help="Load only this CSV; repeat to load multiple files",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_DEFINITIONS),
        help="Dataset shown at startup",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    paths = [path.resolve() for path in args.csv_paths] if args.csv_paths else None
    missing = [path for path in paths or [] if not path.is_file()]
    if missing:
        raise SystemExit(f"CSV file does not exist: {missing[0]}")
    app = ComparisonDashboard(root, paths, args.dataset)
    app.run()


if __name__ == "__main__":
    main()
