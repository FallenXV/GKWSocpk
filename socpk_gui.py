#!/usr/bin/env python3
"""Polished comparison dashboard for every SoCPK CSV in a project."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import signal
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _ensure_macos_tk() -> None:
    """Relaunch with uv's Tk-enabled Python when Homebrew omitted Tk."""
    try:
        import _tkinter  # noqa: F401
        return
    except ModuleNotFoundError:
        if sys.platform != "darwin":
            return

    if os.environ.get("SOCPK_TK_BOOTSTRAPPED") == "1":
        raise SystemExit(
            "Tk is unavailable in the selected macOS Python. Install a Python "
            "distribution with Tk support or run: uv run --python 3.10 "
            "--with-requirements requirements.txt python socpk_gui.py"
        )
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit(
            "This macOS Python does not include Tk. Install uv (https://docs.astral.sh/uv/) "
            "and rerun this command, or install the matching Homebrew python-tk package."
        )

    project_root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment["SOCPK_TK_BOOTSTRAPPED"] = "1"
    # The GUI fallback must not resolve against PyPI at launch time.  The
    # regular venv already contains the project dependencies; this command is
    # only needed because Homebrew's Python may not ship with Tk.  uv can use
    # its local package cache to provide the same dependencies to its Tk build.
    command = [
        uv, "run", "--offline", "--python", "3.10", "--with-requirements",
        str(project_root / "requirements.txt"), "python",
        str(Path(__file__).resolve()), *sys.argv[1:],
    ]
    os.execvpe(uv, command, environment)


_ensure_macos_tk()

from cpu_benchmarks import CPU_BENCHMARKS, CORE_COLUMNS, cpu_profile_label
from battery_soc import canonical_soc_name

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "PingFang SC",
    "Heiti SC",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Segoe UI",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font as tkfont

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
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



class AxisHeader(AnchoredOffsetbox):
    """Keep title and note on one baseline, including resized and exported charts."""

    def __init__(self, axis, title: str, note: str):
        self.title = TextArea(title, textprops=dict(color=TEXT, fontsize=11, fontweight="bold"))
        self.note = TextArea(note, textprops=dict(color=MUTED, fontsize=9))
        self.row = HPacker(children=[self.title, self.note], align="baseline", pad=0, sep=10)
        super().__init__("lower left", child=self.row, frameon=False, pad=0, borderpad=0,
                         bbox_to_anchor=(0, 1.02), bbox_transform=axis.transAxes)

    def get_bbox(self, renderer):
        # Measure with the active renderer so PNG, SVG and PDF fit just like the GUI.
        for area, size in ((self.title, 11), (self.note, 9)):
            area.get_children()[0].set_fontsize(size)
        gap = renderer.points_to_pixels(self.row.sep)
        for _ in range(4):
            width = self.title.get_bbox(renderer).width + self.note.get_bbox(renderer).width
            if width + gap <= self.axes.bbox.width:
                break
            scale = max(1, self.axes.bbox.width - gap) / max(width, 1) * 0.98
            for area in (self.title, self.note):
                text = area.get_children()[0]
                text.set_fontsize(text.get_fontsize() * scale)
        return super().get_bbox(renderer)


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
class CoreFilter:
    # Empty sets mean unrestricted; selections within each field are ORed.
    groups: frozenset[str] = frozenset()
    names: frozenset[str] = frozenset()


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


SOC_AVERAGE_COLUMNS = (
    "soc_device_count",
    "soc_avg_capacity_wh",
    "soc_avg_power_w",
    "soc_avg_min_per_wh",
    "soc_avg_runtime_hours",
)

SOC_AVERAGE_VIEWS = frozenset({"SoC Average Power Draw", "SoC Average Efficiency"})


def add_soc_average_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize SoC names and precompute one device-weighted average per processor."""
    frame = frame.copy()
    raw_soc = frame.get("soc", pd.Series("", index=frame.index)).fillna("").astype(str)
    if "chipset" in frame:
        chipset = frame["chipset"].fillna("").astype(str)
        raw_soc = raw_soc.where(raw_soc.str.strip().ne(""), chipset)
    frame["soc"] = raw_soc.map(canonical_soc_name)
    frame = frame.drop(columns=[column for column in SOC_AVERAGE_COLUMNS if column in frame])

    metrics = ["capacityWh", "avgPowerW", "minPerWh", "hours"]
    profile_rows = []
    for label, rows in frame.groupby("__label", sort=False):
        socs = [value for value in rows["soc"] if value]
        if not socs:
            continue
        profile_rows.append({
            "__label": label,
            "soc": socs[0],
            **{metric: pd.to_numeric(rows[metric], errors="coerce").mean() for metric in metrics},
        })
    if not profile_rows:
        for column in SOC_AVERAGE_COLUMNS:
            frame[column] = np.nan
        return frame

    profiles = pd.DataFrame(profile_rows)
    averages = (
        profiles.groupby("soc", as_index=False)
        .agg(
            soc_device_count=("__label", "nunique"),
            soc_avg_capacity_wh=("capacityWh", "mean"),
            soc_avg_power_w=("avgPowerW", "mean"),
            soc_avg_min_per_wh=("minPerWh", "mean"),
            soc_avg_runtime_hours=("hours", "mean"),
        )
    )
    return frame.merge(averages, on="soc", how="left")


def soc_average_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per available precomputed processor average."""
    if "soc" not in frame or any(column not in frame for column in SOC_AVERAGE_COLUMNS):
        frame = add_soc_average_columns(frame)
    columns = ["soc", *SOC_AVERAGE_COLUMNS]
    available = frame[frame["soc"].fillna("").astype(str).str.strip().ne("")]
    return (
        available[columns]
        .drop_duplicates("soc")
        .sort_values("soc", key=lambda values: values.str.casefold())
        .reset_index(drop=True)
    )


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
        views=(
            "Runtime vs capacity",
            "Energy efficiency",
            "Average power draw",
            "SoC Average Power Draw",
            "SoC Average Efficiency",
        ),
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
    """Find project CSVs, keeping only the newest file in a snapshot family."""
    ignored = {".git", ".venv", "venv", "__pycache__", ".idea", ".pytest_cache"}
    files: list[Path] = []
    for path in root.rglob("*.csv"):
        if any(part in ignored for part in path.relative_to(root).parts):
            continue
        files.append(path)

    # new_snapshot() appends this suffix rather than overwriting prior data.
    # Group those siblings so automatic discovery loads the newest snapshot,
    # instead of merging an older canonical file first.
    suffix = re.compile(r"_\d{8}T\d{12}Z_\d+$")
    newest: dict[Path, Path] = {}
    for path in files:
        family_stem = suffix.sub("", path.stem)
        family = path.with_name(family_stem + path.suffix)
        incumbent = newest.get(family)
        if incumbent is None or path.stat().st_mtime_ns > incumbent.stat().st_mtime_ns:
            newest[family] = path
    return sorted(newest.values(), key=lambda item: str(item).casefold())


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
        if kind == "Battery":
            merged = add_soc_average_columns(merged)
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
        self.core_filters: dict[str, CoreFilter] = {}
        self.selection_modes: dict[str, str] = {}
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
        self.show_soc_averages_var = tk.BooleanVar(master=self.root, value=False)
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
        style.configure("Core.TButton", padding=(4, 7), width=0, background=PANEL_2,
                        foreground=TEXT, font=("Segoe UI", 9))
        style.map("Core.TButton", background=[("active", "#22304e")])
        style.configure("ActiveCore.TButton", padding=(4, 7), width=0, background=ACCENT,
                        foreground="#ffffff", font=("Segoe UI Semibold", 9))
        style.map("ActiveCore.TButton", background=[("active", "#9387ff")])
        style.configure("ActiveMaster.TButton", padding=(6, 8), width=6, background=ACCENT,
                        foreground=TEXT, font=("Segoe UI Semibold", 9))
        style.map("ActiveMaster.TButton", background=[("active", "#9387ff")])
        style.configure(
            "Nav.TButton",
            padding=(8, 10),
            width=0,
            background=PANEL,
            foreground=MUTED,
            font=("Segoe UI Semibold", 10),
        )
        style.map("Nav.TButton", background=[("active", PANEL_2)], foreground=[("active", TEXT)])
        style.configure(
            "ActiveNav.TButton",
            padding=(8, 10),
            width=0,
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
        style.configure(
            "Soc.Toggle.TCheckbutton",
            background=PANEL,
            foreground=MUTED,
            font=("Segoe UI Semibold", 8),
            padding=(8, 1),
        )
        style.map(
            "Soc.Toggle.TCheckbutton",
            background=[("active", PANEL), ("disabled", PANEL)],
            foreground=[("selected", CYAN), ("disabled", "#59647b")],
        )

    def _build_layout(self) -> None:
        shell = tk.Frame(self.root, bg=APP_BG)
        shell.pack(fill=tk.BOTH, expand=True, padx=16, pady=(12, 14))

        header = tk.Frame(shell, bg=APP_BG)
        header.pack(fill=tk.X, pady=(0, 8))

        header.columnconfigure(1, weight=1)
        brand = tk.Frame(header, bg=APP_BG)
        brand.grid(row=0, column=0, sticky="w", padx=(0, 8))
        tk.Label(brand, text="SoCPK", bg=APP_BG, fg=TEXT,
                 font=("Segoe UI Semibold", 24)).pack(anchor="w")

        nav = tk.Frame(header, bg=PANEL, padx=3, pady=3)
        nav.grid(row=0, column=1, sticky="ew")
        for index, (key, definition) in enumerate(DATASET_DEFINITIONS.items()):
            nav.columnconfigure(index, weight=1)
            button = ttk.Button(
                nav, text=definition.tab_label or key.upper(), style="Nav.TButton",
                command=lambda chosen=key: self.set_dataset(chosen),
            )
            button.grid(row=0, column=index, sticky="ew", padx=1)
            self.nav_buttons[key] = button

        actions = tk.Frame(header, bg=APP_BG)
        actions.grid(row=0, column=2, sticky="e", padx=(10, 0))
        ttk.Button(actions, text="Reload data", command=self.reload_data).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(actions, text="Export chart", style="Accent.TButton",
                   command=self.export_chart).grid(row=0, column=1)
        self.source_note = tk.Label(actions, text="", bg=APP_BG, fg=MUTED, font=("Segoe UI", 8))
        self.source_note.grid(row=1, column=0, columnspan=2, sticky="e", pady=(3, 0))

        self.hero = tk.Frame(shell, bg=APP_BG)
        self.hero.pack(fill=tk.X, pady=(0, 8))
        hero_text = self.hero_text = tk.Frame(self.hero, bg=APP_BG)
        self.hero.columnconfigure(0, weight=1)
        self.hero.columnconfigure(1, weight=0)
        hero_text.grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.kicker_label = tk.Label(
            hero_text,
            text="",
            bg=APP_BG,
            fg=CYAN,
            font=("Segoe UI Semibold", 10),
        )
        self.kicker_label.configure(wraplength=300, justify=tk.LEFT)
        self.kicker_label.pack(anchor="w")
        self.title_label = tk.Label(
            hero_text,
            text="",
            bg=APP_BG,
            fg=TEXT,
            font=("Segoe UI Semibold", 20),
            wraplength=330, justify=tk.LEFT,
        )
        self.title_label.pack(anchor="w", pady=(2, 0))

        self.stats_frame = tk.Frame(self.hero, bg=APP_BG)
        self.stats_frame.grid(row=0, column=1, sticky="e")
        self.stat_values: dict[str, tk.Label] = {}
        for key, label, color in (
            ("profiles", "PROFILES", ACCENT),
            ("points", "DATA POINTS", CYAN),
            ("leader", "CURRENT LEADER", GREEN),
        ):
            self._make_stat_card(self.stats_frame, key, label, color)
        self.hero.bind("<Configure>", self._resize_leader)

        body = tk.Frame(shell, bg=APP_BG)
        body.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(body, bg=PANEL, width=285, padx=14, pady=12)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
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
        self.view_combo.pack(fill=tk.X, pady=(4, 10))
        self.view_combo.bind("<<ComboboxSelected>>", self.on_view_change)

        tk.Label(
            sidebar,
            text="SEARCH PROFILES",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w")
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(sidebar, textvariable=self.search_var)
        self.search_entry.pack(fill=tk.X, pady=(4, 8))
        self.search_var.trace_add("write", self.on_search_change)

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
        self.selection_buttons = {}
        for mode, title, command in (("top5", "Top 5", self.select_top_five),
                                     ("all", "All shown", self.select_visible),
                                     ("clear", "Clear", self.clear_selection)):
            button = ttk.Button(quick, text=title, style="Compact.TButton", command=command)
            button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            self.selection_buttons[mode] = button
        self.core_filter_panel = tk.Frame(sidebar, bg=PANEL)
        tk.Label(self.core_filter_panel, text="FILTER CORES", bg=PANEL, fg=MUTED,
                 font=("Segoe UI Semibold", 9)).pack(anchor="w", pady=(2, 5))
        core_buttons = tk.Frame(self.core_filter_panel, bg=PANEL)
        core_buttons.pack(fill=tk.X)
        self.core_group_buttons = {}
        for index, (group, label) in enumerate(
            (("", "All"), ("super", "Super"), ("large", "Large"), ("medium", "Medium"), ("small", "Small"))
        ):
            button = ttk.Button(core_buttons, text=label, style="Core.TButton",
                                command=lambda value=group: self.set_core_group(value))
            button.grid(row=0, column=index, sticky="ew", padx=1, pady=2)
            core_buttons.columnconfigure(index, weight=1)
            self.core_group_buttons[group] = button
        self.core_name_var = tk.StringVar(value="All core names")
        self.core_name_button = ttk.Menubutton(self.core_filter_panel, textvariable=self.core_name_var)
        self.core_name_menu = tk.Menu(self.core_name_button, tearoff=False, bg=PANEL_2, fg=TEXT,
                                     activebackground=ACCENT, activeforeground=TEXT)
        self.core_name_button.configure(menu=self.core_name_menu)
        self.core_name_button.pack(fill=tk.X, pady=(4, 0))
        self.core_name_checks = {}

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

        chart_panel = tk.Frame(body, bg=PANEL, padx=6, pady=6)
        chart_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(11.5, 6.2), dpi=100, facecolor=PANEL)
        grid = self.figure.add_gridspec(
            1, 2, width_ratios=(2.15, 1), left=0.065, right=0.975, top=0.90, bottom=0.12, wspace=0.20
        )
        self.chart_grid = grid
        self.main_axis = self.figure.add_subplot(grid[0, 0])
        self.rank_axis = self.figure.add_subplot(grid[0, 1])
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_panel)
        self.canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ranking_scrollbar = ttk.Scrollbar(
            chart_panel,
            orient=tk.VERTICAL,
            command=self.on_ranking_scrollbar,
        )
        self.ranking_scrollbar.pack(
            side=tk.RIGHT,
            fill=tk.Y,
            padx=(4, 0),
            before=self.canvas.get_tk_widget(),
        )
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
        column = len(self.stat_values)
        parent.columnconfigure(column, weight=1 if key == "leader" else 0)
        card.grid(row=0, column=column, sticky="nsew", padx=(9, 0))
        tk.Frame(card, bg=color, width=3, height=38).pack(side=tk.LEFT, padx=(0, 11))
        content = tk.Frame(card, bg=PANEL)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(
            content,
            text=label,
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 8),
        ).pack(anchor="w")
        value_row = tk.Frame(content, bg=PANEL)
        value_row.pack(anchor="w", fill=tk.X)
        value = tk.Label(
            value_row,
            text="—",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 13),
        )
        value.configure(anchor="w", justify=tk.LEFT)
        value.pack(side=tk.LEFT, anchor="w", fill=tk.X, expand=True)
        if key == "points":
            self.soc_average_toggle = ttk.Checkbutton(
                value_row,
                text="SoC averages",
                variable=self.show_soc_averages_var,
                command=self.on_soc_average_toggle,
                style="Soc.Toggle.TCheckbutton",
            )
        if key == "leader":
            self.leader_font = tkfont.Font(family="Segoe UI Semibold", size=13)
            value.configure(width=1, wraplength=0, font=self.leader_font)
        self.stat_values[key] = value

    def _resize_leader(self, _event=None) -> None:
        value = self.stat_values["leader"]
        self.leader_font.configure(size=13)
        other_cards = sum(card.winfo_reqwidth() + 9 for card in self.stats_frame.winfo_children()[:2])
        available = max(140, self.hero.winfo_width() - self.hero_text.winfo_reqwidth() - other_cards - 21)
        desired = self.leader_font.measure(value.cget("text")) + 64
        width = min(desired, available)
        self.stats_frame.columnconfigure(2, minsize=int(width) + 9)
        for size in range(13, 7, -1):
            self.leader_font.configure(size=size)
            if self.leader_font.measure(value.cget("text")) <= width - 64:
                break

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
        self._sync_soc_average_toggle()
        self.selection_modes.setdefault(key, "top5" if choose_defaults else "manual")
        self.selected.setdefault(key, set())
        self._sync_core_filters()
        self.search_var.set("")
        if key not in self.collections:
            self.visible_labels = []
            self.profile_list.delete(0, tk.END)
            self._draw_unavailable_dataset(definition)
            return
        self.refresh_profile_list()
        self.draw_charts()

    def _core_filtered_frame(self, key: str) -> pd.DataFrame:
        frame = self.collections[key]
        filters = self.core_filters.get(key, CoreFilter())
        if "Core_Group" in frame and filters.groups:
            frame = frame[frame["Core_Group"].isin(filters.groups)]
        if "Core" in frame and filters.names:
            frame = frame[frame["Core"].isin(filters.names)]
        return frame

    def _sync_core_filters(self) -> None:
        frame = self.collections.get(self.dataset_key)
        if frame is None or "Core_Group" not in frame:
            self.core_filter_panel.pack_forget()
            return
        self.core_filter_panel.pack(fill=tk.X, pady=(0, 8), before=self.selection_note)
        filters = self.core_filters.get(self.dataset_key, CoreFilter())
        groups = set(frame["Core_Group"])
        names = sorted(frame["Core"].dropna().unique(), key=str.casefold)
        filters = CoreFilter(filters.groups & groups, filters.names & set(names))
        self.core_filters[self.dataset_key] = filters
        for value, button in self.core_group_buttons.items():
            active = value in filters.groups if value else not filters.groups
            button.configure(style="ActiveCore.TButton" if active else "Core.TButton")
            button.state(["!disabled"] if not value or value in groups else ["disabled"])
        self.core_name_menu.delete(0, tk.END)
        self.core_name_checks = {name: tk.BooleanVar(value=name in filters.names) for name in names}
        self.all_core_names_var = tk.BooleanVar(value=not filters.names)
        self.core_name_menu.add_checkbutton(label="All core names", variable=self.all_core_names_var,
                                           command=lambda: self.toggle_core_name(""))
        self.core_name_menu.add_separator()
        for name, variable in self.core_name_checks.items():
            self.core_name_menu.add_checkbutton(label=name, variable=variable,
                                               command=lambda chosen=name: self.toggle_core_name(chosen))
        self.core_name_var.set(f"{len(filters.names)} core names selected" if filters.names else "All core names")

    def set_core_group(self, group: str) -> None:
        filters = self.core_filters.get(self.dataset_key, CoreFilter())
        groups = filters.groups ^ {group} if group else frozenset()
        self.core_filters[self.dataset_key] = CoreFilter(frozenset(groups), filters.names if group else frozenset())
        self._sync_core_filters()
        self.refresh_profile_list()
        self.draw_charts()

    def toggle_core_name(self, name: str) -> None:
        filters = self.core_filters.get(self.dataset_key, CoreFilter())
        names = filters.names ^ {name} if name else frozenset()
        self.core_filters[self.dataset_key] = CoreFilter(filters.groups, frozenset(names))
        self._sync_core_filters()
        self.refresh_profile_list()
        self.draw_charts()

    def _apply_selection_mode(self) -> None:
        mode = self.selection_modes.get(self.dataset_key, "manual")
        if mode in {"top5", "all"}:
            labels = self.visible_labels
            if mode == "top5":
                visible = set(labels)
                labels = [label for label in self._ranked_labels(self.dataset_key) if label in visible][:5]
            self.selected[self.dataset_key] = set(labels)
        elif mode == "clear":
            self.selected[self.dataset_key] = set()
        for name, button in self.selection_buttons.items():
            button.configure(style="ActiveMaster.TButton" if name == mode else "Compact.TButton")

    def on_search_change(self, *_args) -> None:
        self.refresh_profile_list()
        self.draw_charts()

    def on_view_change(self, _event=None) -> None:
        self._sync_soc_average_toggle()
        self.refresh_profile_list()
        self.draw_charts()

    def _sync_soc_average_toggle(self) -> None:
        if not hasattr(self, "soc_average_toggle"):
            return
        if self.dataset_key != "Battery":
            self.soc_average_toggle.pack_forget()
            return
        if not self.soc_average_toggle.winfo_manager():
            self.soc_average_toggle.pack(side=tk.LEFT, padx=(8, 0))
        frame = self.collections.get("Battery")
        has_soc = frame is not None and "soc" in frame and frame["soc"].astype(bool).any()
        supported_view = self.view_var.get() in {"Energy efficiency", "Average power draw"}
        self.soc_average_toggle.state(["!disabled"] if has_soc and supported_view else ["disabled"])

    def on_soc_average_toggle(self) -> None:
        if self.dataset_key == "Battery":
            self.draw_charts()

    def _ranked_labels(self, key: str) -> list[str]:
        frame = self._core_filtered_frame(key)
        if key in CURVE_DATASETS:
            definition = DATASET_DEFINITIONS[key]
            view = self.view_var.get() if key == self.dataset_key and hasattr(self, "view_var") else definition.views[0]
            metric = definition.score_column if view == "Performance curve" else "Efficiency"
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
            self._core_filtered_frame(self.dataset_key)["__label"].dropna().unique(),
            key=str.casefold,
        )
        self.visible_labels = [label for label in labels if needle in label.casefold()]
        self._apply_selection_mode()
        chosen = self.selected.setdefault(self.dataset_key, set())
        self.profile_list.delete(0, tk.END)
        for index, label in enumerate(self.visible_labels):
            self.profile_list.insert(tk.END, label)
            if label in chosen:
                self.profile_list.selection_set(index)
        self._update_selection_note()

    def on_profile_select(self, _event=None) -> None:
        self.selection_modes[self.dataset_key] = "manual"
        self._apply_selection_mode()
        chosen_visible = {self.visible_labels[index] for index in self.profile_list.curselection()}
        visible_set = set(self.visible_labels)
        self.selected[self.dataset_key] = (
            self.selected.get(self.dataset_key, set()) - visible_set
        ) | chosen_visible
        self.draw_charts()

    def select_visible(self) -> None:
        self.selection_modes[self.dataset_key] = "all"
        self.refresh_profile_list()
        self.draw_charts()

    def select_top_five(self) -> None:
        if self.dataset_key not in self.collections:
            return
        self.selection_modes[self.dataset_key] = "top5"
        self.refresh_profile_list()
        self.draw_charts()

    def clear_selection(self) -> None:
        self.selection_modes[self.dataset_key] = "clear"
        self.refresh_profile_list()
        self.draw_charts()

    def _selected_frame(self) -> pd.DataFrame:
        frame = self._core_filtered_frame(self.dataset_key)
        labels = self.selected.get(self.dataset_key, set())
        return frame[frame["__label"].isin(labels)].copy()

    def _style_axis(self, axis, title: str, subtitle: str = "") -> None:
        axis.clear()
        axis.set_facecolor(PLOT_BG)
        axis.add_artist(AxisHeader(axis, title, subtitle))
        axis.tick_params(colors=MUTED, labelsize=9, length=0, pad=4)
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
        self.chart_grid.update(top=1 - 32 / height, bottom=54 / height)
        self.canvas.draw_idle()

    def draw_charts(self) -> None:
        if self.dataset_key not in self.collections:
            return
        selected_frame = self._selected_frame()
        chart_frame = selected_frame
        if self.dataset_key == "Battery" and self.view_var.get() in SOC_AVERAGE_VIEWS:
            # These are dataset-level aggregates, so they remain useful even if
            # the device selector is narrowed or cleared.
            chart_frame = self.collections["Battery"].copy()
        self.line_artists = []
        self.hover_points = []
        self.hover_marker = None
        self.hover_annotation = None
        self.hover_vertical = None
        self.hover_horizontal = None
        if chart_frame.empty:
            self._draw_empty()
        elif self.dataset_key in CURVE_DATASETS:
            self._draw_curve_charts(chart_frame)
        else:
            self._draw_battery_charts(chart_frame)
        if self.hover_points:
            self._install_hover_artists()
        self._update_stats(chart_frame)
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
        self._resize_leader()
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
            rank_column, rank_label = score_column, "Peak score"
        elif view == "Efficiency vs score":
            x_column, y_column = score_column, "Efficiency"
            x_label, y_label = score_label, "Score per watt"
            rank_column, rank_label = "Efficiency", "Peak score/W"
        else:
            x_column, y_column = "Board_Power_W", "Efficiency"
            x_label, y_label = "Board power (W)", "Score per watt"
            rank_column, rank_label = "Efficiency", "Peak score/W"

        self._style_axis(
            self.main_axis,
            definition.kicker.replace("-", " ").upper() + " CURVE",
            f"{frame['__label'].nunique()} profiles",
        )
        self.main_axis.set_xlabel(x_label, labelpad=6)
        self.main_axis.set_ylabel(y_label, labelpad=6)

        labels = sorted(frame["__label"].unique(), key=str.casefold)
        for index, label in enumerate(labels):
            subset = frame[frame["__label"] == label].sort_values(x_column)
            color = PALETTE[index % len(PALETTE)]
            for source_index, (_source, source_frame) in enumerate(
                subset.groupby("__source", sort=False)
            ):
                source_frame = source_frame.sort_values(x_column)
                has_curve = len(source_frame) >= 4
                dotted = kind in {"SPEC INT", "SPEC FP"} and 1 < len(source_frame) < 4
                has_line = has_curve or dotted
                legend_label = self._legend_label(label) if source_index == 0 else "_nolegend_"
                if has_line:
                    (line,) = self.main_axis.plot(
                        source_frame[x_column],
                        source_frame[y_column],
                        color=color,
                        linewidth=1.8 if dotted else 2.4,
                        linestyle=":" if dotted else "-",
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
                    label="_nolegend_" if has_line else legend_label,
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
        note = " · dotted lines join sparse samples" if kind in {"SPEC INT", "SPEC FP"} else ""
        self.chart_note.configure(text=f"{len(frame):,} curve points shown{note}")

    def _draw_battery_charts(self, frame: pd.DataFrame) -> None:
        view = self.view_var.get()
        if "soc" not in frame or any(column not in frame for column in SOC_AVERAGE_COLUMNS):
            frame = add_soc_average_columns(frame)
        if view in SOC_AVERAGE_VIEWS:
            self._draw_soc_average_charts(frame, view)
            return
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
                soc=("soc", "first"),
                soc_device_count=("soc_device_count", "max"),
                soc_avg_capacity_wh=("soc_avg_capacity_wh", "mean"),
                soc_avg_power_w=("soc_avg_power_w", "mean"),
                soc_avg_min_per_wh=("soc_avg_min_per_wh", "mean"),
                soc_avg_runtime_hours=("soc_avg_runtime_hours", "mean"),
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
        show_soc_averages = (
            bool(getattr(self, "show_soc_averages_var", None))
            and self.show_soc_averages_var.get()
            and view in {"Energy efficiency", "Average power draw"}
        )

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
        self.main_axis.set_xlabel(x_label, labelpad=6)
        self.main_axis.set_ylabel(y_label, labelpad=6)
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
                            + (f"SoC: {row['soc']}\n" if row["soc"] else "")
                            + f"Runtime: {float(row['hours']):.2f} h "
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
        soc_average_count = 0
        if show_soc_averages:
            average_y_columns = {
                "avgPowerW": "soc_avg_power_w",
                "minPerWh": "soc_avg_min_per_wh",
                "hours": "soc_avg_runtime_hours",
            }
            average_y = average_y_columns[y_column]
            soc_rows = summary[summary["soc"].fillna("").astype(str).str.strip().ne("")]
            soc_rows = soc_rows.drop_duplicates("soc").sort_values("soc", key=lambda values: values.str.casefold())
            for index, (_, row) in enumerate(soc_rows.iterrows()):
                y_value = float(row[average_y])
                if not np.isfinite(y_value):
                    continue
                soc = str(row["soc"])
                device_count = int(row["soc_device_count"])
                color = PALETTE[index % len(PALETTE)]
                average_line = self.main_axis.axhline(
                    y_value,
                    color=color,
                    linestyle=(0, (5, 3)),
                    linewidth=1.8,
                    alpha=0.9,
                    zorder=2,
                    picker=5,
                )
                average_line._socpk_label = (
                    f"{soc} average: {y_value:.2f} {y_label} · "
                    f"{device_count} device{'s' if device_count != 1 else ''}"
                )
                self.line_artists.append(average_line)
                self.main_axis.text(
                    0.985,
                    y_value,
                    f"{self._short_label(soc, 22)}  {y_value:.2f}",
                    transform=self.main_axis.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    alpha=0.95,
                    bbox={"facecolor": PLOT_BG, "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
                    zorder=7,
                )
                soc_average_count += 1

        legend_handles = []
        if measured_count:
            legend_handles.extend((
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
            ))
        if soc_average_count:
            legend_handles.append(Line2D(
                [], [], linestyle=(0, (5, 3)), linewidth=1.8, color=ORANGE,
                label="Processor average (y-axis)",
            ))
        if legend_handles:
            legend = self.main_axis.legend(
                handles=legend_handles,
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
                + (f" · {soc_average_count} processor averages" if show_soc_averages else "")
            )
        )

    def _draw_soc_average_charts(self, frame: pd.DataFrame, view: str) -> None:
        averages = soc_average_summary(frame)
        if averages.empty:
            self.ranking_values = pd.Series(dtype=float)
            self.ranking_offset = 0
            self._style_axis(self.main_axis, "No processor averages available")
            self.main_axis.text(
                0.5, 0.5, "Collect battery data with --auto-soc",
                transform=self.main_axis.transAxes, ha="center", va="center", color=MUTED,
            )
            self.main_axis.set_xticks([])
            self.main_axis.set_yticks([])
            self._style_axis(self.rank_axis, "Processor ranking", "waiting for SoC data")
            self.rank_axis.set_xticks([])
            self.rank_axis.set_yticks([])
            self.chart_note.configure(text="No precomputed SoC averages in the loaded battery data")
            return

        if view == "SoC Average Power Draw":
            x_column, y_column = "soc_avg_capacity_wh", "soc_avg_power_w"
            x_label, y_label = "Average battery capacity (Wh)", "Average power draw (W)"
            title = "Processor-average power draw"
            rank_label, higher = "Average power (W)", False
        else:
            x_column, y_column = "soc_avg_power_w", "soc_avg_min_per_wh"
            x_label, y_label = "Average power draw (W)", "Average minutes per Wh"
            title = "Processor-average efficiency"
            rank_label, higher = "Average minutes per Wh", True

        self._style_axis(
            self.main_axis,
            title,
            f"{len(averages)} processors · precomputed across the loaded battery dataset",
        )
        self.main_axis.set_xlabel(x_label, labelpad=6)
        self.main_axis.set_ylabel(y_label, labelpad=6)
        show_labels = len(averages) <= 24
        for index, (_, row) in enumerate(averages.iterrows()):
            soc = str(row["soc"])
            x_value = float(row[x_column])
            y_value = float(row[y_column])
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            color = PALETTE[index % len(PALETTE)]
            point = self.main_axis.scatter(
                x_value,
                y_value,
                s=92,
                color=color,
                edgecolor=PLOT_BG,
                linewidth=1.4,
                zorder=4,
                picker=True,
            )
            point._socpk_label = soc
            self.line_artists.append(point)
            device_count = int(row["soc_device_count"])
            self.hover_points.append(
                HoverPoint(
                    x=x_value,
                    y=y_value,
                    label=soc,
                    details=(
                        f"{soc} · {device_count} device{'s' if device_count != 1 else ''}\n"
                        f"Average capacity: {float(row['soc_avg_capacity_wh']):.2f} Wh\n"
                        f"Average power: {float(row['soc_avg_power_w']):.2f} W\n"
                        f"Average efficiency: {float(row['soc_avg_min_per_wh']):.2f} min/Wh\n"
                        f"Average runtime: {float(row['soc_avg_runtime_hours']):.2f} h"
                    ),
                    color=color,
                )
            )
            if show_labels:
                self.main_axis.annotate(
                    self._short_label(soc, 25),
                    (x_value, y_value),
                    xytext=(7, 7),
                    textcoords="offset points",
                    color=TEXT,
                    fontsize=8,
                    alpha=0.92,
                )

        ranking = averages.set_index("soc")[y_column]
        self._draw_ranking(ranking, rank_label, higher_is_better=higher)
        self.chart_note.configure(
            text=f"{len(averages)} precomputed processor averages · {int(averages['soc_device_count'].sum())} device profiles"
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
        self._style_axis(
            self.rank_axis,
            "Ranking",
            f"{'↑' if higher_is_better else '↓'} better · {start + 1}–{end}/{total}",
        )
        label_width = max(16, int(self.rank_axis.get_window_extent().width / 5.5))
        short_labels = [self._short_label(self._legend_label(label), label_width) for label in values.index]
        selected_labels = sorted(self.ranking_values.index, key=str.casefold)
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
            height=0.40,
            alpha=0.9,
        )
        self.rank_axis.set_yticks([])
        self.rank_axis.set_ylim(-0.5, max(len(values) - 0.25, 0.75))
        for position, label in zip(row_positions, short_labels):
            self.rank_axis.text(0, position + 0.24, label, ha="left", va="bottom", color=TEXT, fontsize=8)
        self.rank_axis.set_xlabel(self.ranking_metric_label, labelpad=6)
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

        self._sync_ranking_scrollbar(total, page_size, start)

    def _sync_ranking_scrollbar(self, total: int, page_size: int, start: int) -> None:
        scrollbar = getattr(self, "ranking_scrollbar", None)
        if scrollbar is None:
            return
        if total <= page_size or not total:
            scrollbar.set(0.0, 1.0)
            scrollbar.state(["disabled"])
            return
        scrollbar.state(["!disabled"])
        scrollbar.set(start / total, min(1.0, (start + page_size) / total))

    def _set_ranking_offset(self, offset: int) -> None:
        total = len(self.ranking_values)
        page_size = min(self.ranking_page_size, total)
        maximum_offset = max(0, total - page_size)
        new_offset = min(max(0, int(offset)), maximum_offset)
        if new_offset == self.ranking_offset:
            return
        self.ranking_offset = new_offset
        self._render_ranking()
        self.canvas.draw_idle()

    def on_ranking_scrollbar(self, *args: str) -> None:
        """Handle native Tk scrollbar commands for the ranking subplot."""
        if len(args) < 2 or len(self.ranking_values) <= self.ranking_page_size:
            return
        command, value = args[0], args[1]
        if command == "moveto":
            try:
                offset = round(float(value) * len(self.ranking_values))
            except ValueError:
                return
        elif command == "scroll":
            try:
                amount = int(float(value))
            except ValueError:
                return
            unit = args[2] if len(args) > 2 else "units"
            stride = self.ranking_page_size if unit == "pages" else 3
            offset = self.ranking_offset + amount * stride
        else:
            return
        self._set_ranking_offset(offset)

    def on_ranking_scroll(self, event) -> None:
        if event.inaxes is not self.rank_axis:
            return
        total = len(self.ranking_values)
        if total <= self.ranking_page_size:
            return
        step = getattr(event, "step", 0)
        if not step:
            step = 1 if getattr(event, "button", None) == "up" else -1
        direction = 1 if step > 0 else -1
        distance = max(1, int(round(abs(float(step)) * 3)))
        self._set_ranking_offset(self.ranking_offset - direction * distance)

    def _update_stats(self, frame: pd.DataFrame) -> None:
        all_frame = self.collections[self.dataset_key]
        if self.dataset_key == "Battery" and self.view_var.get() in SOC_AVERAGE_VIEWS:
            shown_averages = soc_average_summary(frame)
            all_averages = soc_average_summary(all_frame)
            self.stat_values["profiles"].configure(
                text=f"{len(shown_averages)} / {len(all_averages)}"
            )
            self.stat_values["points"].configure(text=f"{len(shown_averages):,}")
        else:
            selected_count = frame["__label"].nunique()
            self.stat_values["profiles"].configure(
                text=f"{selected_count} / {all_frame['__label'].nunique()}"
            )
            self.stat_values["points"].configure(text=f"{len(frame):,}")
        if frame.empty:
            leader_text = "—"
        elif self.dataset_key in CURVE_DATASETS:
            metric = (DATASET_DEFINITIONS[self.dataset_key].score_column
                      if self.view_var.get() == "Performance curve" else "Efficiency")
            leader = frame.groupby("__label")[metric].max().idxmax()
            leader_text = self._leader_label(str(leader), frame)
        elif self.view_var.get() in SOC_AVERAGE_VIEWS:
            averages = soc_average_summary(frame)
            if averages.empty:
                leader_text = "—"
            else:
                metric = ("soc_avg_power_w" if self.view_var.get() == "SoC Average Power Draw"
                          else "soc_avg_min_per_wh")
                best_index = averages[metric].idxmin() if metric == "soc_avg_power_w" else averages[metric].idxmax()
                leader_text = str(averages.loc[best_index, "soc"])
        else:
            metric = {
                "Runtime vs capacity": "hours",
                "Energy efficiency": "minPerWh",
                "Average power draw": "avgPowerW",
            }[self.view_var.get()]
            grouped = frame.groupby("__label")[metric].mean()
            leader = grouped.idxmin() if metric == "avgPowerW" else grouped.idxmax()
            leader_text = self._leader_label(str(leader), frame)
        self.stat_values["leader"].configure(text=leader_text)
        if hasattr(self, "hero"):
            self._resize_leader()

        source_count = all_frame["__source"].nunique()
        warning_note = f" · {len(self.load_warnings)} warning(s)" if self.load_warnings else ""
        self.source_note.configure(
            text=f"{source_count} source file{'s' if source_count != 1 else ''}{warning_note}"
        )

    def _update_selection_note(self) -> None:
        if not self.dataset_key:
            return
        selected_count = (self._selected_frame()["__label"].nunique()
                          if self.dataset_key in self.collections else 0)
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

    def _export_filename(self) -> str:
        definition = DATASET_DEFINITIONS[self.dataset_key]
        mode = self.selection_modes.get(self.dataset_key, "manual")
        parts = ["socpk", definition.kicker, self.view_var.get(), mode]
        if self.dataset_key in {"SPEC INT", "SPEC FP"}:
            filters = self.core_filters.get(self.dataset_key, CoreFilter())
            parts.append("groups-" + "-".join(sorted(filters.groups)) if filters.groups else "all-groups")
            parts.append("cores-" + "-".join(sorted(filters.names)) if filters.names else "all-cores")
        search = self.search_var.get().strip()
        if search:
            parts.append("search-" + search)
        if mode == "manual":
            labels = sorted(self.selected.get(self.dataset_key, set()))
            signature = hashlib.sha256("\n".join(labels).encode()).hexdigest()[:8]
            parts.append(f"{len(labels)}-profiles-{signature}")
        stem = "-".join(re.sub(r"[^\w]+", "-", part.casefold()).strip("-") for part in parts)
        # Many selected cores can exceed a filesystem's filename limit.
        if len(stem.encode("utf-8")) > 180:
            suffix = hashlib.sha256(stem.encode()).hexdigest()[:10]
            stem = stem.encode()[:168].decode("utf-8", errors="ignore").rstrip("-") + "-" + suffix
        return stem + ".png"

    def export_chart(self) -> None:
        default_name = self._export_filename()
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

    @staticmethod
    def _leader_label(label: str, frame: pd.DataFrame) -> str:
        matches = frame[frame["__label"].eq(label)]
        if matches.empty or "Core" not in matches:
            return label
        row = matches.iloc[0]
        parts = [str(row["Core"]), str(row["Core_Group"]).capitalize()]
        variant = row.get("Core_Variant", "")
        if pd.notna(variant) and variant:
            parts.append({"p": "P-core", "e": "E-core"}.get(str(variant), str(variant)))
        return f"{row['CPU']} — " + " · ".join(parts)

    def _legend_label(self, label: str) -> str:
        if self.dataset_key in {"SPEC INT", "SPEC FP"} and " — " in label:
            chip, core = label.split(" — ", 1)
            return f"{chip} — {core.split(' · ', 1)[0]}"
        return label

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

    @staticmethod
    def _exit_on_termination(signum: int, _frame: object) -> None:
        """Exit without entering Tk's unsafe macOS signal finalizer.

        Tk 9.0 installs a macOS signal handler that calls ``Tcl_Exit``.  If
        SIGTERM or SIGINT arrives while ``mainloop`` is active, that handler
        can call back into Python after the thread state has been released and
        abort the interpreter with ``PyEval_RestoreThread``.  External
        termination is already an explicit request to stop, so avoid Tk
        teardown entirely and return the conventional shell status instead.
        """
        os._exit(128 + signum)

    def run(self) -> None:
        # This also makes Ctrl-C from Terminal and Stop from an IDE quiet and
        # predictable on macOS.  The window's close button still uses close().
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._exit_on_termination)
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
