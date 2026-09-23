"""Shared loading, classification and derived columns for every SoCPK CSV.

The Tk dashboard (``socpk_gui``) and the web dashboard (``socpk_web``) read the
same snapshots, so the pandas layer lives here and neither front end owns it.
This module imports no GUI toolkit, which keeps it usable from headless tests
and from the web server on a Python build without Tk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cpu_benchmarks import CPU_BENCHMARKS, CORE_COLUMNS, cpu_profile_label
from battery_soc import canonical_soc_name


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
class CoreFilter:
    # Empty sets mean unrestricted; selections within each field are ORed.
    groups: frozenset[str] = frozenset()
    names: frozenset[str] = frozenset()



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
    # Evidence and fixture CSVs are implementation inputs, not dashboard data.
    # The project-root launch scans recursively, so keep them from being merged
    # with the newest production snapshot family.
    ignored = {
        ".git", ".venv", "venv", "__pycache__", ".idea", ".pytest_cache",
        "artifacts", "tests",
    }
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



# Shown when a tab has no snapshot yet, so both front ends name the same file
# and the same collector command.
DATASET_SOURCE_HINTS = {
    **{
        benchmark.dataset_key: (
            "snapshots/" + benchmark.filename,
            f'python "Performance Benchmark/cpu_curve_parser.py" --benchmark {name}',
        )
        for name, benchmark in CPU_BENCHMARKS.items()
    },
    "GPU": (
        "snapshots/gpu_snl_curves.csv",
        'python "Performance Benchmark/gpu_curve_parser.py"',
    ),
    "Laptop GPU": (
        "snapshots/laptop_gpu_curves.csv",
        'python "Performance Benchmark/laptop_gpu_curve_parser.py"',
    ),
    "Battery": (
        "snapshots/battery_results.csv",
        "python Battery/battery_parser.py",
    ),
}
