"""Fetch Geekbench 6/7 multi-core and SPEC CPU 2026 INT/FP single-core curves from SoCPK's public chart API.

The API supplies power and scores directly. Explicit ``base_url`` arguments
still support legacy SVG sources using the axis helpers below. CSV exports
create separate snapshots and never replace an existing file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote
from typing import Iterable, List, Optional, Dict

import pandas as pd  # type: ignore
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from socpk_client import (  # noqa: E402
    extract_axis_geometry,
    extract_curve_coordinates,
    fetch_chart_page,
    select_curve_series,
    series_label,
    new_snapshot,
)

from cpu_benchmarks import CPU_BENCHMARKS, CORE_COLUMNS  # noqa: E402

__all__ = [
    "extract_axis_scaling",
    "refresh_axis_scaling",
    "discover_cpu_names",
    "parse_cpu_curve",
    "scrape_cpu_curves",
    "scrape_cpu_benchmarks",
]

CPU_PAGE_URL = "https://www.socpk.com/cpucurve/gb6/"
CPU_AXIS_URL = (
    "https://www.socpk.com/assets/curves/cpu-gb6/Geekbench%206_asis.svg"
)
CPU_LAYER_BASE_URL = "https://www.socpk.com/assets/curves/cpu-gb6/cpu/"


###############################################################################
# Axis scaling constants and helpers
###############################################################################

# Legacy GB6 axis ranges.  Runtime refresh updates the SVG plot geometry;
# the site's outlined tick labels are not machine-readable text.
X_START: float = 144.0
X_WIDTH: float = 892.8
POWER_RANGE: float = 22.0

Y_BASE: float = 576.7
Y_HEIGHT: float = 498.96
SCORE_RANGE: float = 15000.0


def extract_axis_scaling(
    base_svg_url: str = CPU_AXIS_URL,
) -> Optional[Dict[str, float]]:
    """Derive dynamic axis scaling from the base GB6 axes layer.

    The base SVG defines the grid and axes for all curves.  This function
    fetches it, extracts the plotting bounds, and combines those bounds
    with the current GB6 power and score ranges.

    Parameters
    ----------
    base_svg_url : str, optional
        URL of the base axes SVG.  Override this if the site
        reorganises its file structure.

    Returns
    -------
    dict or None
        A dictionary with keys ``X_START``, ``X_WIDTH``, ``POWER_RANGE``,
        ``Y_BASE``, ``Y_HEIGHT`` and ``SCORE_RANGE`` if parsing
        succeeds; otherwise ``None``.
    """
    try:
        resp = requests.get(base_svg_url, timeout=10)
        resp.raise_for_status()
    except Exception:
        return None
    return extract_axis_geometry(
        resp.text,
        power_range=POWER_RANGE,
        score_range=SCORE_RANGE,
    )


def refresh_axis_scaling() -> bool:
    """Update global axis constants based on the current base layer.

    Calls :func:`extract_axis_scaling` and, if successful, assigns the
    resulting values to the module‑level constants ``X_START``,
    ``X_WIDTH``, ``POWER_RANGE``, ``Y_BASE``, ``Y_HEIGHT`` and
    ``SCORE_RANGE``.  Returns ``True`` on success, ``False`` on failure.
    """
    params = extract_axis_scaling()
    if not params:
        return False
    globals().update(params)
    return True


def _to_board_power(x: float) -> float:
    """Convert an x pixel coordinate to board power (W)."""
    return (x - X_START) / X_WIDTH * POWER_RANGE


def _to_score(y: float) -> float:
    """Convert a y pixel coordinate to Geekbench score."""
    return (Y_BASE - y) / Y_HEIGHT * SCORE_RANGE


def _benchmark_series(benchmark: str, names=None, *, page=None, core_group=None):
    if benchmark not in CPU_BENCHMARKS:
        raise ValueError(f"Unknown CPU benchmark: {benchmark!r}.")
    definition = CPU_BENCHMARKS[benchmark]
    if core_group and not definition.single_core:
        raise ValueError("Core group filtering is only available for SPEC CPU 2026.")
    series = select_curve_series(
        page if page is not None else fetch_chart_page(definition.slug), names,
        suite=definition.suite, multiple_cores=definition.single_core,
    )
    if core_group:
        series = [item for item in series if item.get("meta", {}).get("coreGroup") == core_group]
    if definition.single_core:
        for item in series:
            meta = item.get("meta", {})
            if not meta.get("coreName") or not meta.get("coreGroup"):
                raise ValueError(f"Missing SPEC core identity for {item['name']}.")
    return series


def _cpu_frame(series, benchmark: str) -> pd.DataFrame:
    definition = CPU_BENCHMARKS[benchmark]
    columns = ["CPU", "Board_Power_W", definition.score_column, "Efficiency", "Benchmark"]
    if definition.single_core:
        columns += list(CORE_COLUMNS)
    rows = []
    for item in series:
        meta = item.get("meta", {})
        identity = {"CPU": series_label(item), "Benchmark": benchmark}
        if definition.single_core:
            identity.update(Core=meta["coreName"], Core_Group=meta["coreGroup"],
                            Core_Variant=meta.get("coreVariant", ""))
        for power, score in item["points"]:
            rows.append({**identity, "Board_Power_W": power, definition.score_column: score,
                         "Efficiency": score / power})
    return pd.DataFrame(rows, columns=columns)


def discover_cpu_names(benchmark: str = "GB6") -> List[str]:
    """Return unique chip names for a benchmark; SPEC chips have several cores."""
    return list(dict.fromkeys(series_label(item)
                              for item in _benchmark_series(benchmark)))


def parse_cpu_curve(
    cpu_name: str,
    base_url: Optional[str] = None,
    *, benchmark: str = "GB6", core_group: str | None = None,
) -> Optional[pd.DataFrame]:
    """Fetch one curve; an explicit base URL selects the legacy SVG parser."""
    if base_url is None:
        return scrape_cpu_curves([cpu_name], benchmark=benchmark, core_group=core_group).drop(columns=["CPU"])
    if benchmark != "GB6" or core_group:
        raise ValueError("Legacy SVG sources support GB6 only.")
    svg_url = f"{base_url.rstrip('/')}/CPU_gb6_{quote(cpu_name, safe='')}.svg"
    try:
        resp = requests.get(svg_url, timeout=10)
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    coordinates = extract_curve_coordinates(resp.text)
    if not coordinates:
        return None
    points = [(_to_board_power(x), _to_score(y)) for x, y in coordinates]
    df = pd.DataFrame(points, columns=["Board_Power_W", "GB6_Multi_Score"])
    # Compute efficiency; avoid division by zero
    df['Efficiency'] = df['GB6_Multi_Score'] / df['Board_Power_W'].replace(0, pd.NA)
    return df


def scrape_cpu_curves(
    cpu_names: Optional[Iterable[str]] = None,
    *,
    default_cpu_names: Optional[List[str]] = None,
    benchmark: str = "GB6",
    core_group: str | None = None,
) -> pd.DataFrame:
    """Fetch one benchmark in its own schema. SPEC exports retain core metadata.

    The library default stays GB6 for existing callers. ``default_cpu_names``
    is retained for compatibility; discovery requires the current API.
    """
    return _cpu_frame(_benchmark_series(benchmark, cpu_names, core_group=core_group), benchmark)


def scrape_cpu_benchmarks(cpu_names=None) -> dict[str, pd.DataFrame]:
    """Fetch every CPU benchmark, using one consistent poll per source page."""
    pages = {}
    frames = {}
    names = list(cpu_names) if cpu_names is not None else None
    for benchmark, definition in CPU_BENCHMARKS.items():
        if definition.slug not in pages:
            pages[definition.slug] = fetch_chart_page(definition.slug)
        frames[benchmark] = _cpu_frame(
            _benchmark_series(benchmark, names, page=pages[definition.slug]), benchmark,
        )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cpus', nargs='*', default=None,
                        help="Chip names; SPEC includes every published core of each chip.")
    parser.add_argument('--benchmark', choices=[*CPU_BENCHMARKS, "all"], default="GB6",
                        help="Benchmark to export (default: GB6); all writes separate CSVs.")
    parser.add_argument('--core-group', choices=["super", "large", "medium", "small"],
                        help="Select a core group for a single SPEC benchmark.")
    parser.add_argument('--output', help="Snapshot path for a single benchmark; never overwrites.")
    parser.add_argument('--output-dir', default="snapshots",
                        help="Directory for default snapshot filenames (default: snapshots).")
    args = parser.parse_args()
    if args.benchmark == "all" and (args.output or args.core_group):
        parser.error("--output and --core-group require a single --benchmark; use --output-dir for all.")
    try:
        frames = (scrape_cpu_benchmarks(args.cpus) if args.benchmark == "all" else {
            args.benchmark: scrape_cpu_curves(args.cpus, benchmark=args.benchmark, core_group=args.core_group)
        })
    except (requests.RequestException, ValueError) as exc:
        raise SystemExit(f"Could not fetch SoCPK curves: {exc}") from exc
    for benchmark, frame in frames.items():
        if frame.empty:
            print(f"No {benchmark} curves scraped; no file written.")
            continue
        requested = args.output or Path(args.output_dir) / CPU_BENCHMARKS[benchmark].filename
        with new_snapshot(requested) as output:
            frame.to_csv(output, index=False)
        identities = ["CPU", *CORE_COLUMNS] if CPU_BENCHMARKS[benchmark].single_core else ["CPU"]
        count = len(frame[identities].drop_duplicates())
        print(f"{benchmark}: scraped {len(frame)} rows for {count} profiles → {output.name}")


if __name__ == '__main__':
    main()
