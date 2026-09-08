"""Fetch Geekbench 6 CPU curves from SoCPK's public chart API.

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
    CPU_PAGE_SLUG,
    fetch_curve_series,
    curve_frame,
    new_snapshot,
)

__all__ = [
    "extract_axis_scaling",
    "refresh_axis_scaling",
    "discover_cpu_names",
    "parse_cpu_curve",
    "scrape_cpu_curves",
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


def discover_cpu_names() -> List[str]:
    """Return names currently published in the chart API."""
    return [item.get("name_en") or item["name"]
            for item in fetch_curve_series(CPU_PAGE_SLUG, suite="GB6")]


def parse_cpu_curve(
    cpu_name: str,
    base_url: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch one curve; an explicit base URL selects the legacy SVG parser."""
    if base_url is None:
        frame = curve_frame(
            fetch_curve_series(CPU_PAGE_SLUG, [cpu_name], suite="GB6"),
            "CPU", "GB6_Multi_Score",
        )
        return frame.drop(columns=["CPU"])
    if base_url is not None:
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
) -> pd.DataFrame:
    """Fetch all selected curves in one poll, preserving the CSV schema.

    ``default_cpu_names`` is retained for caller compatibility. Discovery
    now requires a working API; obsolete asset lists cannot recover failures.
    """
    return curve_frame(
        fetch_curve_series(CPU_PAGE_SLUG, cpu_names, suite="GB6"),
        "CPU", "GB6_Multi_Score",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cpus', nargs='*', default=None,
                        help="Names to scrape; discovers all if omitted.")
    parser.add_argument('--output', default="snapshots/cpu_gb6_curves.csv",
                        help="Snapshot CSV path; an existing path gets a timestamped sibling.")
    args = parser.parse_args()
    try:
        frame = scrape_cpu_curves(args.cpus)
    except (requests.RequestException, ValueError) as exc:
        raise SystemExit(f"Could not fetch SoCPK curves: {exc}") from exc
    if frame.empty:
        print("No curves scraped; no file written.")
        return
    with new_snapshot(args.output) as output:
        frame.to_csv(output, index=False)
        output_path = output.name
    print(f"Scraped {len(frame)} rows for {frame['CPU'].nunique()} CPUs → {output_path}")


if __name__ == '__main__':
    main()
