"""Fetch laptop Time Spy GPU curves from SoCPK's public chart API.

The API supplies power and scores directly. Explicit ``base_url`` arguments
still support legacy SVG sources using the axis helpers below. CSV exports
create separate snapshots and never replace an existing file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote

import pandas as pd  # type: ignore
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from socpk_client import (  # noqa: E402
    extract_laptop_axis_geometry,
    extract_laptop_curve_coordinates,
    LAPTOP_GPU_PAGE_SLUG,
    fetch_curve_series,
    curve_frame,
    series_label,
    new_snapshot,
)

__all__ = [
    "extract_axis_scaling",
    "refresh_axis_scaling",
    "discover_gpu_names",
    "parse_gpu_curve",
    "scrape_gpu_curves",
]

PAGE_URL = "https://www.socpk.com/laptopgpucurve"
AXIS_URL = (
    "https://www.socpk.com/assets/curves/laptop-gpu/gpu/"
    "CPU_2022_12_28_axis.svg"
)
LAYER_BASE_URL = "https://www.socpk.com/assets/curves/laptop-gpu/gpu/"

X_START: float = 112.5
X_WIDTH: float = 1179.0
POWER_RANGE: float = 220.0
Y_BASE: float = 1445.5
Y_HEIGHT: float = 1343.0
SCORE_RANGE: float = 27500.0


def extract_axis_scaling(
    base_svg_url: str = AXIS_URL,
) -> Optional[Dict[str, float]]:
    """Extract TGP and Time Spy scaling from the laptop axis SVG."""
    try:
        response = requests.get(base_svg_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return None
    return extract_laptop_axis_geometry(response.text)


def refresh_axis_scaling() -> bool:
    """Update module axis constants from the current laptop base layer."""
    params = extract_axis_scaling()
    if not params:
        return False
    globals().update(params)
    return True


def _to_board_power(x: float) -> float:
    return (x - X_START) / X_WIDTH * POWER_RANGE


def _to_score(y: float) -> float:
    return (Y_BASE - y) / Y_HEIGHT * SCORE_RANGE


def discover_gpu_names() -> List[str]:
    """Return names currently published in the chart API."""
    return [series_label(item) for item in fetch_curve_series(LAPTOP_GPU_PAGE_SLUG)]


def parse_gpu_curve(
    gpu_name: str,
    base_url: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch one curve; an explicit base URL selects the legacy SVG parser."""
    if base_url is None:
        return curve_frame(
            fetch_curve_series(LAPTOP_GPU_PAGE_SLUG, [gpu_name]), None, "GPU_Score",
        )
    svg_url = f"{base_url.rstrip('/')}/{quote(gpu_name, safe='')}.svg"
    try:
        response = requests.get(svg_url, timeout=10)
    except requests.RequestException:
        return None
    if not response.ok:
        return None

    coordinates = extract_laptop_curve_coordinates(response.text)
    if not coordinates:
        return None
    points = [(_to_board_power(x), _to_score(y)) for x, y in coordinates]
    frame = pd.DataFrame(points, columns=["Board_Power_W", "GPU_Score"])
    frame["Efficiency"] = (
        frame["GPU_Score"]
        / frame["Board_Power_W"].replace(0, pd.NA)
    )
    return frame


def scrape_gpu_curves(
    gpu_names: Optional[Iterable[str]] = None,
    *,
    default_gpu_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fetch all selected curves in one poll, preserving the CSV schema.

    ``default_gpu_names`` is retained for caller compatibility. Discovery
    now requires a working API; obsolete asset lists cannot recover failures.
    """
    return curve_frame(
        fetch_curve_series(LAPTOP_GPU_PAGE_SLUG, gpu_names),
        "GPU", "GPU_Score",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--gpus', nargs='*', default=None,
                        help="Names to scrape; discovers all if omitted.")
    parser.add_argument('--output', default="snapshots/laptop_gpu_curves.csv",
                        help="Snapshot CSV path; an existing path gets a timestamped sibling.")
    args = parser.parse_args()
    try:
        frame = scrape_gpu_curves(args.gpus)
    except (requests.RequestException, ValueError) as exc:
        raise SystemExit(f"Could not fetch SoCPK curves: {exc}") from exc
    if frame.empty:
        print("No curves scraped; no file written.")
        return
    with new_snapshot(args.output) as output:
        frame.to_csv(output, index=False)
    print(f"Scraped {len(frame)} rows for {frame['GPU'].nunique()} GPUs → {output.name}")


if __name__ == '__main__':
    main()
