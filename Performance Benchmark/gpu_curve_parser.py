"""Fetch phone Steel Nomad Light GPU curves from SoCPK's public chart API.

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
    GPU_PAGE_SLUG,
    fetch_curve_series,
    curve_frame,
    new_snapshot,
)

__all__ = [
    "extract_axis_scaling",
    "refresh_axis_scaling",
    "discover_gpu_names",
    "parse_gpu_curve",
    "scrape_gpu_curves",
]

GPU_PAGE_URL = "https://www.socpk.com/gpucurve/"
GPU_AXIS_URL = (
    "https://www.socpk.com/assets/curves/gpu-snl/"
    "3DMark%20Steel%20Nomad%20Light_asis.svg"
)
GPU_LAYER_BASE_URL = "https://www.socpk.com/assets/curves/gpu-snl/gpu/"

###############################################################################
# Axis scaling defaults and helpers
###############################################################################

# Legacy SNL axis ranges and fallback geometry.  Runtime refresh updates
# geometry; the site's outlined tick labels are not machine-readable text.
X_START: float = 144.0
X_WIDTH: float = 892.8
POWER_RANGE: float = 18.0  # Current SNL chart maximum board power in watts

Y_BASE: float = 576.7
Y_HEIGHT: float = 498.96
SCORE_RANGE: float = 4000.0  # Default maximum GPU score (SNL)


def extract_axis_scaling(
    base_svg_url: str = GPU_AXIS_URL,
) -> Optional[Dict[str, float]]:
    """Extract axis scaling parameters from the GPU base axes layer.

    Parameters
    ----------
    base_svg_url : str, optional
        URL of the base SVG containing the grid and axes.  Override
        this if the site reorganises its files.

    Returns
    -------
    dict or None
        Mapping of ``X_START``, ``X_WIDTH``, ``POWER_RANGE``,
        ``Y_BASE``, ``Y_HEIGHT`` and ``SCORE_RANGE``.  Returns
        ``None`` if the SVG cannot be fetched or parsed.
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
    """Update global axis constants from the GPU base layer.

    Returns ``True`` if the update succeeds, ``False`` otherwise.
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
    """Convert a y pixel coordinate to GPU score."""
    return (Y_BASE - y) / Y_HEIGHT * SCORE_RANGE


def discover_gpu_names() -> List[str]:
    """Return names currently published in the chart API."""
    return [item.get("name_en") or item["name"]
            for item in fetch_curve_series(GPU_PAGE_SLUG)]


def parse_gpu_curve(
    gpu_name: str,
    base_url: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch one curve; an explicit base URL selects the legacy SVG parser."""
    if base_url is None:
        frame = curve_frame(
            fetch_curve_series(GPU_PAGE_SLUG, [gpu_name]),
            "GPU", "GPU_Score",
        )
        return frame.drop(columns=["GPU"])
    if base_url is not None:
        svg_url = (
            f"{base_url.rstrip('/')}/3dmark_snl_{quote(gpu_name, safe='')}.svg"
        )
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
    df = pd.DataFrame(points, columns=["Board_Power_W", "GPU_Score"])
    df['Efficiency'] = df['GPU_Score'] / df['Board_Power_W'].replace(0, pd.NA)
    return df


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
        fetch_curve_series(GPU_PAGE_SLUG, gpu_names),
        "GPU", "GPU_Score",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--gpus', nargs='*', default=None,
                        help="Names to scrape; discovers all if omitted.")
    parser.add_argument('--output', default="snapshots/gpu_snl_curves.csv",
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
        output_path = output.name
    print(f"Scraped {len(frame)} rows for {frame['GPU'].nunique()} GPUs → {output_path}")


if __name__ == '__main__':
    main()
