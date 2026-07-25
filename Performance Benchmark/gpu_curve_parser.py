"""
gpu_curve_parser.py
===================

This module provides functions and a command‑line interface to scrape
SocPK’s GPU (Steel Nomad Light) efficiency curves.  Each GPU curve is
published as its own SVG under
``https://www.socpk.com/assets/curves/gpu-snl/gpu/``.  The parser reads
the base axes layer to derive the conversion from pixel positions to
physical board power (W) and GPU performance score, and can process
both continuous curves and scatter‑point curves.  A convenience
function ``scrape_gpu_curves`` aggregates data from multiple chips
into a single pandas DataFrame.

Key features
------------

* **Current axis geometry.**  The script extracts horizontal and
  vertical plot bounds from the current base axes layer.
* **Support for continuous and scatter curves.**  Some GPUs (e.g. Adreno
  in Snapdragon chips) publish full curves, while others provide a
  handful of scatter points.  ``parse_gpu_curve`` handles both.
* **Efficiency calculation.**  The returned DataFrame includes an
  ``Efficiency`` column equal to ``GPU_Score / Board_Power_W`` (NaN
  when power is zero).
* **Command‑line interface.**  Running this file directly allows you
  to scrape multiple GPUs and save the results to a CSV.

Example
-------

Scrape a few GPUs into a CSV from the command line::

    python gpu_curve_parser.py --gpus "A19 Pro" "SD8 Elite Gen5" --output gpu_curves.csv

Then analyse and plot with the companion ``curve_analysis`` script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlparse
from typing import Iterable, List, Optional, Dict

import pandas as pd  # type: ignore
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from socpk_client import (  # noqa: E402
    extract_axis_geometry,
    extract_curve_coordinates,
    fetch_curve_manifest,
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

# Fallback list used only when discovery fails
_FALLBACK_GPU_NAMES: List[str] = [
    "A16", "A17 Pro", "A18", "A18 Pro", "A19", "A19 Pro",
    "SD8 Elite Gen5", "SD8 Elite (9600)", "SD8 Elite (8533)", "SD8 Gen5",
    "SD8 Gen3", "SD8 Gen2", "SD8 Gen1", "SD8+ Gen1", "SD8s Gen3",
    "SD7+ Gen2", "SD780G", "D9500", "D9400 (10667)", "D9400 (8533)",
    "D9300+", "D9200+", "D9000", "D8400 MAX", "D8300 Ultra", "D8200",
    "D8100", "D1200", "K9020", "Tensor G4", "Tensor G3", "E2400",
    "XRing O1",
]

###############################################################################
# Axis scaling defaults and helpers
###############################################################################

# Current SNL axis ranges and fallback geometry.  Runtime refresh updates
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


def discover_gpu_names(
    layer_url: str = GPU_PAGE_URL,
    *,
    fallback_pages: Optional[List[str]] = None,
) -> List[str]:
    """Return GPU names published in the current SoCPK curve manifest."""
    del fallback_pages  # Retained for API compatibility with older callers.
    try:
        return list(_gpu_layer_urls(layer_url))
    except (requests.RequestException, ValueError):
        return []


def _gpu_layer_urls(page_url: str = GPU_PAGE_URL) -> Dict[str, str]:
    """Return current curve names mapped to absolute SVG URLs."""
    manifest = fetch_curve_manifest(page_url)
    config = manifest.get("gpuSnl")
    if not isinstance(config, dict):
        raise ValueError("SoCPK manifest contains no gpuSnl configuration.")

    result = {}
    for layer in config.get("layers", []):
        src = layer.get("src") if isinstance(layer, dict) else None
        if not isinstance(src, str):
            continue
        filename = unquote(urlparse(src).path.rsplit("/", 1)[-1])
        if not filename.startswith("3dmark_snl_") or not filename.endswith(".svg"):
            continue
        name = filename[len("3dmark_snl_"):-len(".svg")]
        result[name] = urljoin(page_url, src)
    return result


def _gpu_curve_url(gpu_name: str) -> str:
    try:
        layers = _gpu_layer_urls()
        by_casefold = {name.casefold(): url for name, url in layers.items()}
        manifest_url = by_casefold.get(gpu_name.casefold())
        if manifest_url:
            return manifest_url
    except (requests.RequestException, ValueError):
        pass
    return f"{GPU_LAYER_BASE_URL}3dmark_snl_{quote(gpu_name, safe='')}.svg"


def parse_gpu_curve(
    gpu_name: str,
    base_url: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Download and parse a single GPU efficiency curve.

    Parameters
    ----------
    gpu_name : str
        Human‑readable name of the GPU/SoC.
    base_url : str, optional
        Base directory containing the GPU SVG layers.  The default
        points to SocPK’s canonical location.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame with columns ``['Board_Power_W', 'GPU_Score',
        'Efficiency']``.  Returns ``None`` if the SVG cannot be
        fetched or contains no usable points.

    Notes
    -----
    Continuous curves are exported as a ``<path>`` element inside
    ``<g id="line2d_1">``; scatter curves use multiple ``<use>``
    elements.  Both formats are handled automatically.
    """
    if base_url is not None:
        svg_url = (
            f"{base_url.rstrip('/')}/3dmark_snl_{quote(gpu_name, safe='')}.svg"
        )
    else:
        svg_url = _gpu_curve_url(gpu_name)
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
    """Scrape multiple GPU curves into a single DataFrame.

    Parameters
    ----------
    gpu_names : iterable of str, optional
        Names of GPUs/SoCs to scrape.  If ``None`` (default), the
        function auto-discovers available GPU curves from SocPK; if
        that yields no results it uses ``default_gpu_names`` (when
        provided) or an internal fallback list.
    default_gpu_names : list of str, optional
        Optional fallback list used when ``gpu_names`` is ``None`` and
        discovery fails.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ``['GPU', 'Board_Power_W',
        'GPU_Score', 'Efficiency']``.  May be empty if no curves were
        successfully scraped.
    """
    # Update axis scaling to adapt to chart changes
    try:
        refresh_axis_scaling()
    except Exception:
        pass
    if gpu_names is not None:
        names = list(gpu_names)
    else:
        names = discover_gpu_names()
        if not names and default_gpu_names is not None:
            names = default_gpu_names
        if not names:
            names = _FALLBACK_GPU_NAMES
    frames: List[pd.DataFrame] = []
    for name in names:
        try:
            df = parse_gpu_curve(name)
        except Exception as exc:
            print(f"Error parsing {name}: {exc}")
            continue
        if df is None or df.empty:
            continue
        df['GPU'] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['GPU', 'Board_Power_W', 'GPU_Score', 'Efficiency'])
    combined = pd.concat(frames, ignore_index=True)
    # Ensure efficiency column exists
    if 'Efficiency' not in combined.columns:
        combined['Efficiency'] = combined['GPU_Score'] / combined['Board_Power_W'].replace(0, pd.NA)
    return combined[['GPU', 'Board_Power_W', 'GPU_Score', 'Efficiency']]


def main() -> None:
    """Command‑line interface for scraping GPU curves.

    Use ``--gpus`` to specify one or more SoC names to scrape.  If
    omitted, current curve names are discovered.  Use ``--output``
    to write the combined results to a CSV file; otherwise the
    DataFrame is printed to stdout.
    """
    parser = argparse.ArgumentParser(description="Scrape Steel Nomad Light GPU curves from SocPK")
    parser.add_argument('--gpus', nargs='*', default=None,
                        help="Names of GPUs/SoCs to scrape (e.g. 'A19 Pro' 'SD8 Elite Gen5').  "
                             "If omitted, current curves are discovered.")
    parser.add_argument('--output', type=str, default="gpu_curves.csv",
                        help="Path to a CSV file where results will be written.  "
                             "If not provided, the DataFrame is printed.")
    args = parser.parse_args()
    df = scrape_gpu_curves(args.gpus)
    if df.empty:
        print("No GPU curves scraped.  Check network connectivity or update the processor list.")
        return
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Scraped {len(df)} rows for {df['GPU'].nunique()} GPUs → {args.output}")
    else:
        print(df)


if __name__ == '__main__':
    main()
