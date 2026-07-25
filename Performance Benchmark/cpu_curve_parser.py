"""
cpu_curve_parser.py
====================

This module provides both a library and a command‑line interface for
extracting Geekbench 6 multi‑core performance curves from the SocPK
website.  Each processor’s curve is published as a separate SVG
under ``https://www.socpk.com/assets/curves/cpu-gb6/cpu/``.  The
underlying functions convert pixel positions in the SVG into
physical board power (W) and Geekbench score by reading the base
axis layer ``Geekbench 6_asis.svg``.  A convenience function
``scrape_cpu_curves`` loops over a list of chip names and
aggregates the results into a single pandas DataFrame.

Key features
------------

* **Current axis geometry.**  Before parsing curves the script fetches
  the base axis layer and derives its horizontal and vertical bounds.
* **Support for continuous and scatter curves.**  Some chips
  (e.g. Qualcomm Snapdragon) publish a continuous line, while
  others (e.g. Apple A‑series) only provide a few scatter points.
  ``parse_cpu_curve`` handles both cases.
* **Efficiency calculation.**  For each point the script
  computes an ``Efficiency`` column defined as
  ``GB6_Multi_Score / Board_Power_W``.  Points with zero power are
  assigned ``NaN`` efficiency to avoid division by zero.
* **Command‑line interface.**  Running this file as a script allows
  you to scrape multiple CPUs and write the results to a CSV file.

Example
-------

Fetch a single chip into a DataFrame::

    from cpu_curve_parser import parse_cpu_curve, refresh_axis_scaling
    refresh_axis_scaling()  # update scaling from base SVG
    df = parse_cpu_curve("A19 Pro")
    print(df.head())

Scrape a list of chips and save to CSV::

    # Running from the command line
    python cpu_curve_parser.py --cpus "A19" "A19 Pro" --output apple_soc_points.csv

The resulting CSV will contain columns ``CPU``, ``Board_Power_W``,
``GB6_Multi_Score`` and ``Efficiency``.  You can further process
this file with the companion ``curve_analysis`` script.
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
    "discover_cpu_names",
    "parse_cpu_curve",
    "scrape_cpu_curves",
]

CPU_PAGE_URL = "https://www.socpk.com/cpucurve/gb6/"
CPU_AXIS_URL = (
    "https://www.socpk.com/assets/curves/cpu-gb6/Geekbench%206_asis.svg"
)
CPU_LAYER_BASE_URL = "https://www.socpk.com/assets/curves/cpu-gb6/cpu/"

_FALLBACK_CPU_NAMES: List[str] = [
    "A16", "A17 Pro", "A18", "A18 Pro", "A19", "A19 Pro",
    "SD8 Elite Gen5", "SD8 Elite (9600)", "SD8 Elite (8533)",
    "SD8 Gen5", "SD8 Gen3", "SD8 Gen2", "SD8+ Gen1", "SD8s Gen3",
    "SD7+ Gen3", "D9500", "D9400 (10667)", "D9400 (8533)",
    "D9300+", "D9200+", "D8400 MAX", "D8300 Ultra", "K9020",
    "K9010", "K9000S", "K9000SL", "K8000", "Tensor G4",
    "Tensor G3", "E2400", "XRing O1",
]

###############################################################################
# Axis scaling constants and helpers
###############################################################################

# Current GB6 axis ranges.  Runtime refresh updates the SVG plot geometry;
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


def _cpu_layer_urls() -> Dict[str, str]:
    """Return current curve names mapped to absolute SVG URLs."""
    manifest = fetch_curve_manifest(CPU_PAGE_URL)
    config = manifest.get("cpuGb6")
    if not isinstance(config, dict):
        raise ValueError("SoCPK manifest contains no cpuGb6 configuration.")

    result = {}
    for layer in config.get("layers", []):
        src = layer.get("src") if isinstance(layer, dict) else None
        if not isinstance(src, str):
            continue
        filename = unquote(urlparse(src).path.rsplit("/", 1)[-1])
        if not filename.startswith("CPU_gb6_") or not filename.endswith(".svg"):
            continue
        name = filename[len("CPU_gb6_"):-len(".svg")]
        result[name] = urljoin(CPU_PAGE_URL, src)
    return result


def discover_cpu_names() -> List[str]:
    """Return CPU names published in the current SoCPK curve manifest."""
    try:
        return list(_cpu_layer_urls())
    except (requests.RequestException, ValueError):
        return []


def _cpu_curve_url(cpu_name: str) -> str:
    try:
        layers = _cpu_layer_urls()
        by_casefold = {name.casefold(): url for name, url in layers.items()}
        manifest_url = by_casefold.get(cpu_name.casefold())
        if manifest_url:
            return manifest_url
    except (requests.RequestException, ValueError):
        pass
    return f"{CPU_LAYER_BASE_URL}CPU_gb6_{quote(cpu_name, safe='')}.svg"


def parse_cpu_curve(
    cpu_name: str,
    base_url: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Download and parse a single CPU efficiency curve.

    Parameters
    ----------
    cpu_name : str
        Human‑readable name of the CPU (e.g. "SD8 Elite Gen5").
    base_url : str, optional
        Base URL where CPU SVG files are stored.  Override this if
        using a mirror.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with columns ``['Board_Power_W', 'GB6_Multi_Score',
        'Efficiency']``.  Returns ``None`` if the SVG cannot be
        downloaded or contains no data.

    Notes
    -----
    The function first attempts to extract a continuous curve from a
    ``<path>`` element inside a ``<g id="line2d_1">`` group.  If no
    such element exists, it falls back to scatter points defined by
    ``<use x="..." y="...">``.  Only numeric coordinates are used.
    """
    if base_url is not None:
        svg_url = f"{base_url.rstrip('/')}/CPU_gb6_{quote(cpu_name, safe='')}.svg"
    else:
        svg_url = _cpu_curve_url(cpu_name)
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
    """Scrape multiple CPU curves into a single DataFrame.

    Parameters
    ----------
    cpu_names : iterable of str, optional
        Names of CPUs to scrape.  If ``None``, names are discovered from
        the current SoCPK manifest, then optional/default fallbacks are used.
    default_cpu_names : list of str, optional
        Fallback list used only if current manifest discovery fails.

    Returns
    -------
    pandas.DataFrame
        Combined results with columns ``['CPU', 'Board_Power_W',
        'GB6_Multi_Score', 'Efficiency']``.  The returned DataFrame
        may be empty if no curves could be scraped.
    """
    # Update axis scaling first.  If this fails, we use existing
    # defaults; no exception is raised.
    try:
        refresh_axis_scaling()
    except Exception:
        pass
    if cpu_names is not None:
        names = list(cpu_names)
    else:
        names = discover_cpu_names()
        if not names and default_cpu_names is not None:
            names = default_cpu_names
        if not names:
            names = _FALLBACK_CPU_NAMES
    frames: List[pd.DataFrame] = []
    for name in names:
        try:
            df = parse_cpu_curve(name)
        except Exception as exc:
            print(f"Error parsing {name}: {exc}")
            continue
        if df is None or df.empty:
            continue
        df['CPU'] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['CPU', 'Board_Power_W', 'GB6_Multi_Score', 'Efficiency'])
    combined = pd.concat(frames, ignore_index=True)
    # Ensure efficiency column exists (in case parse_cpu_curve didn't compute it)
    if 'Efficiency' not in combined.columns:
        combined['Efficiency'] = combined['GB6_Multi_Score'] / combined['Board_Power_W'].replace(0, pd.NA)
    return combined[['CPU', 'Board_Power_W', 'GB6_Multi_Score', 'Efficiency']]


def main() -> None:
    """Entry point for the command‑line interface.

    Use ``--cpus`` to specify one or more CPU names to scrape.  If
    omitted, current curve names are discovered.  Use ``--output`` to write
    the aggregated results to a CSV file; otherwise the DataFrame is
    printed to stdout.
    """
    parser = argparse.ArgumentParser(description="Scrape Geekbench 6 CPU curves from SocPK")
    parser.add_argument('--cpus', nargs='*', default=None,
                        help="Names of CPUs to scrape (e.g. 'A19 Pro' 'SD8 Elite Gen5').  "
                             "If omitted, current curves are discovered.")
    parser.add_argument('--output', type=str, default="cpu_curves.csv",
                        help="Path to a CSV file where results will be written.  "
                             "If not provided, the DataFrame is printed.")
    args = parser.parse_args()
    df = scrape_cpu_curves(args.cpus)
    if df.empty:
        print("No CPU curves scraped.  Check network connectivity or update the processor list.")
        return
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Scraped {len(df)} rows for {df['CPU'].nunique()} CPUs → {args.output}")
    else:
        print(df)


if __name__ == '__main__':
    main()
