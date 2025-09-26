"""
cpu_curve_parser.py
====================

This module provides both a library and a command‑line interface for
extracting Geekbench 6 multi‑core performance curves from the SocPK
website.  Each processor’s curve is published as a separate SVG
under ``https://www.socpk.com/cpucurve/gb6/layer/cpu/``.  The
underlying functions convert pixel positions in the SVG into
physical board power (W) and Geekbench score by reading the base
axis layer ``Geekbench 6_asis.svg``.  A convenience function
``scrape_cpu_curves`` loops over a list of chip names and
aggregates the results into a single pandas DataFrame.

Key features
------------

* **Dynamic axis scaling.**  Before parsing any curves the script
  fetches the base axis layer and derives the horizontal and
  vertical scaling.  This ensures that if SocPK increases the
  maximum board‑power or score displayed, the computed values will
  still be correct.
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
import re
from urllib.parse import quote
from typing import Iterable, List, Optional, Dict

import pandas as pd  # type: ignore
import requests

__all__ = [
    "extract_axis_scaling",
    "refresh_axis_scaling",
    "parse_cpu_curve",
    "scrape_cpu_curves",
]

###############################################################################
# Axis scaling constants and helpers
###############################################################################

# Default axis mapping values.  These correspond to the GB6 base layer
# as of September 2025.  If the site increases the chart extents, these
# values will be updated at runtime by ``refresh_axis_scaling``.
X_START: float = 144.0
X_WIDTH: float = 892.8
POWER_RANGE: float = 22.0

Y_BASE: float = 576.7
Y_HEIGHT: float = 498.96
SCORE_RANGE: float = 15000.0


def extract_axis_scaling(
    base_svg_url: str = "https://www.socpk.com/cpucurve/gb6/layer/Geekbench%206_asis.svg",
) -> Optional[Dict[str, float]]:
    """Derive dynamic axis scaling from the base GB6 axes layer.

    The base SVG defines the grid and axes for all curves.  The
    horizontal axis is encoded as a path ``M<x_start>,<y_base>h<width>``
    and the vertical axis as ``M<x_start>,<y_base>V<y_top>``.  Text
    labels on the axes indicate the maximum board power and score
    displayed.  This function fetches the SVG, extracts these values
    and returns a mapping of constants.  If the request fails or
    parsing is unsuccessful, ``None`` is returned.

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
    svg = resp.text
    # Match horizontal axis: M<x_start>,<y_base>h<width>
    h_match = re.search(r"M\s*([0-9.]+),([0-9.]+)\s*h\s*([0-9.]+)", svg)
    # Match vertical axis: M<x_start>,<y_base>V<y_top>
    v_match = re.search(r"M\s*([0-9.]+),([0-9.]+)\s*V\s*([0-9.]+)", svg)
    if not h_match or not v_match:
        return None
    try:
        x_start = float(h_match.group(1))
        y_base = float(h_match.group(2))
        x_width = float(h_match.group(3))
        y_top = float(v_match.group(3))
    except ValueError:
        return None
    y_height = abs(y_base - y_top)
    # Extract numeric tick labels to infer ranges.  We interpret
    # values <=100 as board‑power and values >100 as score.  Suffixes
    # like 'k' or Chinese '万' are handled.
    labels = re.findall(r"<text[^>]*>([^<]+)</text>", svg)
    numbers: List[float] = []
    for lab in labels:
        s = lab.strip().replace(',', '')
        s_lower = s.lower()
        mult = 1.0
        if 'k' in s_lower:
            mult = 1000.0
            s_lower = s_lower.replace('k', '')
        if '万' in s_lower:
            mult = 10000.0
            s_lower = s_lower.replace('万', '')
        m = re.match(r"-?[0-9.]+", s_lower)
        if not m:
            continue
        try:
            val = float(m.group(0))
        except ValueError:
            continue
        numbers.append(val * mult)
    board = [n for n in numbers if n <= 100.0]
    score = [n for n in numbers if n > 100.0]
    power_range = max(board) if board else POWER_RANGE
    score_range = max(score) if score else SCORE_RANGE
    return {
        "X_START": x_start,
        "X_WIDTH": x_width,
        "POWER_RANGE": power_range,
        "Y_BASE": y_base,
        "Y_HEIGHT": y_height,
        "SCORE_RANGE": score_range,
    }


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


def parse_cpu_curve(
    cpu_name: str,
    base_url: str = "https://www.socpk.com/cpucurve/gb6/layer/cpu/",
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
    # Construct the filename and URL; quote spaces and special chars.
    encoded = quote(cpu_name, safe='')
    svg_url = f"{base_url}CPU_gb6_{encoded}.svg"
    try:
        resp = requests.get(svg_url, timeout=10)
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    svg = resp.text
    # Try continuous curve
    # Use single quotes outside so that double quotes inside the pattern are
    # not prematurely terminated.  This regex matches a <path> element inside
    # a <g id="line2d_1"> group and captures the entire d attribute.
    path_match = re.search(r'<g id="line2d_1">\s*<path[^>]* d="([^"]+)"', svg)
    points: List[tuple[float, float]] = []
    if path_match:
        d_attr = path_match.group(1)
        # Extract all floats from the path string
        nums = [float(s) for s in re.findall(r"[-+]?[0-9]*\.?[0-9]+", d_attr)]
        coords = list(zip(nums[0::2], nums[1::2]))
        for x, y in coords:
            points.append((_to_board_power(x), _to_score(y)))
    else:
        # Fallback: scatter points defined in <use> tags
        for m in re.finditer(r"<use[^>]+x=\"([0-9.]+)\"[^>]+y=\"([0-9.]+)\"", svg):
            x = float(m.group(1))
            y = float(m.group(2))
            points.append((_to_board_power(x), _to_score(y)))
    if not points:
        return None
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
        Names of CPUs to scrape.  If ``None``, the function uses
        ``default_cpu_names`` if provided; otherwise it falls back to
        a built‑in list of Apple and Android SoCs.
    default_cpu_names : list of str, optional
        Fallback list of CPU names to use when ``cpu_names`` is
        ``None``.  If omitted, a hard‑coded list of processors is
        used.

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
        # Fallback list of common CPUs on the GB6 page
        names = default_cpu_names or [
            "A16", "A17 Pro", "A18", "A18 Pro", "A19", "A19 Pro",
            "SD8 Elite Gen5", "SD8 Elite (9600)", "SD8 Elite (8533)",
            "SD8 Gen3", "SD8 Gen2", "SD8 Gen1", "SD8+ Gen1", "SD7+ Gen3",
            "SD7+ Gen2", "SD7 Gen3", "SD7 Gen2",
            "D9500", "D9400 (10667)", "D9400 (8533)", "D9300+", "D9300 Ultra",
            "D9300", "D9200+", "D9200", "D8400 MAX", "D8300 Ultra", "D8300",
            "D8200", "D8100", "K9020", "K9010", "K9000S", "K9000SL", "K9000",
            "K8000", "Tensor G4", "Tensor G3", "Tensor G2", "E2400", "XRing.01"
        ]
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
    omitted, a default list will be used.  Use ``--output`` to write
    the aggregated results to a CSV file; otherwise the DataFrame is
    printed to stdout.
    """
    parser = argparse.ArgumentParser(description="Scrape Geekbench 6 CPU curves from SocPK")
    parser.add_argument('--cpus', nargs='*', default=None,
                        help="Names of CPUs to scrape (e.g. 'A19 Pro' 'SD8 Elite Gen5').  "
                             "If omitted, a default list is used.")
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