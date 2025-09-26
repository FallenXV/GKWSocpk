"""
gpu_curve_parser.py
===================

This module provides functions and a command‑line interface to scrape
SocPK’s GPU (Steel Nomad Light) efficiency curves.  Each GPU curve is
published as its own SVG under
``https://www.socpk.com/gpucurve/gb6/layer/gpu/``.  The parser reads
the base axes layer to derive the conversion from pixel positions to
physical board power (W) and GPU performance score, and can process
both continuous curves and scatter‑point curves.  A convenience
function ``scrape_gpu_curves`` aggregates data from multiple chips
into a single pandas DataFrame.

Key features
------------

* **Dynamic axis scaling.**  The script extracts the horizontal and
  vertical scaling from the base axes layer, so if the chart ranges
  change in future, the computed values will still be accurate.
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
import re
from urllib.parse import quote
from typing import Iterable, List, Optional, Dict

import pandas as pd  # type: ignore
import requests

__all__ = [
    "extract_axis_scaling",
    "refresh_axis_scaling",
    "parse_gpu_curve",
    "scrape_gpu_curves",
]

###############################################################################
# Axis scaling defaults and helpers
###############################################################################

# Default axis constants derived from a typical GPU efficiency chart on
# SocPK.  These values represent the pixel extents of the axes and the
# maximum board‑power and score ranges.  They will be updated at
# runtime by ``refresh_axis_scaling`` if the base layer can be
# downloaded.
X_START: float = 144.0
X_WIDTH: float = 892.8
POWER_RANGE: float = 22.0  # Default maximum board power in watts

Y_BASE: float = 576.7
Y_HEIGHT: float = 498.96
SCORE_RANGE: float = 4000.0  # Default maximum GPU score (SNL)


def extract_axis_scaling(
    base_svg_url: str = "https://www.socpk.com/gpucurve/layer/3DMark%20Steel%20Nomad%20Light_asis.svg",
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
    svg = resp.text
    # Horizontal axis path: M<x_start>,<y_base>h<width>
    h_match = re.search(r"M\s*([0-9.]+),([0-9.]+)\s*h\s*([0-9.]+)", svg)
    # Vertical axis path: M<x_start>,<y_base>V<y_top>
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
    # Extract numeric tick labels to infer ranges.  Values ≤100 are
    # considered board power; values >100 are considered GPU scores.
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
        'X_START': x_start,
        'X_WIDTH': x_width,
        'POWER_RANGE': power_range,
        'Y_BASE': y_base,
        'Y_HEIGHT': y_height,
        'SCORE_RANGE': score_range,
    }


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


def parse_gpu_curve(
    gpu_name: str,
    base_url: str = "https://www.socpk.com/gpucurve/layer/gpu/",
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
    encoded = quote(gpu_name, safe='')
    svg_url = f"{base_url}3dmark_snl_{encoded}.svg"
    try:
        resp = requests.get(svg_url, timeout=10)
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    svg = resp.text
    # Continuous curve detection
    # Use single quotes outside so that double quotes inside the pattern are
    # not prematurely terminated.  This regex matches a <path> element
    # inside a <g id="line2d_1"> group and captures the d attribute.
    path_match = re.search(r'<g id="line2d_1">\s*<path[^>]* d="([^"]+)"', svg)
    points: List[tuple[float, float]] = []
    if path_match:
        d_attr = path_match.group(1)
        nums = [float(s) for s in re.findall(r"[-+]?[0-9]*\.?[0-9]+", d_attr)]
        coords = list(zip(nums[0::2], nums[1::2]))
        for x, y in coords:
            points.append((_to_board_power(x), _to_score(y)))
    else:
        # Fallback to scatter points via <use>
        for m in re.finditer(r"<use[^>]+x=\"([0-9.]+)\"[^>]+y=\"([0-9.]+)\"", svg):
            x = float(m.group(1))
            y = float(m.group(2))
            points.append((_to_board_power(x), _to_score(y)))
    if not points:
        return None
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
        function uses ``default_gpu_names`` if provided; otherwise
        falls back to a built‑in list of common GPUs.
    default_gpu_names : list of str, optional
        Optional fallback list used when ``gpu_names`` is ``None``.

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
        names = default_gpu_names or [
            "A16", "A17 Pro", "A18", "A18 Pro", "A19", "A19 Pro",
            "SD8 Elite Gen5", "SD8 Elite (9600)", "SD8 Elite (8533)",
            "SD8 Gen3", "SD8 Gen2", "SD8 Gen1", "SD7+ Gen3", "SD7 Gen2",
            "D9500", "D9400 (10647)", "D9400 (8533)", "D9300+", "D9300 Ultra", "D9300",
            "D9200+", "D9200", "D8400 MAX", "D8300 Ultra", "D8300", "D8200", "D8100", "D8000",
            "D7200", "K9020", "K9010", "K9000S", "K9000", "K8000", "Tensor G5", "Tensor G4", "Tensor G3",
            "Tensor G2", "E2400", "E2400+", "XRIng 01"
        ]
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
    omitted, a default list of common GPUs is used.  Use ``--output``
    to write the combined results to a CSV file; otherwise the
    DataFrame is printed to stdout.
    """
    parser = argparse.ArgumentParser(description="Scrape Steel Nomad Light GPU curves from SocPK")
    parser.add_argument('--gpus', nargs='*', default=None,
                        help="Names of GPUs/SoCs to scrape (e.g. 'A19 Pro' 'SD8 Elite Gen5').  "
                             "If omitted, a default list is used.")
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