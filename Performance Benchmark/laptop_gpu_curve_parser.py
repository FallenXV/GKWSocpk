"""
Scrape SoCPK laptop GPU Time Spy graphics efficiency curves.

Laptop assets use an older Excel-exported SVG format and a separate
220 W / 27,500-point axis.  This module intentionally stays independent
from the phone GPU parser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote, unquote, urljoin, urlparse

import pandas as pd  # type: ignore
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from socpk_client import (  # noqa: E402
    extract_laptop_axis_geometry,
    extract_laptop_curve_coordinates,
    fetch_curve_manifest,
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


def _layer_urls() -> Dict[str, str]:
    manifest = fetch_curve_manifest(PAGE_URL)
    config = manifest.get("laptopGpu")
    if not isinstance(config, dict):
        raise ValueError("SoCPK manifest contains no laptopGpu configuration.")

    result = {}
    for layer in config.get("layers", []):
        src = layer.get("src") if isinstance(layer, dict) else None
        if not isinstance(src, str):
            continue
        filename = unquote(urlparse(src).path.rsplit("/", 1)[-1])
        if filename.endswith(".svg"):
            result[filename[:-len(".svg")]] = urljoin(PAGE_URL, src)
    return result


def discover_gpu_names() -> List[str]:
    """Return laptop GPU names from the current SoCPK manifest."""
    try:
        return list(_layer_urls())
    except (requests.RequestException, ValueError):
        return []


def _curve_url(gpu_name: str) -> str:
    try:
        layers = _layer_urls()
        by_casefold = {name.casefold(): url for name, url in layers.items()}
        manifest_url = by_casefold.get(gpu_name.casefold())
        if manifest_url:
            return manifest_url
    except (requests.RequestException, ValueError):
        pass
    return f"{LAYER_BASE_URL}{quote(gpu_name, safe='')}.svg"


def parse_gpu_curve(
    gpu_name: str,
    base_url: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Download and parse one laptop Time Spy graphics curve."""
    if base_url is not None:
        svg_url = f"{base_url.rstrip('/')}/{quote(gpu_name, safe='')}.svg"
    else:
        svg_url = _curve_url(gpu_name)
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
    """Scrape laptop curves into the standard GPU CSV schema."""
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

    frames: List[pd.DataFrame] = []
    for name in names:
        try:
            frame = parse_gpu_curve(name)
        except Exception as exc:
            print(f"Error parsing {name}: {exc}")
            continue
        if frame is None or frame.empty:
            continue
        frame["GPU"] = name
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=["GPU", "Board_Power_W", "GPU_Score", "Efficiency"]
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined[["GPU", "Board_Power_W", "GPU_Score", "Efficiency"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape laptop Time Spy GPU curves from SocPK"
    )
    parser.add_argument(
        "--gpus",
        nargs="*",
        default=None,
        help="Laptop GPUs to scrape (e.g. RTX5090 'RX 6600S'); discovers all if omitted.",
    )
    parser.add_argument(
        "--output",
        default="laptop_gpu_curves.csv",
        help="Output CSV path (default: laptop_gpu_curves.csv)",
    )
    args = parser.parse_args()
    frame = scrape_gpu_curves(args.gpus)
    if frame.empty:
        print("No laptop GPU curves scraped.")
        return
    frame.to_csv(args.output, index=False)
    print(
        f"Scraped {len(frame)} rows for "
        f"{frame['GPU'].nunique()} GPUs → {args.output}"
    )


if __name__ == "__main__":
    main()
