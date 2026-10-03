#!/usr/bin/env python3
"""Fetch SoCPK charging time/input-energy curves into a new CSV snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_charging import CHARGING_COLUMNS, CHARGING_PAGE_SLUG, charging_rows_from_page
from socpk_client import fetch_chart_page, new_snapshot


def scrape_charging_curves(names=None):
    return pd.DataFrame(
        charging_rows_from_page(fetch_chart_page(CHARGING_PAGE_SLUG), names),
        columns=CHARGING_COLUMNS,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", help="Published device names or series IDs; defaults to all.")
    parser.add_argument("--output", default="snapshots/charging_curves.csv",
                        help="CSV destination; existing files get a timestamped sibling.")
    args = parser.parse_args()
    try:
        frame = scrape_charging_curves(args.devices)
        with new_snapshot(args.output) as output:
            frame.to_csv(output, index=False)
    except (requests.RequestException, ValueError, OSError) as exc:
        raise SystemExit(f"Could not collect charging curves: {exc}") from exc
    print(f"Saved {len(frame):,} points for {frame['chargingSeries'].nunique()} devices → {output.name}")


if __name__ == "__main__":
    main()
