# GKWSocpk

Scrapes socpk.com for battery life and efficiency results.

IMPORTANT: Apple SoCs only have a few points and may not be representative of actual efficiency curve.

Note: This is a personal project and is not affiliated with socpk.com. Please respect their terms of service when using this script.

## Requirements

Python 3.13 (with Tk support, for the GUI).

### Setup

Create and activate a virtual environment, then install dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On bash/WSL, activate with `source .venv/bin/activate` instead.

`requirements.txt` holds the direct dependencies; `requirements.lock.txt` is a
`pip freeze` of a known-good resolution — install from it if you need to
reproduce the exact environment.

### Running

With the venv active:

```powershell
python Battery\battery_parser.py
python "Performance Benchmark\cpu_curve_parser.py"
python "Performance Benchmark\gpu_curve_parser.py"
python "Performance Benchmark\laptop_gpu_curve_parser.py"
python socpk_gui.py
```

The parsers use SoCPK's September 2026 chart API, including its public chart
tokens and binary point format. No browser or additional dependencies are
needed. Each poll fetches fresh data and retries once if a token expires or
the chart changes between requests. Legacy battery JS and explicitly supplied
SVG base URLs remain supported.

New exports go under `snapshots/` relative to your working directory:

- `cpu_gb6_curves.csv`: Geekbench **6** multi-core only. The new CPU page also
  includes GB7; those scores are excluded from the existing GB6 schema.
- `gpu_snl_curves.csv`: phone GPU Steel Nomad Light.
- `laptop_gpu_curves.csv`: laptop GPU Time Spy.
- `battery_results.csv`: battery test 5.0 runtime, rated Wh and efficiency.

**Existing files are never overwritten**, including custom `--output`,
`--csv`, and `--json` paths. If a path exists, the export creates a sibling
with a UTC timestamp. Sparse results therefore cannot replace historical
data. Empty curve results produce no file; failed writes remove the incomplete
snapshot. The scripts print the actual saved path.

Curve exports retain the API's published points, which may include fitted
curves; they do not interpolate extra points. Names use the site's English
name when available. Previous CLI abbreviations such as `SD8 Gen3` remain
accepted when they identify a single current series.

The comparison dashboard automatically finds and merges recognized CPU, GPU,
and battery CSVs under the project folder. Use the tabs to switch datasets,
search and multi-select profiles, change chart modes, or export the current
comparison as PNG, SVG, or PDF.

Battery charts retain the pulled SoCPK points and overlay Geekerwan's static
measured usable-capacity results where a device matches. Hollow diamonds mark
the measured values; the measured Wh point is derived by scaling the pulled Wh
capacity by `measured mAh / advertised mAh`. These overlays do not change the
pulled CSV data or the dashboard rankings.

To compare only specific files:

```powershell
python socpk_gui.py --csv cpu_curves.csv --csv gpu_curves.csv
```
