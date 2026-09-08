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
python "Performance Benchmark\cpu_curve_parser.py" --benchmark all
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

- `cpu_gb6_curves.csv`: Geekbench 6 multi-core.
- `cpu_gb7_curves.csv`: Geekbench 7 multi-core.
- `cpu_spec2026_int_curves.csv`: SPEC CPU 2026 single-core integer scores.
- `cpu_spec2026_fp_curves.csv`: SPEC CPU 2026 single-core floating-point scores.
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

CPU benchmarks have separate score columns and separate dashboard tabs. The
existing `GB6_Multi_Score` schema remains supported; GB7 uses `GB7_Multi_Score`,
and SPEC uses `SPEC2026_INT_Score` or `SPEC2026_FP_Score`. SPEC snapshots also
include `Core`, `Core_Group`, and `Core_Variant`, so different cores of the same
chip remain separate profiles. Scores and score/W retain their fractional
precision; the dashboard shows SPEC values to three decimal places.

`--benchmark all` polls both CPU source pages once and writes four separate
snapshots. Without `--benchmark`, the CPU parser still defaults to GB6 for
existing commands. To collect a specific benchmark or core group:

```powershell
python "Performance Benchmark\cpu_curve_parser.py" --benchmark GB7
python "Performance Benchmark\cpu_curve_parser.py" --benchmark SPEC2026_INT --cpus "A19 Pro"
python "Performance Benchmark\cpu_curve_parser.py" --benchmark SPEC2026_FP --core-group medium
```

Use `--output-dir` with `--benchmark all`, or `--output` for a single benchmark.
Core-group choices are the site's `super`, `large`, `medium`, and `small`.
Selecting a chip for SPEC includes all its published cores unless filtered.
The standalone `curve_analysis.py` also recognizes the new schemas and
keeps each SPEC core separate.

The comparison dashboard automatically finds and merges recognized CPU, GPU,
and battery CSVs under the project folder. Use the tabs to switch datasets,
search and multi-select profiles, change chart modes, or export the current
comparison as PNG, SVG, or PDF.

Choose **GB6 MULTI**, **GB7 MULTI**, **SPEC26 INT**, or **SPEC26 FP** to switch
CPU benchmarks. Search by chip or core name, and use the horizontal profile
scrollbar to inspect long core labels. Rankings and comparisons stay within
the selected benchmark. Existing CSVs continue to load alongside snapshots.

Battery charts retain the pulled SoCPK points and overlay Geekerwan's static
measured usable-capacity results where a device matches. Hollow diamonds mark
the measured values; the measured Wh point is derived by scaling the pulled Wh
capacity by `measured mAh / advertised mAh`. These overlays do not change the
pulled CSV data or the dashboard rankings.

To compare only specific files:

```powershell
python socpk_gui.py --csv cpu_curves.csv --csv gpu_curves.csv
```
