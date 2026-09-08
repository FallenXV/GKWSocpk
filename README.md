# GKWSocpk

Scrapes socpk.com for battery life and efficiency results, exports them as CSV
snapshots, and compares the snapshots in a Tk dashboard.

Apple SoCs publish only a few points, so their curves may not represent the
real efficiency curve.

This is a personal project, not affiliated with socpk.com. Respect the site's
terms of service when using it.

## Requirements

Python 3.13, with Tk support for the GUI.

Create and activate a virtual environment, then install dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On bash/WSL, activate with `source .venv/bin/activate`.

`requirements.txt` lists the direct dependencies. `requirements.lock.txt` is a
`pip freeze` of a known-good resolution; install from it to reproduce that
exact environment.

## Collecting data

With the venv active:

```powershell
python Battery\battery_parser.py
python "Performance Benchmark\cpu_curve_parser.py" --benchmark all
python "Performance Benchmark\gpu_curve_parser.py"
python "Performance Benchmark\laptop_gpu_curve_parser.py"
```

The parsers read SoCPK's September 2026 chart API, including its public chart
tokens and binary point format. No browser or extra dependencies are required.
Each poll fetches fresh data and retries once if a token expires or the chart
changes between requests. Legacy battery JS sources and explicitly supplied
SVG base URLs also work.

Exports go to `snapshots/` relative to the working directory:

| File | Contents |
| --- | --- |
| `cpu_gb6_curves.csv` | Geekbench 6 multi-core |
| `cpu_gb7_curves.csv` | Geekbench 7 multi-core |
| `cpu_spec2026_int_curves.csv` | SPEC CPU 2026 single-core integer |
| `cpu_spec2026_fp_curves.csv` | SPEC CPU 2026 single-core floating-point |
| `gpu_snl_curves.csv` | Phone GPU, Steel Nomad Light |
| `laptop_gpu_curves.csv` | Laptop GPU, Time Spy |
| `battery_results.csv` | Battery test 5.0 runtime, rated Wh, efficiency |

Exports never overwrite an existing file, including paths given with
`--output`, `--csv`, or `--json`. If the path is taken, the export writes a
sibling with a UTC timestamp, so a sparse result cannot replace historical
data. An empty curve result writes no file, and a failed write deletes the
partial snapshot. Each script prints the path it wrote.

Curve exports keep the API's published points, which may include fitted
curves, and interpolate nothing. Names use the site's English name where one
exists. CLI abbreviations such as `SD8 Gen3` resolve when they identify a
single current series.

### CPU benchmarks

Each benchmark has its own score column and its own dashboard tab:
`GB6_Multi_Score`, `GB7_Multi_Score`, `SPEC2026_INT_Score`, and
`SPEC2026_FP_Score`. SPEC snapshots also carry `Core`, `Core_Group`, and
`Core_Variant`, which keep different cores of one chip as separate profiles.
Scores and score/W keep their fractional precision; the dashboard prints SPEC
values to three decimals.

`--benchmark all` polls both CPU source pages once and writes four snapshots.
Without `--benchmark`, the parser exports GB6. To collect one benchmark or one
core group:

```powershell
python "Performance Benchmark\cpu_curve_parser.py" --benchmark GB7
python "Performance Benchmark\cpu_curve_parser.py" --benchmark SPEC2026_INT --cpus "A19 Pro"
python "Performance Benchmark\cpu_curve_parser.py" --benchmark SPEC2026_FP --core-group medium
```

`--output-dir` applies to `--benchmark all`; `--output` applies to a single
benchmark. Core groups are the site's `super`, `large`, `medium`, and `small`.
Naming a chip for SPEC pulls all of its published cores unless `--core-group`
narrows them.

### GPU and battery parsers

`gpu_curve_parser.py` and `laptop_gpu_curve_parser.py` take `--gpus` to limit
the scrape and `--output` to set the snapshot path; both discover every
published GPU when `--gpus` is omitted.

`battery_parser.py` takes `--site` to pick the battery dataset (default 5.0),
`--url` to override source URLs, `--csv` and `--json` for snapshot paths, and
`--brand-lang source|en` for output brand names. GSMArena enrichment uses
`--spec 'Brand|Model=url'`, `--spec-map-json`, and `--spec-offline` for
cache-only runs. `--preview` prints an enriched table and `--correlate` prints
Pearson correlations against efficiency.

## Dashboard

```powershell
python socpk_gui.py
```

The dashboard scans the project folder for recognized CPU, GPU, and battery
CSVs and merges them. Tabs switch datasets: GB6 MULTI, GB7 MULTI, SPEC26 INT,
SPEC26 FP, GPU, LAPTOP GPU, and BATTERY. Rankings and comparisons stay within
the selected tab. Curve tabs chart an efficiency curve, a performance curve,
or efficiency vs score; battery tabs chart runtime vs capacity, energy
efficiency, or average power draw. Export the current chart as PNG, SVG, or
PDF.

Search by chip or core name, multi-select profiles from the list, and use the
horizontal scrollbar to read long core labels. Top 5 and All shown stay active
as filters, search, and chart views change. Top 5 ranks matching profiles using
the current chart metric. Picking profiles manually releases the automatic
selection mode; Clear keeps the selection empty until another choice is made.

SPEC tabs add a FILTER CORES panel below those buttons. The Super, Large,
Medium, and Small buttons toggle core groups, and the core-name menu supports
multiple checked names. Matches within each field are combined; a profile
must match both the groups and names selected. All clears the core filters.
Filters and selection modes are kept independently for each benchmark.

Curve legends show the chip and core name; the leader card and hover details
retain core types. Dotted lines join two or three sparse SPEC samples from the
same core and source without adding interpolated data points.

Battery charts show the pulled SoCPK points and overlay Geekerwan's static
measured usable-capacity results where a device matches. Hollow diamonds mark
the measured values; the measured Wh point scales the pulled Wh capacity by
`measured mAh / advertised mAh`. The overlays do not affect the CSV data or
the rankings.

Flags:

```powershell
python socpk_gui.py --csv cpu_gb6_curves.csv --csv gpu_snl_curves.csv
python socpk_gui.py --root . --dataset "SPEC INT"
```

`--csv` (alias `--input`) loads only the named files and can repeat. `--root`
sets the folder to scan. `--dataset` picks the tab shown at startup, and takes
`CPU` (GB6), `CPU GB7`, `SPEC INT`, `SPEC FP`, `GPU`, `Laptop GPU`, or
`Battery`.

## Other scripts

`Performance Benchmark\curve_analysis.py --input <csv>` reads a CPU or GPU
snapshot and plots per-model power, score, and efficiency statistics; `--save`
writes the figures as PNGs instead of displaying them. It reads the current
schemas and keeps each SPEC core separate.

`Performance Benchmark\soc_curve_gui.py` is a launcher kept for older commands
and starts the same dashboard as `socpk_gui.py`.
