# GKWSocpk

Scrapes socpk.com for battery life and efficiency results, exports them as CSV
snapshots, and compares the snapshots in a local web dashboard.

Apple SoCs publish only a few points, so their curves may not represent the
real efficiency curve.

This is a personal project, not affiliated with socpk.com. Respect the site's
terms of service when using it.

## Requirements

Python 3.13. The web dashboard needs only a browser; the Tk fallback
dashboard additionally needs a Python build with Tk support.

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

The parsers read SoCPK's September 2026 chart API, including its binary point
format and both current tokenless and legacy token-protected data responses. No
browser or extra dependencies are required. Each poll fetches fresh data and
retries once if a legacy token expires or the chart changes between requests.
Legacy battery JS sources and explicitly supplied SVG base URLs also work.

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

Use `--auto-soc` to resolve phone models through GSMArena's manufacturer
catalogs and add `soc` to the snapshot. Lookups are deliberately opt-in,
polite, and cached in `.gsm_cache/`; `--auto-soc --spec-offline` reuses only
the cache. Alongside each resolved SoC, the CSV stores its device count and
precomputed mean capacity, average power, minutes/Wh, and runtime for that
snapshot. Manual `--spec` mappings are applied first and remain useful for
regional models that a catalog cannot match safely.

```powershell
python Battery\battery_parser.py --auto-soc
```

## Dashboard

```powershell
python socpk_web.py
```

This starts a local server, prints its address, and opens the dashboard in
your browser. Data is read once and sent to the page as JSON, so filtering,
hovering and redrawing happen in the browser with no round trip to Python.
Nothing is uploaded: the server binds to `127.0.0.1` and only serves files
from `web/` plus the loaded snapshots. Stop it with Ctrl-C.

`socpk_gui.py` remains as a Tk fallback with the same datasets and views. It
is kept for environments without a usable browser; on macOS its Tk canvas is
noticeably slower than the web dashboard. Both front ends load snapshots
through the shared `socpk_data.py`, so they always agree on what a CSV means.

The dashboard scans the project folder for recognized CPU, GPU, and battery
CSVs. Timestamped siblings are treated as one snapshot family and only the
newest is loaded automatically. Tabs switch datasets: GB6 MULTI, GB7 MULTI, SPEC26 INT,
SPEC26 FP, GPU, LAPTOP GPU, and BATTERY. A tab with no snapshot yet stays
dimmed and names the collector command to run. Rankings and comparisons stay
within the selected tab. Curve tabs chart an efficiency curve, a performance
curve, or efficiency vs score; battery tabs chart runtime vs capacity, energy
efficiency, or average power draw. Chart titles identify the benchmark source,
with the profile note beside the title.

Export the current chart as PNG or SVG. Suggested filenames include the
benchmark, view, selection mode, and applied core/search filters; manual
selections include a profile signature. The SVG is generated from the same
draw pass as the on-screen chart, so the two match; for PDF, print the
exported SVG from a browser. The Tk fallback still exports PDF directly.

Search by chip or core name, and multi-select profiles from the list —
click to toggle, shift-click for a range. Top 5 and All shown stay active as
filters, search, and chart views change. Top 5 ranks matching profiles using
the current chart metric. Picking profiles manually releases the automatic
selection mode; Clear keeps the selection empty until another choice is made.

SPEC tabs add a FILTER CORES panel below those buttons. The Super, Large,
Medium, and Small buttons toggle core groups, and the core-name menu supports
multiple checked names. Matches within each field are combined; a profile
must match both the groups and names selected. All clears the core filters.
Filters, selection modes, the chosen view, and the search text are kept
independently for each benchmark, so switching tabs and back restores context.

Curve legends show the chip and core name; the leader card and hover details
retain core types. Dotted lines join two or three sparse SPEC samples from the
same core and source without adding interpolated data points. Hovering near a
point shows a crosshair and its full details; scrolling over the ranking panel
pages through profiles beyond the ones on screen.

Battery charts show the pulled SoCPK points and overlay Geekerwan's static
measured usable-capacity results where a device matches. Hollow diamonds mark
the measured values; the measured Wh point scales the pulled Wh capacity by
`measured mAh / advertised mAh`. The overlays do not affect the CSV data or
the rankings. The Devices tab carries a **Geekerwan measured capacity** switch
that turns them off; it is on by default and appears only when a loaded device
matches the table. The Tk fallback always draws them.

When the battery CSV carries `soc`, the Battery tab's left panel gains
**Devices** and **Processors** tabs. Processors lists every resolved processor
with the number of device profiles behind its average, and the search box,
Top 5, All shown and Clear act on whichever list is showing; Top 5 ranks
processors by the metric the current chart uses. That selection drives both
places processor averages appear:

- The **Overlay average lines** switch above the list draws a labelled
  reference line per selected processor on the Energy efficiency and Average
  power draw views. It is off by default and unavailable on the other views,
  which either have no matching average or already plot the averages.
- The **SoC Average Power Draw** and **SoC Average Efficiency** views plot and
  rank the selected processors. Processors start on All shown, so these views
  still survey the whole loaded battery dataset until the list is narrowed.

Each list keeps its own search text, picks and latched mode, so narrowing
processors never disturbs the device comparison. Old battery CSVs without
`soc` remain loadable and simply show no Processors tab. The Tk fallback keeps
its original **SoC averages** checkbox beside the data-point count, which
overlays the averages of the processors behind the selected devices.

Keyboard: `[` and `]` move between tabs, `/` focuses the search box. The
address bar carries the current tab (`#Battery`), so a tab can be bookmarked.

Flags:

```powershell
python socpk_web.py --csv cpu_gb6_curves.csv --csv gpu_snl_curves.csv
python socpk_web.py --root . --dataset "SPEC INT"
python socpk_web.py --port 0 --no-browser
```

`--csv` (alias `--input`) loads only the named files and can repeat. `--root`
sets the folder to scan. `--dataset` picks the tab shown at startup, and takes
`CPU` (GB6), `CPU GB7`, `SPEC INT`, `SPEC FP`, `GPU`, `Laptop GPU`, or
`Battery`. `--host` and `--port` set the bind address (`--port 0` picks a free
port), `--browser` names a browser to open, and `--no-browser` leaves the
browser closed. `socpk_gui.py` accepts `--root`, `--csv` and `--dataset` with
the same meanings.

A loopback `--host` binds IPv4 and IPv6, because `localhost` resolves to `::1`
before `127.0.0.1` on macOS and a single-family bind leaves the browser
retrying a refused connection first. Any other `--host` binds only itself.

### Safari and HTTPS-Only

Safari's **HTTPS-Only** setting refuses plain HTTP everywhere, loopback
included, so `http://localhost:8765/` fails with `WebKitErrorDomain:305`
before it reaches the dashboard. Either serve over TLS:

```powershell
python socpk_web.py --https
```

which generates a certificate for `localhost`, `127.0.0.1` and `::1` once into
`.cache/socpk-web/` and reuses it for a year. Safari then reaches the
dashboard and asks you to accept the self-signed certificate once; installing
[mkcert](https://github.com/FiloSottile/mkcert) beforehand issues a
system-trusted certificate instead and removes even that prompt.

Otherwise, open another browser:

```powershell
python socpk_web.py --browser chrome
```

or turn HTTPS-Only off in Safari's settings. The Tk fallback needs none of
this.

### Tk fallback on macOS

`socpk_gui.py` detects Python installations that omit Tk (including the
default Homebrew configuration). If `uv` is available, it automatically
relaunches with a user-local Tk-enabled Python and the declared requirements;
that fallback is explicitly offline and uses uv's local package cache, so
starting the dashboard does not contact PyPI. No system Python changes are
needed. Without `uv`, install it or install the Homebrew `python-tk` formula
matching your Python version. The web dashboard needs none of this.

## Other scripts

`Performance Benchmark\curve_analysis.py --input <csv>` reads a CPU or GPU
snapshot and plots per-model power, score, and efficiency statistics; `--save`
writes the figures as PNGs instead of displaying them. It reads the current
schemas and keeps each SPEC core separate.

`Performance Benchmark\soc_curve_gui.py` is a launcher kept for older commands
and starts the Tk fallback dashboard, the same as `socpk_gui.py`.
