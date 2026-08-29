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
python socpk_gui.py
```

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
