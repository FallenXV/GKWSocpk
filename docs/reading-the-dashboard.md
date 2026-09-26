# Reading the SoCPK Comparison Lab: How to Get Real Answers Out of Chip and Battery Data

English | [简体中文](reading-the-dashboard.zh-CN.md)

2026-09-26 · @Jun Zhi

[SoCPK](https://socpk.com) publishes some of the most useful efficiency data in mobile hardware: CPU and GPU power–performance curves measured at the board, and battery runtimes for dozens of current phones. The site is built to show one chart at a time, though, and most of the questions I care about need several chips on the same axes, or a number read off a curve at a specific wattage.

GKWSocpk is a small personal project that scrapes those results into CSV snapshots and puts them in a local comparison dashboard. This post is about using that dashboard well: where each number comes from, how to read each chart, what the Analysis lab computes, and where the data stops supporting a conclusion.

If you just want it running, the [README](../README.md) covers setup. The short version is to collect snapshots with the four parser scripts and then run `python socpk_web.py`. Everything below assumes the dashboard is open in your browser.

The examples use the snapshot I collected on 23 September 2026. Your numbers will change whenever SoCPK adds or revises results.

![The dashboard](images/dashboard.png)

## Where the numbers come from

It helps to know which source supplies each number before you trust it.

| Data | Source | What one row is |
| --- | --- | --- |
| CPU curves (GB6, GB7) | SoCPK multi-core efficiency pages | One published point on a chip's curve: board power, score |
| CPU curves (SPEC 2026 INT/FP) | SoCPK SPEC pages | One point for **one core** of a chip |
| Phone GPU | SoCPK, 3DMark Steel Nomad Light | One point on a GPU's curve |
| Laptop GPU | SoCPK, 3DMark Time Spy Graphics | One point on a GPU's curve |
| Battery | SoCPK battery test 5.0 | One phone: runtime and battery-imprint capacity |
| Phone processor, screen, refresh rate | GSMArena, looked up by `--auto-soc` | Metadata attached to a battery row |
| Metadata corrections | `battery_metadata.json`, reviewed by hand | Fixes to chipset, screen and refresh-rate fields |
| Measured usable capacity | Geekerwan's own measurements, hard-coded | Reference overlay for about 27 phones |

Keep three things in mind:

1. **The curve points are SoCPK's published values, and some are fitted upstream.** The dashboard never refits them. When it needs a value between two points, it draws a straight line between them, and it never extrapolates beyond the published range.
2. **Only the battery-imprint capacity and SoCPK's runtime feed the battery rankings.** The GSMArena lookup and the reviewed corrections only decide which processor group a phone belongs to and what its screen specification is. The Geekerwan measurements are drawn on the charts and never change a ranking.
3. **Every snapshot is kept.** The collectors never overwrite a file. A re-run writes something like `battery_results_20260923T014112457065Z_1.csv` next to the original, and the dashboard loads the newest file in each family. The note in the top-right corner of the header names the files that were actually loaded. If you collect new data while the dashboard is open, click **Reload data**.

## Finding your way around

### Tabs are datasets, and datasets don't mix

Each tab along the top is one CSV family:

| Tab | Benchmark | One profile is… |
| --- | --- | --- |
| GB6 MULTI | Geekbench 6 multi-core | a chip |
| GB7 MULTI | Geekbench 7 multi-core | a chip |
| SPEC26 INT | SPEC CPU 2026 integer | one core of a chip, e.g. *A19 Pro — Everest (V4)* |
| SPEC26 FP | SPEC CPU 2026 floating point | one core of a chip |
| GPU | 3DMark Steel Nomad Light (phones) | a GPU |
| LAPTOP GPU | 3DMark Time Spy Graphics | a laptop GPU |
| BATTERY | Battery test 5.0 | a phone, or a phone on a specific OS build |

Scores are only comparable within a tab. A GB6 score and a GB7 score are different units, and phone and laptop GPUs run different 3DMark tests. The dashboard keeps them apart on purpose, and even refuses to load a CSV that mixes two CPU benchmarks.

The tab is stored in the URL, so `http://localhost:8765/#Battery` or `#SPEC%20INT` opens straight on a dataset. You can bookmark those links. `[` and `]` step between tabs, and `/` jumps to the search box.

### Choosing what to compare

- **Compare** picks the chart. Every CPU and GPU tab has three charts, and BATTERY has five.
- The **profile list** on the left sets what is drawn. Click to toggle a profile, or shift-click to select a range.
- **Top 5** selects the five best profiles *by the current chart's ranking metric*, and it stays active. If you switch from *Performance curve* to *Efficiency curve* with Top 5 on, the selection changes, because the two charts rank by different things. To hold a fixed set of chips across charts, click the profiles yourself.
- **All shown** selects everything the search box and filters currently allow. **Clear** empties the selection.
- On the SPEC tabs, **Filter cores** narrows the list to a core group (Super, Large, Medium, Small) or to named cores. That's how you compare every chip's prime core against the others without the efficiency cores cluttering the chart.

### The parts of every chart

- **Stat cards.** *Profiles* reads "selected / available". *Data points* counts the published points behind the selection. *Current leader* names the best selected profile by the chart's measure.
- **Ranking panel.** The ranking to the right of each chart orders the selected profiles by the metric in its heading, with ↑ or ↓ showing which direction is better. Scroll over it when the list is long.
- **Better corner.** A small cyan triangle marks the corner where better results sit, and the line under the chart says what that corner means.
- **Hover.** Hovering over a point shows its exact values. Clicking a point pins its name to the note under the chart.
- **Efficiency reference.** This switch draws dashed guide lines where they make sense, and the note under the chart describes the line currently shown. Those guides are covered chart by chart below.

## Reading the CPU and GPU curves

Every point on these tabs has three numbers: **board power (W)**, **score**, and **efficiency (score ÷ W)**. The three charts show different pairs of them.

| Performance curve | Efficiency curve | Efficiency vs score |
| --- | --- | --- |
| ![](images/cpu-performance-curve.png) | ![](images/cpu-efficiency-curve.png) | ![](images/cpu-efficiency-vs-score.png) |
| Score against power | Score/W against power | Score/W against score |
| Ranked by peak score | Ranked by peak score/W | Ranked by peak score/W |

### Performance curve: what you get for each watt budget

This is the chart to start with. Read it vertically ("at 5 W, which chip scores most?") or horizontally ("to reach 9,000 points, which chip needs the fewest watts?"). Better is toward the top left.

The ranking uses **peak score**, which is usually the score at the highest published power. That tells you the ceiling. It says nothing about efficiency, and a chip can rank first here only because SoCPK ran it at 18 W.

With **Efficiency reference** on, the chart draws dashed rays from the origin. Every point on a ray has the same score per watt. The bold ray passes through the most efficient selected point, and on CPU tabs the fainter rays mark 50%, 25% and 12.5% of that efficiency. GPU tabs use tighter steps: 87.5%, 75%, 62.5% and 50%. A curve that bends down across the lower rays is losing efficiency as power rises, and the ray it crosses at a given power tells you how much it has lost.

### Efficiency curve: where each chip is happiest

This chart shows score per watt against power. Most chips peak at low power and fall steadily as they're pushed harder. The ranking uses **peak score/W**.

Be careful with that ranking. Peak efficiency usually occurs at the lowest published power, and chips don't all start at the same wattage. In the GB6 snapshot, A20 Pro's lowest point is at 0.52 W and reaches 3,212 points/W. Snapdragon 8 Elite Gen 5's curve starts at 1.37 W and peaks at 1,847 points/W. Part of that gap is real, and part of it only reflects that one chip was measured further down its curve. At equal power, the next chart and the Analysis lab give a fairer comparison.

### Efficiency vs score: comparing at equal performance

This chart plots efficiency against score. It answers "if both chips deliver the same performance, which uses less energy doing it?", which is usually the question behind "which chip is more efficient?". Better is toward the top right. When two curves cross, each chip is more efficient on its own side of the crossing score.

### Sparse curves are the most common trap

SoCPK publishes 120 points for Snapdragon 8 Elite Gen 5 on GB6. For A20 Pro it publishes three: 0.52 W, 2.18 W and 13.50 W. On the chart, the A20 Pro "curve" is two straight segments, and the second one spans more than 11 W.

A real power–performance curve bends downward, and a straight line between two points on such a curve sits below the curve. So when you read an Apple chip between its published points, the dashboard is probably **underestimating** it. It isn't wrong, since it shows only what was published, but a comparison that falls inside one of those long gaps is weak evidence. The Analysis lab reports the size of the gap for every estimate.

A few other limits apply on these tabs:

- **Board power** is SoCPK's measurement basis, and the charts assume it's consistent within a benchmark.
- **A short benchmark run** doesn't show sustained or thermally limited performance.
- **SPEC profiles are single cores.** Comparing *A19 Pro — Everest* with *Snapdragon 8 Elite Gen 5 — Oryon V3 L* compares prime cores and says nothing about the whole CPU.

## The Analysis lab for curves

The **Analysis lab** panel below the chart works on the current selection. Choose a mode from **Analysis** and a **Baseline** profile, then set the targets. Every table can be sorted by clicking a column heading.

### Equal power / performance

Set **Power target (W)** and **Target score**. The first table gives each chip's score at that power. The second gives the minimum power each chip needs to reach the target score, and how much power it saves against the baseline.

Here's the GB6 snapshot at 5 W:

| Profile | Score at 5 W | Evidence |
| --- | --: | --- |
| Snapdragon 8 Elite Gen 5 | 7,539 | Estimated between two close published points |
| Dimensity 9500 | 7,500 | Estimated between two close published points |
| A20 Pro | 7,010 | Estimated: 2.18 W / 5,015 → 13.50 W / 13,025; **gap 11.32 W** |
| A19 Pro | 6,355 | Estimated across an 11.0 W gap |

Read naively, A20 Pro trails the Android flagships by about 7% at 5 W. The **Evidence** column shows why that reading is weak: the A20 Pro figure is a straight line drawn across 11 W with no measurement in between. The same issue affects "power to reach 9,000 points" (A20 Pro 7.81 W, Snapdragon 8 Elite Gen 5 6.82 W).

The column exists so a gap like that can't hide. When it says *Published*, the number is a real point. When it says *Estimated* with a small gap, the number is reliable. When the gap is several watts wide, treat the number as a lower bound.

If a target falls outside a chip's published range, the table says *Outside published range* and doesn't guess. When a curve crosses a target score more than once, the table reports the lowest power at which that score is reached.

### Power trade-offs

For each chip, this table gives the power needed to reach 80%, 90%, 95% and 100% of **its own** peak score, plus the watts spent on the final 10% of performance. That last column is often the most revealing number in the lab, since a chip that spends 6 W on the final 10% is tuned well past its efficient range. Keep in mind that each percentage is relative to that chip's own peak, so the rows don't describe equal performance.

### Published-point frontier

This mode pools every published point from the selected chips and keeps only the non-dominated ones: points where no other selected point has at least the same score at no more power. The plot and table show which chip owns each part of the power range. The frontier uses only published points and never joins them with estimates.

### Baseline / generation comparison

Pick the older chip as **Baseline**, for example A19 Pro as the baseline for A20 Pro, or Snapdragon 8 Elite for Snapdragon 8 Elite Gen 5. For each other chip the lab reports:

- the **shared power range** where both curves have data. Nothing outside that range is compared.
- the **gain at your target power**, as a percentage over the baseline
- **equal-score crossings**, which are the powers where the two curves meet. For example, a new chip might win below 4 W and lose above it.
- a plot of percentage gain against power across the shared range

The same sparse-curve warning applies here. A long straight segment on either curve makes the percentage between its endpoints an estimate.

## Reading the battery charts

Each phone has two measured inputs, **runtime** from SoCPK's battery test 5.0 and **capacity (Wh)** from the battery imprint. Everything else is derived from those two:

- **Average power (W)** = capacity ÷ runtime in hours. Lower is better.
- **Efficiency (min/Wh)** = runtime in minutes ÷ capacity. Higher is better. The two metrics say the same thing in different units: 60 ÷ average power = min/Wh.

For example:

| Phone | Runtime | Capacity | Average power | Efficiency |
| --- | --: | --: | --: | --: |
| iPhone 18 Pro Max | 637 min (10.62 h) | 21.06 Wh | 1.98 W | 30.2 min/Wh |
| Honor WIN RT | 726 min (12.10 h) | 36.88 Wh | 3.05 W | 19.7 min/Wh |

The Honor lasts longer, and the iPhone uses about a third less power. Both statements are true. Most of the battery tab is about keeping those two questions apart.

These numbers describe **the whole phone**: screen, modem, memory, software and chip. They estimate average draw from the rated energy, and nothing measures the chip on its own. Chip-level conclusions need that caveat attached.

### Runtime vs capacity

![Runtime vs capacity](images/battery-runtime-vs-capacity.png)

This chart shows runtime in hours against battery size, ranked by runtime. It answers "which phone lasts longest?", and big batteries are rewarded, as they should be for that question.

The dashed reference line shows the runtime each battery size would give at the power draw of the most efficient selected phone. If the iPhone 18 Pro Max is selected, the line is `runtime = capacity ÷ 1.98 W`. At the Honor's 36.88 Wh that would be 18.6 h, well above its actual 12.1 h. The vertical distance from a phone to the line shows how much runtime it gives up to the most efficient phone. The line is a constant-efficiency scenario and not a prediction for any real phone.

### Energy efficiency

![Energy efficiency](images/battery-energy-efficiency.png)

This chart uses the same layout with runtime in minutes, but ranks by **minutes per Wh**. Here a small phone can beat a big one, because battery size is divided out. Use this ranking when the question is "which phone uses its battery best?"

### Average power draw

![Average power draw](images/battery-average-power-draw.png)

This chart plots average power against battery size, ranked from the lowest draw. Better is toward the bottom right: low draw and a big battery. The reference here is a **Pareto frontier**. Ringed phones are ones where no other selected phone has both a bigger battery and a lower draw, and the dashed line joins them. Any phone above and to the left of the line is beaten on both counts by a phone on it.

### Geekerwan measured capacity

For about 27 phones, hollow diamonds show Geekerwan's measured usable capacity, joined to the SoCPK point by a dotted line. Hover over a diamond to see measured against advertised mAh, the shortfall, and what power and min/Wh would be on the measured basis. Treat it as context: rankings, reference lines and processor averages all stay on the battery-imprint basis. Turn the diamonds off with **Geekerwan measured capacity**.

### Processor averages

When the snapshot was collected with `--auto-soc`, the left panel gains a **Processors** tab and two more charts appear:

| SoC Average Power Draw | SoC Average Efficiency |
| --- | --- |
| ![](images/battery-soc-average-power.png) | ![](images/battery-soc-average-efficiency.png) |

Each processor point is the plain mean of the phones that use it, with each phone counted once. Hover over a point to see how many phones are behind it. That count matters most: a processor average based on one phone is just that phone, not a chip-family result. Phones whose chipset couldn't be identified with confidence are left out of the averages, though they still appear on the device charts.

On the device charts, **Overlay average lines** draws one labelled line per selected processor, so you can see which phones sit above or below their chip's average.

## The Analysis lab for batteries

### Phone power grouped by processor

This mode gives each selected processor's **mean, median and range** of phone power, and how many brands contribute. It also has a **Remove one brand at a time** table, which recomputes each mean without each brand in turn. If a processor's average moves a lot when one brand is removed, the average mostly reflects that brand's phones.

The **Brand**, **screen size** and **listed refresh** filters apply only to this panel. They're useful for rough like-for-like comparisons, such as 6.3-inch phones only. A phone with missing specification data is excluded whenever the matching filter is active. "Listed refresh" is the panel's specification, not the refresh rate used during the test.

The last table lists each device with its provenance, including any correction from `battery_metadata.json` and its source.

### Runtime / generation comparison

This mode splits the runtime difference between each phone and the baseline into two factors:

```
runtime ratio = capacity ratio × inverse power ratio
```

With the iPhone 18 Pro Max as baseline, the Honor WIN RT comes out like this:

| | Value |
| --- | --: |
| Capacity ratio | 1.751 (a 75% bigger battery) |
| Inverse power ratio | 0.651 (it draws 1.54× the power) |
| Runtime ratio | 1.751 × 0.651 = **1.140** |

The Honor lasts 14% longer because its battery advantage outweighs its power disadvantage. The table makes that trade-off explicit instead of leaving it inside one runtime figure.

**Common battery (Wh)** answers the reverse question: what each phone would run for at the same capacity, assuming its average draw stays the same. At 20 Wh, the iPhone would run 10.1 h and the Honor 6.6 h. That is a scenario, not a measurement.

For a generation comparison, set the older model as baseline and match the product line and size yourself, for example iPhone 17 Pro Max against iPhone 18 Pro Max. Software and other hardware still differ between the two.

## Getting the data out

### Charts and tables

- **Export chart** saves the current chart, with its ranking panel, as a PNG or SVG.
- **Export analysis CSV** saves every table in the current Analysis lab mode. The file header records the dataset, the time the data was loaded, your settings, the selected profiles, the source files and the method, so the numbers can be traced later.

### The raw CSVs

The snapshots in `snapshots/` are plain UTF-8 CSVs. The curve files share one layout:

| Column | Meaning |
| --- | --- |
| `CPU` / `GPU` | Chip name |
| `Board_Power_W` | Board power in watts |
| `GB6_Multi_Score`, `GB7_Multi_Score`, `SPEC2026_INT_Score`, `SPEC2026_FP_Score`, `GPU_Score` | The score, named by benchmark |
| `Efficiency` | Score ÷ power |
| `Benchmark` | CPU files only; must match the score column |
| `Core`, `Core_Group`, `Core_Variant` | SPEC files only: core name, group (super/large/medium/small), and p/e variant |

The battery file has one row per phone:

| Column | Meaning |
| --- | --- |
| `brand`, `model`, `os` | Identity. Brands are in SoCPK's language (e.g. 荣耀) unless collected with `--brand-lang en` |
| `minutes`, `hours` | Runtime |
| `capacityWh` | Battery-imprint capacity |
| `avgPowerW`, `avgPowermW`, `minPerWh` | Derived as above |
| `delta_vs_best_*_pct` | Percentage difference from the best phone in the snapshot on power, min/Wh and runtime (0 is the best) |
| `chipset`, `soc`, `cpu`, `gpu`, `screen_size_in`, `refresh_hz`, … | GSMArena metadata, when `--auto-soc` was used |
| `soc_device_count`, `soc_avg_*` | Processor averages at collection time |
| `metadata_review`, `metadata_source`, `metadata_reviewed` | Manual review notes, when present |
| `url`, `spec_url` | The source video and the specification page |

One detail: the reviewed corrections in `battery_metadata.json` are applied when the dashboard loads data, and never written back to the snapshot. A processor average computed from the raw CSV can therefore differ slightly from the dashboard's.

### JSON and Python

While the dashboard runs, `GET /api/data` returns everything the page draws in one JSON document: every dataset, profile, curve point, battery profile and processor average, with corrections applied. `POST /api/reload` rebuilds it from disk.

```sh
curl -s http://localhost:8765/api/data \
  | jq '.datasets[] | select(.key=="Battery") | .socAverages[] | {soc, deviceCount, powerW}'
```

To get the same view in Python without starting a server:

```python
from pathlib import Path
from socpk_data import load_collections
from socpk_web import build_payload

collections, warnings = load_collections(Path("."))   # newest snapshot per family
gb6 = collections["CPU"]                               # pandas DataFrame, raw points
payload = build_payload(collections, warnings, Path("."))  # what the dashboard sees
```

The dataset keys are `CPU`, `CPU GB7`, `SPEC INT`, `SPEC FP`, `GPU`, `Laptop GPU` and `Battery`.

## A short checklist before you quote a number

1. **Is the comparison within one tab?** Scores don't carry across benchmarks.
2. **Is the value published or estimated?** Check the Evidence column, and be suspicious of any gap wider than a watt or two.
3. **Are you comparing at equal power, at equal performance, or at each chip's peak?** These are three different questions, and chips can rank differently on each.
4. **For batteries, is the question about runtime or about power?** A long-lasting phone and an efficient phone can be different phones.
5. **How many phones are behind a processor average?** One phone is an anecdote.
6. **Is it the phone or the chip?** Battery data measures the whole device.

The dashboard is designed to make each of those questions answerable in a couple of clicks.
