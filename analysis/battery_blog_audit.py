"""Reproduce the blog's reviewed September snapshot without fetching new results.

Battery-imprint Wh and runtime are immutable inputs. Corrections affect only
processor assignments and display metadata. Two ambiguous processors stay unassigned.
Run from any directory with the project's Python environment.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Battery.battery_parser import cache_path_for, parse_gsmarena_specs

SOURCE = ROOT / "snapshots/battery_results_20260919T075628417409Z_1.csv"
BRANDS = {"苹果": "Apple", "三星": "Samsung", "谷歌": "Google", "华为": "Huawei",
          "一加": "OnePlus", "红魔": "RedMagic", "努比亚": "Nubia", "小米": "Xiaomi",
          "红米": "Redmi", "荣耀": "Honor"}

# The collector, dashboard and audit share the same reviewed metadata ledger.
FIXES = {(r['brand'], r['model']): (r['fields'], r['reason'], r['source'])
         for r in json.loads((ROOT / 'battery_metadata.json').read_text())}


def cache_file(brand, model):
    key = "phone_" + (brand + "|" + model).lower().replace(" ", "_")
    current = ROOT / ".gsm_cache" / Path(cache_path_for(key)).name
    if current.is_file():
        return current
    # Retain the historical cache layout as a read-only fallback so the audit
    # can still reproduce archived snapshots created before hashed filenames.
    legacy = re.sub(r"[^a-zA-Z0-9._-]+", "_", key).strip("._") + ".html"
    return ROOT / ".gsm_cache" / legacy


def reduction(value, reference):
    return 100 * (1 - value / reference)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE)
    raw[["brand", "model"]] = raw[["brand", "model"]].apply(lambda x: x.str.strip())
    assert len(raw) == 80 and not raw.duplicated(["brand", "model"]).any()
    assert raw[["minutes", "capacityWh"]].gt(0).all().all()
    assert np.allclose(raw.hours, raw.minutes / 60, rtol=0, atol=1e-12)
    assert np.allclose(raw.avgPowerW, raw.capacityWh * 60 / raw.minutes, rtol=0, atol=1e-12)
    assert np.allclose(raw.minPerWh, raw.minutes / raw.capacityWh, rtol=0, atol=1e-12)
    for _, group in raw.groupby("soc"):
        assert np.allclose(group.soc_device_count, len(group))
        for column, metric in [("soc_avg_power_w", "avgPowerW"),
                               ("soc_avg_capacity_wh", "capacityWh"),
                               ("soc_avg_min_per_wh", "minPerWh"),
                               ("soc_avg_runtime_hours", "hours")]:
            assert np.allclose(group[column], group[metric].mean())

    frame = raw.copy()
    records = []
    for index, row in raw.iterrows():
        path = cache_file(row.brand, row.model)
        html = path.read_text()
        soup = BeautifulSoup(html, "html.parser")
        title = soup.select_one('[data-spec="modelname"]').get_text(" ", strip=True)
        specs = parse_gsmarena_specs(html)
        # All imported main-screen/chipset fields must be traceable to the cache,
        # even where the cache itself belongs to the wrong phone.
        for field in ["chipset", "screen_size_in", "resolution_px_w", "resolution_px_h", "refresh_hz"]:
            a, b = row[field], specs.get(field)
            # The original snapshot predates the PWM parser fix. Accept exactly
            # its three documented extraction errors, never arbitrary mismatches.
            legacy_pwm = {('红米', 'K90 Pro Max'): 560, ('小米', '17 Pro'): 160,
                          ('小米', '17 Pro Max'): 160}
            documented_pwm = (field == 'refresh_hz'
                              and legacy_pwm.get((row.brand, row.model)) == a
                              and b == FIXES.get((row.brand, row.model), ({},))[0].get(field))
            assert (pd.isna(a) and b is None) or a == b or documented_pwm, (row.model, field, a, b)
        fields, note, evidence = FIXES.get((row.brand, row.model),
            ({}, "Cache identity and imported fields checked; retained", str(path.relative_to(ROOT))))
        changes = {}
        for field, value in fields.items():
            before = None if pd.isna(row[field]) else row[field]
            if before != value:
                changes[field] = {"before": before, "after": value}
            frame.at[index, field] = value
        records.append({"csv_line": int(index + 2), "brand": BRANDS.get(row.brand, row.brand),
                        "model": row.model, "cached_title": title,
                        "cache": str(path.relative_to(ROOT)), "note": note,
                        "evidence": evidence, "changes": changes})

    for field in ["capacityWh", "minutes", "hours", "avgPowerW", "minPerWh"]:
        assert frame[field].equals(raw[field]), field
    frame["mp"] = frame.resolution_px_w * frame.resolution_px_h / 1e6
    assigned = frame[frame.soc.notna()]
    groups = assigned.groupby("soc").agg(n=("model", "size"), power=("avgPowerW", "mean"),
        minimum=("avgPowerW", "min"), maximum=("avgPowerW", "max"), sd=("avgPowerW", "std"))
    groups = groups.sort_values("power")
    subset = assigned[assigned.refresh_hz == 120].groupby("soc").agg(
        n=("model", "size"), power=("avgPowerW", "mean"))
    key_groups = ["Snapdragon 8 Elite Gen 5", "Snapdragon 8 Elite", "Dimensity 9500"]
    leave_out = {}
    for soc in key_groups:
        rows = assigned[assigned.soc == soc]
        changes = [{"brand": BRANDS.get(brand, brand), "omitted": len(part),
                    "mean_after": rows[rows.brand != brand].avgPowerW.mean(),
                    "change_w": rows[rows.brand != brand].avgPowerW.mean() - rows.avgPowerW.mean()}
                   for brand, part in rows.groupby("brand")]
        leave_out[soc] = sorted(changes, key=lambda x: abs(x["change_w"]), reverse=True)
    apple = frame[frame.brand == "苹果"]
    recent = apple[apple.soc.isin(["Apple A19", "Apple A19 Pro", "Apple A20 Pro"])]
    other = frame[frame.brand != "苹果"]
    assert len(recent) == 6 and len(other) == 69
    assert recent.avgPowerW.max() < other.avgPowerW.min()
    generations = []
    for suffix in ["Pro", "Pro Max"]:
        for old, new in [(15, 16), (16, 17), (17, 18)]:
            a = apple[apple.model == f"iPhone {old} {suffix}"].avgPowerW.iloc[0]
            b = apple[apple.model == f"iPhone {new} {suffix}"].avgPowerW.iloc[0]
            generations.append({"model": suffix, "from": old, "to": new,
                                "old_w": a, "new_w": b, "lower_pct": reduction(b, a)})
    comparisons = [{"reference": soc, "reference_w": groups.loc[soc, "power"],
                    "a20_lower_pct": reduction(groups.loc["Apple A20 Pro", "power"], groups.loc[soc, "power"]),
                    "a19_lower_pct": reduction(groups.loc["Apple A19 Pro", "power"], groups.loc[soc, "power"])}
                   for soc in ["Kirin 9030S", *key_groups, "Google Tensor G5"]]
    display_means = frame.assign(apple=frame.brand == "苹果").groupby("apple")[["refresh_hz", "mp"]].mean()
    summary = {
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "capacity_basis": "Battery-imprint Wh, confirmed by the author; retained unchanged",
        "scope": "All 80 rows checked for arithmetic, raw aggregates, cache identity and imported fields. Selected incorrect specifications checked against cited sources. No repeat battery testing or independent verification of all original manufacturer specifications.",
        "devices": len(frame), "assigned_devices": len(assigned), "processor_groups": len(groups),
        "singleton_groups": int((groups.n == 1).sum()), "median_group_size": float(groups.n.median()),
        "raw_processor_groups": raw.soc.nunique(), "records_with_changes": sum(bool(r["changes"]) for r in records),
        "ranking": json.loads(groups.reset_index().to_json(orient="records")),
        "refresh_120": json.loads(subset.reset_index().to_json(orient="records")),
        "comparisons": comparisons, "leave_one_brand_out": leave_out,
        "correlations": frame[["capacityWh", "screen_size_in", "mp", "refresh_hz", "avgPowerW"]].corr().avgPowerW.to_dict(),
        "display_means": json.loads(display_means.reset_index().to_json(orient="records")),
        "recent_apple_max_w": recent.avgPowerW.max(), "non_apple_min_w": other.avgPowerW.min(),
        "separation_w": other.avgPowerW.min() - recent.avgPowerW.max(),
        "generations": generations,
        "pro_only_a19_w": apple[apple.model.isin(["iPhone 17 Pro", "iPhone 17 Pro Max"])].avgPowerW.mean(),
        "raw_device_audit": records,
        "reviewed_devices": json.loads(frame[["brand", "model", "soc", "minutes", "capacityWh", "avgPowerW", "screen_size_in", "resolution_px_w", "resolution_px_h", "refresh_hz"]].to_json(orient="records")),
    }
    target = args.output_dir / "battery_blog_audit.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# Battery blog audit", "", summary["scope"], "", f"Source: `{summary['source']}`. SHA-256: `{summary['source_sha256']}`.",
        "", "Battery-imprint Wh is the capacity source of truth, as confirmed by the author. All original capacities, runtimes and device power estimates are preserved. No mAh overlay is applied.",
        "", f"80 devices; {len(assigned)} assigned to {len(groups)} processor groups; two unresolved variants retained in device comparisons. {summary['records_with_changes']} records have metadata changes.",
        "", "Run `python analysis/battery_blog_audit.py` to regenerate this audit and its companion JSON. The script makes no network requests.",
        "", "## Processor ranking", "", "| Processor | Devices | Mean W | Observed range W |", "| --- | ---: | ---: | --- |"]
    for soc, row in groups.iterrows():
        lines.append(f"| {soc} | {int(row.n)} | {row.power:.6f} | {row.minimum:.6f}–{row.maximum:.6f} |")
    lines += ["", "## All-device sweep", "", "'Retained' means the local source chain was checked, not that a device was independently retested. Source links on corrected entries document the correction.",
              "", "| CSV line | Device | Cached page title | Review result |", "| ---: | --- | --- | --- |"]
    for row in records:
        changes = "; ".join(f"{k}: {v['before']} → {v['after']}" for k, v in row["changes"].items())
        source = row["evidence"]
        if not source.startswith("https:"):
            source = "../" + source
        lines.append(f"| {row['csv_line']} | {row['brand']} {row['model']} | {row['cached_title']} | [{row['note']}]({source}){' — ' + changes if changes else ''} |")
    (args.output_dir / "battery_blog_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ["raw_device_audit", "reviewed_devices", "leave_one_brand_out"]}, ensure_ascii=False, indent=2))
    print("LARGEST BRAND OMISSIONS", json.dumps({k:v[0] for k,v in leave_out.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
