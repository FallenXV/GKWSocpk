#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Battery Life Parser & Efficiency Calculator (with % difference vs #1)
- Fetch SoCPK "续航 3.5" JS data
- Robustly parse arr=[...]
- Compute avg power & minutes/Wh
- Print CJK/English width-aware tables
- Add Δ vs #1 (%) per ranking
- Save results.csv by default
"""

import re
import csv
import json
import argparse
import requests
import unicodedata
from typing import List, Dict
from urllib.parse import urljoin

# ==========================
# Config
# ==========================
DEFAULT_URLS = [
    "https://www.socpk.com/batlife/3.5/50cl1st.js?22",
    "https://www.socpk.com/batlife/3.5/",
]
DEFAULT_CSV = "results.csv"

# Map brands from Chinese to English
BRAND_MAP_ZH_TO_EN = {
    "苹果": "Apple",
    "三星": "Samsung",
    "小米": "Xiaomi",
    "OPPO": "OPPO",
    "一加": "OnePlus",
    "华为": "Huawei",
    "荣耀": "Honor",
    "vivo": "vivo",
    "iQOO": "iQOO",
}

def map_brand_name(brand: str, lang: str) -> str:
    b = (brand or "").strip()
    if lang == "en":
        return BRAND_MAP_ZH_TO_EN.get(b, b)
    return b  # 'source'

# ==========================
# Unicode width utilities
# ==========================
try:
    from wcwidth import wcswidth  # optional, nicer width handling
except ImportError:
    def _char_width(ch: str) -> int:
        if unicodedata.combining(ch):
            return 0
        if unicodedata.east_asian_width(ch) in ("F", "W"):
            return 2
        return 1
    def wcswidth(s: str) -> int:
        return sum(_char_width(c) for c in s)

def ljust_v(s: str, width: int) -> str:
    pad = max(0, width - wcswidth(s))
    return s + " " * pad

def rjust_v(s: str, width: int) -> str:
    pad = max(0, width - wcswidth(s))
    return " " * pad + s

# ==========================
# Fetching & array extraction
# ==========================
def fetch_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (BatteryParser)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() in ("iso-8859-1", "ascii"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def discover_js_from_html(html: str, base_url: str) -> List[str]:
    srcs = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html, flags=re.I)
    return [urljoin(base_url, s) for s in srcs]

def extract_array_literal(js_text: str) -> str:
    """
    Return the FULL bracket-balanced literal after `arr =`.
    Ignores quotes/comments while balancing.
    """
    m = re.search(r'\b(?:var|let|const)\s+arr\s*=\s*', js_text)
    if not m:
        raise ValueError("Could not find `arr =` in JS.")
    i = m.end()
    while i < len(js_text) and js_text[i].isspace():
        i += 1
    while i < len(js_text) and js_text[i] != '[':
        if not js_text[i].isspace():
            raise ValueError("Expected '[' after `arr =`.")
        i += 1
    if i >= len(js_text) or js_text[i] != '[':
        raise ValueError("Opening '[' not found for array.")

    start = i
    depth = 0
    in_single = in_double = False
    escaped = False
    i -= 1
    while True:
        i += 1
        if i >= len(js_text):
            raise ValueError("Unclosed '[' while extracting array.")
        ch = js_text[i]

        if (in_single or in_double) and not escaped and ch == '\\':
            escaped = True
            continue
        if escaped:
            escaped = False
            continue

        if in_single:
            if ch == "'":
                in_single = False
            continue
        if in_double:
            if ch == '"':
                in_double = False
            continue

        if ch == '/' and i + 1 < len(js_text):
            nxt = js_text[i + 1]
            if nxt == '/':
                i += 2
                while i < len(js_text) and js_text[i] not in ('\n', '\r'):
                    i += 1
                continue
            if nxt == '*':
                i += 2
                while i + 1 < len(js_text) and not (js_text[i] == '*' and js_text[i + 1] == '/'):
                    i += 1
                i += 1
                continue

        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue

        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                end = i + 1
                return js_text[start:end]

# ==========================
# Safe comment stripping & row parsing
# ==========================
def strip_js_comments_safely(s: str) -> str:
    out = []
    i, n = 0, len(s)
    in_single = in_double = escaped = False
    while i < n:
        ch = s[i]
        if (in_single or in_double) and not escaped and ch == "\\":
            out.append(ch); escaped = True; i += 1; continue
        if in_single:
            out.append(ch)
            if not escaped and ch == "'": in_single = False
            escaped = False; i += 1; continue
        if in_double:
            out.append(ch)
            if not escaped and ch == '"': in_double = False
            escaped = False; i += 1; continue
        if ch == "'": in_single = True; out.append(ch); i += 1; continue
        if ch == '"': in_double = True; out.append(ch); i += 1; continue
        if ch == "/" and i+1 < n and s[i+1] == "/":
            i += 2
            while i < n and s[i] not in ("\n", "\r"): i += 1
            continue
        if ch == "/" and i+1 < n and s[i+1] == "*":
            i += 2
            while i+1 < n and not (s[i] == "*" and s[i+1] == "/"): i += 1
            i += 2 if i < n else 0
            continue
        out.append(ch); i += 1
    return "".join(out)

def parse_rows_from_js(array_literal: str):
    """
    Extract rows like:
      ['brand','model',minutes,'os','',capacityWh,'url']
    using a targeted regex.
    """
    text = strip_js_comments_safely(array_literal)
    row_re = re.compile(
        r"""\[\s*'(?P<brand>[^']*)'\s*,\s*'(?P<model>[^']*)'\s*,\s*
             (?P<minutes>\d+(?:\.\d+)?)\s*,\s*
             '(?P<os>[^']*)'\s*,\s*'(?P<unused>[^']*)'\s*,\s*
             (?P<cap>\d+(?:\.\d+)?)\s*,\s*
             '(?P<url>[^']*)'\s*\]""",
        re.VERBOSE
    )
    rows = []
    for m in row_re.finditer(text):
        rows.append([
            m.group("brand"),
            m.group("model"),
            float(m.group("minutes")),
            m.group("os"),
            m.group("unused"),
            float(m.group("cap")),
            m.group("url"),
        ])
    if not rows:
        raise ValueError("Regex parser found 0 rows. The source format may have changed.")
    return rows

# ==========================
# Transform & metrics (+Δ%)
# ==========================
def rows_to_records(arr, brand_lang: str) -> List[Dict]:
    out = []
    for row in arr:
        if not (isinstance(row, (list, tuple)) and len(row) >= 6):
            continue
        brand_src = str(row[0]).strip()
        brand = map_brand_name(brand_src, brand_lang)
        model = str(row[1]).strip()
        osver = str(row[3]).strip()
        url = str(row[6]).strip() if len(row) > 6 else ""
        try:
            minutes = float(row[2]); capacityWh = float(row[5])
        except:
            continue
        if minutes <= 0 or capacityWh <= 0:
            continue

        avgPowerW = capacityWh * 60.0 / minutes     # lower is better
        minPerWh  = minutes / capacityWh            # higher is better

        out.append({
            "brand": brand, "model": model, "os": osver,
            "minutes": minutes, "hours": minutes/60.0,
            "capacityWh": capacityWh,
            "avgPowerW": avgPowerW,
            "avgPowermW": avgPowerW*1000.0,
            "minPerWh": minPerWh,
            "url": url,
            # place-holders for Δ%; filled later
            "delta_vs_best_power_pct": None,
            "delta_vs_best_min_per_wh_pct": None,
            "delta_vs_best_minutes_pct": None,
        })
    return out

def compute_deltas(records: list[dict]) -> None:
    """
    Fill Δ% columns vs #1 using a sign convention where 'less/shorter' is negative.
    - Power (lower is better):   Δ = (value - best) / best * 100
      -> less power than #1 => negative
    - Min/Wh (higher is better): Δ = (value - best) / best * 100
      -> fewer min/Wh than #1 => negative
    - Minutes (higher is better): Δ = (value - best) / best * 100
      -> shorter battery life than #1 => negative
    """
    if not records:
        return

    best_power = min(r["avgPowerW"] for r in records)   # lower is better
    best_minwh = max(r["minPerWh"]  for r in records)   # higher is better
    best_mins  = max(r["minutes"]   for r in records)   # higher is better

    for r in records:
        r["delta_vs_best_power_pct"] = ((r["avgPowerW"] - best_power) / best_power) * 100.0
        r["delta_vs_best_min_per_wh_pct"] = ((r["minPerWh"] - best_minwh) / best_minwh) * 100.0
        r["delta_vs_best_minutes_pct"] = ((r["minutes"] - best_mins) / best_mins) * 100.0


# ==========================
# Pretty-print tables (Δ%)
# ==========================
def print_table_viz(title: str, rows: List[Dict], delta_key: str):
    if not rows:
        return
    cols = [
        ("Rank",           lambda r,i: str(i+1),               "right"),
        ("Brand",          lambda r,i: r["brand"],             "left"),
        ("Model",          lambda r,i: r["model"],             "left"),
        ("OS",             lambda r,i: r["os"],                "left"),
        ("Minutes",        lambda r,i: f"{int(r['minutes'])}", "right"),
        ("Capacity (Wh)",  lambda r,i: f"{r['capacityWh']:.2f}", "right"),
        ("Avg Power (W)",  lambda r,i: f"{r['avgPowerW']:.3f}",  "right"),
        ("Avg Power (mW)", lambda r,i: f"{r['avgPowermW']:.0f}", "right"),
        ("Min/Wh",         lambda r,i: f"{r['minPerWh']:.2f}",   "right"),
        ("Δ vs #1 (%)",    lambda r,i: f"{r[delta_key]:+.2f}" if r[delta_key] is not None else "—", "right"),
    ]
    matrix = []
    headers = [h for (h,_,_) in cols]
    matrix.append(headers)
    for i, r in enumerate(rows):
        matrix.append([getter(r, i) for (_, getter, _) in cols])
    col_w = [max(wcswidth(str(row[c])) for row in matrix) for c in range(len(cols))]

    print(title)
    header_line = " | ".join(
        (ljust_v if cols[c][2] == "left" else rjust_v)(headers[c], col_w[c])
        for c in range(len(cols))
    )
    print(header_line)
    print("-+-".join("-" * col_w[c] for c in range(len(cols))))
    for ridx in range(1, len(matrix)):
        line = " | ".join(
            (ljust_v if cols[c][2] == "left" else rjust_v)(matrix[ridx][c], col_w[c])
            for c in range(len(cols))
        )
        print(line)
    print()

# ==========================
# Main
# ==========================
def main():
    ap = argparse.ArgumentParser(description="Parse SoCPK JS and compute efficiency with Δ vs #1")
    ap.add_argument("--url", action="append", help="Override source URL(s)")
    ap.add_argument("--csv", help=f"Output CSV path (default: {DEFAULT_CSV})")
    ap.add_argument("--json", help="Optional JSON output path")
    ap.add_argument("--brand-lang", choices=["source","en"], default="en",
                    help="Brand language: source or en (default=source)")
    args = ap.parse_args()

    urls = args.url if args.url else DEFAULT_URLS

    # Fetch JS that defines arr=[...]
    js_text = None
    for url in urls:
        try:
            txt = fetch_text(url)
            if url.endswith("/") or "<html" in txt.lower():
                for ju in discover_js_from_html(txt, url):
                    try:
                        jst = fetch_text(ju)
                        if re.search(r'\barr\s*=', jst):
                            js_text = jst; break
                    except:
                        continue
            else:
                if re.search(r'\barr\s*=', txt):
                    js_text = txt
            if js_text:
                break
        except:
            continue
    if not js_text:
        raise SystemExit("Could not fetch a JS file containing arr=[...].")

    array_lit = extract_array_literal(js_text)
    raw_rows = parse_rows_from_js(array_lit)
    records = rows_to_records(raw_rows, brand_lang=args.brand_lang)
    if not records:
        raise SystemExit("No valid rows parsed.")

    # Compute Δ% vs best for all rows (used by all views)
    compute_deltas(records)

    # Rankings
    by_lowest_power = sorted(records, key=lambda p: p["avgPowerW"])                    # best = min
    by_min_per_wh   = sorted(records, key=lambda p: p["minPerWh"], reverse=True)       # best = max
    by_minutes      = sorted(records, key=lambda p: p["minutes"], reverse=True)        # best = max

    print_table_viz("▶ Ranked by Efficiency (Lowest Avg Power Draw First)",
                    by_lowest_power, delta_key="delta_vs_best_power_pct")
    print_table_viz("▶ Ranked by Minutes per Wh (Higher is Better)",
                    by_min_per_wh, delta_key="delta_vs_best_min_per_wh_pct")
    print_table_viz("▶ Ranked by Total Battery Life (Minutes)",
                    by_minutes, delta_key="delta_vs_best_minutes_pct")

    # CSV (complete dataset with all three delta columns)
    csv_path = args.csv or DEFAULT_CSV
    fieldnames = [
        "brand","model","os","minutes","hours","capacityWh",
        "avgPowerW","avgPowermW","minPerWh","url",
        "delta_vs_best_power_pct",
        "delta_vs_best_min_per_wh_pct",
        "delta_vs_best_minutes_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"Wrote CSV: {csv_path}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON: {args.json}")

if __name__ == "__main__":
    main()
