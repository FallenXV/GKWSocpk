#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Battery Life Parser & Efficiency + GSMArena Enrichment

Features
- Fetch SoCPK 续航 3.5 JS, robustly extract arr=[...]
- Compute:
    * Avg Power (W) = capacityWh*60 / minutes
    * Minutes per Wh (min/Wh) = minutes / capacityWh
    * Δ vs #1 (%) with negative meaning "less/shorter"
- CJK/English width-aware table printing
- CSV output (default: results.csv), optional JSON
- Brand language mapping (--brand-lang source|en)
- GSMArena enrichment via manual URL mappings:
    * screen_size_in, resolution_px_w/h, refresh_hz
    * chipset, cpu, gpu, battery_mAh
  Use --spec "Brand|Model=URL" or --spec-map-json file.

Requires:
    pip install beautifulsoup4 requests
Optional:
    pip install wcwidth
Tested on Python 3.10+ (works on 3.8+ with minor typing tweaks)
"""

import os
import re
import csv
import json
import time
import argparse
import requests
import unicodedata
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin

# -----------------------------
# SoCPK data sources & defaults
# -----------------------------
DEFAULT_URLS = [
    "https://www.socpk.com/batlife/3.5/50cl1st.js?22",
    "https://www.socpk.com/batlife/3.5/",
]
DEFAULT_CSV = "results.csv"

DEFAULT_SPEC_URLS: Dict[str, str] = {
    # key = slug_key(brand, model)
    # iPhone family examples
    "apple|iphone 17 pro max": "https://www.gsmarena.com/apple_iphone_17_pro_max-13964.php",
    "apple|iphone 17 pro":     "https://www.gsmarena.com/apple_iphone_17_pro-13963.php",
    "apple|iphone 17":         "https://www.gsmarena.com/apple_iphone_17-13962.php",
    "apple|iphone 16 pro max": "https://www.gsmarena.com/apple_iphone_16_pro_max-13912.php",
    "apple|iphone 16 pro":     "https://www.gsmarena.com/apple_iphone_16_pro-13911.php",
    "apple|iphone 16":         "https://www.gsmarena.com/apple_iphone_16-13910.php",
    "apple|iphone 16e":        "https://www.gsmarena.com/apple_iphone_16e-13909.php",  # if exists
    "apple|iphone 16 plus":    "https://www.gsmarena.com/apple_iphone_16_plus-13908.php",
    "apple|iphone 15 pro":     "https://www.gsmarena.com/apple_iphone_15_pro-13778.php",
    "apple|iphone 15":         "https://www.gsmarena.com/apple_iphone_15-13777.php",
    # # Samsung
    # "samsung|s25 ultra":       "https://www.gsmarena.com/samsung_galaxy_s25_ultra-xxxx.php",  # replace xxxx
    # "samsung|s25 edge":        "https://www.gsmarena.com/samsung_galaxy_s25_edge-xxxx.php",
    # # OPPO
    # "oppo|find x8s":           "https://www.gsmarena.com/oppo_find_x8s_5g-13769.php",
    # "oppo|find x8 ultra":      "https://www.gsmarena.com/oppo_find_x8_ultra-xxxx.php",
    # # OnePlus
    # "oneplus|ace5 pro":        "https://www.gsmarena.com/oneplus_ace5_pro-xxxx.php",
    # "oneplus|ace5 至尊版":       "https://www.gsmarena.com/oneplus_ace5_supreme-xxxx.php",
    # # Xiaomi
    # "xiaomi|15":               "https://www.gsmarena.com/xiaomi_15-xxxx.php",
    # "xiaomi|15 pro":           "https://www.gsmarena.com/xiaomi_15_pro-xxxx.php",
    # "xiaomi|15s pro":          "https://www.gsmarena.com/xiaomi_15s_pro-xxxx.php",
    # # Huawei / Honor / vivo / iQOO etc
    # "huawei|pura 80":          "https://www.gsmarena.com/huawei_pura_80-xxxx.php",
    # "huawei|pura 80 ultra":    "https://www.gsmarena.com/huawei_pura_80_ultra-xxxx.php",
    # "honor|gt pro":            "https://www.gsmarena.com/honor_gt_pro-xxxx.php",
    # "vivo|x200 ultra":         "https://www.gsmarena.com/vivo_x200_ultra-xxxx.php",
    # "vivo|x200 pro mini":      "https://www.gsmarena.com/vivo_x200_pro_mini-xxxx.php",
    # "iqoo|z10 turbo+":         "https://www.gsmarena.com/iqoo_z10_turbo_plus-xxxx.php",
}

# -----------------------------
# Brand language mapping
# -----------------------------
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
REV_BRAND_MAP_EN_TO_ZH = {}
for zh, en in BRAND_MAP_ZH_TO_EN.items():
    REV_BRAND_MAP_EN_TO_ZH.setdefault(en, set()).add(zh)

def map_brand_name(brand: str, lang: str) -> str:
    b = (brand or "").strip()
    if lang == "en":
        return BRAND_MAP_ZH_TO_EN.get(b, b)  # if zh known -> en; else unchanged
    return b  # 'source' (as-is)

# -----------------------------
# Unicode width utilities (CJK)
# -----------------------------
try:
    from wcwidth import wcswidth  # best accuracy if available
except Exception:
    def _char_width(ch: str) -> int:
        if unicodedata.combining(ch):
            return 0
        if unicodedata.east_asian_width(ch) in ("F", "W"):
            return 2
        return 1
    def wcswidth(s: str) -> int:
        return sum(_char_width(c) for c in s)

def ljust_v(s: str, width: int) -> str:
    pad = max(0, width - wcswidth(str(s)))
    return str(s) + " " * pad

def rjust_v(s: str, width: int) -> str:
    pad = max(0, width - wcswidth(str(s)))
    return " " * pad + str(s)

# -----------------------------
# Fetching helpers
# -----------------------------
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

# -----------------------------
# Robust arr=[ ... ] extraction
# -----------------------------
def extract_array_literal(js_text: str) -> str:
    """
    Return the full bracket-balanced array literal after `arr =`.
    Ignores brackets inside strings and comments.
    """
    m = re.search(r'\b(?:var|let|const)\s+arr\s*=\s*', js_text)
    if not m:
        raise ValueError("Could not find `arr =` in JS.")
    i = m.end()
    # move to first '['
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
            raise ValueError("Unclosed array '[' while extracting.")
        ch = js_text[i]

        if (in_single or in_double) and not escaped and ch == '\\':
            escaped = True
            continue
        if escaped:
            escaped = False
            continue

        if in_single:
            if ch == "'": in_single = False
            continue
        if in_double:
            if ch == '"': in_double = False
            continue

        if ch == '/' and i + 1 < len(js_text):
            nxt = js_text[i + 1]
            if nxt == '/':  # line comment
                i += 2
                while i < len(js_text) and js_text[i] not in ('\n', '\r'):
                    i += 1
                continue
            if nxt == '*':  # block comment
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

# -----------------------------
# Comment stripping & row parse
# -----------------------------
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

def parse_rows_from_js(array_literal: str) -> List[List]:
    """
    Extract rows:
      ['brand','model',minutes,'os','',capacityWh,'url']
    using a targeted regex (robust to commas/whitespace).
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

# -----------------------------
# Records & metrics
# -----------------------------
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
        except:  # invalid numbers
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
            # deltas (filled later)
            "delta_vs_best_power_pct": None,
            "delta_vs_best_min_per_wh_pct": None,
            "delta_vs_best_minutes_pct": None,
        })
    return out

def compute_deltas(records: List[Dict]) -> None:
    """
    Fill Δ% columns vs #1 where 'less/shorter' is negative.
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

# -----------------------------
# Pretty-print tables
# -----------------------------
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

# -----------------------------
# GSMArena enrichment
# -----------------------------
try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except Exception as e:
    raise SystemExit("Missing dependency: beautifulsoup4. Install with: pip install beautifulsoup4") from e

SPEC_CACHE_DIR = ".gsm_cache"  # local cache dir

def slug_key(brand: str, model: str) -> str:
    return (brand.strip() + "|" + model.strip()).lower().replace(" ", "_")

def ensure_cache_dir():
    os.makedirs(SPEC_CACHE_DIR, exist_ok=True)

def cache_path_for(key: str) -> str:
    return os.path.join(SPEC_CACHE_DIR, f"{key}.html")

def polite_fetch(url: str, sleep_s: float = 1.0) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Personal Battery Research; +https://example.invalid)",
        "Accept-Language": "en-US,en;q=0.8",
    }
    time.sleep(max(0.0, sleep_s))
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() in ("iso-8859-1", "ascii"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def get_gsmarena_html(url: str, key: str, offline_only: bool = False) -> Optional[str]:
    ensure_cache_dir()
    cpath = cache_path_for(key)
    if os.path.exists(cpath):
        try:
            return open(cpath, "r", encoding="utf-8").read()
        except Exception:
            pass
    if offline_only:
        return None
    try:
        html = polite_fetch(url, sleep_s=1.0)
        with open(cpath, "w", encoding="utf-8") as f:
            f.write(html)
        return html
    except Exception:
        return None

def parse_gsmarena_specs(html: str) -> Dict:
    """
    Parse key specs from a GSMArena device page.
    Reads '#specs-list' table rows: each 'tr' contains '.ttl' (label) and '.nfo' (value).
    Many '.nfo' elements have a 'data-spec' attribute (e.g., chipset, displaysize, displayres, cpu, gpu, batsize).
    """
    soup = BeautifulSoup(html, "html.parser")
    data = {}
    host = soup.select_one("#specs-list") or soup  # fallback to entire doc if selector differs

    for row in host.select("tr"):
        ttl = row.select_one(".ttl")
        nfo = row.select_one(".nfo")
        if not ttl or not nfo:
            continue
        label = ttl.get_text(strip=True)
        value = nfo.get_text(" ", strip=True)
        dspec = nfo.get("data-spec")
        if dspec:
            data[f"data:{dspec}"] = value
        data[f"label:{label}"] = value

    out = {}
    # Screen size (inches) from displaysize or Size
    displaysize = data.get("data:displaysize") or data.get("label:Size") or ""
    m = re.search(r'([0-9.]+)\s*(?:inches|inch)', displaysize, re.I)
    if m: out["screen_size_in"] = float(m.group(1))

    # Resolution from displayres or Resolution
    displayres = data.get("data:displayres") or data.get("label:Resolution") or ""
    m = re.search(r'(\d{3,4})\s*x\s*(\d{3,4})', displayres)
    if m:
        out["resolution_px_w"] = int(m.group(1))
        out["resolution_px_h"] = int(m.group(2))

    # Refresh Hz from Size/Type lines
    displaytype = data.get("data:displaytype") or data.get("label:Type") or ""
    rr_m = re.search(r'(\d{2,3})\s*Hz', displaytype + " " + displaysize)
    if rr_m:
        out["refresh_hz"] = int(rr_m.group(1))

    # Chipset / CPU / GPU
    chipset = data.get("data:chipset") or data.get("label:Chipset")
    if chipset: out["chipset"] = chipset
    cpu = data.get("data:cpu") or data.get("label:CPU")
    if cpu: out["cpu"] = cpu
    gpu = data.get("data:gpu") or data.get("label:GPU")
    if gpu: out["gpu"] = gpu

    # Battery mAh
    batsize = data.get("data:batsize") or data.get("label:Battery") or ""
    m = re.search(r'(\d{3,5})\s*mAh', batsize.replace(",", ""))
    if m: out["battery_mAh"] = int(m.group(1))

    return out

def candidate_spec_keys(brand_now: str, model: str) -> List[str]:
    """
    Generate tolerant keys so the user can provide --spec for either zh brand or English brand,
    regardless of --brand-lang used for printing.
    """
    keys = {slug_key(brand_now, model)}
    # try EN->ZH variants
    if brand_now in REV_BRAND_MAP_EN_TO_ZH:
        for zh in REV_BRAND_MAP_EN_TO_ZH[brand_now]:
            keys.add(slug_key(zh, model))
    # try ZH->EN variants
    for zh, en in BRAND_MAP_ZH_TO_EN.items():
        if brand_now == zh:
            keys.add(slug_key(en, model))
            break
    return list(keys)

def enrich_with_gsmarena(records: List[Dict], spec_urls: Dict[str, str],
                         offline_only: bool = False) -> None:
    """
    For each record, look up GSMArena URL by tolerant key and enrich in place.
    """
    for r in records:
        # Find any key that exists in spec_urls (brand can be zh or en)
        chosen_key = None
        url = None
        for k in candidate_spec_keys(r["brand"], r["model"]):
            if k in spec_urls:
                chosen_key = k
                url = spec_urls[k]
                break
        if not url:
            continue
        html = get_gsmarena_html(url, chosen_key, offline_only=offline_only)
        r["spec_url"] = url
        if not html:
            continue
        specs = parse_gsmarena_specs(html)
        r.update(specs)

# -----------------------------
# Simple correlation helpers
# -----------------------------
def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs)/n
    my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    denx = sum((x-mx)**2 for x in xs)
    deny = sum((y-my)**2 for y in ys)
    if denx <= 0 or deny <= 0:
        return None
    return num / ((denx*deny) ** 0.5)

def print_correlations(records: List[Dict]) -> None:
    """
    Print Pearson correlation between efficiency metrics and enriched numeric specs.
    """
    def col(vals):
        return [v for v in vals if v is not None]

    # Build aligned vectors only where both fields exist
    def gather(xkey: str, ykey: str) -> Tuple[List[float], List[float]]:
        xs, ys = [], []
        for r in records:
            x = r.get(xkey)
            y = r.get(ykey)
            if x is not None and y is not None:
                xs.append(float(x)); ys.append(float(y))
        return xs, ys

    candidates = [
        ("screen_size_in", "Screen size (in)"),
        ("refresh_hz", "Refresh rate (Hz)"),
        ("battery_mAh", "Battery (mAh)"),
        ("total_pixels", "Total pixels (w*h)"),
    ]

    # derive total_pixels
    for r in records:
        w = r.get("resolution_px_w"); h = r.get("resolution_px_h")
        r["total_pixels"] = (w*h) if (isinstance(w, int) and isinstance(h, int)) else None

    print("▶ Correlation with Minutes/Wh and Avg Power (Pearson r)")
    print("Metric                 |  r(min/Wh)  |  r(avgPowerW)")
    print("-----------------------+------------+-------------")
    for key, label in candidates:
        x1, y1 = gather(key, "minPerWh")
        x2, y2 = gather(key, "avgPowerW")
        r1 = _pearson(x1, y1)
        r2 = _pearson(x2, y2)
        r1s = f"{r1: .3f}" if r1 is not None else "   —  "
        r2s = f"{r2: .3f}" if r2 is not None else "   —  "
        print(f"{label:<23} | {r1s:>10} | {r2s:>11}")
    print()

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="SoCPK Battery Efficiency + GSMArena enrichment")
    ap.add_argument("--url", action="append", help="Override SoCPK source URL(s)")
    ap.add_argument("--csv", help=f"Output CSV path (default: {DEFAULT_CSV})")
    ap.add_argument("--json", help="Optional JSON output path")
    ap.add_argument("--brand-lang", choices=["source","en"], default="source",
                    help="Brand language for output names: source or en (default=source)")
    # GSMArena mapping inputs
    ap.add_argument("--spec", action="append", default=[],
                    help="Map Brand|Model to GSMArena URL, e.g.: "
                         "--spec 'Apple|iPhone 17 Pro Max=https://www.gsmarena.com/apple_iphone_17_pro_max-13964.php'")
    ap.add_argument("--spec-map-json",
                    help="JSON file mapping 'brand|model' -> url (keys case/space-insensitive)")
    ap.add_argument("--spec-offline", action="store_true",
                    help="Use cached HTML only; do not fetch (use .gsm_cache)")
    # Enriched preview / correlations
    ap.add_argument("--preview", action="store_true",
                    help="Print a short enriched preview table")
    ap.add_argument("--correlate", action="store_true",
                    help="Compute simple Pearson correlations vs efficiency")
    args = ap.parse_args()

    # 1) Fetch SoCPK JS that defines arr=[...]
    urls = args.url if args.url else DEFAULT_URLS
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

    # 2) Extract & parse rows
    array_lit = extract_array_literal(js_text)
    raw_rows = parse_rows_from_js(array_lit)

    # 3) Build records + metrics
    records = rows_to_records(raw_rows, brand_lang=args.brand_lang)
    if not records:
        raise SystemExit("No valid rows parsed.")

    compute_deltas(records)

    # 4) Build GSMArena spec URL mapping
    spec_urls: Dict[str, str] = {}

    # a) seed with default mapping
    for k, v in DEFAULT_SPEC_URLS.items():
        spec_urls[k] = v

    # b) inline --spec entries override defaults
    for entry in args.spec or []:
        if "=" not in entry:
            print(f"[warn] skipping malformed --spec: {entry}")
            continue
        left, url = entry.split("=", 1)
        if "|" not in left:
            print(f"[warn] skipping malformed --spec: {entry}")
            continue
        b, m = left.split("|", 1)
        spec_urls[slug_key(b, m)] = url.strip()

    # c) JSON map entries override defaults
    if args.spec_map_json:
        try:
            with open(args.spec_map_json, "r", encoding="utf-8") as f:
                file_map = json.load(f)
            for k, v in file_map.items():
                spec_urls[k.strip().lower().replace(" ", "_")] = v
        except Exception as e:
            print(f"[warn] failed to load spec map JSON: {e}")

    # 6) Enrich in place
    enrich_with_gsmarena(records, spec_urls, offline_only=args.spec_offline)

    # 7) Rankings & print
    by_lowest_power = sorted(records, key=lambda p: p["avgPowerW"])
    by_min_per_wh   = sorted(records, key=lambda p: p["minPerWh"], reverse=True)
    by_minutes      = sorted(records, key=lambda p: p["minutes"], reverse=True)

    print_table_viz("▶ Ranked by Efficiency (Lowest Avg Power Draw First)",
                    by_lowest_power, delta_key="delta_vs_best_power_pct")
    print_table_viz("▶ Ranked by Minutes per Wh (Higher is Better)",
                    by_min_per_wh, delta_key="delta_vs_best_min_per_wh_pct")
    print_table_viz("▶ Ranked by Total Battery Life (Minutes)",
                    by_minutes, delta_key="delta_vs_best_minutes_pct")

    # 8) Optional enriched preview & correlations
    if args.preview:
        def preview_enriched(records: List[Dict], title: str, limit: int = 12):
            cols = [
                ("Brand",  lambda r,i: r["brand"], "left"),
                ("Model",  lambda r,i: r["model"], "left"),
                ("Size (in)", lambda r,i: f'{r.get("screen_size_in","—")}', "right"),
                ("Res",    lambda r,i: f'{r.get("resolution_px_w","—")}x{r.get("resolution_px_h","—")}', "right"),
                ("Hz",     lambda r,i: f'{r.get("refresh_hz","—")}', "right"),
                ("Chipset",lambda r,i: r.get("chipset","—"), "left"),
                ("mAh",    lambda r,i: f'{r.get("battery_mAh","—")}', "right"),
                ("Min/Wh", lambda r,i: f'{r["minPerWh"]:.2f}', "right"),
            ]
            matrix = []; headers = [h for (h,_,_) in cols]; matrix.append(headers)
            for i, rr in enumerate(records[:limit]):
                matrix.append([get(rr, i) for (_,get,_) in cols])
            col_w = [max(wcswidth(str(row[c])) for row in matrix) for c in range(len(cols))]
            print(title)
            print(" | ".join((ljust_v if cols[c][2]=="left" else rjust_v)(headers[c], col_w[c]) for c in range(len(cols))))
            print("-+-".join("-"*w for w in col_w))
            for ridx in range(1, len(matrix)):
                print(" | ".join((ljust_v if cols[c][2]=="left" else rjust_v)(matrix[ridx][c], col_w[c]) for c in range(len(cols))))
            print()

        preview_enriched(sorted(records, key=lambda r: r["minPerWh"], reverse=True),
                         "▶ Enriched preview (top by Minutes/Wh)")

    if args.correlate:
        print_correlations(records)

    # 9) CSV / JSON
    extra_spec_fields = [
        "screen_size_in", "resolution_px_w", "resolution_px_h",
        "refresh_hz", "chipset", "cpu", "gpu", "battery_mAh", "spec_url"
    ]
    fieldnames = [
        "brand","model","os","minutes","hours","capacityWh",
        "avgPowerW","avgPowermW","minPerWh","url",
        "delta_vs_best_power_pct","delta_vs_best_min_per_wh_pct","delta_vs_best_minutes_pct",
    ] + extra_spec_fields

    csv_path = args.csv or DEFAULT_CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            row = {k: r.get(k) for k in fieldnames}
            w.writerow(row)
    print(f"Wrote CSV: {csv_path}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON: {args.json}")

if __name__ == "__main__":
    main()
