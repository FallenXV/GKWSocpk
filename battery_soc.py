"""Shared helpers for resolving and grouping phone system-on-chip names."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup


GSM_ARENA_ROOT = "https://www.gsmarena.com/"

BRAND_ALIASES = {
    "苹果": "apple",
    "三星": "samsung",
    "谷歌": "google",
    "华为": "huawei",
    "一加": "oneplus",
    "真我": "realme",
    "红魔": "zte",
    "努比亚": "zte",
    "小米": "xiaomi",
    "红米": "xiaomi",
    "荣耀": "honor",
    "iqoo": "vivo",
    "nubia": "zte",
    "redmagic": "zte",
    "redmi": "xiaomi",
}

MODEL_BRAND_PREFIXES = {
    "apple": ("apple",),
    "google": ("google", "pixel"),
    "honor": ("honor",),
    "huawei": ("huawei",),
    "oneplus": ("oneplus",),
    "oppo": ("oppo",),
    "realme": ("realme",),
    "samsung": ("samsung", "galaxy"),
    "vivo": ("vivo", "iqoo"),
    "xiaomi": ("xiaomi", "redmi"),
    "zte": ("zte", "nubia", "redmagic"),
}


def canonical_brand(brand: str) -> str:
    """Return the GSMArena maker name used for a source brand."""
    folded = str(brand or "").strip().casefold()
    return BRAND_ALIASES.get(folded, folded)


def canonical_soc_name(value: object) -> str:
    """Reduce verbose chipset descriptions to a stable, readable SoC name."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.casefold() == "nan":
        return ""
    text = re.sub(r"\s*\([^)]*\bnm\b[^)]*\)\s*$", "", text, flags=re.I)
    text = re.sub(r"^(?:Qualcomm\s+)?SM\d+[A-Z0-9-]*\s+", "", text, flags=re.I)
    text = re.sub(r"^Qualcomm\s+(?=Snapdragon\b)", "", text, flags=re.I)
    text = re.sub(r"^Mediatek\s+MT\d+[A-Z0-9-]*\s+", "MediaTek ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    if re.match(r"^snapdragon\b", text, re.I):
        return "Snapdragon" + text[len("Snapdragon"):]
    if re.match(r"^mediatek\s+dimensity\b", text, re.I):
        return "Dimensity" + re.sub(r"^mediatek\s+dimensity", "", text, flags=re.I)
    if re.match(r"^dimensity\b", text, re.I):
        return "Dimensity" + text[len("Dimensity"):]
    return text


def _model_forms(brand: str, model: str) -> set[str]:
    text = str(model or "").strip().casefold()
    # Catalogs often add a sales-region qualifier that is absent from SoCPK.
    # It does not identify a different hardware model for this lookup.
    text = re.sub(r"\(\s*(?:china|global|india|international)\s*\)", " ", text, flags=re.I)
    replacements = {
        "至尊版": " ultra ",
        "竞速版": " racing ",
        "探索版": " explorer ",
        "标准版": " ",
        "+": " plus ",
        "特别版": " special edition ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\b5g\b", " ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()
    forms = {re.sub(r"[^a-z0-9]+", "", text)}
    for prefix in MODEL_BRAND_PREFIXES.get(canonical_brand(brand), ()):
        compact_prefix = re.sub(r"[^a-z0-9]+", "", prefix.casefold())
        forms |= {
            form[len(compact_prefix):]
            for form in tuple(forms)
            if form.startswith(compact_prefix) and len(form) > len(compact_prefix)
        }
    return {form for form in forms if form}


def choose_phone_url(brand: str, model: str, candidates: Iterable[tuple[str, str]]) -> str | None:
    """Choose a catalog URL only when its model name is an unambiguous close match."""
    wanted = _model_forms(brand, model)
    scored: list[tuple[float, str]] = []
    for candidate_name, candidate_url in candidates:
        candidate_forms = _model_forms(brand, candidate_name)
        if wanted & candidate_forms:
            scored.append((1.0, candidate_url))
            continue
        score = max(
            (SequenceMatcher(None, left, right).ratio() for left in wanted for right in candidate_forms),
            default=0.0,
        )
        scored.append((score, candidate_url))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < 0.91:
        return None
    # Region suffixes are intentionally removed by _model_forms, so two
    # different catalog pages can both be exact matches.  Do not turn that
    # uncertainty into a URL choice based on lexical ordering.
    if len({url for score, url in scored if score == 1.0}) > 1:
        return None
    if len(scored) > 1 and scored[0][0] < 0.98 and scored[0][0] - scored[1][0] < 0.04:
        return None
    return scored[0][1]


def parse_maker_links(html: str) -> dict[str, str]:
    """Parse the GSMArena manufacturer directory."""
    soup = BeautifulSoup(html, "html.parser")
    result: dict[str, str] = {}
    for anchor in soup.select(".st-text a[href]"):
        name = re.sub(r"\s+\d+\s+devices?\s*$", "", anchor.get_text(" ", strip=True), flags=re.I)
        result[name.casefold()] = urljoin(GSM_ARENA_ROOT, anchor["href"])
    return result


def parse_phone_catalog(html: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Return phone model links and additional catalog pages."""
    soup = BeautifulSoup(html, "html.parser")
    phones = [
        (anchor.get_text(" ", strip=True), urljoin(GSM_ARENA_ROOT, anchor["href"]))
        for anchor in soup.select(".makers li a[href]")
    ]
    pages = [
        urljoin(GSM_ARENA_ROOT, anchor["href"])
        for anchor in soup.select(".nav-pages a[href]")
        if anchor.get("href") != "#"
    ]
    return phones, list(dict.fromkeys(pages))
