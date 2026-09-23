"""Reviewed metadata overlays; never modify battery energy or runtime inputs."""
from __future__ import annotations
import json
from pathlib import Path

BRANDS = {'苹果': 'apple', '三星': 'samsung', '华为': 'huawei', '一加': 'oneplus',
          '小米': 'xiaomi', '红米': 'redmi', '荣耀': 'honor', '谷歌': 'google'}

def brand_name(value):
    name = str(value).strip().casefold()
    return BRANDS.get(name, name)

REVIEWED = {(brand_name(r['brand']), r['model'].casefold()): r for r in
            json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))}

def review_record(record):
    """Return a copy carrying narrowly matched, attributable metadata corrections."""
    result = dict(record)
    review = REVIEWED.get((brand_name(result.get('brand', '')), str(result.get('model', '')).strip().casefold()))
    if review:
        result.update(review['fields'])
        # An explicit unresolved assignment must not fall back to the raw chipset.
        if 'soc' in review['fields'] and review['fields']['soc'] is None:
            result['soc'] = ''
            result['chipset'] = ''
        result['metadata_review'] = review['reason']
        result['metadata_source'] = review['source']
        result['metadata_reviewed'] = review['reviewed']
    return result
