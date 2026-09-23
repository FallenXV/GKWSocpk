# Battery blog audit

All 80 rows checked for arithmetic, raw aggregates, cache identity and imported fields. Selected incorrect specifications checked against cited sources. No repeat battery testing or independent verification of all original manufacturer specifications.

Source: `snapshots/battery_results_20260919T075628417409Z_1.csv`. SHA-256: `9b234136a80b4023827aaa5f571434b56938f5b256c12ca4ffa8f592de8bcac4`.

Battery-imprint Wh is the capacity source of truth, as confirmed by the author. All original capacities, runtimes and device power estimates are preserved. No mAh overlay is applied.

80 devices; 78 assigned to 21 processor groups; two unresolved variants retained in device comparisons. 13 records have metadata changes.

Run `python analysis/battery_blog_audit.py` to regenerate this audit and its companion JSON. The script makes no network requests.

## Processor ranking

| Processor | Devices | Mean W | Observed range W |
| --- | ---: | ---: | --- |
| Apple A20 Pro | 2 | 2.003999 | 1.983956–2.024043 |
| Apple A19 Pro | 3 | 2.150550 | 2.106742–2.222356 |
| Apple A19 | 1 | 2.236364 | 2.236364–2.236364 |
| Kirin 9030S | 2 | 2.378736 | 2.328125–2.429348 |
| Apple A18 | 1 | 2.428070 | 2.428070–2.428070 |
| Kirin 9020 | 2 | 2.542288 | 2.503125–2.581452 |
| Apple A18 Pro | 2 | 2.544508 | 2.511712–2.577305 |
| Apple A17 Pro | 2 | 2.566212 | 2.540000–2.592424 |
| Kirin 9010S | 4 | 2.612726 | 2.557394–2.680074 |
| Kirin 9030 Pro | 1 | 2.709717 | 2.709717–2.709717 |
| Snapdragon 8 Gen 5 | 2 | 2.716345 | 2.537690–2.895000 |
| Dimensity 9500 | 9 | 2.846062 | 2.680800–3.113300 |
| Snapdragon 8 Elite Gen 5 | 19 | 2.867096 | 2.600962–3.151883 |
| Dimensity 9400+ | 3 | 2.873561 | 2.714155–2.997628 |
| Snapdragon 8 Elite | 19 | 2.891058 | 2.503390–3.094382 |
| Dimensity 9500s | 1 | 2.932653 | 2.932653–2.932653 |
| Dimensity 8500 Ultra | 1 | 2.973512 | 2.973512–2.973512 |
| Dimensity 9500 Monster | 1 | 2.979695 | 2.979695–2.979695 |
| Xring O1 | 1 | 3.033040 | 3.033040–3.033040 |
| Kirin 9000S | 1 | 3.227528 | 3.227528–3.227528 |
| Google Tensor G5 | 1 | 3.377778 | 3.377778–3.377778 |

## All-device sweep

'Retained' means the local source chain was checked, not that a device was independently retested. Source links on corrected entries document the correction.

| CSV line | Device | Cached page title | Review result |
| ---: | --- | --- | --- |
| 2 | Honor WIN RT | Honor Win RT | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__win_rt.html) |
| 3 | Honor WIN | Honor Win | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__win.html) |
| 4 | Redmi K90 至尊版 | Xiaomi Redmi K90 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__k90.html) |
| 5 | Redmi Turbo 5 Max | Xiaomi Redmi Turbo 5 Max | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__turbo_5_max.html) |
| 6 | Redmi K90 Max | Xiaomi Redmi K90 Max | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__k90_max.html) |
| 7 | RedMagic 11 Pro | ZTE nubia RedMagic 11 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__11_pro.html) |
| 8 | Xiaomi 17 Max | Xiaomi 17 Max | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__17_max.html) |
| 9 | iQOO Z11 Turbo | vivo iQOO Z11 Turbo | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_iqoo_z11_turbo.html) |
| 10 | iQOO Z10 Turbo+ | vivo iQOO Z10 Turbo+ | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_iqoo_z10_turbo.html) |
| 11 | OnePlus Ace6T | OnePlus Ace 6T | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__ace6t.html) |
| 12 | OnePlus 15T | OnePlus 15T | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__15t.html) |
| 13 | OnePlus Ace6 至尊版 | OnePlus Ace 6 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__ace6.html) |
| 14 | iQOO Neo 11 | vivo iQOO Neo11 (China) | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_iqoo_neo_11.html) |
| 15 | Honor Magic8 Pro | Honor Magic8 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__magic8_pro.html) |
| 16 | OPPO Find X9 Pro | Oppo Find X9s Pro | [Wrong Find X9s Pro match](https://www.oppo.com/cn/smartphones/series-find-x/find-x9-pro/specs/) — screen_size_in: 6.32 → 6.78; resolution_px_w: 1216 → 1272; resolution_px_h: 2640 → 2772; refresh_hz: 144.0 → 120 |
| 17 | Huawei Nova 16 | Huawei nova 16 | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__nova_16.html) |
| 18 | Huawei Nova 16 Pro | Huawei nova 16 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__nova_16_pro.html) |
| 19 | iQOO 15 Ultra | vivo iQOO 15 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_iqoo_15_ultra.html) |
| 20 | iQOO 15T | vivo iQOO 15T | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_iqoo_15t.html) |
| 21 | Redmi K90 Pro Max | Xiaomi Redmi K90 Pro Max | [PWM frequency parsed as refresh rate](../.gsm_cache/phone__k90_pro_max.html) — refresh_hz: 560.0 → 120 |
| 22 | OnePlus Ace6 | OnePlus Ace 6 Ultra | [Ace 6 Ultra cache collision](https://www.oneplus.com/cn/ace-6) — soc: Dimensity 9500 → Snapdragon 8 Elite; screen_size_in: 6.78 → 6.83; resolution_px_h: 2772 → 2800 |
| 23 | Nubia Z80 Ultra | ZTE nubia Z80 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__z80_ultra.html) |
| 24 | Huawei Pura 90 Pro | Huawei Pura 90 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__pura_90_pro.html) |
| 25 | Redmi K80 至尊版 | Xiaomi Redmi K80 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__k80.html) |
| 26 | Honor GT Pro | Honor GT Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__gt_pro.html) |
| 27 | Huawei Nova 15 Pro | Huawei nova 15 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__nova_15_pro.html) |
| 28 | Xiaomi 17T Pro | Xiaomi 17T Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__17t_pro.html) |
| 29 | iQOO 15 | vivo iQOO 15 | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_iqoo_15.html) |
| 30 | OPPO Find X9s Pro | Oppo Find X9s Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_oppo_find_x9s_pro.html) |
| 31 | iQOO Neo 10 Pro+ | vivo iQOO Neo10 Pro+ (China) | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_iqoo_neo_10_pro.html) |
| 32 | OnePlus 15 | OnePlus 15 | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__15.html) |
| 33 | Huawei Pura 90 Pro Max | Huawei Pura 90 Pro Max | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__pura_90_pro_max.html) |
| 34 | Honor Magic8 | Honor Magic8 | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__magic8.html) |
| 35 | Realme GT8 | Realme GT8 (China) | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_realme_gt8.html) |
| 36 | Huawei Pura 90 | Huawei Pura 90 | [Normalise 9010s/9010S case](../.gsm_cache/phone__pura_90.html) — soc: Kirin 9010s → Kirin 9010S |
| 37 | vivo X300s | vivo X300s | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_vivo_x300s.html) |
| 38 | Xiaomi 17 Pro Max | Xiaomi 17 Pro Max | [PWM frequency parsed as refresh rate](../.gsm_cache/phone__17_pro_max.html) — refresh_hz: 160.0 → 120 |
| 39 | Apple iPhone 17 Pro Max | Apple iPhone 17 Pro Max | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__iphone_17_pro_max.html) |
| 40 | Xiaomi 17 | Xiaomi Redmi 17 | [Wrong Redmi 17 match](https://www.mi.com/sg/product/xiaomi-17/specs/) — soc: Snapdragon 4 Gen 5 → Snapdragon 8 Elite Gen 5; screen_size_in: 6.9 → 6.3; resolution_px_w: 720 → 1220; resolution_px_h: 1600 → 2656 |
| 41 | OPPO Find X9 | Oppo Find X9s | [Wrong Find X9s match](https://www.oppo.com/cn/smartphones/series-find-x/find-x9/specs/) — soc: Dimensity 9500s → Dimensity 9500 |
| 42 | Xiaomi 17T | Xiaomi 17T | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__17t.html) |
| 43 | Redmi K90 | Xiaomi Redmi K90 Ultra | [K90 Ultra cache collision](https://www.antutu.com/doc/136907.htm) — screen_size_in: 6.83 → 6.59; resolution_px_w: 1280 → 1156; resolution_px_h: 2772 → 2510; refresh_hz: 165.0 → 120 |
| 44 | OPPO Find X9 Ultra | Oppo Find X9 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_oppo_find_x9_ultra.html) |
| 45 | Huawei Mate 80 | Huawei Mate 80 | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__mate_80.html) |
| 46 | OnePlus Ace5 至尊版 | OnePlus Ace 5 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__ace5.html) |
| 47 | vivo X300 Pro | vivo X300 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_vivo_x300_pro.html) |
| 48 | vivo X300 | vivo X300 | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_vivo_x300.html) |
| 49 | Realme GT8 Pro | Realme GT 8 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_realme_gt8_pro.html) |
| 50 | Huawei Mate 70 Pro+ | Huawei Mate 70 Pro+ | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__mate_70_pro.html) |
| 51 | Huawei Mate 80 Pro | Huawei Mate 80 Pro | [Tested Kirin 9030/9030 Pro variant unidentified](../.gsm_cache/phone__mate_80_pro.html) — soc: Kirin 9030 Pro (6 nm) (512GB / 1TB 16GB RAM) Kirin 9030 (6 nm) (256GB / 512GB 12GB RAM) → None |
| 52 | Huawei Mate 80 Pro Max | Huawei Mate 80 Pro Max | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__mate_80_pro_max.html) |
| 53 | vivo X300 Ultra | vivo X300 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_vivo_x300_ultra.html) |
| 54 | Xiaomi 17 Ultra | Xiaomi 17 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__17_ultra.html) |
| 55 | iQOO 13 | vivo iQOO 13 | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_iqoo_13.html) |
| 56 | OPPO Find X8 Ultra | Oppo Find X8 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_oppo_find_x8_ultra.html) |
| 57 | Redmi K80 Pro | Xiaomi Redmi K80 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__k80_pro.html) |
| 58 | OnePlus Ace5 Pro | OnePlus Ace 5 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__ace5_pro.html) |
| 59 | Xiaomi 17 Pro | Xiaomi 17 Pro | [PWM frequency parsed as refresh rate](../.gsm_cache/phone__17_pro.html) — refresh_hz: 160.0 → 120 |
| 60 | Honor Magic7 Pro | Honor Magic7 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__magic7_pro.html) |
| 61 | OnePlus 13 | OnePlus 13 | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__13.html) |
| 62 | Xiaomi 15s Pro | Xiaomi 15S Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__15s_pro.html) |
| 63 | Xiaomi 15 Pro | Xiaomi 15 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__15_pro.html) |
| 64 | Apple iPhone 17 Pro | Apple iPhone 17 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__iphone_17_pro.html) |
| 65 | Samsung S26 Ultra | Samsung Galaxy S26 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__s26_ultra.html) |
| 66 | vivo X200 Ultra | vivo X200 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone_vivo_x200_ultra.html) |
| 67 | Apple iPhone 16 Pro Max | Apple iPhone 16 Pro Max | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__iphone_16_pro_max.html) |
| 68 | Xiaomi 15 | OnePlus 15 | [OnePlus 15 cache collision and Redmi 15 URL](https://www.mi.com/global/product/xiaomi-15/specs/) — soc: Snapdragon 8 Elite Gen 5 → Snapdragon 8 Elite; screen_size_in: 6.78 → 6.36; resolution_px_w: 1272 → 1200; resolution_px_h: 2772 → 2670; refresh_hz: 165.0 → 120 |
| 69 | Samsung S25 Ultra | Samsung Galaxy S25 Ultra | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__s25_ultra.html) |
| 70 | Samsung S26 | Samsung Galaxy S26 | [Tested Snapdragon/Exynos variant unidentified](../.gsm_cache/phone__s26.html) — soc: Snapdragon 8 Elite Gen 5 (3 nm) - US/CA/CN Exynos 2600 (2 nm) - ROW → None |
| 71 | Apple iPhone 15 Pro Max | Apple iPhone 15 Pro Max | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__iphone_15_pro_max.html) |
| 72 | Apple iPhone 17 | Apple iPhone 17 | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__iphone_17.html) |
| 73 | Huawei Mate 60 Pro | Huawei Mate 60 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__mate_60_pro.html) |
| 74 | Samsung S25 Edge | Samsung Galaxy S25 Edge | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__s25_edge.html) |
| 75 | Google Pixel 10 Pro XL | Google Pixel 10 Pro XL | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__pixel_10_pro_xl.html) |
| 76 | Apple iPhone 16 | Apple iPhone 16 | [Missing refresh rate](https://www.dxomark.com/apple-iphone-16-display-test/) — refresh_hz: None → 60 |
| 77 | Apple iPhone 16 Pro | Apple iPhone 16 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__iphone_16_pro.html) |
| 78 | Apple iPhone Air | Apple iPhone Air | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__iphone_air.html) |
| 79 | Apple iPhone 15 Pro | Apple iPhone 15 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__iphone_15_pro.html) |
| 80 | Apple iPhone 18 Pro | Apple iPhone 18 Pro | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__iphone_18_pro.html) |
| 81 | Apple iPhone 18 Pro Max | Apple iPhone 18 Pro Max | [Cache identity and imported fields checked; retained](../.gsm_cache/phone__iphone_18_pro_max.html) |
