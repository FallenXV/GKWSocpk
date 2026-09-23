# Fact check of Untitled.md — completed revision

Updated 22 September 2026 after the author's clarification: **battery-imprint Wh is the capacity source of truth**. Those Wh values and every original runtime are preserved. The post compares only chips represented in the snapshot, with every performance percentage stated as a reduction in power consumption.

[Revised post](Untitled.md) · [Full 80-device audit](analysis/battery_blog_audit.md) · [Reproducible analysis](analysis/battery_blog_audit.py)

The full sweep checked all 80 records for unique device identities, positive inputs, runtime/hour conversion, power and minutes-per-Wh calculations, all original processor aggregate columns, and the specification fields imported from cached pages. Cached device identities were reviewed across the whole set. Incorrect matches were checked against the sources listed in the audit. This is a review of the source chain and calculations, not independent repeat testing of the phones or verification of every manufacturer's specification.

**What changed**

| Issue | Applied revision |
| --- | --- |
| Six incorrect specification matches | Corrected Xiaomi 17, Xiaomi 15, OnePlus Ace 6, OPPO Find X9, OPPO Find X9 Pro and Redmi K90 metadata in the article analysis |
| PWM interpreted as refresh rate | Corrected Redmi K90 Pro Max, Xiaomi 17 Pro and Xiaomi 17 Pro Max to 120 Hz |
| Missing iPhone 16 refresh rate | Filled with 60 Hz using DXOMARK's original display test |
| Kirin 9010S/9010s split | Merged case variants |
| Ambiguous Galaxy S26 and Mate 80 Pro | Retained in device results; excluded from processor averages |
| Raw and adjusted statistics mixed | Recomputed every table from one documented correction set |
| Confidence intervals treated as representativeness | Replaced with observed ranges and clearly scoped brand-removal sensitivity |
| Battery-label accusation | Removed; rankings retain authoritative battery-imprint Wh |
| mAh overlay presented as measured Wh | Separated from the comparison and excluded from all published calculations |
| Mixed percentage directions | All comparison percentages now mean lower power relative to the stated reference |
| Unsupported causal explanations | Removed claims allocating savings to silicon, displays, process nodes or product price |
| Comparisons to processors absent from the data | Removed; scope is the fixed September snapshot |
| Unavailable chart links | Replaced with explicit numerical tables |
| Reproduction fetched new data | Added an offline, fixed-input analysis command |

There are **13 records with metadata changes**, including the two unresolved chipset assignments. The original snapshot and production dashboard/parser files are unchanged. The article uses the separate correction overlay in `analysis/battery_blog_audit.py`.

**Final processor comparison**

| Processor | Assigned phones | Mean W | A20 Pro lower consumption | A19 Pro lower consumption |
| --- | ---: | ---: | ---: | ---: |
| Apple A20 Pro | 2 | 2.003999 | — | — |
| Apple A19 Pro | 3 | 2.150550 | — | — |
| Snapdragon 8 Elite Gen 5 | 19 | 2.867096 | 30.1% | 25.0% |
| Snapdragon 8 Elite | 19 | 2.891058 | 30.7% | 25.6% |
| Dimensity 9500 | 9 | 2.846062 | 29.6% | 24.4% |

The reviewed processor ranking contains **21 groups covering 78 phones**, with **nine singleton groups** and a median group size of **two**. All 80 phones remain in device-level comparisons.

The six A19/A19 Pro/A20 Pro phones still have lower estimated power than all 69 non-Apple phones. Their maximum is 2.236364 W; the non-Apple minimum is 2.328125 W. The separation is 0.091761 W. This is a descriptive result for the sample, not proof that selection or measurement uncertainty cannot affect it.

**Display and brand checks**

| Group | All-device mean | Corrected 120 Hz subset | Largest brand-removal change |
| --- | ---: | ---: | ---: |
| Snapdragon 8 Elite Gen 5 | 2.867096 W, n=19 | 2.878754 W, n=9 | −0.043781 W, omit Xiaomi's five |
| Snapdragon 8 Elite | 2.891058 W, n=19 | 2.909094 W, n=11 | +0.029088 W, omit Samsung's two |
| Dimensity 9500 | 2.846062 W, n=9 | 2.822986 W, n=4 | −0.033405 W, omit OnePlus's one |

The 120 Hz filter does not consistently reduce the group means and does not measure panel power. The earlier claim that display differences account for only a few percent has been removed.

After the metadata corrections, Pearson correlations with the power estimate are 0.516214 for imprint Wh, 0.323561 for screen diagonal, 0.237811 for pixel count and 0.300596 for maximum refresh rate. All 80 records now have values for these fields. These associations are descriptive; battery Wh is also mathematically part of the power metric.

**Reproduce and verify**

```bash
python analysis/battery_blog_audit.py
```

The script validates the source arithmetic, preserves capacities and runtimes, recalculates the reviewed results, and writes `analysis/battery_blog_audit.json` plus the 80-device audit. It uses the pinned input and local cached pages; no new battery results are fetched.

Snapshot SHA-256: `9b234136a80b4023827aaa5f571434b56938f5b256c12ca4ffa8f592de8bcac4`.

Remaining limits are stated in the post: the two test-unit chipset variants are unidentified, no repeat-test uncertainty is supplied, maximum display refresh is not the actual test-time refresh trace, and whole-device power cannot be attributed to individual components. The dashboard still displays its original metadata until the data pipeline is separately repaired.
