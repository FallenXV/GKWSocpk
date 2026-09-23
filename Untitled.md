# Average Phone Power by SoC: A20 Pro, A19 Pro and the September Snapshot

2026-09-22 · @Jun Zhi

Across 80 phones in the SoCPK battery test 5.0 snapshot, the two A20 Pro phones average 2.004 W and the three A19 Pro phones average 2.151 W. After correcting chipset assignments, Snapdragon 8 Elite Gen 5 averages 2.867 W and Dimensity 9500 averages 2.846 W. **A20 Pro's average power consumption is 30.1% lower than the Snapdragon group and 29.6% lower than the Dimensity group. A19 Pro's is 25.0% and 24.4% lower, respectively.**

Those are substantial differences. They describe complete phones on this test, with power estimated from the energy printed on each battery and its recorded runtime. They do not isolate the contribution of the chip.

## It is a phone number filed under a chip name

The metric is one division:

```latex
P_{\text{avg}} = \frac{\text{battery-imprint energy (Wh)}}{\text{runtime (h)}}
```

The Wh values come from the battery imprint, which is the capacity source of truth for this comparison. The iPhone 18 Pro Max has a recorded capacity of 21.063 Wh and runs for 637 minutes, or 10.617 hours. Dividing the two gives 1.984 W.

This is an estimate of average whole-device power on a consistent battery-imprint basis. It includes the display, processor, memory, modem, radios and power-delivery losses. The dataset does not measure those components separately, and it does not record a direct integral of battery energy delivered during each test.

The processor figure is the unweighted mean of the individual phone estimates. Every phone counts once. “Snapdragon 8 Elite Gen 5 averages 2.867 W” means that the 19 phones assigned to that chip in the reviewed dataset average 2.867 W on this calculation.

Throughout this post, **lower power consumption is better**. Every percentage reduction uses the comparison phone or group as its denominator:

```latex
\text{power reduction} = \left(1 - \frac{P_{\text{lower}}}{P_{\text{reference}}}\right) \times 100\%
```

## The ranking

The source snapshot contains 80 device profiles. Correcting the specification matches leaves 78 phones assigned to 21 processor groups. Two phones have unresolved chipset variants and remain in the device comparisons, but are excluded from processor averages.

| Processor | Phones | Average power |
| --- | --: | --: |
| Apple A20 Pro | 2 | 2.004 W |
| Apple A19 Pro | 3 | 2.151 W |
| Apple A19 | 1 | 2.236 W |
| Kirin 9030S | 2 | 2.379 W |
| Apple A18 | 1 | 2.428 W |
| Kirin 9020 | 2 | 2.542 W |
| Apple A18 Pro | 2 | 2.545 W |
| Apple A17 Pro | 2 | 2.566 W |
| Kirin 9010S | 4 | 2.613 W |
| Kirin 9030 Pro | 1 | 2.710 W |
| Snapdragon 8 Gen 5 | 2 | 2.716 W |
| Dimensity 9500 | 9 | 2.846 W |
| Snapdragon 8 Elite Gen 5 | 19 | 2.867 W |
| Dimensity 9400+ | 3 | 2.874 W |
| Snapdragon 8 Elite | 19 | 2.891 W |
| Dimensity 9500s | 1 | 2.933 W |
| Dimensity 8500 Ultra | 1 | 2.974 W |
| Dimensity 9500 Monster | 1 | 2.980 W |
| Xring O1 | 1 | 3.033 W |
| Kirin 9000S | 1 | 3.228 W |
| Google Tensor G5 | 1 | 3.378 W |

Apple holds six of the top eight positions among the assigned groups. Even the A17 Pro group averages less power than every assigned Snapdragon and Dimensity group. That is a comparison of group means; individual phones can cross that ordering.

Kirin 9030S is the lowest-consuming non-Apple group, at 2.379 W across the two Pura 90 Pro models. It sits between A19 and A18 in the ranking, outside the lower range occupied by A19, A19 Pro and A20 Pro.

Four of the larger Snapdragon and Dimensity groups cluster within 0.045 W: Dimensity 9500, Snapdragon 8 Elite Gen 5, Dimensity 9400+ and Snapdragon 8 Elite. Their similar means describe the phones in this sample; they do not establish equal chip efficiency or performance.

At the other end, Tensor G5's 3.378 W is one Pixel 10 Pro XL result. A20 Pro's group mean is 40.7% lower. Neither that single Pixel nor the single Xiaomi 15s Pro behind Xring O1 is enough to characterise an entire processor family.

## The separation exists at the device level

**All six phones using A19, A19 Pro or A20 Pro have lower estimated power consumption than all 69 non-Apple phones in this snapshot.**

The highest-consuming of those six iPhones is the base iPhone 17 at 2.236 W. The lowest-consuming non-Apple phone is the Huawei Pura 90 Pro at 2.328 W. The separation is 0.092 W.

That observation does not depend on averaging phones into chip groups. It still depends on the devices selected, the recorded runtimes and the capacity basis. It is a result for this sample, not a guarantee about every phone that uses those processors.

Lower consumption also does not automatically mean the longest runtime. The six recent iPhones carry between 12.260 and 21.063 Wh. The Honor WIN RT lasts 12.1 hours on a 36.88 Wh pack, longer than any iPhone in this snapshot, while averaging 3.048 W. Runtime depends on both available energy and the rate at which the phone consumes it.

## Put the comparison and its denominator together

A19 Pro, Snapdragon 8 Elite Gen 5 and Dimensity 9500 provide a comparison within the 2025 flagship-chip generation represented here. Matching the generation does not match the rest of the phone, but it avoids treating the A20 Pro comparison as a same-generation result.

| Comparison group | Phones | Average power | A19 Pro consumes less by |
| --- | --: | --: | --: |
| Kirin 9030S | 2 | 2.379 W | 9.6% |
| Snapdragon 8 Elite Gen 5 | 19 | 2.867 W | 25.0% |
| Dimensity 9500 | 9 | 2.846 W | 24.4% |

The useful headline for A19 Pro is therefore **about a quarter less power than the Elite Gen 5 and Dimensity 9500 groups**. A20 Pro extends the observed reduction to about 30%. These comparisons are restricted to the chips and phones present in the snapshot.

## The largest iPhone step is from A18 Pro to A19 Pro

Following the Pro and Pro Max product lines gives a more comparable view of Apple's progression than mixing them with the Air. It still compares different phones, with changes to displays, batteries, other components and software.

| Transition | Pro: lower power | Pro Max: lower power |
| --- | --: | --: |
| iPhone 15 to iPhone 16, A17 Pro to A18 Pro | 1.1% | 0.6% |
| iPhone 16 to iPhone 17, A18 Pro to A19 Pro | 15.5% | 18.3% |
| iPhone 17 to iPhone 18, A19 Pro to A20 Pro | 4.6% | 5.8% |

The largest reduction among these iPhone Pro transitions comes with the iPhone 17 generation. The iPhone 18 generation lowers consumption again in both sizes, by a smaller amount.

The A19 Pro group includes the iPhone Air at 2.222 W. Restrict it to the iPhone 17 Pro and Pro Max and their mean is 2.115 W. Against those two models, the A20 Pro pair's 2.004 W is **5.2% lower**.

This is a reduction in the phone-level metric. It is not a controlled experiment replacing only the SoC. For example, screen sizes changed between the iPhone 15 and 16 Pro generations, and the snapshot lists iOS 26.5-family builds on the older phones and iOS 27.0 on the iPhone 18 pair. The data cannot allocate the improvement between silicon, display, software and other changes.

## What the sample sizes tell us

Both Snapdragon 8 Elite groups contain 19 phones; A20 Pro contains two. Nine of the 21 assigned processor groups have only one phone, and the median group size is two.

| Group | Phones | Mean | Observed device range |
| --- | --: | --: | --- |
| Apple A20 Pro | 2 | 2.004 W | 1.984–2.024 W |
| Apple A19 Pro | 3 | 2.151 W | 2.107–2.222 W |
| Snapdragon 8 Elite Gen 5 | 19 | 2.867 W | 2.601–3.152 W |
| Snapdragon 8 Elite | 19 | 2.891 W | 2.503–3.094 W |
| Dimensity 9500 | 9 | 2.846 W | 2.681–3.113 W |

The larger groups cover more implementations. That makes them more informative about variation within this dataset, but does not make them a random or sales-weighted sample of the market. Closely related phones can share hardware and software, and the snapshot does not provide repeated-run variability for each device. Formal confidence intervals across models would not resolve those limitations.

One useful sensitivity check is to remove each brand label in turn and recalculate the mean. The largest absolute changes are:

| Group | Largest-changing omission | Original mean | Mean after omission |
| --- | --- | --: | --: |
| Snapdragon 8 Elite Gen 5 | Xiaomi, five phones | 2.867 W | 2.823 W |
| Snapdragon 8 Elite | Samsung, two phones | 2.891 W | 2.920 W |
| Dimensity 9500 | OnePlus, one phone | 2.846 W | 2.813 W |

None of those omissions changes its group's mean by more than 0.044 W. The Apple comparison survives this particular sensitivity check. That establishes stability to removing one listed brand, not immunity to product mix or selection bias. Brand labels are also not necessarily independent manufacturers.

## Keep battery-imprint Wh as the capacity basis

The power rankings use the Wh printed on the battery throughout. A separately stored Geekerwan table provides measured and advertised mAh pairs for 27 matching phones, and the dashboard can display a scaled-capacity overlay.

That overlay is a separate calculation. It multiplies battery-imprint Wh by measured mAh divided by advertised mAh. Those quantities can use different capacity definitions: advertised typical capacity is not necessarily the rated capacity printed on the battery. A ratio between the two mAh figures therefore does not automatically replace the imprint Wh with measured delivered energy.

For this comparison, the imprint values remain unchanged. The overlay is not used in any table or percentage in this post, and the data do not establish that some fraction of Apple's lead comes from dishonest battery labelling.

## A 120 Hz filter is a subset, not a display-power measurement

The original refresh-rate metadata contained three parsing errors: a 2560 Hz PWM entry became 560 Hz, and two 2160 Hz PWM entries became 160 Hz. All three pages actually list 120 Hz refresh. Several wrong device matches also supplied incorrect screen specifications. The base iPhone 16's missing refresh entry has been filled with 60 Hz.

Using the reviewed metadata and the same chipset assignments as the ranking above:

| Processor | All assigned phones | Phones with a listed maximum refresh rate of 120 Hz |
| --- | --: | --: |
| Apple A20 Pro | 2.004 W, n=2 | 2.004 W, n=2 |
| Apple A19 Pro | 2.151 W, n=3 | 2.151 W, n=3 |
| Kirin 9030S | 2.379 W, n=2 | 2.379 W, n=2 |
| Snapdragon 8 Elite Gen 5 | 2.867 W, n=19 | 2.879 W, n=9 |
| Dimensity 9500 | 2.846 W, n=9 | 2.823 W, n=4 |
| Snapdragon 8 Elite | 2.891 W, n=19 | 2.909 W, n=11 |

The A20 Pro and A19 Pro groups remain lower-consuming within this subset. Dimensity 9500's subset mean is slightly lower than its full-group mean; the two Snapdragon subset means are slightly higher. There is no consistent reduction from selecting 120 Hz models.

This filter does not put the panels on equal footing. A listed maximum refresh rate is not a record of the rate used throughout a test, and it does not control brightness, panel technology, screen area or resolution. The table cannot tell us how many watts the display accounts for.

For completeness, the descriptive correlations across the 80 phones, after the documented metadata corrections, are:

| Variable | Pearson correlation with estimated power |
| --- | --: |
| Battery-imprint Wh | +0.52 |
| Screen diagonal | +0.32 |
| Main-panel pixel count | +0.24 |
| Listed maximum refresh rate | +0.30 |

These are associations, not component-level explanations. Battery Wh also appears in the numerator of the power calculation itself, so its correlation is not independent evidence that bigger batteries cause higher consumption.

## What changed in the source labels

The review checked all 80 records against their cached specification pages and recalculated the device and group arithmetic. It found six incorrect device-specification matches:

- Xiaomi 17 was matched to Redmi 17. Its chipset assignment is now Snapdragon 8 Elite Gen 5, with the main display specifications corrected.
- Xiaomi 15 had a Redmi 15 URL and inherited OnePlus 15 specifications through a cache collision. Its chipset is now Snapdragon 8 Elite, with the display corrected.
- OnePlus Ace 6 inherited Ace 6 Ultra specifications through another cache collision. It now belongs to Snapdragon 8 Elite, with the display corrected.
- OPPO Find X9 was matched to Find X9s. It now belongs to Dimensity 9500.
- OPPO Find X9 Pro was matched to Find X9s Pro. Its chipset group is unchanged, but its display is corrected to 6.78 inches and 120 Hz.
- Redmi K90 inherited K90 Ultra display specifications. Its chipset group is unchanged, but its display is corrected to 6.59 inches and 120 Hz.

The corrections use [Xiaomi's 17 specifications](https://www.mi.com/sg/product/xiaomi-17/specs/), [Xiaomi's 15 specifications](https://www.mi.com/global/product/xiaomi-15/specs/), [OnePlus's Ace 6 specifications](https://www.oneplus.com/cn/ace-6), [OPPO's Find X9 specifications](https://www.oppo.com/cn/smartphones/series-find-x/find-x9/specs/), [OPPO's Find X9 Pro specifications](https://www.oppo.com/cn/smartphones/series-find-x/find-x9-pro/specs/), and [AnTuTu's K90 listing](https://www.antutu.com/doc/136907.htm). The iPhone 16 refresh rate is documented in [DXOMARK's display test](https://www.dxomark.com/apple-iphone-16-display-test/).

Kirin 9010S and 9010s are merged as one four-phone group. The three PWM parsing errors are corrected from the existing cached pages. Two records remain unresolved:

| Phone | Possible chipset in source | Device power |
| --- | --- | --: |
| Samsung Galaxy S26 | Snapdragon 8 Elite Gen 5 or Exynos 2600 | 2.443 W |
| Huawei Mate 80 Pro | Kirin 9030 or Kirin 9030 Pro | 2.594 W |

The records do not identify the tested variants. Neither phone is assigned to a processor average. Both remain among the 80 device results, including the 69 non-Apple phones in the non-overlap comparison.

The A19 Pro group also combines the Air's five-core GPU configuration with the Pro models' six-core configuration. That grouping follows the shared chip name and should not be read as identical hardware. [Apple's Air comparison](https://www.apple.com/iphone-air/).

## What the data supports

The recent iPhones occupy the lowest power positions on this battery-imprint/runtime metric. A19 Pro averages about a quarter less consumption than the Elite Gen 5 and Dimensity 9500 groups; A20 Pro averages about 30% less. The separation is visible in individual device results and survives the listed brand-removal and 120 Hz subset checks.

The dataset cannot say how much of that difference comes from the processor, display, modem or software. Its strongest result is the observed difference between the tested phones, on a common calculation with a documented capacity basis.

### Reproducing this

The fixed input is `snapshots/battery_results_20260919T075628417409Z_1.csv`, collected on 2026-09-19. Runtime and battery-imprint Wh are preserved. A separate analysis applies the documented metadata corrections and excludes the two unresolved variants from processor averages.

```bash
python analysis/battery_blog_audit.py
```

The script checks all 80 rows and the original aggregate columns, then writes the reviewed figures to `analysis/battery_blog_audit.json` and an [80-device audit with the full ranking](analysis/battery_blog_audit.md). It uses the fixed snapshot and local cache without fetching new battery results.

To inspect the original, uncorrected snapshot in the dashboard:

```bash
python socpk_web.py --csv snapshots/battery_results_20260919T075628417409Z_1.csv --dataset Battery
```

The original dashboard processor labels and display metadata will differ from this reviewed analysis. The Geekerwan capacity overlay does not change the ranking and is not the capacity basis used here.
