# Model card: Bagré Sentinel-2 land-cover segmentation

> Complete this file after the real runs. Delete every placeholder that is not supported by an artifact. Do not report metrics from a validation split as test performance.

## Model

- Architecture: U-Net with a ResNet34 encoder, ImageNet initialization.
- Inputs: Sentinel-2 L2A B2, B3, B4, and B8, dry-season median composite, 10 m resolution.
- Outputs: five merged land-cover classes.
- Baseline: per-pixel Random Forest using the same four spectral bands.
- Best checkpoint selected by: validation mean IoU.

## Intended use

This is a research and portfolio demonstration of a reproducible segmentation pipeline. It is not an operational land-cover mapping system, a field-validated accuracy assessment, or a novel architecture.

## Study area and data

- Study area: configurable rectangle around Bagré, Burkina Faso. The rectangle is not an official irrigation-scheme boundary.
- Composite dates: 2021-01-01 to 2021-05-01, end date exclusive.
- Reference labels: ESA WorldCover 2021 v200.
- Spatial split: western training region, central validation region, eastern test region, with discarded buffer zones.
- Processed patch counts: [VERIFY FROM metadata.json].

## Merged classes

| Merged class | ESA WorldCover source values |
|---|---|
| Cropland | 40 |
| Natural vegetation | 10, 20, 30, 90, 95, 100 |
| Built-up | 50 |
| Bare/sparse | 60 |
| Water | 80 |

Unsupported values, including snow/ice, are ignored.

## Results

[INSERT THE GENERATED results.csv WITHOUT ROUNDING AWAY CLASS DIFFERENCES]

The test region is spatially held out. Mean IoU averages only classes present in the test reference labels. Always inspect per-class support before interpreting it.

## Limitations

- WorldCover is itself a model-derived product. Scores measure agreement with WorldCover, not absolute land-cover accuracy against field observations.
- A single composite and one AOI do not establish temporal or geographic generalization.
- Spectral similarity, mixed pixels, small fields, irrigation season, and class imbalance can cause errors. Replace this list with observed failure modes after inspecting the figures.
- Spatially separated regions can have different class distributions. Report absent or rare test classes explicitly.

## Reproducibility

- Configuration: [COMMIT HASH]
- Data preparation metadata: [PATH]
- U-Net checkpoint: [PATH]
- Random Forest model: [PATH]
- Evaluation artifacts: [PATH]
