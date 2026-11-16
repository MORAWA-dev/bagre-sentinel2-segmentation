# Bagré Sentinel-2 Land-Cover Segmentation

## Reported Colab benchmark

The models were evaluated on a geographic holdout against ESA WorldCover 2021.

| Model | Macro IoU | Pixel agreement |
| --- | ---: | ---: |
| U-Net | 31.97% | 71.03% |
| Random Forest | 19.15% | 56.21% |

Both models scored 0% water IoU. The test split contained 519 water-labeled pixels. Independent field accuracy and performance outside the study area remain unmeasured.


A reproducible minimum project for five-class semantic segmentation of Sentinel-2 imagery around Bagré, Burkina Faso. The main result is an honest comparison between U-Net and a per-pixel Random Forest on a spatially held-out test region.

**Current repository status:** implementation scaffold, not a completed experiment. No performance claim is valid until the GeoTIFF exports, training runs, and evaluation artifacts exist.

## What this project demonstrates

- Google Earth Engine export of a cloud-masked Sentinel-2 L2A composite and ESA WorldCover 2021.
- Exact raster-grid validation, class remapping, tiling, and geographically disjoint splits.
- A Random Forest baseline and a U-Net with a ResNet34 encoder, both using B2, B3, B4, and B8.
- Dice plus cross-entropy loss, geometric augmentation, AdamW, mixed precision, checkpoint resuming, and early stopping.
- mIoU, per-class IoU, per-class F1, overall accuracy, confusion matrices, training curves, and fixed-seed prediction figures.
- Explicit treatment of label and generalization limitations.

This is not a production system, a new architecture, or a field-validated land-cover accuracy assessment.

## Repository layout

```text
configs/bagre.yaml                 Experiment configuration
gee/01_export_bagre.js             GEE data export
notebooks/Bagre_Colab_Runner.ipynb Guided Colab workflow
scripts/prepare_data.py            Raster checks, tiling, spatial splits
scripts/train_rf.py                Random Forest baseline
scripts/train_unet.py              U-Net training and test evaluation
scripts/compare_models.py          Results table and figures
src/bagre_segmentation/            Reusable implementation
tests/                             Unit and synthetic integration tests
docs/                              Model-card and interview templates
```

## Data configuration

| Choice | First run |
|---|---|
| AOI | Rectangle `[-0.75, 11.25, -0.35, 11.65]` around Bagré |
| Composite | January to April 2021, end date 2021-05-01 exclusive |
| Bands | B2, B3, B4, B8 |
| Resolution and CRS | 10 m, EPSG:32630 |
| Reference labels | ESA WorldCover 2021 v200 |
| Classes | Cropland, natural vegetation, built-up, bare/sparse, water |
| Patches | 256 × 256 pixels, 128-pixel training stride |
| Split | 60% western train, 20% central validation, 20% eastern test, with buffers |

The AOI is a practical starting rectangle, not an official irrigation-scheme boundary. Inspect the RGB preview in GEE and adjust it if needed. Keep the rectangle modest for the first complete run.

### Class mapping

| Merged ID | Class | WorldCover values |
|---:|---|---|
| 0 | Cropland | 40 |
| 1 | Natural vegetation | 10, 20, 30, 90, 95, 100 |
| 2 | Built-up | 50 |
| 3 | Bare/sparse | 60 |
| 4 | Water | 80 |
| 255 | Ignore | All unsupported values and invalid pixels |

## Step 1: Export the data

1. Create a Google Drive folder named `bagre-segmentation`, with a `raw` subfolder.
2. Open [Google Earth Engine Code Editor](https://code.earthengine.google.com/).
3. Paste `gee/01_export_bagre.js` and run it.
4. Inspect the AOI, Sentinel-2 RGB map, observation count, and optional WorldCover layer.
5. Start both export tasks. Move the resulting files into `raw`:

```text
bagre_s2_2021_dryseason.tif
bagre_worldcover_2021.tif
```

The script uses Cloud Score+ and SCL masking. The year and season match the 2021 WorldCover reference product. The composite and labels use the same export region, CRS, and scale.

## Step 2: Run the guided Colab workflow

1. Upload the extracted repository folder into Drive so its path is:

```text
/content/drive/MyDrive/bagre-segmentation/bagre-sentinel2-segmentation
```

2. Open `notebooks/Bagre_Colab_Runner.ipynb` in Colab.
3. Select **Runtime → Change runtime type → GPU**.
4. Run the cells in order. The notebook mounts Drive, installs the package, runs tests, prepares the data, trains both models, and creates the comparison artifacts.
5. Inspect `data_summary.png` and `spatial_split.png` after preparation. If a split lacks a class or has very little support, stop and revise the AOI or spatial split before training. Do not replace the spatial split with random patches.

Expected Drive artifacts:

```text
bagre-segmentation/
├── raw/
├── processed/
│   ├── metadata.json
│   ├── tiles.csv
│   ├── dataset_fingerprint.json
│   ├── test_manifest.json
│   ├── data_summary.png
│   └── spatial_split.png
├── rf/
│   ├── dataset_fingerprint.json
│   ├── test_manifest.json
│   ├── metrics.json
│   └── test_predictions.npy
├── unet/
│   ├── best.pt
│   ├── last.pt
│   ├── history.csv
│   ├── dataset_fingerprint.json
│   ├── test_manifest.json
│   ├── metrics.json
│   └── test_predictions.npy
└── comparison/
    ├── results.csv
    ├── prediction_examples.png
    ├── unet_confusion_matrix.png
    ├── rf_confusion_matrix.png
    └── unet_training_curves.png
```

## Manual commands

Run from the repository root after `pip install -e '.[dev]'`:

```bash
python scripts/prepare_data.py \
  --image /path/to/bagre_s2_2021_dryseason.tif \
  --labels /path/to/bagre_worldcover_2021.tif \
  --output-dir /path/to/processed

python scripts/train_rf.py \
  --data-dir /path/to/processed \
  --output-dir /path/to/rf

python scripts/train_unet.py \
  --data-dir /path/to/processed \
  --output-dir /path/to/unet

python scripts/compare_models.py \
  --data-dir /path/to/processed \
  --unet-dir /path/to/unet \
  --rf-dir /path/to/rf \
  --output-dir /path/to/comparison
```

Resume an interrupted U-Net run:

```bash
python scripts/train_unet.py \
  --data-dir /path/to/processed \
  --output-dir /path/to/unet \
  --resume /path/to/unet/last.pt
```

The default U-Net batch size is 4 with four gradient-accumulation steps, giving an effective batch size of 16 without assuming that a batch of 16 fits the GPU. If memory is still insufficient, reduce `batch_size` to 2 and keep the same effective batch size by increasing accumulation to 8.

## Evaluation rules

- Select the U-Net checkpoint using validation mIoU only.
- Evaluate the test region once after model selection. Do not tune against test scores.
- Macro IoU averages only classes with nonzero reference support. Both models use the same supported-class set, while support and prediction counts are reported for every class.
- Validation and test patches never overlap. If a raster dimension is not divisible by the patch size, the partial evaluation remainder is omitted rather than duplicated through an overlapping edge patch.
- Compare RF and U-Net on the same test pixels and the same four input bands. Both runs carry the prepared dataset fingerprint and ordered test manifest; comparison refuses any fingerprint or manifest mismatch.
- Treat the generated figures as diagnostic evidence. Their test patches are selected with a fixed random seed, not cherry-picked.
- If RF wins, report that result and investigate why. A competitive pixel baseline is a valid conclusion.

## Reproducibility checks

```bash
pytest
```

The tests cover class remapping, ignore handling, manually verifiable metrics, split disjointness, a four-band U-Net forward pass, and an end-to-end synthetic raster-preparation run. These tests validate code behavior, not remote-sensing accuracy.

## CV promotion gate

Do not add a completed-project bullet yet. After both runs and a review of the artifacts, use:

> Trained a U-Net with a ResNet34 encoder for five-class semantic segmentation of Sentinel-2 imagery around Bagré, Burkina Faso, using geographically disjoint train, validation, and test regions; benchmarked against a Random Forest baseline and evaluated with mIoU, per-class IoU, F1, confusion matrices, and qualitative error analysis.

Add the quantified sentence only from `comparison/results.csv`:

> Achieved test mIoU of [VERIFIED U-NET VALUE], compared with [VERIFIED RF VALUE] for the four-band Random Forest baseline, across five merged ESA WorldCover classes.

Keep the limitation in the README, model card, and interview explanation: **scores measure agreement with ESA WorldCover labels, not accuracy against field ground truth.**

## Data references

- [ESA WorldCover datasets](https://github.com/ESA-WorldCover/esa-worldcover-datasets)
- [ESA WorldCover 2021 Product User Manual](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf)
- [Sentinel-2 SR Harmonized in Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [Cloud Score+ in Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED)
- [Bagré irrigation context, FAO](https://www.fao.org/4/x5598e/X5598E04.htm)

## Scope after the MVP

Do not add Sentinel-1, more bands, other encoders, transformers, deployment, or a web application until the complete RF-versus-U-Net pipeline is reproducible and the first results are written up.
