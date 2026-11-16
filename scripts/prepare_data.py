import argparse

from bagre_segmentation.config import json_safe, load_config, save_json
from bagre_segmentation.prepare import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Align, remap, spatially split, and tile Bagre rasters.")
    parser.add_argument("--config", default="configs/bagre.yaml")
    parser.add_argument("--image", required=True, help="Four-band Sentinel-2 GeoTIFF")
    parser.add_argument("--labels", required=True, help="ESA WorldCover GeoTIFF")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    metadata = prepare_dataset(args.image, args.labels, args.output_dir, config)
    save_json(json_safe(metadata), f"{args.output_dir}/metadata.json")
    print("Prepared splits:", metadata["split_counts"])


if __name__ == "__main__":
    main()
