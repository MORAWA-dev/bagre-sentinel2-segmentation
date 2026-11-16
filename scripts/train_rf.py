import argparse

from bagre_segmentation.config import load_config
from bagre_segmentation.random_forest import train_random_forest


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the four-band RF pixel baseline.")
    parser.add_argument("--config", default="configs/bagre.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metrics = train_random_forest(args.data_dir, args.output_dir, load_config(args.config))
    print("RF test mIoU:", metrics["mean_iou"])


if __name__ == "__main__":
    main()
