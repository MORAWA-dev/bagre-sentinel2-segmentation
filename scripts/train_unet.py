import argparse

from bagre_segmentation.config import load_config
from bagre_segmentation.training import train_unet


def main():
    parser = argparse.ArgumentParser(description="Train U-Net, evaluate the best checkpoint, and save artifacts.")
    parser.add_argument("--config", default="configs/bagre.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None, help="Optional path to last.pt")
    args = parser.parse_args()
    metrics = train_unet(args.data_dir, args.output_dir, load_config(args.config), args.resume)
    print("U-Net test mIoU:", metrics["mean_iou"])


if __name__ == "__main__":
    main()
