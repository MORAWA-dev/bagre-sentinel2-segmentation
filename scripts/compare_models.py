import argparse

from bagre_segmentation.compare import compare_models
from bagre_segmentation.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Create the final benchmark table and figures.")
    parser.add_argument("--config", default="configs/bagre.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--unet-dir", required=True)
    parser.add_argument("--rf-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    compare_models(
        args.data_dir, args.unet_dir, args.rf_dir, args.output_dir,
        seed=int(config["seed"]), num_examples=int(config["evaluation"]["num_examples"])
    )
    print(f"Comparison saved to {args.output_dir}")


if __name__ == "__main__":
    main()
