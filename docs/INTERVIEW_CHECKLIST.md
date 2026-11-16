# Interview defense checklist

Answer these from your actual run, not from generic deep-learning claims.

1. Why does U-Net use spatial context while the RF baseline classifies each pixel independently?
2. What do the encoder and decoder contribute? What is lost through downsampling and recovered through skip connections?
3. Why use a pretrained ResNet34 encoder? How is the first convolution adapted for four channels? Inspect the installed library implementation before answering.
4. Why can random patch splitting inflate segmentation scores? Show your split map and buffer rule.
5. Why is mIoU the primary metric? What can overall accuracy hide under class imbalance?
6. Which classes are absent or rare in each split? Use `metadata.json` and `data_summary.png`.
7. Why combine Dice and cross-entropy? What evidence would justify changing the loss?
8. What does a 256 by 256 patch represent on the ground at 10 m resolution? What context can it contain or miss?
9. Why do the scores measure agreement with WorldCover rather than field accuracy?
10. Did U-Net outperform RF overall and for every class? If not, explain the result without calling it a failure.
11. What do the training and validation curves show about convergence and overfitting?
12. Which errors are visible in the fixed, randomly selected test figures? Consider boundaries, mixed pixels, small fields, water, and built-up areas.
13. Why does the primary RF baseline use the same four bands as U-Net? How would NDVI/NDWI change the comparison?
14. What would you test next: field labels, another season, another AOI, or SAR? Choose one and justify it.

## Claim boundary

Safe after completion: “Trained a U-Net for five-class semantic segmentation of Sentinel-2 imagery and benchmarked it against a Random Forest using spatially held-out test data.”

Not supported by this MVP: production deployment, field-validated accuracy, a novel architecture, state-of-the-art performance, or generalization across Burkina Faso.
