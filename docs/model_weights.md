# Model Weights

This repository intentionally does not commit large detector or segmentation weights.

The original local prototype used:

- GroundingDINO Swin-T weights: `weights/groundingdino_swint_ogc.pth`
- EfficientSAM ViT-S weights: `weights/efficient_sam_vits.pt`

The public code can run in two modes:

1. Lightweight Rubik fallback mode, which detects saturated cube stickers and is enough for dry-run verification on the included demo video.
2. Optional open-vocabulary mode, which uses GroundingDINO when the package and weight file are available.

Use `scripts/download_weights.sh` as a helper for the GroundingDINO checkpoint. EfficientSAM is optional in this public showcase version.

Recommended local layout:

```text
weights/
├── groundingdino_swint_ogc.pth
└── efficient_sam_vits.pt
```

The `weights/` directory is ignored by Git.
