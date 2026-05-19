#!/usr/bin/env bash
set -euo pipefail

mkdir -p weights

if [ ! -f weights/groundingdino_swint_ogc.pth ]; then
  curl -L \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    -o weights/groundingdino_swint_ogc.pth
fi

cat <<'MSG'
GroundingDINO weight check complete.

EfficientSAM is optional for this public showcase version. If you use it,
place efficient_sam_vits.pt under weights/ and keep it out of Git.
MSG
