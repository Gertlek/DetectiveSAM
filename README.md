# DetectiveSAM Poster Demo

This repo is the poster-demo bundle for DetectiveSAM. It is intentionally narrow: inference only, bundled checkpoints, and a small set of ready-to-run examples for live demos.

## What is bundled

- Inference checkpoints under `checkpoints/`
- SAM2 config and weights under `sam2configs/`
- Poster demo pairs under `demo/cocoglide/`, `demo/flux_test/`, and `demo/qwen_test/`
- A drop-in single-image slot at `demo/user_image/demo_input.png`

Built-in checkpoint aliases:

- `detective_sam`
- `detective_sam_sota`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Poster Demo Flows

### 1. Live single-image demo

Place your image at `demo/user_image/demo_input.png`, then run:

```bash
python -m detectivesam_inference.predict \
  --checkpoint detective_sam \
  --output-dir outputs/poster_user_image
```

In this mode the CLI reuses the target image as its own source reference so the demo stays runnable with a single image.

### 2. Bundled baseline example

If `demo/user_image/demo_input.png` is absent, the default `predict` command falls back to the bundled CocoGlide sample `banana_28809`.

```bash
python -m detectivesam_inference.predict \
  --checkpoint detective_sam \
  --output-dir outputs/poster_baseline
```

### 3. Bundled SOTA examples

Flux example:

```bash
python -m detectivesam_inference.predict \
  --checkpoint detective_sam_sota \
  --source demo/flux_test/source/548.png \
  --target demo/flux_test/target/548.png \
  --mask demo/flux_test/mask/548.png \
  --output-dir outputs/poster_flux
```

Qwen example:

```bash
python -m detectivesam_inference.predict \
  --checkpoint detective_sam_sota \
  --source demo/qwen_test/source/166.png \
  --target demo/qwen_test/target/166.png \
  --mask demo/qwen_test/mask/166.png \
  --output-dir outputs/poster_qwen
```

## Outputs

Each `predict` run writes a compact set of visual artifacts plus a JSON summary:

- `<name>_comparison.png`
- `<name>_probability.png`
- `<name>_pred_mask.png`
- `<name>_pred_overlay.png`
- `<name>_summary.json`

If a ground-truth mask is provided, the run also saves:

- `<name>_gt_mask.png`
- `<name>_gt_overlay.png`

The `evaluate` command writes `summary.json` plus a few visualization examples under `visualizations/`.

## Notes

- The runtime selects `cuda` automatically when available and otherwise runs on CPU.
- Checkpoint settings come from the YAML sidecars in `checkpoints/`; you only need the alias or checkpoint path.
- This repo does not include training code or training-only dependencies.
