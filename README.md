# DetectiveSAM

DetectiveSAM is an inference-only image forgery localization bundle built around SAM2. This GitHub repo keeps the lightweight code, configs, and demo assets. The full runnable bundle with model weights is hosted on Hugging Face:

- https://huggingface.co/Gertlek/DetectiveSAM

## What is bundled

- Inference checkpoint configs under `checkpoints/`
- SAM2 config under `sam2configs/`
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

Then download the large weights from Hugging Face:

```bash
pip install -U huggingface_hub
hf download Gertlek/DetectiveSAM \
  checkpoints/model_epoch22_batch999_score1.1114.pth \
  checkpoints/detective_sam_sota.pth \
  sam2configs/sam2.1_hiera_base_plus.pt \
  --local-dir .
```

The expected checkpoint paths are also documented in `checkpoints/README.md` and `sam2configs/README.md`.

## Hugging Face Usage

For the simplest setup, clone the Hugging Face repo directly:

```bash
git lfs install
git clone https://huggingface.co/Gertlek/DetectiveSAM
cd DetectiveSAM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m detectivesam_inference.predict \
  --checkpoint detective_sam \
  --output-dir outputs/poster_baseline
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

### 4. Bundled CocoGlide subset sweep

Use this to evaluate the bundled banana and train CocoGlide demo pairs.

```bash
python -m detectivesam_inference.evaluate \
  --checkpoint detective_sam \
  --dataset-root demo/cocoglide \
  --output-dir outputs/poster_eval_cocoglide \
  --num-visualizations 2
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
- The GitHub repo does not bundle `.pth` or `.pt` weight files; download them from the Hugging Face repo before running inference.
- This repo does not include training code or training-only dependencies.
