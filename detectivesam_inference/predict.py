from __future__ import annotations

import argparse
import json
from pathlib import Path

from detectivesam_inference.dataset import prepare_sample
from detectivesam_inference.metrics import compute_f1, compute_iou
from detectivesam_inference.runtime import DetectiveSAMRunner, get_repo_root
from detectivesam_inference.visualization import save_prediction_outputs


DEFAULT_DEMO_NAME = "banana_28809"


def resolve_demo_defaults(repo_root: Path) -> tuple[Path | None, Path, Path | None, str]:
    user_demo_target = repo_root / "demo" / "user_image" / "demo_input.png"
    fallback_source = repo_root / "demo" / "cocoglide" / "source" / f"{DEFAULT_DEMO_NAME}.png"
    fallback_target = repo_root / "demo" / "cocoglide" / "target" / f"{DEFAULT_DEMO_NAME}.png"
    fallback_mask = repo_root / "demo" / "cocoglide" / "mask" / f"{DEFAULT_DEMO_NAME}.png"

    if user_demo_target.exists():
        return None, user_demo_target, None, "single_image"
    return fallback_source, fallback_target, fallback_mask, "pair"


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description="Run DetectiveSAM on a single source/target pair.")
    parser.add_argument(
        "--checkpoint",
        default="detective_sam",
        help="Checkpoint path or alias. Built-in aliases: detective_sam, detective_sam_sota.",
    )
    parser.add_argument("--source", default=None, help="Optional source image. If omitted, target is reused as source.")
    parser.add_argument(
        "--target",
        default=None,
        help="Target image. If omitted, uses demo/user_image/demo_input.png when present, else falls back to the bundled CocoGlide pair.",
    )
    parser.add_argument("--mask", default=None, help="Optional ground-truth mask for metrics.")
    parser.add_argument("--output-dir", default=str(repo_root / "outputs" / "predict_demo"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def resolve_input_paths(args: argparse.Namespace, repo_root: Path) -> tuple[Path, Path, Path | None, str]:
    demo_source, demo_target, demo_mask, demo_mode = resolve_demo_defaults(repo_root)
    if args.target is not None:
        target_path = Path(args.target)
        source_path = Path(args.source) if args.source else target_path
        mask_path = Path(args.mask) if args.mask else None
        return source_path, target_path, mask_path, "custom"

    target_path = demo_target
    source_path = Path(args.source) if args.source else (demo_source or target_path)
    mask_path = Path(args.mask) if args.mask else demo_mask
    return source_path, target_path, mask_path, demo_mode


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    source_path, target_path, mask_path, demo_mode = resolve_input_paths(args, repo_root)
    reference_mode = "pair" if source_path != target_path else "target_as_source"

    runner = DetectiveSAMRunner(checkpoint_path=args.checkpoint, device=args.device)
    sample = prepare_sample(
        source_path=source_path,
        target_path=target_path,
        mask_path=mask_path,
        img_size=runner.config.img_size,
        perturbation_type=runner.config.perturbation_type,
        perturbation_intensity=runner.config.perturbation_intensity,
    )
    prediction = runner.predict_sample(sample, threshold=args.threshold)

    gt_mask = sample.mask.squeeze().numpy().astype("uint8") if sample.mask is not None else None
    summary = {
        "sample": sample.name,
        "checkpoint": str(runner.checkpoint_path.resolve()),
        "demo_mode": demo_mode,
        "reference_mode": reference_mode,
        "source": str(sample.source_path),
        "target": str(sample.target_path),
        "mask": str(sample.mask_path) if sample.mask_path is not None else None,
        "threshold": args.threshold,
        "metrics": {
            "iou": compute_iou(prediction.pred_mask, gt_mask) if gt_mask is not None else None,
            "f1": compute_f1(prediction.pred_mask, gt_mask) if gt_mask is not None else None,
        },
    }

    output_dir = Path(args.output_dir)
    save_prediction_outputs(
        output_dir=output_dir,
        name=sample.name,
        source_image=sample.source_image,
        target_image=sample.target_image,
        probability_map=prediction.probability,
        pred_mask=prediction.pred_mask,
        gt_mask=gt_mask,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{sample.name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
