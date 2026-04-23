from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from detectivesam_inference.dataset import PairDataset
from detectivesam_inference.metrics import compute_f1, compute_iou, summarize_results
from detectivesam_inference.runtime import DetectiveSAMRunner, get_repo_root
from detectivesam_inference.visualization import save_prediction_outputs


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description="Evaluate DetectiveSAM on a dataset root with source/target/mask folders.")
    parser.add_argument(
        "--checkpoint",
        default="detective_sam",
        help="Checkpoint path or alias. Built-in aliases: detective_sam, detective_sam_sota.",
    )
    parser.add_argument("--dataset-root", default=str(repo_root / "demo" / "cocoglide"))
    parser.add_argument("--output-dir", default=str(repo_root / "outputs" / "eval_demo"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-visualizations", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = DetectiveSAMRunner(checkpoint_path=args.checkpoint, device=args.device)
    dataset = PairDataset(
        root_dir=args.dataset_root,
        img_size=runner.config.img_size,
        perturbation_type=runner.config.perturbation_type,
        perturbation_intensity=runner.config.perturbation_intensity,
        max_samples=args.max_samples,
    )

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    per_sample_results: list[dict[str, float | str | None]] = []
    for index, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        prediction = runner.predict_sample(sample, threshold=args.threshold)
        gt_mask = sample.mask.squeeze().numpy().astype("uint8") if sample.mask is not None else None
        per_sample_results.append(
            {
                "name": sample.name,
                "iou": compute_iou(prediction.pred_mask, gt_mask) if gt_mask is not None else None,
                "f1": compute_f1(prediction.pred_mask, gt_mask) if gt_mask is not None else None,
            }
        )

        if index < args.num_visualizations:
            save_prediction_outputs(
                output_dir=vis_dir,
                name=sample.name,
                source_image=sample.source_image,
                target_image=sample.target_image,
                probability_map=prediction.probability,
                pred_mask=prediction.pred_mask,
                gt_mask=gt_mask,
            )

    payload = {
        "checkpoint": str(runner.checkpoint_path.resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "threshold": args.threshold,
        "summary": summarize_results(per_sample_results),
        "samples": per_sample_results,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload["summary"], indent=2))
    print(f"Detailed results written to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
