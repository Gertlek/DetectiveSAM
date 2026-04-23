from __future__ import annotations

import math
from pathlib import Path

import pytest

from detectivesam_inference.checkpoint import resolve_checkpoint_path
from detectivesam_inference.dataset import PairDataset, prepare_sample
from detectivesam_inference.metrics import compute_f1, compute_iou, summarize_results
from detectivesam_inference.runtime import DetectiveSAMRunner, get_repo_root


def assert_exact(value: float | None, expected: float | None, *, abs_tol: float = 1e-12) -> None:
    assert value is not None
    assert expected is not None
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=abs_tol)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return get_repo_root()


@pytest.fixture(scope="module")
def baseline_runner() -> DetectiveSAMRunner:
    return DetectiveSAMRunner(checkpoint_path="detective_sam", device="cpu")


@pytest.fixture(scope="module")
def sota_runner() -> DetectiveSAMRunner:
    return DetectiveSAMRunner(checkpoint_path="detective_sam_sota", device="cpu")


def predict_metrics(
    runner: DetectiveSAMRunner,
    *,
    source_path: Path,
    target_path: Path,
    mask_path: Path,
) -> tuple[float, float]:
    sample = prepare_sample(
        source_path=source_path,
        target_path=target_path,
        mask_path=mask_path,
        img_size=runner.config.img_size,
        perturbation_type=runner.config.perturbation_type,
        perturbation_intensity=runner.config.perturbation_intensity,
    )
    prediction = runner.predict_sample(sample, threshold=0.5)
    true_mask = sample.mask.squeeze().numpy().astype("uint8")
    return compute_iou(prediction.pred_mask, true_mask), compute_f1(prediction.pred_mask, true_mask)


def test_checkpoint_alias_resolution(repo_root: Path) -> None:
    assert resolve_checkpoint_path("detective_sam", repo_root) == repo_root / "checkpoints" / "model_epoch22_batch999_score1.1114.pth"
    assert resolve_checkpoint_path("detective_sam_sota", repo_root) == repo_root / "checkpoints" / "detective_sam_sota.pth"


def test_baseline_banana_demo_metrics(repo_root: Path, baseline_runner: DetectiveSAMRunner) -> None:
    demo_root = repo_root / "demo" / "cocoglide"
    iou, f1 = predict_metrics(
        baseline_runner,
        source_path=demo_root / "source" / "banana_28809.png",
        target_path=demo_root / "target" / "banana_28809.png",
        mask_path=demo_root / "mask" / "banana_28809.png",
    )
    assert_exact(iou, 0.8566427949370513)
    assert_exact(f1, 0.9227868680750683)


def test_sota_flux_demo_metrics(repo_root: Path, sota_runner: DetectiveSAMRunner) -> None:
    demo_root = repo_root / "demo" / "flux_test"
    iou, f1 = predict_metrics(
        sota_runner,
        source_path=demo_root / "source" / "548.png",
        target_path=demo_root / "target" / "548.png",
        mask_path=demo_root / "mask" / "548.png",
    )
    assert_exact(iou, 0.8703024868799283)
    assert_exact(f1, 0.9306542583192329)


def test_sota_qwen_demo_metrics(repo_root: Path, sota_runner: DetectiveSAMRunner) -> None:
    demo_root = repo_root / "demo" / "qwen_test"
    iou, f1 = predict_metrics(
        sota_runner,
        source_path=demo_root / "source" / "166.png",
        target_path=demo_root / "target" / "166.png",
        mask_path=demo_root / "mask" / "166.png",
    )
    assert_exact(iou, 0.8297306693388413)
    assert_exact(f1, 0.9069429542203147)


def test_baseline_cocoglide_eval_summary(repo_root: Path, baseline_runner: DetectiveSAMRunner) -> None:
    dataset = PairDataset(
        root_dir=repo_root / "demo" / "cocoglide",
        img_size=baseline_runner.config.img_size,
        perturbation_type=baseline_runner.config.perturbation_type,
        perturbation_intensity=baseline_runner.config.perturbation_intensity,
    )

    per_sample_results: list[dict[str, float | str | None]] = []
    for sample in dataset:
        prediction = baseline_runner.predict_sample(sample, threshold=0.5)
        true_mask = sample.mask.squeeze().numpy().astype("uint8")
        per_sample_results.append(
            {
                "name": sample.name,
                "iou": compute_iou(prediction.pred_mask, true_mask),
                "f1": compute_f1(prediction.pred_mask, true_mask),
            }
        )

    summary = summarize_results(per_sample_results)
    assert summary["num_samples"] == 5
    assert summary["num_samples_with_gt"] == 5
    assert_exact(summary["mean_iou"], 0.5092573070481035)
    assert_exact(summary["mean_f1"], 0.6509390858765342)

    expected_by_name = {
        "airplane_139871": (0.41829717560376584, 0.5898582931686339),
        "banana_28809": (0.8566427949370513, 0.9227868680750683),
        "giraffe_296969": (0.22833093957714018, 0.3717743031951054),
        "train_221213": (0.547253866814856, 0.7073872989458688),
        "tv_453722": (0.49576175830770386, 0.662888665997994),
    }
    for result in per_sample_results:
        expected_iou, expected_f1 = expected_by_name[result["name"]]
        assert_exact(result["iou"], expected_iou)
        assert_exact(result["f1"], expected_f1)
