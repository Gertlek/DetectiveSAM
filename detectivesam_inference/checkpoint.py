from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CHECKPOINT = Path("checkpoints/model_epoch22_batch999_score1.1114.pth")
CHECKPOINT_ALIASES = {
    "detective_sam": DEFAULT_CHECKPOINT,
    "detective_sam_sota": Path("checkpoints/detective_sam_sota.pth"),
}


@dataclass(frozen=True)
class InferenceConfig:
    img_size: int
    prompt_dim: int
    downscale: int
    dropout_rate: float
    perturbation_type: str
    perturbation_intensity: float
    sam_config_file: str
    sam_checkpoint: str

    @property
    def max_streams(self) -> int:
        return count_perturbation_streams(self.perturbation_type)


def resolve_checkpoint_path(checkpoint_value: str | Path | None, repo_root: str | Path) -> Path:
    repo_root = Path(repo_root)
    if checkpoint_value is None:
        return repo_root / DEFAULT_CHECKPOINT

    checkpoint_str = str(checkpoint_value)
    if checkpoint_str in CHECKPOINT_ALIASES:
        return repo_root / CHECKPOINT_ALIASES[checkpoint_str]

    checkpoint_path = Path(checkpoint_value)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    if checkpoint_path.exists():
        return checkpoint_path.resolve()

    repo_candidate = repo_root / checkpoint_path
    if repo_candidate.exists():
        return repo_candidate

    aliased_checkpoint = repo_root / "checkpoints" / f"{checkpoint_str}.pth"
    if aliased_checkpoint.exists():
        return aliased_checkpoint
    return repo_candidate


def resolve_repo_path(path_value: str | Path, repo_root: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    repo_root = Path(repo_root)
    direct = repo_root / path
    if direct.exists():
        return direct

    sam_config = repo_root / "sam2configs" / path.name
    if sam_config.exists():
        return sam_config
    return direct


def load_inference_config(checkpoint_path: str | Path) -> InferenceConfig:
    params = _load_params_file(checkpoint_path)
    return InferenceConfig(
        img_size=int(_resolve_param(params, "img_size", section="training_config", default=512)),
        prompt_dim=int(
            _resolve_param(
                params,
                "prompt_dim",
                section="model_config",
                default=_resolve_param(params, "prompt", section="model_config", default=128),
            )
        ),
        downscale=int(_resolve_param(params, "downscale", section="model_config", default=16)),
        dropout_rate=float(
            _resolve_param(
                params,
                "dropout_rate",
                section="model_config",
                default=_resolve_param(params, "dropout", section="model_config", default=0.1),
            )
        ),
        perturbation_type=str(_resolve_param(params, "perturbation_type", section="data_config", default="none")),
        perturbation_intensity=float(
            _resolve_param(params, "perturbation_intensity", section="data_config", default=0.0)
        ),
        sam_config_file=str(
            _resolve_param(params, "sam_config_file", section="sam_config", default="sam2.1_hiera_b+.yaml")
        ),
        sam_checkpoint=str(
            _resolve_param(
                params,
                "sam_checkpoint",
                section="sam_config",
                default="sam2configs/sam2.1_hiera_base_plus.pt",
            )
        ),
    )


def count_perturbation_streams(perturbation_type: str) -> int:
    if perturbation_type == "none":
        return 0
    if "+" in perturbation_type:
        return len(perturbation_type.split("+"))
    if "/" in perturbation_type:
        return len(perturbation_type.split("/"))
    return 1


def _load_params_file(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    candidate_paths = [
        checkpoint_path.with_name(f"{checkpoint_path.stem}_params.yaml"),
        checkpoint_path.parent / "model_params.yaml",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            if not isinstance(loaded, dict):
                raise ValueError(f"Checkpoint params file must deserialize to a mapping: {candidate}")
            return loaded
    raise FileNotFoundError(
        f"Could not find a params file for checkpoint {checkpoint_path}. "
        f"Checked: {', '.join(str(path) for path in candidate_paths)}"
    )


def _resolve_param(
    params: dict[str, Any],
    key: str,
    *,
    section: str,
    default: Any,
) -> Any:
    if key in params:
        return params[key]
    return params.get(section, {}).get(key, default)
