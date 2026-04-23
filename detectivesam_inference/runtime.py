from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from detectivesam_inference.checkpoint import (
    InferenceConfig,
    load_inference_config,
    resolve_checkpoint_path,
    resolve_repo_path,
)
from detectivesam_inference.dataset import PreparedSample
from detectivesam_inference.models.forgerylocalizer import ForgeryLocalizer


@dataclass(frozen=True)
class PredictionResult:
    probability: np.ndarray
    pred_mask: np.ndarray


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def select_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_sam2_config(config_dir: str | Path) -> None:
    config_dir = str(Path(config_dir).resolve())
    hydra = GlobalHydra.instance()
    current_dir = getattr(initialize_sam2_config, "_current_dir", None)
    if hydra.is_initialized():
        if current_dir == config_dir:
            return
        hydra.clear()
    initialize_config_dir(config_dir=config_dir, version_base=None)
    initialize_sam2_config._current_dir = config_dir


class DetectiveSAMRunner:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.repo_root = get_repo_root()
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path, self.repo_root)
        self.device = select_device(device)
        self.config = load_inference_config(self.checkpoint_path)
        self.model = self._load_model()

    def _load_model(self) -> ForgeryLocalizer:
        sam_config_path = resolve_repo_path(self.config.sam_config_file, self.repo_root)
        sam_checkpoint_path = resolve_repo_path(self.config.sam_checkpoint, self.repo_root)
        initialize_sam2_config(sam_config_path.parent)

        model = ForgeryLocalizer(
            sam_config=sam_config_path.name,
            sam_checkpoint=str(sam_checkpoint_path),
            prompt_dim=self.config.prompt_dim,
            downscale=self.config.downscale,
            dropout_rate=self.config.dropout_rate,
            max_streams=self.config.max_streams,
            device=str(self.device),
        ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def autocast_context(self):
        if self.device.type == "cuda":
            return torch.amp.autocast(device_type="cuda")
        return nullcontext()

    def predict_sample(
        self,
        sample: PreparedSample,
        threshold: float = 0.5,
    ) -> PredictionResult:
        orig = sample.orig.unsqueeze(0).to(self.device)
        streams = [stream.unsqueeze(0).to(self.device) for stream in sample.streams]

        with torch.inference_mode():
            with self.autocast_context():
                logits = self.model(orig, streams)

        probability = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
        pred_mask = (probability > threshold).astype("uint8")
        return PredictionResult(probability=probability, pred_mask=pred_mask)
