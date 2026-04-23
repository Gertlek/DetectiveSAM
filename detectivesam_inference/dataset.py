from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sam2.utils.transforms import SAM2Transforms
from torch.utils.data import Dataset

from detectivesam_inference.perturbations import (
    add_gaussian_noise_deterministic,
    apply_blur_to_image_tensor,
    apply_jpeg_compression_to_tensor,
)


LEGACY_CONTRASTIVE_FLAG = False


@dataclass
class PreparedSample:
    name: str
    source_path: Path
    target_path: Path
    mask_path: Path | None
    source_image: Image.Image
    target_image: Image.Image
    orig: torch.Tensor
    streams: list[torch.Tensor]
    mask: torch.Tensor | None


def parse_perturbation_types(perturbation_type: str) -> list[str]:
    if perturbation_type == "none":
        return []
    if "+" in perturbation_type:
        return [item.strip() for item in perturbation_type.split("+")]
    if "/" in perturbation_type:
        return [item.strip() for item in perturbation_type.split("/")]
    return [perturbation_type.strip()]


def compute_perturbation_params(perturbation_intensity: float) -> dict[str, float | int]:
    return {
        "blur_sigma": perturbation_intensity * 2.0,
        "jpeg_quality": max(10, int(95 - (perturbation_intensity * 56.67))),
        "noise_std": perturbation_intensity * 0.2,
    }


def create_combined_mask(
    mask_rgba: np.ndarray,
) -> np.ndarray:
    if mask_rgba.ndim == 2:
        return (mask_rgba // 255).astype(np.uint8)

    if mask_rgba.ndim == 3 and mask_rgba.shape[2] == 4:
        alpha = mask_rgba[:, :, 3]
        alpha_is_opaque = alpha.sum() == alpha.size * 255
        if alpha_is_opaque:
            foreground = (mask_rgba[:, :, 0] > 0).astype(np.uint8)
            return cv2.resize(foreground, (512, 512), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        _, binary = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        return (1 - (binary // 255)).astype(np.uint8)

    return (mask_rgba[:, :, 0] > 0).astype(np.uint8)


def resize_triplet(
    source_image: Image.Image,
    target_image: Image.Image,
    mask_image: Image.Image | None,
    img_size: int,
) -> tuple[Image.Image, Image.Image, Image.Image | None]:
    source_resized = source_image.resize((img_size, img_size), Image.BILINEAR)
    target_resized = target_image.resize((img_size, img_size), Image.BILINEAR)
    if mask_image is None:
        return source_resized, target_resized, None
    return source_resized, target_resized, mask_image.resize((img_size, img_size), Image.NEAREST)


def build_streams(
    target_image: Image.Image,
    perturbation_type: str,
    perturbation_intensity: float,
    seed: int,
) -> list[torch.Tensor]:
    perturbations = parse_perturbation_types(perturbation_type)
    params = compute_perturbation_params(perturbation_intensity)
    orig_tensor = TF.to_tensor(target_image)

    streams: list[torch.Tensor] = []
    for perturbation in perturbations:
        if perturbation == "gaussian_blur":
            streams.append(apply_blur_to_image_tensor(orig_tensor, sigma=float(params["blur_sigma"])))
        elif perturbation == "jpeg_compression":
            streams.append(apply_jpeg_compression_to_tensor(orig_tensor, quality=int(params["jpeg_quality"])))
        elif perturbation == "gaussian_noise":
            streams.append(
                add_gaussian_noise_deterministic(
                    orig_tensor,
                    std=float(params["noise_std"]),
                    seed=seed + len(streams),
                )
            )
        else:
            raise ValueError(f"Unsupported perturbation type: {perturbation}")
    return streams


def build_sample_seed(
    source_path: Path,
    target_path: Path,
    mask_path: Path | None,
    perturbation_type: str,
    perturbation_intensity: float,
) -> int:
    sample_key = "|".join(
        [
            source_path.parent.parent.name,
            source_path.stem,
            target_path.parent.parent.name,
            target_path.stem,
            mask_path.stem if mask_path is not None else "no-mask",
            perturbation_type,
            f"{perturbation_intensity:.8f}",
            str(LEGACY_CONTRASTIVE_FLAG),
        ]
    )
    digest = hashlib.sha256(sample_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)


def prepare_sample(
    source_path: str | Path,
    target_path: str | Path,
    mask_path: str | Path | None,
    img_size: int,
    perturbation_type: str,
    perturbation_intensity: float,
) -> PreparedSample:
    source_path = Path(source_path)
    target_path = Path(target_path)
    mask_path = Path(mask_path) if mask_path is not None else None

    source_image = Image.open(source_path).convert("RGB")
    target_image = Image.open(target_path).convert("RGB")
    mask_image = Image.open(mask_path) if mask_path is not None else None
    source_image, target_image, mask_image = resize_triplet(source_image, target_image, mask_image, img_size)

    sample_seed = build_sample_seed(
        source_path=source_path,
        target_path=target_path,
        mask_path=mask_path,
        perturbation_type=perturbation_type,
        perturbation_intensity=perturbation_intensity,
    )
    transforms = SAM2Transforms(resolution=img_size, mask_threshold=0.0)

    orig_tensor = TF.to_tensor(target_image)
    orig = transforms.transforms(orig_tensor).unsqueeze(0).squeeze(0)
    streams_raw = build_streams(
        target_image=target_image,
        perturbation_type=perturbation_type,
        perturbation_intensity=perturbation_intensity,
        seed=sample_seed,
    )
    streams = [transforms.transforms(stream).unsqueeze(0).squeeze(0) for stream in streams_raw]

    mask_tensor = None
    if mask_image is not None:
        binary_mask = create_combined_mask(
            mask_rgba=np.array(mask_image),
        )
        if binary_mask.shape != (img_size, img_size):
            binary_mask = cv2.resize(binary_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.tensor(binary_mask, dtype=torch.float32).unsqueeze(0)

    return PreparedSample(
        name=target_path.stem,
        source_path=source_path,
        target_path=target_path,
        mask_path=mask_path,
        source_image=source_image,
        target_image=target_image,
        orig=orig,
        streams=streams,
        mask=mask_tensor,
    )


class PairDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        img_size: int,
        perturbation_type: str,
        perturbation_intensity: float,
        max_samples: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.source_dir = self.root_dir / "source"
        self.target_dir = self.root_dir / "target"
        self.mask_dir = self.root_dir / "mask"
        self.img_size = img_size
        self.perturbation_type = perturbation_type
        self.perturbation_intensity = perturbation_intensity

        if not self.source_dir.exists() or not self.target_dir.exists():
            raise FileNotFoundError(f"{root_dir} must contain source/ and target/ directories")

        target_files = sorted(
            path
            for path in self.target_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if max_samples is not None:
            target_files = target_files[:max_samples]
        self.target_files = target_files

    def __len__(self) -> int:
        return len(self.target_files)

    def __getitem__(self, index: int) -> PreparedSample:
        target_path = self.target_files[index]
        source_path = self.source_dir / target_path.name
        if not source_path.exists():
            png_fallback = self.source_dir / f"{target_path.stem}.png"
            jpg_fallback = self.source_dir / f"{target_path.stem}.jpg"
            source_path = png_fallback if png_fallback.exists() else jpg_fallback
        if not source_path.exists():
            raise FileNotFoundError(f"Could not find source image for {target_path.name}")

        mask_candidate = self.mask_dir / f"{target_path.stem}.png"
        mask_path = mask_candidate if mask_candidate.exists() else None
        return prepare_sample(
            source_path=source_path,
            target_path=target_path,
            mask_path=mask_path,
            img_size=self.img_size,
            perturbation_type=self.perturbation_type,
            perturbation_intensity=self.perturbation_intensity,
        )
