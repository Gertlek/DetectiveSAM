from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def ensure_uint8(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255
    if mask.max() <= 1.0:
        return (mask * 255).astype(np.uint8)
    return mask.astype(np.uint8)


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(ensure_uint8(mask), mode="L")


def probability_to_image(probability: np.ndarray) -> Image.Image:
    clipped = np.clip(probability, 0.0, 1.0)
    return Image.fromarray((clipped * 255).astype(np.uint8), mode="L")


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    overlay[mask.astype(bool)] = (1.0 - alpha) * overlay[mask.astype(bool)] + alpha * np.array(color, dtype=np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def concat_images(images: list[Image.Image]) -> Image.Image:
    widths, heights = zip(*(image.size for image in images))
    canvas = Image.new("RGB", (sum(widths), max(heights)), color=(255, 255, 255))
    x_offset = 0
    for image in images:
        canvas.paste(image.convert("RGB"), (x_offset, 0))
        x_offset += image.width
    return canvas


def save_prediction_outputs(
    output_dir: str | Path,
    name: str,
    source_image: Image.Image,
    target_image: Image.Image,
    probability_map: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probability_image = probability_to_image(probability_map)
    pred_mask_image = mask_to_image(pred_mask)
    pred_overlay = overlay_mask(target_image, pred_mask, color=(255, 0, 0))

    comparison_images = [
        source_image.convert("RGB"),
        target_image.convert("RGB"),
        ImageOps.colorize(probability_image, black="black", white="white").convert("RGB"),
        pred_overlay,
    ]

    probability_image.save(output_dir / f"{name}_probability.png")
    pred_mask_image.save(output_dir / f"{name}_pred_mask.png")
    pred_overlay.save(output_dir / f"{name}_pred_overlay.png")

    if gt_mask is not None:
        gt_mask_image = mask_to_image(gt_mask)
        gt_overlay = overlay_mask(target_image, gt_mask, color=(0, 255, 0))
        gt_mask_image.save(output_dir / f"{name}_gt_mask.png")
        gt_overlay.save(output_dir / f"{name}_gt_overlay.png")
        comparison_images.append(gt_overlay)

    concat_images(comparison_images).save(output_dir / f"{name}_comparison.png")
