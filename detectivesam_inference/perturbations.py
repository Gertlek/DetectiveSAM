from __future__ import annotations

import io

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def create_spatial_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    x_pos, y_pos = torch.meshgrid(coords, coords, indexing="ij")
    gaussian = torch.exp(-(x_pos**2 + y_pos**2) / (2 * sigma**2))
    gaussian /= gaussian.sum()
    return gaussian.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def apply_spatial_gaussian_blur(
    video_tensor: torch.Tensor,
    sigma: float,
    kernel_sizes: tuple[int, ...] = (5,),
) -> torch.Tensor:
    batch_size, channels, frames, height, width = video_tensor.shape
    blurred = video_tensor
    for kernel_size in kernel_sizes:
        kernel = create_spatial_gaussian_kernel(kernel_size, sigma, channels, video_tensor.device)
        padding = kernel_size // 2
        reshaped = blurred.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        reshaped = F.conv2d(reshaped, kernel, padding=padding, groups=channels)
        blurred = reshaped.reshape(batch_size, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    return blurred


def add_gaussian_noise_deterministic(
    image: torch.Tensor,
    *,
    mean: float = 0.0,
    std: float = 0.1,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=image.device)
    generator.manual_seed(seed)
    noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype) * std + mean
    return torch.clamp(image + noise, 0.0, 1.0)


def apply_blur_to_image_tensor(image_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return image_tensor.clone()
    video_tensor = image_tensor.unsqueeze(0).unsqueeze(2)
    blurred = apply_spatial_gaussian_blur(video_tensor, sigma=sigma, kernel_sizes=(5,))
    return blurred.squeeze(0).squeeze(1)


def apply_jpeg_compression(image: Image.Image, quality: int = 75) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_jpeg_compression_to_tensor(image_tensor: torch.Tensor, quality: int) -> torch.Tensor:
    image = TF.to_pil_image(image_tensor)
    return TF.to_tensor(apply_jpeg_compression(image, quality=quality))
