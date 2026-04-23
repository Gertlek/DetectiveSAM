from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam2.modeling.backbones.image_encoder import ImageEncoder
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms

from detectivesam_inference.models.adapters import MaskAdapter, SharedAdapter


MODEL_CHANNELS = [256, 32, 64]
MASK_ADAPTER_RESOLUTION = (128, 128)
FeaturePyramid = list[torch.Tensor]
StreamPyramid = list[list[torch.Tensor]]


class ForgeryLocalizer(nn.Module):
    """Inference-only DetectiveSAM model."""

    def __init__(
        self,
        sam_config: str,
        sam_checkpoint: str,
        prompt_dim: int = 256,
        output_resolution: tuple[int, int] = (512, 512),
        downscale: int = 4,
        dropout_rate: float = 0.1,
        max_streams: int = 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.output_resolution = output_resolution

        sam = self._build_sam(
            sam_config=sam_config,
            sam_checkpoint=sam_checkpoint,
            output_resolution=output_resolution,
            device=device,
        )
        self.no_mem_embed = sam.no_mem_embed if hasattr(sam, "no_mem_embed") else None
        self.encoder: ImageEncoder = sam.image_encoder
        self.decoder: MaskDecoder = sam.sam_mask_decoder
        self.sam_prompt_encoder: PromptEncoder = sam.sam_prompt_encoder
        self.sam_prompt_encoder.image_embedding_size = MASK_ADAPTER_RESOLUTION

        self._freeze_sam_components()
        self.adapters = SharedAdapter(
            hidden_dim=prompt_dim,
            in_channels_list=MODEL_CHANNELS,
            dropout_rate=dropout_rate,
            max_streams=max_streams,
        )
        self.mask_adapter = MaskAdapter(
            hidden_dim=prompt_dim,
            output_resolution=MASK_ADAPTER_RESOLUTION,
            downscale=downscale,
            in_channels_list=MODEL_CHANNELS,
            dropout_rate=dropout_rate,
            max_streams=max_streams,
        )
        self.transforms = SAM2Transforms(resolution=sam.image_size, mask_threshold=0.0)

    @staticmethod
    def _build_sam(
        *,
        sam_config: str,
        sam_checkpoint: str,
        output_resolution: tuple[int, int],
        device: str,
    ) -> SAM2Base:
        sam: SAM2Base = build_sam2(sam_config, sam_checkpoint, device=device)
        sam.image_size = output_resolution[0]
        return sam

    def _freeze_sam_components(self) -> None:
        for module in (self.encoder, self.decoder, self.sam_prompt_encoder):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def _project_sam_features(self, backbone_features: FeaturePyramid) -> tuple[torch.Tensor, FeaturePyramid]:
        image_embeddings = backbone_features[-1]
        if self.no_mem_embed is not None:
            image_embeddings = image_embeddings + self.no_mem_embed.reshape(1, 256, 1, 1).detach()

        high_res_features = [
            self.decoder.conv_s0(backbone_features[0]),
            self.decoder.conv_s1(backbone_features[1]),
        ]
        return image_embeddings, high_res_features

    def _encode_original_and_streams(
        self,
        orig: torch.Tensor,
        streams: list[torch.Tensor],
    ) -> tuple[FeaturePyramid, StreamPyramid]:
        orig_backbone_features = self.encoder(orig)["backbone_fpn"]
        orig_image_embeddings, orig_high_res_features = self._project_sam_features(orig_backbone_features)

        stream_image_embeddings: list[torch.Tensor] = []
        stream_high_res_features_0: list[torch.Tensor] = []
        stream_high_res_features_1: list[torch.Tensor] = []
        for stream in streams:
            stream_backbone_features = self.encoder(stream)["backbone_fpn"]
            stream_image_embedding, stream_high_res_features = self._project_sam_features(stream_backbone_features)
            stream_image_embeddings.append(stream_image_embedding)
            stream_high_res_features_0.append(stream_high_res_features[0])
            stream_high_res_features_1.append(stream_high_res_features[1])

        unadapted = [orig_image_embeddings, orig_high_res_features[0], orig_high_res_features[1]]
        streams_unadapted = [stream_image_embeddings, stream_high_res_features_0, stream_high_res_features_1]
        return unadapted, streams_unadapted

    def _apply_adapters(
        self,
        unadapted: FeaturePyramid,
        streams_unadapted: StreamPyramid,
    ) -> FeaturePyramid:
        return [
            self.adapters(streams_unadapted[scale_idx], unadapted[scale_idx], scale_idx)
            for scale_idx in range(len(unadapted))
        ]

    def _prepare_decoder_inputs(
        self,
        adapted: FeaturePyramid,
        mask_prompt: torch.Tensor,
    ) -> tuple[torch.Tensor, FeaturePyramid, torch.Tensor]:
        # Preserve the original interpolation steps so inference stays bit-exact.
        image_embeddings = F.interpolate(
            adapted[0],
            size=adapted[0].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high_res_features = [
            F.interpolate(
                feature,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for feature in adapted[1:]
        ]
        mask_prompt = F.interpolate(
            mask_prompt,
            size=mask_prompt.shape[-2:],
            mode="nearest",
        )
        return image_embeddings, high_res_features, mask_prompt

    def forward(self, orig: torch.Tensor, streams: list[torch.Tensor]) -> torch.Tensor:
        unadapted, streams_unadapted = self._encode_original_and_streams(orig, streams)
        adapted = self._apply_adapters(unadapted, streams_unadapted)

        mask_prompt = self.mask_adapter(adapted, streams_unadapted, unadapted)
        image_embeddings, high_res_features, mask_prompt = self._prepare_decoder_inputs(adapted, mask_prompt)

        self.sam_prompt_encoder.image_embedding_size = image_embeddings.shape[-2:]
        sparse_prompt_embeddings, dense_prompt_embeddings = self.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=mask_prompt,
        )

        mask_logits, _, _, _ = self.decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        return self.transforms.postprocess_masks(mask_logits, torch.Size(self.output_resolution))
