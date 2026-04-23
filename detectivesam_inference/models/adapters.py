from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


FeaturePyramid = list[torch.Tensor]
StreamPyramid = list[list[torch.Tensor]]


class SharedAdapter(nn.Module):
    """Applies a residual adapter to each feature scale."""

    def __init__(
        self,
        in_channels_list: list[int],
        hidden_dim: int,
        dropout_rate: float = 0.1,
        max_streams: int = 2,
    ) -> None:
        super().__init__()
        max_streams = max(max_streams, 1)

        self.mlps_tune = nn.ModuleList(
            nn.Conv2d(max_streams * channels, hidden_dim, kernel_size=1)
            for channels in in_channels_list
        )
        self.mlps_bottleneck = nn.ModuleList(
            nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1), nn.GELU())
            for _ in in_channels_list
        )
        self.mlp_up = nn.ModuleList(
            nn.Conv2d(hidden_dim, channels, kernel_size=1)
            for channels in in_channels_list
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(
        self,
        stream_features: list[torch.Tensor],
        unadapted: torch.Tensor,
        scale_idx: int,
    ) -> torch.Tensor:
        fused_streams = torch.cat(stream_features, dim=1) if stream_features else unadapted
        hidden = self.mlps_tune[scale_idx](fused_streams)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.mlps_bottleneck[scale_idx](hidden)
        delta = self.mlp_up[scale_idx](hidden)
        return unadapted + delta


class RefineBlock(nn.Module):
    """Refines the coarse mask with low-level features."""

    def __init__(
        self,
        hidden_dim: int,
        low_channels: int,
        out_channels: int = 1,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim + low_channels, hidden_dim, kernel_size=3, padding=1)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(
        self,
        attention_features: torch.Tensor,
        low_features: torch.Tensor,
        coarse_upsampled: torch.Tensor,
    ) -> torch.Tensor:
        refined = torch.cat([attention_features, low_features], dim=1)
        refined = self.conv1(refined)
        refined = self.activation1(refined)
        refined = self.dropout1(refined)
        refined = self.conv2(refined)
        refined = self.activation2(refined)
        refined = self.dropout2(refined)
        delta = self.conv3(refined)
        return coarse_upsampled + delta


class CoarseProcessingBlock(nn.Module):
    """Adds transformer-based coarse reasoning before refinement."""

    def __init__(
        self,
        hidden_dim: int,
        attn_dim: int,
        n_heads: int,
        num_encoder_layers: int,
        dropout_rate: float,
        downscale: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coarse_down = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=downscale, stride=downscale, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
        )
        self.pos_embed_conv = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.pos_dropout = nn.Dropout2d(p=dropout_rate)
        self.feat_proj = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Dropout(p=dropout_rate),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=n_heads,
            dim_feedforward=attn_dim * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.transformer_out = nn.Sequential(
            nn.Linear(attn_dim, hidden_dim),
            nn.Dropout(p=dropout_rate),
        )
        self.residual_gate_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),
        )
        self.cached_pos_encodings: dict[tuple[int, int], torch.Tensor] = {}

    def _generate_pos_encoding(self, height: int, width: int) -> torch.Tensor:
        device = self.pos_embed_conv.weight.device
        y_pos = torch.linspace(-1, 1, height, device=device).view(height, 1).expand(height, width)
        x_pos = torch.linspace(-1, 1, width, device=device).view(1, width).expand(height, width)
        pos_grid = torch.stack([y_pos, x_pos], dim=0).unsqueeze(0)
        return self.pos_embed_conv(pos_grid)

    def _get_positional_encoding(self, batch_size: int, height: int, width: int) -> torch.Tensor:
        key = (height, width)
        device = self.pos_embed_conv.weight.device
        if key not in self.cached_pos_encodings:
            self.cached_pos_encodings[key] = self._generate_pos_encoding(height, width).detach()

        cached_encoding = self.cached_pos_encodings[key]
        if cached_encoding.device != device:
            cached_encoding = cached_encoding.to(device)
            self.cached_pos_encodings[key] = cached_encoding
        return cached_encoding.expand(batch_size, -1, -1, -1)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        coarse_features = self.coarse_down(fused)
        batch_size, _, height, width = coarse_features.shape

        pos_embed = self._get_positional_encoding(batch_size, height, width)
        pos_embed = self.pos_dropout(pos_embed)
        coarse_with_position = coarse_features + pos_embed

        feature_sequence = coarse_with_position.flatten(2).permute(0, 2, 1)
        feature_sequence = self.feat_proj(feature_sequence)
        transformer_output = self.transformer_encoder(feature_sequence)
        hidden = self.transformer_out(transformer_output)
        hidden = hidden.permute(0, 2, 1).view(batch_size, self.hidden_dim, height, width)

        gate_input = torch.cat([hidden, coarse_features], dim=1)
        residual_gate = torch.sigmoid(self.residual_gate_conv(gate_input))
        return residual_gate * hidden + (1 - residual_gate) * coarse_features


class FineProcessingBlock(nn.Module):
    """Produces the coarse mask and uncertainty map."""

    def __init__(self, hidden_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.feature_refinement = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
        )
        self.coarse_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(
        self,
        hidden: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.feature_refinement(hidden)
        coarse_logit = self.coarse_head(hidden)
        uncertainty_logit = self.uncertainty_head(hidden)

        coarse_mask = F.interpolate(
            coarse_logit,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        uncertainty_map = F.interpolate(
            uncertainty_logit,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return hidden, coarse_mask, torch.sigmoid(uncertainty_map)


class FeatureFusionBlockSpatial(nn.Module):
    """Fuses original, adapted, and perturbed features with per-pixel attention."""

    def __init__(
        self,
        in_channels_list: list[int],
        hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        max_streams: int = 2,
        attn_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.num_streams = 2 + max_streams
        self.att_conv = nn.ModuleList()
        self.proj_conv = nn.ModuleList()

        for channels in in_channels_list:
            total_channels = channels * self.num_streams
            mid_channels = max(total_channels // attn_reduction, 8)
            self.att_conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        total_channels,
                        mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=self.num_streams,
                        bias=False,
                    ),
                    nn.GELU(),
                    nn.Conv2d(mid_channels, self.num_streams, kernel_size=1, bias=False),
                )
            )
            self.proj_conv.append(
                nn.Sequential(
                    nn.Conv2d(channels, hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout2d(p=dropout_rate),
                )
            )

        fusion_channels = hidden_dim * len(in_channels_list)
        self.fuse_project = nn.Sequential(
            nn.Conv2d(fusion_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(
        self,
        adapted: FeaturePyramid,
        unadapted: FeaturePyramid,
        streams_unadapted: StreamPyramid,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        fused_scales = []
        for scale_idx, (att_head, projection) in enumerate(zip(self.att_conv, self.proj_conv)):
            streams = [adapted[scale_idx], unadapted[scale_idx], *streams_unadapted[scale_idx]]
            logits = att_head(torch.cat(streams, dim=1))
            weights = F.softmax(logits, dim=1).unsqueeze(2)
            fused = (torch.stack(streams, dim=1) * weights).sum(dim=1)
            fused = projection(fused)
            fused = F.interpolate(
                fused,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            fused_scales.append(fused)

        return self.fuse_project(torch.cat(fused_scales, dim=1))


class MaskAdapter(nn.Module):
    """Builds the prompt mask passed into the SAM decoder."""

    def __init__(
        self,
        hidden_dim: int = 256,
        downscale: int = 16,
        output_resolution: tuple[int, int] = (128, 128),
        in_channels_list: list[int] | None = None,
        attn_dim: int = 16,
        n_heads: int = 4,
        num_encoder_layers: int = 2,
        dropout_rate: float = 0.1,
        max_streams: int = 2,
    ) -> None:
        super().__init__()
        channels = in_channels_list or [256, 32, 64]
        self.downscale = downscale
        self.output_resolution = output_resolution

        self.feature_fusion = FeatureFusionBlockSpatial(
            in_channels_list=channels,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            max_streams=max_streams,
        )
        self.coarse_processor = CoarseProcessingBlock(
            hidden_dim=hidden_dim,
            attn_dim=attn_dim,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            dropout_rate=dropout_rate,
            downscale=downscale,
        )
        self.fine_processor = FineProcessingBlock(hidden_dim, dropout_rate)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine_head = RefineBlock(
            hidden_dim=hidden_dim,
            low_channels=32,
            out_channels=1,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        adapted: FeaturePyramid,
        streams_unadapted: StreamPyramid,
        unadapted: FeaturePyramid,
    ) -> torch.Tensor:
        output_height, output_width = self.output_resolution
        output_size = (output_height, output_width)
        coarse_size = (output_height // self.downscale, output_width // self.downscale)

        fused = self.feature_fusion(adapted, unadapted, streams_unadapted, output_size)
        hidden = self.coarse_processor(fused)
        if hidden.shape[-2:] != coarse_size:
            hidden = F.adaptive_avg_pool2d(hidden, coarse_size)

        hidden, coarse_mask, uncertainty_map = self.fine_processor(hidden, output_size)
        attention_features = F.interpolate(hidden, size=output_size, mode="bilinear", align_corners=False)
        low_features = F.interpolate(unadapted[1], size=output_size, mode="bilinear", align_corners=False)
        refined_mask = self.refine_head(attention_features, low_features, coarse_mask)

        spatial_gate = self.spatial_gate(torch.cat([coarse_mask, uncertainty_map], dim=1))
        return spatial_gate * refined_mask + (1 - spatial_gate) * coarse_mask
