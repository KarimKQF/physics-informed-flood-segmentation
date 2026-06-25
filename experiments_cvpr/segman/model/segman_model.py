"""SegMAN (CVPR 2025) segmentor for Sen1Floods11 flood/water segmentation.

Wraps the *official* SegMAN encoder + MMSCopE decoder (loaded via pure-PyTorch
shims) into a single ``nn.Module`` that:

* accepts a 15-channel input (Sentinel-2 13-band + Sentinel-1 2-band SAR) via an
  inflated stem,
* returns logits ``[B, num_classes, H, W]`` upsampled to the input/mask
  resolution (so the loss is computed at full resolution),
* keeps the architecture fixed across all loss variants.

The global attention in stage 4 uses a learned position bias tied to a fixed
``image_size``; Sen1Floods11 tiles are 512x512, so ``image_size`` must equal the
training crop (default 512).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from segman_kernels.compat import load_segman_decoder_module, load_segman_encoder_module

# variant -> (encoder builder name, encoder embed_dims, decoder channels, decoder feat_proj_dim)
VARIANTS: dict[str, dict[str, Any]] = {
    "t": {"enc": "SegMANEncoder_t", "embed_dims": [32, 64, 144, 192], "channels": 128, "feat_proj_dim": 192},
    "s": {"enc": "SegMANEncoder_s", "embed_dims": [64, 144, 288, 512], "channels": 144, "feat_proj_dim": 288},
    "b": {"enc": "SegMANEncoder_b", "embed_dims": [96, 160, 364, 560], "channels": 180, "feat_proj_dim": 320},
    "l": {"enc": "SegMANEncoder_l", "embed_dims": [96, 192, 432, 640], "channels": 224, "feat_proj_dim": 432},
}


class SegMANSegmentor(nn.Module):
    def __init__(
        self,
        variant: str = "s",
        in_chans: int = 15,
        num_classes: int = 2,
        image_size: int = 512,
        dropout_ratio: float = 0.1,
        drop_path_rate: float = 0.1,
        decoder_channels: int | None = None,
        feat_proj_dim: int | None = None,
        use_checkpoint: bool = False,
        pretrained_encoder: str | None = None,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown SegMAN variant {variant!r}; choose from {list(VARIANTS)}")
        spec = VARIANTS[variant]
        self.variant = variant
        self.image_size = image_size
        self.in_chans = in_chans
        self.num_classes = num_classes

        enc_mod = load_segman_encoder_module()
        dec_mod = load_segman_decoder_module()

        ckpt_stages = [2, 2, 2, 0] if use_checkpoint else [0, 0, 0, 0]
        enc_builder = getattr(enc_mod, spec["enc"])
        self.encoder = enc_builder(
            pretrained=None,
            image_size=image_size,
            in_chans=in_chans,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            use_checkpoint=ckpt_stages,
        )
        # drop the classification head we will not use (keeps state_dict clean)
        self.encoder.classifier = nn.Identity()

        ch = decoder_channels if decoder_channels is not None else spec["channels"]
        fpd = feat_proj_dim if feat_proj_dim is not None else spec["feat_proj_dim"]
        self.decoder = dec_mod.SegMANDecoder(
            in_channels=list(spec["embed_dims"]),
            in_index=[0, 1, 2, 3],
            channels=ch,
            feat_proj_dim=fpd,
            dropout_ratio=dropout_ratio,
            num_classes=num_classes,
            norm_cfg={"type": "BN", "requires_grad": True},
            align_corners=False,
            image_size=image_size,
        )

        if pretrained_encoder:
            self.load_pretrained_encoder(pretrained_encoder)

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the encoder and return the 4-stage feature pyramid."""
        f = self.encoder.patch_embed(x)
        feats: list[torch.Tensor] = []
        for i, layer in enumerate(self.encoder.layers):
            f = layer(f)
            if i % 2 == 0:  # layers = [stage0, down0, stage1, down1, ...]; collect pre-downsample
                feats.append(f)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        logits = self.decoder(feats)  # decoder emits at stride 8
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    @torch.no_grad()
    def load_pretrained_encoder(self, path: str) -> dict[str, Any]:
        """Load an official SegMAN ImageNet encoder checkpoint, inflating the
        3-channel stem to ``in_chans`` by averaging across the RGB inputs."""
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict_ema", ckpt.get("state_dict", ckpt))
        stem_key = "patch_embed.0.weight"
        if stem_key in state and self.in_chans != 3:
            w = state[stem_key]  # [out, 3, k, k]
            if w.shape[1] == 3:
                mean_w = w.mean(dim=1, keepdim=True)  # [out,1,k,k]
                state[stem_key] = mean_w.repeat(1, self.in_chans, 1, 1) * (3.0 / self.in_chans)
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        return {"missing": list(missing), "unexpected": list(unexpected)}


def build_segman(model_cfg: dict[str, Any]) -> SegMANSegmentor:
    """Construct a :class:`SegMANSegmentor` from a config ``model`` block."""
    return SegMANSegmentor(
        variant=str(model_cfg.get("variant", "s")),
        in_chans=int(model_cfg.get("in_chans", 15)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        image_size=int(model_cfg.get("image_size", 512)),
        dropout_ratio=float(model_cfg.get("dropout_ratio", 0.1)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.1)),
        decoder_channels=model_cfg.get("decoder_channels"),
        feat_proj_dim=model_cfg.get("feat_proj_dim"),
        use_checkpoint=bool(model_cfg.get("use_checkpoint", False)),
        pretrained_encoder=model_cfg.get("pretrained_encoder"),
    )
