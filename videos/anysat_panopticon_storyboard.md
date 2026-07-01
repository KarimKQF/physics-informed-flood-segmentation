# AnySat and Panopticon Storyboard

Video title: "AnySat and Panopticon: SOTA 2025 Multimodal EO Models for Physics-Informed Flood Segmentation"

Target duration: 8 to 12 minutes.

Style: dark background, sparse text, synthetic satellite-like rectangles,
channel stacks, arrows, modular blocks, and comparison tables. No copyrighted
imagery is used.

## Scene List

### S00_Opening

Purpose: establish the project and the non-RGB input.

Visuals:

- Left: Sentinel-2 stack labeled `13 optical bands`.
- Left-middle: Sentinel-1 stack labeled `2 SAR channels`.
- Center: `X in R^{15 x 512 x 512}`.
- Right: flood segmentation output with class labels.
- Bottom: red cross over `RGB-only models`.

Voiceover point:

- Our data are multispectral and multimodal.
- RGB-only assumptions are not scientifically natural here.

Approximate duration: 45-60 seconds.

### S01_WhyRGBIsProblem

Purpose: explain why RGB shortcuts are limited.

Visuals:

- Top row: `RGB image -> 3 channels -> classical vision backbone`.
- Middle row: `S2 bands + S1 SAR -> 15 channels -> not RGB`.
- Bottom row: three limited solutions:
  - select RGB-like bands;
  - project 15 channels to 3;
  - inflate first convolution from 3 to 15.

Voiceover point:

- A satellite band is a physical measurement, not just a color slot.
- Shortcuts may work but are not ideal.

Approximate duration: 50-70 seconds.

### S02_AnySatConcept

Purpose: introduce AnySat as heterogeneous EO representation learning.

Visuals:

- Inputs: Sentinel-2, Sentinel-1, aerial, Landsat, MODIS, other sensors.
- Center block: `AnySat`.
- Output block: `shared EO representation`.
- Downstream tasks: land cover, crop, change, flood, burn scar, deforestation.

Voiceover point:

- AnySat is not one model for one fixed sensor.
- It learns across EO resolutions, scales, and modalities.

Approximate duration: 55-75 seconds.

### S03_AnySatArchitecture

Purpose: explain JEPA-style self-supervision and scale-adaptive encoding.

Visuals:

- Synthetic satellite tile as context view.
- Masked/alternate view as target view.
- `context view -> encoder -> predicted embedding`.
- `target view -> target embedding`.
- Feature-space prediction loss.
- Bottom callout: `scale-adaptive spatial encoders`.

Voiceover point:

- Predict target representations, not pixels.
- Adapt spatial encoding across sensor resolutions.

Approximate duration: 70-90 seconds.

### S04_AnySatForOurProject

Purpose: map AnySat into the flood segmentation protocol.

Visuals:

- `S2 optical + S1 SAR -> AnySat wrapper -> segmentation head -> logits [B,2,H,W]`.
- Below logits: `softmax -> p_water`.
- Separate DEM block crossed out as model input.
- Loss branches:
  - `p_water + GT mask -> CE/DiceCE`;
  - `p_water + DEM -> Topo loss + metrics`.
- Risk list: band audit, channel mapping, dense logits.

Voiceover point:

- AnySat changes the backbone but not the loss protocol.
- DEM remains loss-only and metric-only.

Approximate duration: 70-90 seconds.

### S05_PanopticonConcept

Purpose: explain channel identity.

Visuals:

- Rows for channels:
  - S2 B2, B3, B4, NIR, SWIR;
  - S1 VV, VH.
- Each row has a channel identity: wavelength or SAR mode.
- Center: `channels + channel identities`.
- Right: `spectral/channel-aware patch embedding -> DINOv2-like backbone`.

Voiceover point:

- The model should know what each channel physically means.

Approximate duration: 55-75 seconds.

### S06_PanopticonArchitecture

Purpose: compare standard DINOv2 with Panopticon.

Visuals:

- Top row: `RGB patches -> patch embedding -> ViT / DINOv2`.
- Bottom row:
  - arbitrary channel tokens;
  - cross-attention over channels;
  - patch tokens;
  - DINOv2-like transformer.
- Bottom idea pills:
  - same geolocation = sensor views;
  - channel subsampling = robustness;
  - cross-attention = flexible fusion.

Voiceover point:

- Panopticon extends DINOv2 to any-sensor EO with channel-aware fusion.

Approximate duration: 70-90 seconds.

### S07_PanopticonForOurProject

Purpose: map Panopticon into the 15-channel segmentation protocol.

Visuals:

- `13 S2 bands with wavelengths`.
- `2 S1 SAR VV/VH mode IDs`.
- `Panopticon backbone -> patch features -> dense decoder/head -> [B,2,H,W]`.
- Benefits list:
  - no RGB compression;
  - arbitrary channel handling;
  - good Sentinel-1 + Sentinel-2 fit.
- Risks list:
  - needs dense head;
  - patch upsampling;
  - DEM stays out.

Voiceover point:

- Panopticon is a strong conceptual fit but requires segmentation adaptation.

Approximate duration: 70-90 seconds.

### S08_Comparison

Purpose: compare candidate backbones.

Visuals:

Table columns:

- Model
- Main idea
- Strength
- Risk
- Role

Rows:

- AnySat
- Panopticon
- SegMAN
- TerraMind

Voiceover point:

- Different backbones; same required output: dense logits `[B,2,H,W]`.

Approximate duration: 60-80 seconds.

### S09_PhysicsPipeline

Purpose: emphasize the fixed scientific protocol.

Visuals:

- `Any model -> dense logits [B,2,H,W] -> softmax -> p_water`.
- Four losses:
  - CE;
  - Dice+CE;
  - Dice+CE+Topo real DEM;
  - Dice+CE+Topo shuffled DEM.
- DEM block with cross-out note: not model input.
- Highlight: shuffled DEM separates physics from regularization.

Voiceover point:

- Model changes, protocol stays fixed.
- DEM-shuffled is mandatory.

Approximate duration: 75-95 seconds.

### S10_FinalRoadmap

Purpose: close with a conservative research roadmap.

Visuals:

Roadmap nodes:

1. Finish SegMAN N=100 diagnostic.
2. Complete TerraMind baseline.
3. Audit AnySat.
4. Audit Panopticon.
5. Integrate only if dense logits are clean.
6. Compare real DEM vs shuffled DEM.

Final callout:

`Not RGB-only. Not flood-only. Multimodal and sensor-aware.`

Voiceover point:

- AnySat and Panopticon are general EO foundation models, which is the right
  level of generality for this project.

Approximate duration: 60-80 seconds.

## Expected Duration

Total expected duration after natural narration and pauses:

- Short cut: about 8 minutes.
- Normal narration: about 10 minutes.
- Slower teaching pace: about 11-12 minutes.

## Scientific Guardrails

- DEM is never shown entering AnySat, Panopticon, SegMAN, or TerraMind.
- DEM appears only after logits / water probability, inside the loss and metrics.
- Dense logits `[B,2,H,W]` are the integration requirement for any candidate.
- The DEM-shuffled ablation remains mandatory.

