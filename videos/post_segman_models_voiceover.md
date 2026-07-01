# Voiceover Script — *After SegMAN: VFMNet, EoMT, ViT-P and TerraMind*

**Target duration:** 7–9 minutes
**Tone:** calm, precise, pedagogical. Pause on each model name and each equation.
**Timing column** is approximate; tune to the rendered animation.

The same narration lines are embedded as `# VOICEOVER:` comments at the top of
each `section_*` method in `post_segman_models_explanation.py`, so the audio and
the visuals stay in sync.

**Scientific invariant (must stay true in every section):**
> The DEM is never given as a model input in our current controlled protocol.
> It remains a supervision signal through the topographic loss and a physical
> consistency metric only.

---

### 0 · Title (0:00 – 0:25)

> SegMAN is our current main model. But the architecture is not the
> contribution. So the real question is whether our physics-informed loss
> survives across very different model families. After SegMAN, four of them:
> VFMNet, EoMT, ViT-P, and TerraMind.

---

### 1 · Why look beyond SegMAN (0:25 – 1:05)

> The architecture is not the contribution. We hold the loss protocol fixed and
> swap the model. If the topographic loss only helps one architecture, it is an
> artefact. If it helps across families — dense networks, foundation models, EO
> models — it is a property worth reporting.

---

### 2 · The shared protocol (1:05 – 1:50)

> The protocol is the same for every model. Sentinel-1 and Sentinel-2 go in. The
> model emits two logits per pixel, and a softmax gives the water mask. And here
> is the rule that does not change: the DEM is *not* a model input. It lives only
> inside the topographic loss and the physical consistency metrics.

---

### 3 · VFMNet / VFMSeg (1:50 – 2:25)

> First family: VFMNet, or VFMSeg, from AAAI 2025. It builds segmentation on top
> of large visual foundation models, aiming for generalization and
> out-of-distribution robustness. Its limitation for us: that pretraining is
> mostly on natural RGB images.

---

### 3b · VFMNet — how it works

> How does VFMNet work? It does not learn features from scratch. It takes one or
> more large, pretrained visual foundation models — think DINOv2 or SAM — and
> keeps the backbone frozen. A light adapter fuses those rich features, and only
> a small segmentation head is trained. The bet: foundation-model features
> already generalize, so they transfer to new, out-of-distribution scenes.

---

### 4 · EoMT (2:25 – 3:00)

> Second: EoMT, a CVPR 2025 Highlight. Encoder-only Mask Transformer. A plain
> ViT encoder does the segmentation directly, with mask classification instead
> of a separate heavy decoder. Simple and fast — but ViT-scale compute, and a
> mask-classification head rather than dense logits.

---

### 4b · EoMT — how it works

> How does EoMT work? Take the patch tokens of a plain ViT and append a few
> learnable query tokens. The encoder's own self-attention lets those queries
> read the patches — there is no separate mask decoder. At the output, each query
> predicts a class and a mask embedding; the mask itself is the dot product of
> that embedding with the patch features. Encoder-only, yet it does full mask
> classification.

---

### 5 · ViT-P (3:00 – 3:35)

> Third: ViT-P, a CVPR 2026, SOTA-level universal image segmentation reference.
> Top reported accuracy — but it is the newest and the hardest to integrate, so
> we treat it as a future reference rather than an immediate baseline.

---

### 5b · ViT-P — how it works

> How does ViT-P work? A plain ViT backbone produces dense features, and the
> model predicts a set of pairs — a class and a mask for each query. That single
> mask-classification view covers semantic, instance and panoptic segmentation
> with one model: that is what "universal" means. It is the newest of the four,
> so treat the exact internals as a moving state-of-the-art reference rather than
> a fixed recipe.

---

### 6 · Our data are not RGB (3:35 – 4:15)

> Notice something. VFMNet, EoMT and ViT-P are general-vision models, trained
> mostly on natural RGB images. But our data are not RGB. We use Sentinel-1 SAR
> and Sentinel-2 multispectral. That distribution gap is exactly why an Earth
> Observation foundation model matters.

---

### 7 · TerraMind — intuition & pipeline (4:15 – 5:15)

> TerraMind is an Earth Observation foundation model — not a generic RGB
> segmentation model. It is built around satellite and geospatial modalities, so
> it is naturally closer to our domain. Sentinel-1 and Sentinel-2 go into
> TerraMind; it produces EO features; a segmentation head turns those into two
> logits per pixel and a water mask.
>
> TerraMind is the most domain-aligned model for Earth Observation, but SegMAN
> is currently the main architecture for clean loss-comparison experiments,
> because it is already integrated and produces dense logits directly.

---

### 7a · TerraMind — how it works

> How does TerraMind work? Unlike an RGB model, it is pretrained across many
> Earth-Observation modalities — radar, optical, and more — each tokenized and
> fused inside a shared transformer with cross-modal attention. That pretraining
> teaches it EO-specific statistics: SAR speckle, multispectral reflectance. We
> then fine-tune a segmentation head for water. Important: TerraMind *can* ingest
> a DEM as a modality, but in our controlled protocol we feed only Sentinel-1 and
> Sentinel-2 — the DEM stays out of the model and lives only in the loss.

---

### 7b · TerraMind — role & continuity (5:15 – 6:05)

> Its role in the project: TerraMind is not abandoned. It remains our Earth
> Observation, Sen1Floods11 baseline — a specialized satellite reference that
> lets us compare a recent general architecture, SegMAN, against a model designed
> for EO data, and it preserves continuity with our previous TerraMind work.
>
> We already ran TerraMind and TerraTorch with Dice, Dice plus CE, topo on the
> real DEM, and topo on the shuffled DEM. It showed strong full-data performance
> and interesting low-data behaviour, including collapse and rescue patterns.
> SegMAN does not replace TerraMind; it complements it. One caution: as a
> multimodal EO model, TerraMind may reason over several modalities — so we must
> be explicit about which modalities it is given.

---

### 8 · TerraMind — risks & limitations (6:05 – 6:45)

> But TerraMind carries real risks. The TerraTorch pipeline is heavier. Modality
> handling must be controlled carefully. And critically: if the DEM is ever given
> as an input, the experiment no longer tests the same hypothesis as our
> loss-only DEM setup. Integration and reproducibility are also harder than with
> SegMAN. So the invariant holds: the DEM stays a loss-and-metric signal, never a
> model input.

---

### 9 · Comparison (6:45 – 7:30)

> Side by side. SegMAN: integrated, dense logits now. VFMNet: foundation-model
> generalization. EoMT: a ViT encoder-only design. ViT-P: future SOTA. And
> TerraMind: the Earth Observation foundation model, most aligned with Sentinel
> data and Sen1Floods11 — at the cost of a heavier pipeline and careful modality
> control, with a DEM-leakage risk if it is not configured carefully.

---

### 10 · Physics-informed transfer (7:30 – 8:05)

> Whichever model we pick, the physics-informed transfer is identical.
> Sentinel-1 and Sentinel-2 are the inputs. The model emits logits. The
> topographic loss reads the DEM only to penalise physically inconsistent
> predictions. The DEM is never given as a model input in our controlled
> protocol — it remains a supervision signal and a physical consistency metric.

---

### 11 · Roadmap (8:05 – 8:40)

> The roadmap, in order. One: SegMAN, the current main model, already running
> multi-seed. Two: TerraMind, the EO and Sen1Floods11 baseline to preserve and
> complete. Three: VFMNet, the AAAI line for foundation-model generalization.
> Four: EoMT, the second CVPR architecture. Five: ViT-P, a SOTA reference for
> later, with higher integration risk.

---

### 12 · Conclusion (8:40 – 9:15)

> After SegMAN, the goal is not to chase scores blindly. The goal is to test
> whether our topographic physics-informed loss remains useful across different
> model families: dense segmentation architectures, Earth Observation foundation
> models, foundation-model generalization systems, ViT encoder-only segmentation,
> and universal mask-classification frameworks.
>
> And throughout, the DEM is never given as a model input — it stays a
> supervision signal and a physical consistency metric.

---

**Closing card:** *After SegMAN — VFMNet · EoMT · ViT-P · TerraMind · physics-informed flood segmentation.*

---

## Production notes

- Total scripted runtime lands around **8:30–9:15** read at a measured pace.
- Pronounce model names clearly: "VFMNet" (V-F-M-Net), "EoMT" (E-o-M-T), "ViT-P"
  (V-i-T-P), "TerraMind".
- Leave ~0.5 s of silence after each equation / pipeline appears before
  continuing, so the viewer can read it.
- TTS generation (matches the SegMAN video workflow):
  `edge-tts --voice en-US-JennyNeural --rate=-5%` over the blockquote lines.
