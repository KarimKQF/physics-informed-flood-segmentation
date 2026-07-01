# Storyboard — *After SegMAN: VFMNet, EoMT, ViT-P and TerraMind*

A scene-by-scene visual plan for the multi-model explainer
(`post_segman_models_explanation.py`). Each section maps 1:1 to a `section_*`
method and to a standalone `S0x_*` scene class for isolated rendering.

**Format:** dark navy background (`#0b0f1a`), 16:9, 1080p60 for final.

This explainer is a sequel to the SegMAN video. It situates SegMAN among four
other model families that we may evaluate under the **same** physics-informed
loss protocol, and it adds **TerraMind** as the Earth Observation member.

---

## Colour language

| Colour | Hex | Meaning |
|--------|-----|---------|
| Cyan | `#22d3ee` | water · **SegMAN** (current main model) |
| Blue | `#0ea5e9` | Sentinel-2 optical / deep water |
| Violet | `#a78bfa` | Sentinel-1 SAR · **EoMT** |
| Sky | `#38bdf8` | **VFMNet / VFMSeg** |
| Amber | `#fbbf24` | equation highlight · **ViT-P** |
| Emerald | `#34d399` | **TerraMind** (Earth Observation) |
| Tan | `#d4a373` | elevation / DEM |
| Green | `#22c55e` | strength / physically coherent |
| Orange | `#f97316` | caution |
| Red | `#ef4444` | risk / "DEM is not an input" |
| Slate | `#243042` | background / land |
| Light grey | `#e5e7eb` | primary text |

**Per-model accent map:** SegMAN = cyan, VFMNet = sky, EoMT = violet,
ViT-P = amber, TerraMind = emerald. These colours are reused consistently in the
comparison table, roadmap and conclusion.

---

## 0 · Title — `S00_Title` (~25 s)

- **Visual:** "After SegMAN" in large cyan; the four model names on one line
  (`VFMNet - EoMT - ViT-P - TerraMind`); amber underline; subtitle fades up.
- **Motion:** `Write` → `FadeIn` + `Create` underline → `FadeIn` subtitle.

## 1 · Why look beyond SegMAN — `S01_Motivation` (~40 s)

- **Visual:** bold amber thesis "The architecture is NOT the contribution.";
  the research question; four staggered bullets (fixed protocol / swap model /
  one model = artefact / across families = property).

## 2 · The shared protocol — `S02_Protocol` (~45 s)

- **Visual:** horizontal flow `S1+S2 (15ch) → any model → logits[B,2,H,W]`.
- **Key beat:** a tan DEM block below the model; a red arrow into the model is
  **crossed out**; red caption "The DEM is NOT a model input", grey sub-caption
  "(used only in the topographic loss & physical consistency metrics)".
- **Purpose:** burns in the invariant for the whole video.

## 3 · VFMNet / VFMSeg — `S03_VFMNet` (~35 s)

- **Visual:** sky-blue model header + "AAAI 2025" venue tag + one-line idea; four
  bullets (reuses VFMs / OOD strength / RGB-centric risk / our line).

## 3b · VFMNet — how it works — `S03b_VFMNetHow` (~40 s)

- **Visual:** pipeline `image → frozen VFM backbone(s) → rich features → light
  adapter → seg head`, with "frozen -- not trained" over the backbone and
  "trained" over the head; four mechanism bullets.

## 4 · EoMT — `S04_EoMT` (~35 s)

- **Visual:** violet model header + "CVPR 2025 Highlight"; four bullets
  (ViT encoder segments directly / simple-fast strength / ViT compute + mask-cls
  risk / second CVPR architecture).

## 4b · EoMT — how it works — `S04b_EoMTHow` (~40 s)

- **Visual:** a row of slate **patch tokens** + two violet **query tokens** →
  one `shared ViT self-attention (no separate decoder)` block → two outputs:
  `class` and `mask = query · patch`. Caption: "queries read patches via the
  ViT's own attention".

## 5 · ViT-P — `S05_ViTP` (~35 s)

- **Visual:** amber model header + "CVPR 2026"; four bullets (universal seg /
  SOTA strength / highest integration risk / future reference).

## 5b · ViT-P — how it works — `S05b_ViTPHow` (~40 s)

- **Visual:** `image → ViT backbone → N queries (class, mask)`; below, three
  task chips (semantic / instance / panoptic) under "one model -- universal
  segmentation"; four mechanism bullets (incl. red "newest, moving SOTA target").

## 6 · Our data are not RGB — `S06_NotRGB` (~40 s) — *pivot to EO*

- **Visual:** left column = three general-vision chips (VFMNet, EoMT, ViT-P) with
  "mostly natural RGB"; right column = our inputs (S1 SAR 2ch, S2 multispectral
  13ch). A red "domain gap" arrow between them.
- **Footer (amber):** "=> an Earth Observation foundation model is naturally
  aligned."

## 7 · TerraMind — intuition & pipeline — `S07_TerraMind` (~60 s) — *new core*

- **Visual:** emerald "TerraMind" header + "Earth Observation foundation model";
  one-line idea "built for satellite / geospatial data — not generic RGB".
- **Pipeline:** `S1/S2 modalities → TerraMind EO foundation → EO features →
  seg head/decoder → logits[B,2,H,W]` → cyan water mask below.
- **Key amber callout (verbatim intent):** "Most domain-aligned for EO — but
  SegMAN stays the main loss-comparison model (integrated, dense logits
  directly)."

## 7a · TerraMind — how it works — `S07a_TerraMindHow` (~50 s)

- **Visual:** modality chips (S1 SAR, S2 optical, other EO modalities) → a
  `shared transformer (cross-modal attention)` block → `fine-tuned seg head`;
  top label "pretrained on EO modalities -> learns SAR speckle & multispectral
  reflectance".
- **Amber guard (footer):** "In our protocol we feed only S1 / S2. DEM stays out
  of the model -- loss only." (TerraMind *can* take a DEM modality, but we do not
  give it one.)

## 7b · TerraMind — role & continuity — `S07b_TerraMindRole` (~50 s)

- **Visual:** left column "Role in the project" (5 green bullets: not abandoned /
  EO+Sen1Floods11 baseline / specialized satellite reference / compare
  SegMAN vs EO model / continuity). Right column "Previously, with TerraMind /
  TerraTorch:" with four loss chips (Dice, Dice+CE, Topo real, Topo shuffled),
  "strong full-data; low-data collapse/rescue", and "SegMAN complements
  TerraMind — it does not replace it."
- **Footer (orange):** modality caution — be explicit about which modalities
  TerraMind is given.

## 8 · TerraMind — risks & limitations — `S08_TerraMindRisks` (~40 s)

- **Visual:** orange title "Risks to control"; bullets (heavier TerraTorch
  pipeline / careful modality handling / **DEM as input = no longer the same
  hypothesis** in red / harder integration & reproducibility).
- **Footer (amber):** "Invariant: DEM stays a loss/metric signal — never a model
  input."

## 9 · Comparison — `S09_Comparison` (~45 s)

- **Visual:** 4-column table (Model · Strength · Risk · Role) with a header rule
  and **five rows**: SegMAN, VFMNet/VFMSeg, EoMT, ViT-P, TerraMind. Each model
  name carries its accent colour and a tiny idea subtitle. The **TerraMind row is
  highlighted** with an emerald surrounding rectangle.
- TerraMind row content:
  - Strength: "aligned w/ S1/S2 + Sen1Floods11"
  - Risk: "heavy pipeline; DEM-leak if misconfigured"
  - Role: "specialized EO baseline + continuity"

## 10 · Physics-informed transfer — `S10_Transfer` (~40 s)

- **Visual:** `S1+S2 → model(any family) → logits → Dice+CE+Topo loss`; a tan DEM
  block feeds the **loss** with a green "topographic loss only" arrow, while a
  red crossed-out arrow shows the DEM does **not** feed the model.
- **Footer (amber):** "The DEM is never a model input — only a loss /
  physical-consistency signal."

## 11 · Roadmap — `S11_Roadmap` (~35 s)

- **Visual:** ordered, colour-coded list:
  1. SegMAN (cyan) — current main model, already running multi-seed
  2. TerraMind (emerald) — EO / Sen1Floods11 baseline to preserve & complete
  3. VFMNet/VFMSeg (sky) — AAAI line for OOD / FM generalization
  4. EoMT (violet) — second CVPR architecture, ViT encoder-only
  5. ViT-P (amber) — SOTA-level future reference, higher integration risk

## 12 · Conclusion — `S12_Conclusion` (~35 s) — *cautious close*

- **Visual:** amber thesis "Not chasing scores — testing a property."; the
  question; five colour-coded family bullets (dense seg / EO foundation models /
  FM generalization / ViT encoder-only / universal mask-classification).
- **Invariant line (amber):** "The DEM is never a model input — only a loss /
  physical-consistency signal."
- **Outro card:** "After SegMAN — VFMNet · EoMT · ViT-P · TerraMind."

---

## Pacing summary

| Section | Scene class | ~Duration |
|---------|-------------|-----------|
| 0 Title | `S00_Title` | 0:25 |
| 1 Motivation | `S01_Motivation` | 0:40 |
| 2 Protocol | `S02_Protocol` | 0:45 |
| 3 VFMNet | `S03_VFMNet` | 0:35 |
| 3b VFMNet how | `S03b_VFMNetHow` | 0:40 |
| 4 EoMT | `S04_EoMT` | 0:35 |
| 4b EoMT how | `S04b_EoMTHow` | 0:40 |
| 5 ViT-P | `S05_ViTP` | 0:35 |
| 5b ViT-P how | `S05b_ViTPHow` | 0:40 |
| 6 Not RGB | `S06_NotRGB` | 0:40 |
| 7 TerraMind | `S07_TerraMind` | 1:00 |
| 7a TerraMind how | `S07a_TerraMindHow` | 0:50 |
| 7b TerraMind role | `S07b_TerraMindRole` | 0:50 |
| 8 TerraMind risks | `S08_TerraMindRisks` | 0:40 |
| 9 Comparison | `S09_Comparison` | 0:45 |
| 10 Transfer | `S10_Transfer` | 0:40 |
| 11 Roadmap | `S11_Roadmap` | 0:35 |
| 12 Conclusion | `S12_Conclusion` | 0:35 |
| **Total** | | **≈ 11:30** |

---

## Rendering

```bash
# Full video, high quality (1080p60):
manim -pqh videos/post_segman_models_explanation.py PostSegMANModelsVideo

# Fast preview while iterating (480p15):
manim -pql videos/post_segman_models_explanation.py PostSegMANModelsVideo

# Render the TerraMind chapter on its own:
manim -pqh videos/post_segman_models_explanation.py S07_TerraMind
```

**Prerequisites (CPU-only — no GPU used):**

```bash
pip install manim          # Manim Community Edition
```

Plus a LaTeX distribution for the equations (on Windows: install **MiKTeX**).
On this host the `.venv_manim` environment already has Manim 0.20.1; FFmpeg
(Gyan build) and MiKTeX are on PATH for rendering. `Text.set_default(font="Arial")`
is set at the top of the script because DejaVu Sans is not installed here.

Output lands under `media/videos/post_segman_models_explanation/` (or under the
`--media_dir` you pass).
