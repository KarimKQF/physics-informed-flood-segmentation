# Storyboard — *SegMAN for Physics-Informed Flood Segmentation*

A scene-by-scene visual plan for the Manim explainer
(`segman_physics_informed_explanation.py`). Each section maps 1:1 to a
`section_*` method and to a standalone `S0x_*` scene class for isolated
rendering.

**Format:** dark navy background (`#0b0f1a`), 16:9, 1080p60 for final.

---

## Colour language (used consistently throughout)

| Colour | Hex | Meaning |
|--------|-----|---------|
| Cyan | `#22d3ee` | water / flood |
| Blue | `#0ea5e9` | deep water · Sentinel-2 optical |
| Violet | `#a78bfa` | Sentinel-1 / SAR |
| Slate | `#243042` | background / land |
| Tan | `#d4a373` | elevation / DEM |
| Green | `#22c55e` | physically coherent / "real DEM" branch |
| Orange | `#f97316` | caution / "shuffled DEM" branch |
| Red | `#ef4444` | penalty / inconsistency / "no input" |
| Amber | `#fbbf24` | equation highlight · key callouts |
| Light grey | `#e5e7eb` | primary text |

---

## 0 · Title — `S00_Title` (~18 s)

- **Visual:** "SegMAN" in large cyan, amber underline draws left-to-right,
  subtitle fades up.
- **Motion:** `Write` → `Create` underline → `FadeIn` subtitle.
- **Exit:** fade all.

## 1 · The flood segmentation problem — `S01_Problem` (~40 s)

- **Visual:** left = 10×12 pseudo-satellite grid (blues/greens/tans);
  right = binary water mask with a meandering cyan river; amber "segment" arrow
  between them.
- **Lower third:** 3-chip legend — water=1, background=0, ignore=-1.
- **Motion:** FadeIn image → GrowArrow → FadeIn mask → FadeIn legend.

## 2 · The input tensor — `S02_Input` (~50 s)

- **Visual:** 15 stacked planes offset diagonally (13 blue S2 + 2 violet SAR),
  braces label each group. Equation `X ∈ ℝ^{15×H×W}` below.
- **Key beat:** a separate tan DEM plane on the right; a red arrow toward the
  cube is **crossed out**; bold red caption "DEM is NOT fed to SegMAN", grey
  sub-caption "(used only in the topographic loss & metrics)".
- **Motion:** lagged FadeIn of planes → Write equation → braces → DEM → red
  cross + caption.
- **Purpose:** burns in the critical invariant.

## 3 · SegMAN pipeline — `S03_Pipeline` (~45 s)

- **Visual:** horizontal flow of rounded blocks:
  `X(15ch) → Stem → SegMAN Encoder → Multi-scale features → MMSCopE Decoder →
  logits[B,2,H,W]`, each appearing with its connecting arrow.
- **Tail:** `→softmax→` mini cyan mask labelled "water mask".
- **Callout:** amber surrounding rectangle around Encoder + features, labelled
  "local attention + state-space scan" (teases sections 4–5).

## 4 · Local attention — `S04_Attention` (~50 s)

- **Visual left:** 5×5 neighbourhood grid; centre pixel cyan, labelled `i`;
  caption "neighborhood N(i)". Eight amber arrows from neighbours to centre,
  thickness/opacity ∝ attention weight.
- **Visual right:** stacked `q_i=W_Q x_i` (cyan), `k_j=W_K x_j` (amber),
  `v_j=W_V x_j` (violet); then the attention sum with the scaled-softmax term
  highlighted.
- **Beat:** `Indicate` centre pixel on the line "which nearby pixels matter".

## 5 · State-space scan — `S05_StateSpace` (~45 s)

- **Visual:** a row of 9 token squares (top). A cyan hidden-state dot `h_t`
  sweeps left→right beneath them; each token turns blue as it is consumed;
  short amber link drops from token to state at each step.
- **Equations:** `h_t = f(h_{t-1}, x_t)` (cyan), `y_t = g(h_t)` (green).
- **Beat:** three direction labels appear — left→right, right→left,
  top→bottom; caption "long-range spatial context, linear cost".

## 6 · Multi-scale fusion — `S06_MultiScale` (~35 s)

- **Visual:** four squares of decreasing size labelled H/4, H/8, H/16, H/32
  (cyan→blue→violet→amber). Grey arrows converge into one green "dense
  prediction" square.
- **Motion:** lagged FadeIn of scale maps → merge arrows → output.

## 7 · Segmentation head — `S07_Head` (~30 s)

- **Visual:** two logit maps `Z_background` (slate) and `Z_water` (blue), amber
  arrow → softmax equation
  `p_water(i) = e^{z_water} / (e^{z_background} + e^{z_water})` (water term in
  cyan) → a small cyan probability heatmap "p_water in [0,1]".

## 8 · Four loss variants — `S08_Losses` (~45 s)

- **Visual:** four stacked rounded cards, each title + equation:
  1. **CE** (grey) — `L = L_CE`
  2. **Dice+CE** (cyan) — `L_DiceCE = L_Dice + α L_CE`
  3. **Dice+CE+Topo real** (green) — `L_total = L_DiceCE + λ_topo L_topo(DEM_real)`
  4. **Dice+CE+Topo shuffled** (orange) — `… L_topo(DEM_shuffled)`
- **Footer:** `λ_topo = 0.5` in amber; `Indicate` cards 3 and 4.

## 9 · Topographic-loss intuition — `S09_TopoLoss` (~55 s) — *emotional core*

- **Visual:** ground line; two tan elevation bars — tall (left, `h_i`) and
  short (right, `h_j`). Top condition `h_i > h_j + margin`.
- **Inconsistent case:** cyan water pixel on the **tall** bar (`p_i≈1`), dry
  slate pixel on the **short** bar (`p_j≈0`). Red surrounding flash + red
  equation `penalty ∝ p_i(1−p_j)`; caption "water perched above dry ground =
  inconsistent".
- **Coherent case:** swap fills — water moves to the **low** bar, dry to the
  **high** bar; labels flip to `p_i≈0`, `p_j≈1`; green box; equation transforms
  to `p_i(1−p_j) ≈ 0`; caption "physically coherent → no penalty".

## 10 · DEM-shuffled ablation — `S10_Ablation` (~40 s)

- **Visual:** two large branch panels — left green "Real DEM"
  (`L_topo(DEM_real)`, "elevation matches the scene"), right orange "Shuffled
  DEM" (`L_topo(DEM_shuffled)`, "elevation from a different tile").
- **Decision logic (footer):**
  - green: "real > shuffled → model exploits physical topography"
  - orange: "real ≈ shuffled → topo term acts as regularization"

## 11 · Transfer to our experiment — `S11_Transfer` (~40 s)

- **Visual:** left column of four nodes — SegMAN-S (fixed), 15-channel S1+S2,
  Dice+CE loss, Topographic loss — grey arrows converge into a tall cyan "flood
  mask" node.
- **Thesis (footer):** "Architecture is NOT the contribution." +
  "Question: does a physics-informed loss help at fixed architecture?" (amber).

## 12 · Current status — `S12_Status` (~40 s) — *cautious close*

- **Visual:** title "Where we stand", four dotted status lines:
  - green — "Seed 0: all variants converged, no collapse."
  - orange — "Seed 0: DEM-shuffled OUTPERFORMED real DEM."
  - red — "⇒ No strong physical claim can be made yet."
  - cyan — "Multi-seed N=50 runs in progress (robustness test)."
- **Amber caption:** "Regularization vs. physics: still an open question."
- **Outro card:** "SegMAN — physics-informed flood segmentation."

---

## Pacing summary

| Section | Scene class | ~Duration |
|---------|-------------|-----------|
| 0 Title | `S00_Title` | 0:18 |
| 1 Problem | `S01_Problem` | 0:40 |
| 2 Input | `S02_Input` | 0:50 |
| 3 Pipeline | `S03_Pipeline` | 0:45 |
| 4 Attention | `S04_Attention` | 0:50 |
| 5 State-space | `S05_StateSpace` | 0:45 |
| 6 Multi-scale | `S06_MultiScale` | 0:35 |
| 7 Head | `S07_Head` | 0:30 |
| 8 Losses | `S08_Losses` | 0:45 |
| 9 Topo loss | `S09_TopoLoss` | 0:55 |
| 10 Ablation | `S10_Ablation` | 0:40 |
| 11 Transfer | `S11_Transfer` | 0:40 |
| 12 Status | `S12_Status` | 0:40 |
| **Total** | | **≈ 8:13** |

To land nearer 6:30, trim the per-step `run_time` in sections 4, 5 and 9, and
reduce the trailing `self.wait(...)` calls (currently 1.4–2.0 s) to ~0.8 s.

---

## Rendering

```bash
# Full video, high quality (1080p60):
manim -pqh videos/segman_physics_informed_explanation.py SegMANPhysicsVideo

# Fast preview while iterating (480p15):
manim -pql videos/segman_physics_informed_explanation.py SegMANPhysicsVideo

# Render one chapter on its own:
manim -pqh videos/segman_physics_informed_explanation.py S09_TopoLoss

# 4K master (slow):
manim -qk videos/segman_physics_informed_explanation.py SegMANPhysicsVideo
```

**Prerequisites (CPU-only — no GPU used):**

```bash
pip install manim          # Manim Community Edition
```

Plus a LaTeX distribution for the equations (on Windows: install **MiKTeX**;
Manim invokes `latex`/`dvisvgm` to typeset every `MathTex`). Fonts referenced:
*DejaVu Sans* and *DejaVu Sans Mono* (bundled with Manim; substitute any
installed sans-serif if unavailable).

Flags: `-p` preview when done, `-q{l,m,h,k}` = low/medium/high/4K quality.
Output lands under `media/videos/segman_physics_informed_explanation/`.
