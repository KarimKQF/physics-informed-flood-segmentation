# Voiceover Script — *SegMAN for Physics-Informed Flood Segmentation*

**Target duration:** 6–8 minutes
**Tone:** calm, precise, pedagogical. Pause on each equation.
**Timing column** is approximate; tune to the rendered animation.

The same narration lines are embedded as `# VOICEOVER:` comments at the top of
each `section_*` method in `segman_physics_informed_explanation.py`, so the
audio and the visuals stay in sync.

---

### 0 · Title (0:00 – 0:18)

> How does SegMAN *see* a flood? And can the laws of topography make it see
> better? Let's build the picture from pixels up.

---

### 1 · The flood segmentation problem (0:18 – 1:00)

> Start with a satellite image. Each pixel is either water, or it is not.
> Our task is **binary semantic segmentation**: assign every pixel one of two
> labels — water, or background. A third label, minus-one, marks pixels we
> ignore: clouds, no-data, unlabelled edges. The output is a clean binary mask
> of the flood.

---

### 2 · The input tensor (1:00 – 1:50)

> What actually goes in? A stack of **fifteen channels**: thirteen Sentinel-2
> optical bands, and two Sentinel-1 radar channels. Optical shows colour and
> vegetation; radar sees through clouds and night — essential during a flood.
>
> One thing is deliberately missing. The elevation map — the **DEM** — is *not*
> one of the inputs. It never enters the network. We keep it aside, because it
> has a different job: it lives only inside the **topographic loss** and the
> topographic metrics. Remember this invariant; the whole study depends on it.

---

### 3 · The SegMAN pipeline (1:50 – 2:35)

> Here is the whole pipeline. The fifteen-channel input passes through a stem,
> then the **SegMAN encoder**, producing features at several scales. A
> multi-scale decoder — MMSCopE — fuses them into **two logits per pixel**, and
> a softmax turns those into a water-probability map.
>
> Two ingredients make the encoder special: **local attention** and a
> **state-space scan**. Let's zoom into each.

---

### 4 · Local attention (2:35 – 3:25)

> Inside the encoder, local attention lets each pixel look at its neighbours.
> For a centre pixel *i*, we form a **query**, q-i. Each neighbour *j*
> contributes a **key**, k-j, and a **value**, v-j.
>
> The dot product of query and key — scaled by the square root of the
> dimension, then soft-maxed — decides how much each neighbour matters. Sum the
> values with those weights, and you have the attention output for pixel *i*.
> In plain terms: the model learns *which nearby pixels signal water* when
> deciding the label of the pixel in the middle.

---

### 5 · State-space scan (3:25 – 4:10)

> Attention is local. To carry information across the *whole* image cheaply,
> SegMAN also uses a **state-space scan** — a Mamba-like recurrence.
>
> A hidden state, h-t, marches along the tokens. At each step it updates from
> the previous state and the current token: h-t equals f of h-t-minus-one and
> x-t. From the state we read an output, y-t equals g of h-t. Run the scan
> left-to-right, right-to-left, and top-to-bottom, and long-range spatial
> context propagates across the image in **linear time**.

---

### 6 · Multi-scale fusion (4:10 – 4:45)

> The encoder produces features at decreasing resolution — a quarter, an
> eighth, a sixteenth, a thirty-second of the image. The decoder **fuses** these
> scales: coarse maps give context — *is this a river basin?* — while fine maps
> give sharp boundaries — *exactly where is the shoreline?* Together they yield
> one dense prediction.

---

### 7 · Segmentation head (4:45 – 5:15)

> At the head, the network emits two logit maps per pixel: one for background,
> one for water. A **softmax** converts them into the water probability:
>
> p-water of *i* equals e-to-the z-water, divided by e-to-the z-background plus
> e-to-the z-water. A single number between zero and one at every pixel.

---

### 8 · Four loss variants (5:15 – 6:00)

> Now the heart of the study. We **hold the architecture fixed** and change only
> the loss. Four variants.
>
> One: plain cross-entropy.
> Two: Dice plus cross-entropy — better for the rare water class.
> Three: that, plus a **topographic term** using the *real* DEM.
> Four: a control — the same topographic term, but with the DEM **shuffled**
> across samples.
>
> In every topo variant, lambda-topo — the weight on the physics term — is fixed
> at one half.

---

### 9 · Topographic-loss intuition (6:00 – 6:55)

> What does the topographic term actually mean? Take two neighbouring pixels.
> Suppose pixel *i* sits **higher** than pixel *j* by more than a small margin.
> If the model then calls the *high* pixel water — p-i near one — and the *low*
> pixel dry — p-j near zero — that is physically suspicious. Water does not
> perch above dry ground; it flows downhill.
>
> So we penalise exactly that pattern: the penalty grows with p-i times
> one-minus-p-j. Flip it around — water in the low pixel, dry on the high one —
> and the penalty vanishes. The loss gently nudges predictions to **respect
> elevation**.

---

### 10 · The DEM-shuffled ablation (6:55 – 7:30)

> But does the network truly *use* the physics, or just the extra constraint?
> We run a control. In one branch the topographic loss sees the **real**
> elevation, aligned to the scene. In the other, the elevation map is **shuffled**
> — taken from a different tile.
>
> The logic is simple. If the *real* DEM helps more than the shuffled one, the
> model is exploiting genuine topography. If shuffling does just as well — or
> better — then the topographic term is mostly acting as a **regulariser**, not
> as physics.

---

### 11 · Transfer to our experiment (7:30 – 8:00)

> Putting it together: SegMAN-S, a fifteen-channel Sentinel input, Dice plus
> cross-entropy, and the topographic loss — with the DEM kept strictly *out* of
> the model.
>
> The architecture is **not** the contribution. The research question is whether
> a physics-informed loss **improves or stabilises** segmentation when the
> architecture is held fixed.

---

### 12 · Current status (8:00 – 8:40)

> Where do we stand? On seed zero, **every variant trained stably** — no
> collapse, clean convergence. But there's a twist: the shuffled-DEM control
> actually **beat** the real DEM. So we cannot yet claim a physical effect.
>
> That's why **multi-seed experiments**, at fifty training samples, are running
> now — to test whether any difference survives across random seeds.
> Regularisation, or real physics? For now, honestly, it remains an open
> question.

---

**Closing card:** *SegMAN — physics-informed flood segmentation.*

---

## Production notes

- Total scripted runtime lands around **8:00–8:40** read at a measured pace;
  trim the longer pauses (sections 4, 9) to hit the 6–8 minute window, or speak
  slightly faster.
- Hyphenated phonetic spellings (q-i, h-t, p-water) are for the narrator only —
  pronounce them as the symbols they denote.
- Leave ~0.5 s of silence after each equation appears before continuing, so the
  viewer can read it.
