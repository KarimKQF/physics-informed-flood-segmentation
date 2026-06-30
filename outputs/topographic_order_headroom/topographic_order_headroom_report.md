# Topographic Order Headroom Diagnostics
## SegMAN-S · Sen1Floods11 · N=100 · seed 0

**Inference-only. SegMAN frozen. DEM used only post-hoc — never a model input.**

---

## 1. Motivation

We do not refute D4 and D8 independently. We test the **shared mechanism** of
their loss class: the reduction of local topographic-order violations.  In this
setting, the unconstrained baseline already exhibits lower violation fraction and
lower violation energy than the reference labels.  Therefore, any loss whose sole
mechanism is to further reduce these violations lacks native label-relative headroom.

## 2. Class definition

A **local topographic-order loss** is any R_φ such that

    v_ij = max(0, p_i − p_j − τ)    when h_i > h_j, j ∈ N(i)

is the penalised quantity (N(i) = D8 steepest-descent, or D4 all-lower 4-neighbors).
Minimising R_φ pushes T = −VF(p̂, z) upward (fewer violations).  The class includes
D8, slope-weighted D8, margin-based D8, D4 variants, etc.

**Parameters:** s₀=1.0  τ=0.05  threshold=0.5
**Checkpoint:** `E:\flood_research\experiments\segman\runs\segman_n100_dice_ce_seed0\checkpoints\best_checkpoint.pt`  (best_ep=34,
best_mIoU=0.8546)

---

## 3. VF native headroom  H_R = VF(pred) − VF(GT)

| split | VF(y) wtd | VF(ŷ₀) wtd | **H_R wtd** | 95% CI | % tiles pred≤GT | VF(y) uwtd | VF(ŷ₀) uwtd | H_R uwtd |
|-------|----------:|----------:|----------:|--------|-----:|----------:|----------:|----------:|
| val | 0.00217 | 0.00062 | **-0.00154** | [-0.00303, -0.00159] | 88% | 0.00401 | 0.00121 | -0.00280 |
| test | 0.00241 | 0.00058 | **-0.00182** | [-0.00344, -0.00120] | 90% | 0.00421 | 0.00115 | -0.00306 |
| bolivia | 0.00321 | 0.00094 | **-0.00227** | [-0.00406, -0.00105] | 80% | 0.00504 | 0.00212 | -0.00292 |

_H_R ≤ 0 ⇒ baseline already violates the D8 constraint **less than the labels** → no native headroom._

### D4 VF headroom

| split | VF_D4(y) wtd | VF_D4(ŷ₀) wtd | H_R_D4 wtd | 95% CI | % tiles pred≤GT |
|-------|----------:|----------:|----------:|--------|-----:|
| val | 0.00180 | 0.00055 | **-0.00125** | [-0.00267, -0.00141] | 88% |
| test | 0.00204 | 0.00051 | **-0.00153** | [-0.00308, -0.00104] | 90% |
| bolivia | 0.00284 | 0.00082 | **-0.00202** | [-0.00368, -0.00089] | 80% |

---

## 4. Violation energy  E_topo = mean(w·v²)

| split | E_D8(y) | E_D8(ŷ₀) | ΔE_D8 | 95% CI | E_D8(p soft) | E_D4(y) | E_D4(ŷ₀) |
|-------|--------:|--------:|------:|--------|--------:|--------:|--------:|
| val | 0.001955 | 0.000564 | **-0.001391** | [-0.002702, -0.001439] | 0.000012 | 0.001628 | 0.000501 |
| test | 0.002174 | 0.000527 | **-0.001647** | [-0.003068, -0.001106] | 0.000015 | 0.001839 | 0.000462 |
| bolivia | 0.002896 | 0.000849 | **-0.002047** | [-0.003721, -0.000907] | 0.000016 | 0.002561 | 0.000736 |

_ΔE ≤ 0 → predictions have lower or equal violation energy than labels._

---

## 5. Distributional tails  P(v > t)

Fraction of D8-active pixels with violation magnitude exceeding threshold t.
Hard labels: v_y = 0.95 at violations (step function); hard ŷ: same.
Soft p: v_p = max(0, p_i − p_d − τ), continuous in [0,1].

### val

| t | P(v_y>t) GT | P(v_ŷ>t) hard | P(v_p>t) soft | ΔP(pred−GT) hard | ΔP(soft−GT) |
|---|---:|---:|---:|---:|---:|
| 0.00 | 0.0040 | 0.0012 | 0.0059 | -0.0028 | 0.0019 |
| 0.05 | 0.0040 | 0.0012 | 0.0016 | -0.0028 | -0.0024 |
| 0.10 | 0.0040 | 0.0012 | 0.0005 | -0.0028 | -0.0035 |
| 0.20 | 0.0040 | 0.0012 | 0.0000 | -0.0028 | -0.0040 |
| 0.30 | 0.0040 | 0.0012 | 0.0000 | -0.0028 | -0.0040 |
| 0.50 | 0.0040 | 0.0012 | 0.0000 | -0.0028 | -0.0040 |
| 0.70 | 0.0040 | 0.0012 | 0.0000 | -0.0028 | -0.0040 |

### test

| t | P(v_y>t) GT | P(v_ŷ>t) hard | P(v_p>t) soft | ΔP(pred−GT) hard | ΔP(soft−GT) |
|---|---:|---:|---:|---:|---:|
| 0.00 | 0.0042 | 0.0011 | 0.0063 | -0.0031 | 0.0021 |
| 0.05 | 0.0042 | 0.0011 | 0.0019 | -0.0031 | -0.0024 |
| 0.10 | 0.0042 | 0.0011 | 0.0006 | -0.0031 | -0.0036 |
| 0.20 | 0.0042 | 0.0011 | 0.0001 | -0.0031 | -0.0041 |
| 0.30 | 0.0042 | 0.0011 | 0.0000 | -0.0031 | -0.0042 |
| 0.50 | 0.0042 | 0.0011 | 0.0000 | -0.0031 | -0.0042 |
| 0.70 | 0.0042 | 0.0011 | 0.0000 | -0.0031 | -0.0042 |

### bolivia

| t | P(v_y>t) GT | P(v_ŷ>t) hard | P(v_p>t) soft | ΔP(pred−GT) hard | ΔP(soft−GT) |
|---|---:|---:|---:|---:|---:|
| 0.00 | 0.0050 | 0.0021 | 0.0118 | -0.0029 | 0.0067 |
| 0.05 | 0.0050 | 0.0021 | 0.0030 | -0.0029 | -0.0020 |
| 0.10 | 0.0050 | 0.0021 | 0.0008 | -0.0029 | -0.0042 |
| 0.20 | 0.0050 | 0.0021 | 0.0000 | -0.0029 | -0.0050 |
| 0.30 | 0.0050 | 0.0021 | 0.0000 | -0.0029 | -0.0050 |
| 0.50 | 0.0050 | 0.0021 | 0.0000 | -0.0029 | -0.0050 |
| 0.70 | 0.0050 | 0.0021 | 0.0000 | -0.0029 | -0.0050 |

---

## 6. Useful violation rate (D8)

Among D8-active prediction violations (predicted water upstream of predicted dry downstream),
what fraction are useful (GT dry = real FP suppression) vs harmful (GT water = recall loss)?

| split | n violations | useful (GT dry) | harmful (GT water) | GT-endorsed | enrichment factor |
|-------|---:|---:|---:|---:|---:|
| val | 23168 | 0.586 | 0.414 | 0.089 | 3.91× |
| test | 21950 | 0.586 | 0.414 | 0.102 | 4.44× |
| bolivia | 5875 | 0.609 | 0.391 | 0.069 | 3.69× |

Enrichment = P(GT dry | D8-violation) / P(GT dry | predicted water).
Value > 1 means violations are enriched in false positives vs average predicted-water pixels.

---

## 7. Interpretation — Canal II-a

The combination of results above closes the label-relative headroom argument for the
**local topographic-order loss class**:

1. **H_R < 0** (robustly, CI entirely negative): predictions violate D8 and D4 *less* than labels.
2. **ΔE_topo < 0**: violation energy is lower for predictions than for labels.
3. **Distributional dominance**: prediction violation tails are below label tails at all thresholds.
4. **Useful violation rate ~59/41**: violations are enriched in FPs (~4×) but the 41% harmful
   component (true water suppressed) explains the observed precision↑/recall↓ wash with no net IoU gain.

Together: the baseline already satisfies the local topographic-order constraint as well as or
better than the reference labels.  Losses whose sole mechanism is to reduce these violations
have **no native label-relative headroom**.

---

## 8. Limitations

- **Label-relative**: 'no headroom' means relative to reference labels y, which may themselves
  be topographically noisy. If the true physical flood state y★ is cleaner, there could be
  headroom relative to y★ — but y★ is unobserved.
- **Local monotone constraints only**: This closes headroom for losses based on reducing
  v_ij = max(0, p_i − p_j − τ) for higher-to-lower neighbouring pixels (D4, D8 and variants).
  **Constraints based on hydrological connectivity, flow accumulation, HAND, basin routing,
  or other orthogonal statistics are outside scope and not tested here.**
- **Single seed**: seed 0 baseline only. Multi-seed variance not estimated here.

---

*Generated by `experiments_cvpr/segman/tools/topographic_order_headroom_diagnostics.py`.*
*Checkpoint: `E:\flood_research\experiments\segman\runs\segman_n100_dice_ce_seed0\checkpoints\best_checkpoint.pt`*
