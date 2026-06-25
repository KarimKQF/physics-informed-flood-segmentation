# N=50 Manifest Composition Audit

**Generated:** 2026-06-24  
**Purpose:** Assess whether manifest composition (water content, country distribution) explains collapse behaviour in seeds 1 and 3.

---

## 1. Water Statistics per Seed

| Seed | Total | Water+ | No-water | % Water+ | Total water px | Mean frac | Median frac | >1% | >5% | >10% | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 50 | 45 | 5 | 90.0% | 1 074 446 | 0.0946 | 0.0207 | 33 | 19 | 12 | Dice collapse, Real rescued, Shuffled all-water |
| 1 | 50 | 46 | 4 | 92.0% | **1 680 776** | **0.1338** | **0.0257** | **35** | **21** | **15** | **collapse everywhere** |
| 2 | 50 | 47 | 3 | 94.0% | 975 785 | 0.0846 | 0.0229 | 34 | 19 | 14 | Dice all-water, Real rescued, Shuffled rescued |
| 3 | 50 | 42 | 8 | 84.0% | 1 102 071 | 0.0950 | 0.0171 | 32 | 17 | 11 | **collapse everywhere** |
| 42 | 50 | 42 | 11 | 84.0% | 724 717 | 0.0629 | 0.0132 | 28 | 15 | 11 | Dice collapse, Real rescued, Shuffled rescued |

Labels: 0=background, 1=water, -1=invalid (excluded from all counts).  
Source: `LabelHand/*.tif` rasterio read.

---

## 2. Country Distribution

| Seed | Ghana | India | Mekong | Nigeria | Pakistan | Paraguay | Somalia | Spain | Sri-Lanka | USA |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 15 | 1 | 3 | 3 | 7 | 3 | 4 | 4 | 8 |
| 1 | 4 | 5 | 4 | 3 | 4 | 9 | 3 | 4 | 5 | 9 |
| 2 | 4 | 5 | 4 | 0 | 4 | 7 | 6 | 6 | 4 | 10 |
| 3 | **10** | **12** | 1 | 1 | 2 | 7 | 5 | 6 | 3 | 3 |
| 42 | 8 | 6 | 4 | 1 | 5 | 5 | 3 | 3 | 5 | 10 |

---

## 3. Key Findings

### Seed 1 has the MOST water
Seed 1 has more total water pixels (1,680,776), higher mean water fraction (0.134), and more high-water tiles (>5%: 21, >10%: 15) than any other seed — including the rescued seeds 2 and 42. Despite this, seed 1 collapses under all conditions tested.

### Seed 42 has the LEAST water and gets rescued
Seed 42 has the lowest total water (724,717 px), lowest mean fraction (0.063), and fewest high-water tiles (>10%: 11). Yet it is rescued by both Physics real DEM and Physics shuffled DEM.

### Seed 3 has fewer water-positive tiles but not dramatically so
Seed 3 has 42/50 water-positive tiles vs 46/50 for seed 1. Median fraction (0.017) is lower than rescued seeds. However, the difference is marginal.

### Country composition shows seed 3 is India/Ghana heavy
Seed 3 has 10 Ghana + 12 India = 44% of tiles from these two regions, vs 9/50 (18%) for seed 1 and 14/50 (28%) for seed 42. Ghana tiles are often spectrally uniform with low water signal; heavy representation may increase difficulty. However, seed 0 also has 15 India tiles and seed 0 is rescued by real DEM.

---

## 4. Interpretation

**The composition hypothesis is not supported for seed 1.** Seed 1 has MORE water than any rescued seed. The collapse cannot be explained by insufficient water signal in the training data.

**Seed 3 has a marginally less favourable composition** (more Ghana/India, lower median water fraction) but the differences are not dramatic enough to explain complete collapse across all conditions.

**The dominant explanation must be optimization dynamics.** Seeds 1 and 3 lock into the all-background attractor before the model can learn water-positive gradients, regardless of whether the training data contains water. The Dice loss gradient is uninformative when the model predicts all-background (Dice ≈ 0 for both classes), creating a flat gradient region. Cross-Entropy does not share this property — it provides a non-zero gradient even at all-background initialization.

**This motivates the Dice+CE ablation:** CE provides non-zero supervision on each pixel class from epoch 1, potentially preventing the initialization-dependent lock-in observed in seeds 1 and 3.

---

## 5. Files

- Labels: `E:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/`
- Manifests: `C:/flood_research/repos/.../manifests/terramind_baseline/low_data_multiseed/flood_train_low_data_n50_seed*.txt`
- Raw data: `results/n50_manifest_composition_audit.json`
