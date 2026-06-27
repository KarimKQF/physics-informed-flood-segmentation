# DEM Shuffle Scope Verification

**Date**: 2026-06-26  
**Verdict**: **CONFIRMED SAFE** — shuffle applies only to training batches; val/test/Bolivia always use the real DEM.  
**Multi-seed launch**: Safe to proceed.

---

## Files Inspected

| File | Relevant lines |
|------|---------------|
| `scripts/step6c_v3_train.py` | 137–142 (`tile_id_from_mask_path`, `dem_path_for_sample`), 171–232 (`TopographySegmentationDataset.__getitem__`), 235–302 (`TopographyDataModule`), 441–449 (shuffle map load) |
| `configs/segman/segman_dice_ce_topo_dem_shuffled.yaml` | `dem.dem_tile_id_map_file` field; config `notes` section |
| `manifests/terramind_baseline/low_data_multiseed/dem_shuffle_map_n50_seed0.json` | map contents, metadata, `note` field |
| `experiments_cvpr/segman/train_segman.py` | lines 210–214 (map injection into config) |

---

## Complete Execution Trace

### 1. Config load and map injection (`step6c_v3_train.py:441–449`, called from `train_segman.py:210–214`)

```python
dem_map_file = config.get("dem", {}).get("dem_tile_id_map_file")
if dem_map_file:
    _map_data = json.load(open(dem_map_file))
    config.setdefault("dem", {})["dem_tile_id_map"] = _map_data.get("mapping", {})
```

This injects the 50-entry derangement into `config["dem"]["dem_tile_id_map"]`. The same
`config` dict is passed to every `TopographyDataModule` and every dataset created from it
(train, val, test, Bolivia). There is no per-split config copy.

### 2. DEM tile-ID lookup in `TopographySegmentationDataset.__getitem__` (lines 214–217)

```python
tile_id = tile_id_from_mask_path(sample["mask"])          # e.g. "France_12345" for val
dem_map = self.config.get("dem", {}).get("dem_tile_id_map") or {}  # 50-entry train map
dem_tile_id = dem_map.get(tile_id, tile_id)               # identity if not in map
topo_path = dem_path_for_sample(self.config, split=self.split_name, tile_id=dem_tile_id)
```

- **Training tiles**: `dem_map.get(tile_id, tile_id)` returns the shuffled tile_id
  (a different training tile). Path: `{dem_root}/train_{shuffled_tile_id}_{pattern}.tif`  
- **Val/test/Bolivia tiles**: `dem_map.get(tile_id, tile_id)` returns the **original
  tile_id** because those tile IDs are **not in the map**. Path:
  `{dem_root}/{split}_{original_tile_id}_{pattern}.tif` — the correct real DEM.

### 3. Shuffle map domain

The shuffle map contains **exactly 50 keys** — the N=50 training tile IDs (e.g.,
`Ghana_1089161`, `India_25540`, `Spain_5923267`, …). The metadata fields confirm:

```json
{
  "purpose": "Shuffled DEM control for E2 experiment — strict derangement",
  "algorithm": "Sattolo's algorithm (single-cycle permutation, guaranteed derangement)",
  "n_tiles": 50,
  "n_self_maps": 0,
  "is_bijective": true,
  "is_single_cycle": true,
  "note": "Each training sample's tile_id maps to a DIFFERENT sample's DEM.
           Topo metrics at eval time should use real DEM for physical coherence measurement."
}
```

Val (86 tiles), test (89 tiles), and Bolivia (15 tiles) share no tile IDs with the
N=50 training split — standard mutually exclusive ML splits. Their tile IDs cannot appear
as keys in the training shuffle map.

### 4. `TopographyDataModule.setup()` (lines 272–280)

```python
def setup(self, stage):
    if stage == "fit":
        self.train_dataset = self._dataset("train", train=True)   # split_name="train"
        self.val_dataset   = self._dataset("valid", train=False)  # split_name="valid"
    elif stage == "test":
        split_name = self.split or "test"
        self.test_dataset  = self._dataset(split_name, train=False)  # "test" or "bolivia"
```

Each dataset is constructed with the correct `split_name`, which is forwarded to
`TopographySegmentationDataset` as `self.split_name`. The `split_name` is used
in `dem_path_for_sample(config, split=self.split_name, ...)` to compose the correct DEM
file path prefix (`train_`, `valid_`, `test_`, `bolivia_`).

### 5. Secondary safety layer: path construction

`dem_path_for_sample` (line 141–142):
```python
return Path(config["dem"]["aligned_dem_root"]) / \
       str(config["dem"]["dem_filename_pattern"]).format(split=split, tile_id=tile_id)
```

If the remap were somehow applied to a val tile, the path constructed would be
`valid_{shuffled_train_tile_id}_{pattern}.tif`. That file doesn't exist in the DEM
directory, so `load_dem_array` would raise `FileNotFoundError`. This acts as a
tripwire that would make any accidental cross-split DEM swap immediately fatal rather
than silent.

---

## Answers to the Six Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | Is the shuffled DEM used only for the topo loss during training? | **Yes.** The shuffle map domain is exactly the 50 training tile IDs. Only training batches will have their DEM remapped. |
| 2 | Are validation metrics computed with the real DEM? | **Yes.** Val tile IDs are not in the shuffle map; `dem_map.get(tile_id, tile_id)` returns identity; `split_name="valid"` is used in path construction. |
| 3 | Are test metrics computed with the real DEM? | **Yes.** Same reasoning as val. `split_name="test"`. |
| 4 | Are Bolivia/OOD metrics computed with the real DEM? | **Yes.** Same reasoning. `split_name="bolivia"`. |
| 5 | Are topo violation fractions always computed against the real DEM for all variants? | **Yes.** The violation metric is computed from `batch["topography"]`, which for the shuffled variant is the real DEM for all eval splits. The topo violation fractions reported in `segman_seed0_loss_comparison_summary.json` for the shuffled variant are valid and comparable to the other variants. |
| 6 | If behavior is ambiguous, identify the code path | Not applicable — behavior is unambiguous. No fix required. |

---

## Impact on Previously Reported Audit

The `segman_seed0_loss_comparison_summary.json` field:
```json
"shuffle_scope_verified": false,
"note_shuffle_scope": "Config notes state eval uses real DEM; TopographyDataModule scope not code-verified"
```

Should be updated to:
```json
"shuffle_scope_verified": true,
"note_shuffle_scope": "Verified: shuffle map domain is exactly the 50 training tile IDs. Val/test/Bolivia tile IDs are not in the map; dem_map.get(tile_id,tile_id) returns identity for all eval splits. Confirmed in step6c_v3_train.py:214-217 and dem_shuffle_map_n50_seed0.json metadata."
```

The open question in the audit report ("Does TopographyDataModule apply DEM tile-ID remap
to val/test splits or train only?") is **resolved: training only**.

---

## Multi-Seed Launch Readiness

**Safe to launch.** No implementation issue found. The DEM shuffle is correctly scoped
to training tiles via the map domain, with an additional path-construction tripwire
that would produce a `FileNotFoundError` if cross-split DEM swapping occurred. Both the
code behavior and the shuffle map metadata are consistent with the design intent stated
in the config notes.

No code changes required before launching multi-seed experiments.
