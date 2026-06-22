# STEP 6A - Physics Topographic Loss Implementation

## Status

Result: PASS

STEP 6A implemented and verified the first differentiable topographic physics
loss for future flood segmentation training. No real model training, physics-loss
training, DARN training, STURM-Flood training, DEM/HAND download, or raw-data
modification was started.

## Files Created

- `src/losses/physics_topographic_loss.py`
- `src/losses/combined_loss.py`
- `src/losses/__init__.py`
- `tests/test_physics_topographic_loss.py`
- `scripts/physics/step6a_synthetic_topographic_loss_smoke.py`
- `configs/physics_loss/terramind_l_upernet_topographic_loss_stub.yaml`
- `configs/physics_loss/terramind_base_unetdecoder_topographic_loss_control_stub.yaml`
- `reports/STEP_6A_physics_topographic_loss_implementation_report.md`

STEP 6A run directory:

`E:/flood_research/experiments/terramind_baseline/runs/step6a_physics_topographic_loss_implementation`

Run subfolders created:

- `reports/`
- `metrics/`
- `smoke/`
- `configs/`
- `logs/`

## Mathematical Formulation

The implemented total loss is:

```text
L_total = L_CE + lambda_topo * L_topo
```

The topographic loss penalizes local high-to-low violations:

```text
L_topo = (1 / |P|) * sum_{(i,j) in P} w_ij * p_i_water * (1 - p_j_water)
P = {(i,j): h_i > h_j + elevation_margin}
```

where:

- `p_i_water` is the softmax water probability for pixel `i`;
- class index `1` is treated as water;
- `h_i` and `h_j` are aligned topographic values;
- invalid labels where `target == -1` are ignored;
- non-finite topographic values are ignored;
- `w_ij` is either an elevation-difference weight or `1.0`, depending on
  `use_elevation_weight`.

The default neighborhood is 4-neighborhood. 8-neighborhood is supported.

## Implementation Details

`TopographicInconsistencyLoss` accepts:

- `logits` shaped `[B, C, H, W]`;
- `target` shaped `[B, H, W]` or `[B, 1, H, W]`;
- `topography` shaped `[B, H, W]` or `[B, 1, H, W]`.

It is differentiable with respect to logits and returns a safe zero connected to
the logits graph when there are no valid high-to-low neighbor pairs.

`CombinedSegmentationPhysicsLoss` combines:

- `torch.nn.CrossEntropyLoss(ignore_index=-1)`;
- optional class weights;
- `TopographicInconsistencyLoss`;
- a configurable `lambda_topo`.

The returned dictionary contains:

- `loss_total`
- `loss_seg`
- `loss_topo`
- `lambda_topo`

The code is model-agnostic and can later be used with both:

- main future target: `TerraMind-L pretrained + UPerNet`
- control baseline: `TerraMind base pretrained + UNetDecoder`

## Unit Tests

Command executed with the project virtual environment because bare `python` is
not available in this Windows shell:

```powershell
E:\flood_research\venvs\terramind-gpu\Scripts\python.exe -m pytest tests/test_physics_topographic_loss.py -q
```

Result:

```text
9 passed in 2.82s
```

Coverage included:

- basic shape test;
- backward/differentiability;
- ignore-index behavior;
- no valid high-to-low pairs;
- positive penalty for an inconsistent high-water / low-dry case;
- lower penalty for a coherent high-dry / low-water case;
- non-finite topography safety;
- CPU/CUDA consistency when CUDA is available;
- combined loss correctness.

Pytest log:

`E:/flood_research/experiments/terramind_baseline/runs/step6a_physics_topographic_loss_implementation/logs/step6a_pytest.log`

## Synthetic Smoke

Command executed:

```powershell
E:\flood_research\venvs\terramind-gpu\Scripts\python.exe scripts\physics\step6a_synthetic_topographic_loss_smoke.py
```

Result:

```text
loss_total=0.101080
loss_seg=0.077794
loss_topo=0.465714
lambda_topo=0.050000
gradient_l1=0.074228
```

Smoke summary JSON:

`E:/flood_research/experiments/terramind_baseline/runs/step6a_physics_topographic_loss_implementation/smoke/step6a_synthetic_loss_smoke_summary.json`

The smoke used synthetic tensors only and did not read Sen1Floods11 imagery,
DEM, HAND, checkpoints, or raw data.

## Config Stubs

Main SOTA target stub:

`configs/physics_loss/terramind_l_upernet_topographic_loss_stub.yaml`

- model: `TerraMind-L pretrained + UPerNet`
- baseline run: STEP 5O
- lambda candidates: `0.01`, `0.05`, `0.1`
- status: `not_ready_for_training_until_topographic_alignment_is_validated`

Control stub:

`configs/physics_loss/terramind_base_unetdecoder_topographic_loss_control_stub.yaml`

- model: `TerraMind base pretrained + UNetDecoder`
- baseline run: STEP 5I
- lambda candidates: `0.01`, `0.05`, `0.1`
- status: `secondary_control_experiment`

## Limitations

This loss is a local topographic plausibility prior. It does not model:

- flow accumulation;
- drainage connectivity;
- levees, roads, culverts, buildings, or barriers;
- rainfall intensity;
- hydrodynamics;
- uncertainty in flood labels;
- whether the topography source is aligned with Sentinel-1/Sentinel-2 pixels.

It should be treated as a first differentiable regularizer, not a complete
physics model.

## Topographic Data Requirements

Before any physics-loss training, STEP 6B must validate or build topographic
inputs that are:

- spatially aligned to each Sen1Floods11 tile;
- resampled to the segmentation grid;
- finite or explicitly masked;
- consistently georeferenced;
- documented with nodata handling;
- compatible with the train/valid/test/Bolivia split manifests.

No DEM/HAND/topographic data was downloaded or generated in STEP 6A.

## Why No Training Was Started

STEP 6A is implementation-only by design. Training with this loss before
validating topographic alignment would risk optimizing against misregistered
height data and contaminating the comparison between classical and
physics-informed baselines.

Guardrail status:

- Physics-loss training started: False
- DARN started: False
- STURM-Flood training started: False
- Raw data modified: False
- DEM/HAND/topographic download started: False
- Real model training started: False

## Next Step

STEP 6B should validate or build aligned topographic inputs for Sen1Floods11.
Only after human validation should a launcher be created for any future
long-running physics-informed training step.
