# STEP 5D - TerraMind Tiny Micro-Overfit Smoke Test

## Status

STEP 5D is complete. This was a tiny micro-overfit smoke test only; no full training, UPerNet, DARN, STURM-Flood training, physics loss, raw-data modification, or checkpoint creation was performed.

Result: PASS

## Scope

- Dataset: Sen1Floods11 train split only
- Model: terramind_v1_tiny
- Decoder: UNetDecoder
- Backbone pretrained: False
- Ignore index: -1
- Optimizer: AdamW
- Learning rate: 0.0003
- Precision requested/used: 16-mixed / 16-mixed
- Batch size: 1
- Num workers: 0
- Seed: 42
- Max steps / completed steps: 60 / 60
- Trainable parameters: 2361714
- Total parameters: 8433330

## Selected Samples

Explicitly excluded fully invalid LabelHand samples:
Ghana_234935, Ghana_26376, Ghana_277, Ghana_5079, Ghana_83483.

| Sample ID | Valid pixels | Water pixels | Background pixels | Invalid pixels | Water share of valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| USA_1068362 | 262124 | 31182 | 230942 | 20 | 11.9% |
| India_773682 | 252793 | 30959 | 221834 | 9351 | 12.25% |
| India_391908 | 211832 | 25989 | 185843 | 50312 | 12.27% |
| Sri-Lanka_523539 | 218481 | 24669 | 193812 | 43663 | 11.29% |

## Before/After Metrics

Metrics were computed with the existing project metrics modules under src/metrics/, using ignore_index=-1.

| Phase | Loss | Accuracy | IoU background | IoU water | mIoU | F1 water | Precision water | Recall water | Valid pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | 0.6391 | 0.8352 | 0.8342 | 0.0352 | 0.4347 | 0.068 | 0.1046 | 0.0504 | 945230 |
| After | 0.2815 | 0.9266 | 0.9219 | 0.4509 | 0.6864 | 0.6216 | 0.8075 | 0.5053 | 945230 |

Loss decreased: True

## Per-Sample Change

| Sample ID | mIoU before | mIoU after | F1 water before | F1 water after |
| --- | ---: | ---: | ---: | ---: |
| USA_1068362 | 0.4222 | 0.772 | 0.0547 | 0.7494 |
| India_773682 | 0.4304 | 0.6118 | 0.0536 | 0.4876 |
| India_391908 | 0.4437 | 0.601 | 0.0916 | 0.4614 |
| Sri-Lanka_523539 | 0.4468 | 0.7607 | 0.0806 | 0.7362 |

## Runtime And GPU

- Device: NVIDIA GeForce RTX 4070
- CUDA available: True
- Training elapsed seconds: 24.831
- Total elapsed seconds: 42.246
- GPU memory allocated/reserved MB: 90.24 / 180.0
- GPU peak allocated/reserved MB: 152.99 / 180.0

## Artifacts

- Script: E:/flood_research/experiments/terramind_baseline/scripts/step5d_tiny_micro_overfit.py
- Config: $(@{step=5D; status=done; training_scope=tiny micro-overfit smoke test only; full_training_started=False; upernet_started=False; darn_started=False; sturm_training_started=False; physics_loss_started=False; raw_data_modified=False; repo_root=C:\Users\Karim\Desktop\flood-segmentation-training\physics-informed-flood-segmentation; config_path=E:\flood_research\experiments\terramind_baseline\configs\terramind_v1_tiny_sen1floods11_micro_overfit.yaml; base_config_path=E:/flood_research/experiments/terramind_baseline/configs/terramind_v1_tiny_sen1floods11_gpu_smoke.yaml; micro_manifest=E:\flood_research\experiments\terramind_baseline\manifests\flood_train_micro_overfit.txt; venv_path=E:/flood_research/venvs/terramind-gpu; device=NVIDIA GeForce RTX 4070; cuda_available=True; cuda_device_count=1; model=terramind_v1_tiny; decoder=UNetDecoder; backbone_pretrained=False; ignore_index=-1; optimizer=AdamW; learning_rate=0.0003; requested_precision=16-mixed; precision_used=16-mixed; batch_size=1; num_workers=0; max_steps=60; steps_completed=60; seed=42; selected_samples=System.Object[]; train_dataset_len=4; trainable_param_count=2361714; total_param_count=8433330; initial_loss=0.6391364336013794; final_loss=0.28154638409614563; loss_decreased=True; before_metrics=; after_metrics=; metrics_csv=E:\flood_research\experiments\terramind_baseline\metrics\micro_overfit_metrics.csv; prediction_artifact_dir=E:\flood_research\experiments\terramind_baseline\predictions\micro_overfit; log_path=E:\flood_research\experiments\terramind_baseline\logs\STEP_5D_micro_overfit.log; checkpoint_saved=False; train_elapsed_seconds=24.831; elapsed_seconds=42.246; gpu_memory_allocated_mb=90.24; gpu_memory_reserved_mb=180.0; gpu_peak_memory_allocated_mb=152.99; gpu_peak_memory_reserved_mb=180.0; micro_overfit_passed=True; next_step_allowed=False; human_validation_required=True}.config_path.Replace('\','/'))
- Micro manifest: $(@{step=5D; status=done; training_scope=tiny micro-overfit smoke test only; full_training_started=False; upernet_started=False; darn_started=False; sturm_training_started=False; physics_loss_started=False; raw_data_modified=False; repo_root=C:\Users\Karim\Desktop\flood-segmentation-training\physics-informed-flood-segmentation; config_path=E:\flood_research\experiments\terramind_baseline\configs\terramind_v1_tiny_sen1floods11_micro_overfit.yaml; base_config_path=E:/flood_research/experiments/terramind_baseline/configs/terramind_v1_tiny_sen1floods11_gpu_smoke.yaml; micro_manifest=E:\flood_research\experiments\terramind_baseline\manifests\flood_train_micro_overfit.txt; venv_path=E:/flood_research/venvs/terramind-gpu; device=NVIDIA GeForce RTX 4070; cuda_available=True; cuda_device_count=1; model=terramind_v1_tiny; decoder=UNetDecoder; backbone_pretrained=False; ignore_index=-1; optimizer=AdamW; learning_rate=0.0003; requested_precision=16-mixed; precision_used=16-mixed; batch_size=1; num_workers=0; max_steps=60; steps_completed=60; seed=42; selected_samples=System.Object[]; train_dataset_len=4; trainable_param_count=2361714; total_param_count=8433330; initial_loss=0.6391364336013794; final_loss=0.28154638409614563; loss_decreased=True; before_metrics=; after_metrics=; metrics_csv=E:\flood_research\experiments\terramind_baseline\metrics\micro_overfit_metrics.csv; prediction_artifact_dir=E:\flood_research\experiments\terramind_baseline\predictions\micro_overfit; log_path=E:\flood_research\experiments\terramind_baseline\logs\STEP_5D_micro_overfit.log; checkpoint_saved=False; train_elapsed_seconds=24.831; elapsed_seconds=42.246; gpu_memory_allocated_mb=90.24; gpu_memory_reserved_mb=180.0; gpu_peak_memory_allocated_mb=152.99; gpu_peak_memory_reserved_mb=180.0; micro_overfit_passed=True; next_step_allowed=False; human_validation_required=True}.micro_manifest.Replace('\','/'))
- Metrics CSV: $(@{step=5D; status=done; training_scope=tiny micro-overfit smoke test only; full_training_started=False; upernet_started=False; darn_started=False; sturm_training_started=False; physics_loss_started=False; raw_data_modified=False; repo_root=C:\Users\Karim\Desktop\flood-segmentation-training\physics-informed-flood-segmentation; config_path=E:\flood_research\experiments\terramind_baseline\configs\terramind_v1_tiny_sen1floods11_micro_overfit.yaml; base_config_path=E:/flood_research/experiments/terramind_baseline/configs/terramind_v1_tiny_sen1floods11_gpu_smoke.yaml; micro_manifest=E:\flood_research\experiments\terramind_baseline\manifests\flood_train_micro_overfit.txt; venv_path=E:/flood_research/venvs/terramind-gpu; device=NVIDIA GeForce RTX 4070; cuda_available=True; cuda_device_count=1; model=terramind_v1_tiny; decoder=UNetDecoder; backbone_pretrained=False; ignore_index=-1; optimizer=AdamW; learning_rate=0.0003; requested_precision=16-mixed; precision_used=16-mixed; batch_size=1; num_workers=0; max_steps=60; steps_completed=60; seed=42; selected_samples=System.Object[]; train_dataset_len=4; trainable_param_count=2361714; total_param_count=8433330; initial_loss=0.6391364336013794; final_loss=0.28154638409614563; loss_decreased=True; before_metrics=; after_metrics=; metrics_csv=E:\flood_research\experiments\terramind_baseline\metrics\micro_overfit_metrics.csv; prediction_artifact_dir=E:\flood_research\experiments\terramind_baseline\predictions\micro_overfit; log_path=E:\flood_research\experiments\terramind_baseline\logs\STEP_5D_micro_overfit.log; checkpoint_saved=False; train_elapsed_seconds=24.831; elapsed_seconds=42.246; gpu_memory_allocated_mb=90.24; gpu_memory_reserved_mb=180.0; gpu_peak_memory_allocated_mb=152.99; gpu_peak_memory_reserved_mb=180.0; micro_overfit_passed=True; next_step_allowed=False; human_validation_required=True}.metrics_csv.Replace('\','/'))
- Summary JSON: $(E:\flood_research\experiments\terramind_baseline\metrics\micro_overfit_summary.json.Replace('\','/'))
- Log: $(@{step=5D; status=done; training_scope=tiny micro-overfit smoke test only; full_training_started=False; upernet_started=False; darn_started=False; sturm_training_started=False; physics_loss_started=False; raw_data_modified=False; repo_root=C:\Users\Karim\Desktop\flood-segmentation-training\physics-informed-flood-segmentation; config_path=E:\flood_research\experiments\terramind_baseline\configs\terramind_v1_tiny_sen1floods11_micro_overfit.yaml; base_config_path=E:/flood_research/experiments/terramind_baseline/configs/terramind_v1_tiny_sen1floods11_gpu_smoke.yaml; micro_manifest=E:\flood_research\experiments\terramind_baseline\manifests\flood_train_micro_overfit.txt; venv_path=E:/flood_research/venvs/terramind-gpu; device=NVIDIA GeForce RTX 4070; cuda_available=True; cuda_device_count=1; model=terramind_v1_tiny; decoder=UNetDecoder; backbone_pretrained=False; ignore_index=-1; optimizer=AdamW; learning_rate=0.0003; requested_precision=16-mixed; precision_used=16-mixed; batch_size=1; num_workers=0; max_steps=60; steps_completed=60; seed=42; selected_samples=System.Object[]; train_dataset_len=4; trainable_param_count=2361714; total_param_count=8433330; initial_loss=0.6391364336013794; final_loss=0.28154638409614563; loss_decreased=True; before_metrics=; after_metrics=; metrics_csv=E:\flood_research\experiments\terramind_baseline\metrics\micro_overfit_metrics.csv; prediction_artifact_dir=E:\flood_research\experiments\terramind_baseline\predictions\micro_overfit; log_path=E:\flood_research\experiments\terramind_baseline\logs\STEP_5D_micro_overfit.log; checkpoint_saved=False; train_elapsed_seconds=24.831; elapsed_seconds=42.246; gpu_memory_allocated_mb=90.24; gpu_memory_reserved_mb=180.0; gpu_peak_memory_allocated_mb=152.99; gpu_peak_memory_reserved_mb=180.0; micro_overfit_passed=True; next_step_allowed=False; human_validation_required=True}.log_path.Replace('\','/'))
- Predictions and panels: $(@{step=5D; status=done; training_scope=tiny micro-overfit smoke test only; full_training_started=False; upernet_started=False; darn_started=False; sturm_training_started=False; physics_loss_started=False; raw_data_modified=False; repo_root=C:\Users\Karim\Desktop\flood-segmentation-training\physics-informed-flood-segmentation; config_path=E:\flood_research\experiments\terramind_baseline\configs\terramind_v1_tiny_sen1floods11_micro_overfit.yaml; base_config_path=E:/flood_research/experiments/terramind_baseline/configs/terramind_v1_tiny_sen1floods11_gpu_smoke.yaml; micro_manifest=E:\flood_research\experiments\terramind_baseline\manifests\flood_train_micro_overfit.txt; venv_path=E:/flood_research/venvs/terramind-gpu; device=NVIDIA GeForce RTX 4070; cuda_available=True; cuda_device_count=1; model=terramind_v1_tiny; decoder=UNetDecoder; backbone_pretrained=False; ignore_index=-1; optimizer=AdamW; learning_rate=0.0003; requested_precision=16-mixed; precision_used=16-mixed; batch_size=1; num_workers=0; max_steps=60; steps_completed=60; seed=42; selected_samples=System.Object[]; train_dataset_len=4; trainable_param_count=2361714; total_param_count=8433330; initial_loss=0.6391364336013794; final_loss=0.28154638409614563; loss_decreased=True; before_metrics=; after_metrics=; metrics_csv=E:\flood_research\experiments\terramind_baseline\metrics\micro_overfit_metrics.csv; prediction_artifact_dir=E:\flood_research\experiments\terramind_baseline\predictions\micro_overfit; log_path=E:\flood_research\experiments\terramind_baseline\logs\STEP_5D_micro_overfit.log; checkpoint_saved=False; train_elapsed_seconds=24.831; elapsed_seconds=42.246; gpu_memory_allocated_mb=90.24; gpu_memory_reserved_mb=180.0; gpu_peak_memory_allocated_mb=152.99; gpu_peak_memory_reserved_mb=180.0; micro_overfit_passed=True; next_step_allowed=False; human_validation_required=True}.prediction_artifact_dir.Replace('\','/'))
- Checkpoint saved: False

## Warnings And Notes

- An initial script attempt failed before optimization because the TerraTorch sample ilename field was a dictionary rather than a direct path string. The script was patched before optimization, then rerun successfully.
- TerraTorch emitted a deprecation warning for gb_modality; no action was taken in this smoke step.
- PyTorch/TerraTorch emitted a 	riton not found FLOP-counting warning; it did not block the run.
- Repository test collection is still known to fail because urban_runoff.data is missing. This is unrelated to the STEP 5D micro-overfit execution.

## Gate

Next recommended step: Wait for human validation before setting up or starting STEP 5E first real TerraMind baseline training.

Human validation required before starting STEP 5E - first real TerraMind baseline training.
