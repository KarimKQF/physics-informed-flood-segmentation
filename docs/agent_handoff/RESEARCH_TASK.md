# Research Task Template

## Task Owner

- Scientific request written by:
- Implementation owner:
- Human validation required: yes

## Standing Rules

1. Do not launch full training unless explicitly instructed.
2. If a long job is launched, it must run in background, report PID and log path, then stop immediately.
3. Do not monitor logs continuously.
4. Do not use DEM as model input unless explicitly approved.
5. DEM must currently be used only inside the topographic loss.
6. Do not modify raw data.
7. Do not start DARN or STURM-Flood training unless explicitly approved.
8. Preserve all previous logs, configs, metrics, reports, and checkpoints.
9. Every experiment must have a unique run directory.
10. Every implementation task must end with a short report.

## Objective

Describe the research or implementation objective in one short paragraph.

## Scientific Rationale

Explain why this task matters for the experiment or paper.

## Required Controls

- Dataset:
- Split protocol:
- Model:
- Decoder:
- Feature indices:
- Losses:
- Input modalities:
- DEM usage:
- Training recipe:
- Evaluation splits:

## Success Criteria

- Criterion 1:
- Criterion 2:
- Criterion 3:

## Explicit Non-Goals

- Do not:
- Do not:
- Do not:

## Requested Outputs

- Config path:
- Script path:
- Run directory:
- Metrics:
- Report:

## Human Approval Needed Before

- Full training:
- New dataset training:
- DEM-as-input experiments:
- DARN/STURM-Flood experiments:
