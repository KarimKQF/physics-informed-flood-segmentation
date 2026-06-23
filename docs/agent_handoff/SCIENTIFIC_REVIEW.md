# Scientific Review Template

## Reviewer

- Reviewer:
- Date:
- Reviewed Codex report:

## Standing Rules Checked

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

## Scientific Validity

- Protocol parity acceptable: yes/no
- Dataset split preserved: yes/no
- Model parity preserved: yes/no
- DEM usage acceptable: yes/no
- Loss comparison isolated: yes/no

## Findings

| Finding | Severity | Evidence |
|---|---|---|
|  |  |  |

## Decision

Choose one:

- Approved for next step.
- Needs implementation revision.
- Needs additional diagnostics.
- Blocked pending human decision.

## Required Changes

- Change 1:
- Change 2:

## Human Validation Notes

Summarize what the human user must approve before the next major action.
