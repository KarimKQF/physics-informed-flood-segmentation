# AnySat / Panopticon Render Instructions

Files:

- `videos/anysat_panopticon_explanation.py`
- `videos/anysat_panopticon_voiceover.md`
- `videos/anysat_panopticon_storyboard.md`

## Syntax Check

Run:

```powershell
python -m py_compile videos/anysat_panopticon_explanation.py
```

This is the only command requested during creation. It does not render video and
does not touch training.

## Quick Preview Render

Use this only when explicitly ready to render:

```powershell
manim -pql videos/anysat_panopticon_explanation.py AnySatPanopticonVideo
```

## High Quality Render

```powershell
manim -pqh videos/anysat_panopticon_explanation.py AnySatPanopticonVideo
```

## Individual Scene Renders

```powershell
manim -pql videos/anysat_panopticon_explanation.py S00_Opening
manim -pql videos/anysat_panopticon_explanation.py S01_WhyRGBIsProblem
manim -pql videos/anysat_panopticon_explanation.py S02_AnySatConcept
manim -pql videos/anysat_panopticon_explanation.py S03_AnySatArchitecture
manim -pql videos/anysat_panopticon_explanation.py S04_AnySatForOurProject
manim -pql videos/anysat_panopticon_explanation.py S05_PanopticonConcept
manim -pql videos/anysat_panopticon_explanation.py S06_PanopticonArchitecture
manim -pql videos/anysat_panopticon_explanation.py S07_PanopticonForOurProject
manim -pql videos/anysat_panopticon_explanation.py S08_Comparison
manim -pql videos/anysat_panopticon_explanation.py S09_PhysicsPipeline
manim -pql videos/anysat_panopticon_explanation.py S10_FinalRoadmap
```

## Notes

- Rendering with Manim is CPU-oriented here; do not launch GPU workloads.
- The script uses synthetic diagrams only: no copyrighted images.
- The video does not load models, checkpoints, datasets, or experiment runs.
- The active SegMAN N=100 diagnostic chain should not be stopped or monitored
  while rendering decisions are being made.

