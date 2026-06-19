# OpenStreetMap Building Masks Experiment

This module is an isolated experiment for fetching and visualizing building
footprints from OpenStreetMap. It prepares a future reliability mask for the
advanced topographic loss without changing the main dataset workflow, training
loop, dataloader, losses, DEM handling, or Sen1Floods11/STURM-Flood scripts.

## Scientific Motivation

The robust topographic constraint will later use a reliability coefficient:

```text
m_ij = q_i q_j c_ij
```

Here, `q_i` should be `0` on pixels where the topographic constraint is not
reliable, for example buildings, roofs, or other artificial surfaces, and `1`
elsewhere. This module only prepares the building information needed for that
future mask.

OpenStreetMap gives vector polygons. The future loss will need a raster mask:

```text
1 = building
0 = non-building
```

The conversion from polygons to a raster must eventually be aligned with the
same CRS, transform, width, and height as the imagery or DEM used by the model.

## Install

From the repository root:

```bash
pip install -r experiments/building_masks/requirements_building_masks.txt
```

These dependencies are kept outside the main `pyproject.toml` on purpose.

## Configure

Edit:

```text
experiments/building_masks/configs/building_osm_config.yaml
```

Two query modes are available.

Place mode:

```yaml
query_mode: place
place_name: "Saint-Étienne-du-Rouvray, Seine-Maritime, Normandie, France"
```

Bounding-box mode:

```yaml
query_mode: bbox
bbox:
  north: 49.40
  south: 49.35
  east: 1.15
  west: 1.05
```

## Fetch Buildings

From the repository root:

```bash
python experiments/building_masks/scripts/fetch_osm_buildings.py
```

The script writes:

```text
experiments/building_masks/outputs/geojson/osm_buildings.geojson
```

It prints the number of buildings, CRS, bounds, and output path.

## Visualize Buildings

From the repository root:

```bash
python experiments/building_masks/scripts/visualize_osm_buildings.py
```

The script writes:

```text
experiments/building_masks/outputs/figures/osm_buildings.png
experiments/building_masks/outputs/maps/osm_buildings_map.html
```

Open the HTML file in a browser to inspect the interactive map.

## Example: Saint-Etienne-du-Rouvray

Saint-Etienne-du-Rouvray can be fetched directly by place name:

```yaml
query_mode: place
place_name: "Saint-Étienne-du-Rouvray, Seine-Maritime, Normandie, France"
```

From the repository root:

```bash
python experiments/building_masks/scripts/fetch_osm_buildings.py
python experiments/building_masks/scripts/visualize_osm_buildings.py
```

The configured outputs are:

```text
experiments/building_masks/outputs/geojson/osm_buildings_saint_etienne_du_rouvray.geojson
experiments/building_masks/outputs/figures/osm_buildings_saint_etienne_du_rouvray.png
experiments/building_masks/outputs/maps/osm_buildings_saint_etienne_du_rouvray.html
```

## Optional Rasterization

From the repository root:

```bash
python experiments/building_masks/scripts/rasterize_osm_buildings.py
```

Without `raster.reference_raster`, the script creates a standalone test mask
from the configured bbox or the building bounds at `raster.resolution_meters`.
This is useful for checking the pipeline, but it is not the final `q_i` mask.

For the final research workflow, set `raster.reference_raster` to a real image
or DEM. The output mask will then use the same CRS, transform, width, and height
as that reference raster.

## Why This Is Isolated

The current goal is only to verify access to OSM building polygons and produce
clear visual outputs. Nothing here is imported by the model code. The main
dataset workflow remains unchanged until the building reliability mask is ready
to be integrated deliberately.
