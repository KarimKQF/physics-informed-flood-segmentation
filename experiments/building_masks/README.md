# OpenStreetMap Building Masks

This directory contains an isolated experiment for fetching, visualizing, and
rasterizing OpenStreetMap building footprints. It is not yet connected to the
training pipeline, dataloader, model, losses, DEM workflow, Sen1Floods11, or
STURM-Flood scripts.

## Rasterizing OSM buildings on a reference raster grid

The OSM GeoJSON contains vector building polygons, while the segmentation model
works on rasters. To use buildings later, the polygons must be rasterized on the
exact grid of a reference raster. This guarantees the same CRS, transform,
resolution, width, height, bounds, and pixel layout.

This step is required before constructing the future spatial weighting mask
`q_i`, where building pixels can be treated as unreliable locations for a more
robust topographic constraint.

Example:

```bash
python experiments/building_masks/scripts/rasterize_osm_buildings_to_reference.py \
  --buildings experiments/building_masks/outputs/geojson/osm_buildings_saint_etienne_du_rouvray.geojson \
  --reference path/to/reference_image.tif \
  --output experiments/building_masks/outputs/building_mask_aligned.tif \
  --all-touched \
  --overwrite
```

Then validate the alignment:

```bash
python experiments/building_masks/scripts/validate_building_mask_alignment.py \
  --reference path/to/reference_image.tif \
  --mask experiments/building_masks/outputs/building_mask_aligned.tif
```

The aligned building mask is not yet used by the training pipeline. It is a
preparatory step toward constructing the future spatial weighting mask `q_i`.

## Development reference raster

Until no real local Sentinel, DEM, or STURM-Flood raster is available, a
georeferenced development reference raster can be generated from the OSM
building footprint extent.

This raster is only a development support. It is not a real satellite raster. It
only exists to test the geospatial alignment chain:

```text
OSM building GeoJSON -> reference raster -> aligned building mask -> validation -> figure
```

Create the development reference raster:

```powershell
python experiments\building_masks\scripts\create_reference_raster_from_buildings.py `
  --buildings experiments\building_masks\outputs\geojson\osm_buildings_saint_etienne_du_rouvray.geojson `
  --output experiments\building_masks\outputs\reference\reference_saint_etienne_10m.tif `
  --target-crs EPSG:2154 `
  --resolution 10 `
  --buffer 100 `
  --overwrite
```

Rasterize OSM buildings on that reference grid:

```powershell
python experiments\building_masks\scripts\rasterize_osm_buildings_to_reference.py `
  --buildings experiments\building_masks\outputs\geojson\osm_buildings_saint_etienne_du_rouvray.geojson `
  --reference experiments\building_masks\outputs\reference\reference_saint_etienne_10m.tif `
  --output experiments\building_masks\outputs\building_mask_aligned_saint_etienne_10m.tif `
  --all-touched `
  --overwrite
```

Validate the alignment:

```powershell
python experiments\building_masks\scripts\validate_building_mask_alignment.py `
  --reference experiments\building_masks\outputs\reference\reference_saint_etienne_10m.tif `
  --mask experiments\building_masks\outputs\building_mask_aligned_saint_etienne_10m.tif
```

Create an overlay figure:

```powershell
python experiments\building_masks\scripts\visualize_aligned_building_mask.py `
  --reference experiments\building_masks\outputs\reference\reference_saint_etienne_10m.tif `
  --mask experiments\building_masks\outputs\building_mask_aligned_saint_etienne_10m.tif `
  --buildings experiments\building_masks\outputs\geojson\osm_buildings_saint_etienne_du_rouvray.geojson `
  --output experiments\building_masks\outputs\figures\building_mask_aligned_overlay.png `
  --overwrite
```
