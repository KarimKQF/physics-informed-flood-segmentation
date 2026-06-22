# STEP 6B2 - DEM Source Acquisition Plan

## Purpose

STEP 6B proved that Sen1Floods11 S1/S2/LabelHand grids are internally aligned,
but no DEM/HAND/elevation source exists locally. STEP 6B2 prepares the smallest
safe DEM acquisition plan needed to rerun sample topographic alignment.

No DEM was downloaded in this planning step.

## Required Coverage

Source inventory:

`E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/inventory/sen1floods11_geospatial_inventory.csv`

Coverage summary:

- Samples: 441
- Event/location count: 11
- CRS: `EPSG:4326`
- Min longitude: `-95.666984498`
- Max longitude: `106.107564056`
- Min latitude: `-25.250564658`
- Max latitude: `40.290518471`
- Required 1-degree DEM cells: 53

Required cell manifest:

- `E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/manifests/required_dem_cells.csv`
- `E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/manifests/required_dem_cells.json`

Cells per event/location:

| Event/location | Required cells |
| --- | ---: |
| Bolivia | 5 |
| Ghana | 6 |
| India | 6 |
| Mekong | 4 |
| Nigeria | 4 |
| Pakistan | 6 |
| Paraguay | 5 |
| Somalia | 5 |
| Spain | 1 |
| Sri-Lanka | 5 |
| USA | 6 |

Cells per split:

| Split | Required cells |
| --- | ---: |
| train | 47 |
| valid | 37 |
| test | 32 |
| bolivia | 5 |

## Source Choice

Preferred source: Copernicus DEM GLO-30.

Fallback source: SRTM Global 1 arc-second.

The selected planning source for STEP 6B2 is:

`copernicus_glo30`

Rationale:

- Copernicus DEM GLO-30 is a global approximately 30 m digital surface model,
  appropriate for a first local monotonic topographic regularizer.
- SRTM Global 1 arc-second is also approximately 30 m and is an acceptable
  fallback if Copernicus access is not practical.
- HAND would be hydrologically better for flood plausibility because it encodes
  height above drainage rather than absolute elevation, but it requires either
  an existing HAND product or hydrologic derivation from DEM plus drainage/flow
  processing. It is optional for the first version of the physics loss.

Reference URLs checked:

- Copernicus Data Space DEM collection: `https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM`
- Copernicus DEM on AWS Open Data: `https://registry.opendata.aws/copernicus-dem/`
- NASA SRTM Global 1 arc-second catalog: `https://www.earthdata.nasa.gov/data/catalog/lpcloud-srtmgl1-003`
- USGS SRTM 1 Arc-Second Global page: `https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1`

## Local Storage Targets

Preferred Copernicus target:

`E:/flood_research/data/raw/dem/copernicus_glo30/`

Fallback SRTM target:

`E:/flood_research/data/raw/dem/srtm_1arcsec/`

These folders were created as empty local targets. No raw Sen1Floods11 files were
modified.

## Credentials and Automation

Copernicus:

- Copernicus Data Space access may require registered-user access.
- Some public Copernicus DEM GLO-30 tiles are available through public cloud
  mirrors, but the helper does not assume a configured cloud access method.
- Automatic download is therefore not enabled by default.

SRTM:

- NASA/USGS SRTM access commonly uses Earthdata Search or USGS EarthExplorer.
- Credentials or manual download may be required depending on the access route.
- Automatic download is therefore not enabled by default.

STEP 6B2 can proceed automatically only after an explicit access method is
configured and human validation approves download. The current safe state is
manual provisioning or credential setup.

## Dry-Run Result

Command executed:

```powershell
E:\flood_research\venvs\terramind-gpu\Scripts\python.exe scripts\physics\step6b2_prepare_dem_source.py --source copernicus_glo30 --output-root E:\flood_research\data\raw\dem --dry-run
```

Result:

```text
source=copernicus_glo30
source_label=Copernicus DEM GLO-30
required_cells=53
existing_cells=0
missing_cells=53
automatic_download_supported=False
credential_hint=Use Copernicus Data Space registration or an explicitly configured public-data access method before enabling downloads.
```

Planned manifests:

- `E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/manifests/planned_dem_download_manifest.csv`
- `E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/manifests/planned_dem_download_manifest.json`

## Manual Instructions

To unblock STEP 6B2, provide one of:

1. Copernicus DEM GLO-30 tiles for all 53 required cells under:
   `E:/flood_research/data/raw/dem/copernicus_glo30/`
2. SRTM 1 arc-second tiles for all 53 required cells under:
   `E:/flood_research/data/raw/dem/srtm_1arcsec/`
3. A single validated regional DEM mosaic covering the total bounds under a
   clearly named folder in `E:/flood_research/data/raw/dem/`.

After files are present, rerun DEM source verification and then STEP 6B3 sample
alignment/QC.
