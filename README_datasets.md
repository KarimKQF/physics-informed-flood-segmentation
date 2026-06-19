# Dataset Preparation

Ce document prepare les datasets classiques de base pour la recherche
`Physics-Informed Loss Functions for Urban Runoff`.

La premiere etape sert a recuperer ou inspecter les datasets de segmentation
d'eau / inondation et a verifier que les images et masques sont lisibles. Les
DEM viennent ensuite, dans un manifest separe, apres alignement spatial strict.

## Pourquoi Sen1Floods11 ?

Sen1Floods11 est un dataset de reference pour la cartographie d'inondations a
partir d'imagerie satellite, notamment Sentinel-1. Il est utile pour etablir une
baseline robuste avant d'ajouter une contrainte topographique.

## Pourquoi STURM-Flood ?

STURM-Flood est associe a des donnees d'inondation plus recentes et permet de
tester la generalisation sur une autre source de donnees. Il servira de second
point de comparaison pour verifier que la Topographic Loss n'est pas uniquement
adaptee a un dataset.

## Arborescence Des Donnees

```text
data/
  raw/
    Sen1Floods11/
    STURM-Flood/
  processed/
  dem/
  README.md
```

Les donnees lourdes ne doivent pas etre versionnees dans Git. Le `.gitignore`
ignore `data/raw/`, `data/processed/`, `data/dem/`, les archives, les rasters et
les tableaux binaires.

## External data storage on D:

Real datasets must not be stored inside the repository. Heavy data should live
under:

```text
D:/urban_runoff_data/raw
D:/urban_runoff_data/processed
D:/urban_runoff_data/manifests
D:/urban_runoff_data/logs
```

Initialize and inspect the external data root:

```powershell
python scripts\inspect_external_data_root.py --config configs\local_paths.yaml
```

Inspect Sen1Floods11 remote sources without downloading:

```powershell
python scripts\inspect_sen1floods11_remote.py --config configs\local_paths.yaml
```

Download only a small Sen1Floods11 subset to `D:/urban_runoff_data`:

```powershell
python scripts\download_sen1floods11_subset_to_d.py `
  --config configs\local_paths.yaml `
  --max-files 50 `
  --include-tif-only `
  --overwrite
```

Inspect a GeoTIFF manifest:

```powershell
python scripts\inspect_geotiff_manifest.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_manifest.csv `
  --max-samples 10
```

Run a GeoTIFF training smoke test:

```powershell
python experiments\smoke_tests\smoke_test_geotiff_training.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_manifest.csv `
  --max-samples 8
```

## Installation Des Dependances

Depuis la racine du projet:

```bash
pip install -e ".[dev]"
```

Les scripts utilisent `requests`, `tqdm`, `pyyaml` et `rasterio`. Pour
Sen1Floods11, il faut aussi installer Google Cloud SDK pour disposer de `gsutil`.

## Sen1Floods11 download

### Windows prerequisites

Install Google Cloud SDK / Google Cloud CLI. Official Google documentation says
that `gsutil` is installed as part of Google Cloud CLI on Windows.

PowerShell installer command:

```powershell
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")

& $env:Temp\GoogleCloudSDKInstaller.exe
```

During installation:

- keep the bundled Python option unless you have a specific reason not to;
- enable adding Google Cloud CLI to `PATH` if the installer asks;
- finish the installation;
- close and reopen PowerShell, Git Bash, WSL, or VS Code.

Check:

```bash
gcloud version
gsutil version
```

If `gcloud` works but `gsutil` does not, check the installed Google Cloud CLI
components or rerun the installer.

Inspecter le bucket sans telecharger:

```bash
bash scripts/download_sen1floods11.sh --list
```

Telecharger / synchroniser:

```bash
bash scripts/download_sen1floods11.sh
```

PowerShell equivalent:

```powershell
New-Item -ItemType Directory -Force -Path data\raw\Sen1Floods11
gsutil -m rsync -r gs://sen1floods11 data/raw/Sen1Floods11
```

Bash / WSL equivalent:

```bash
mkdir -p data/raw/Sen1Floods11
gsutil -m rsync -r gs://sen1floods11 data/raw/Sen1Floods11
```

The download can be relaunched with the same `rsync` command if it is
interrupted. Heavy data must not be committed to Git. This step does not yet
download DEMs or building masks.

Inspecter les fichiers locaux:

```bash
python scripts/inspect_sen1floods11.py
```

If `gsutil rsync` or `gcloud storage rsync` is blocked on Windows, use the local
STAC catalog files to download GeoTIFF assets over HTTPS:

```powershell
python scripts/download_sen1floods11_from_catalog.py --dry-run --limit 10
python scripts/download_sen1floods11_from_catalog.py --collection sen1floods11_hand_labeled_source --collection sen1floods11_hand_labeled_label
```

This avoids bucket-level `storage.buckets.get` permissions and does not use
`rsync`.

## STURM-Flood

Le dataset est associe au DOI:

```text
10.5281/zenodo.12748982
```

Verifier les fichiers disponibles sans telecharger:

```bash
python scripts/download_sturm_flood.py --dry-run
```

Par defaut, ce dry-run est offline-first pour rester rapide et non bloquant. Pour
interroger Zenodo et afficher les fichiers/tailles si le reseau repond:

```bash
python scripts/download_sturm_flood.py --dry-run --resolve-metadata
```

Telecharger si l'API Zenodo expose les fichiers:

```bash
python scripts/download_sturm_flood.py
```

Si le telechargement automatique ne trouve pas les fichiers, les recuperer
manuellement depuis:

```text
https://zenodo.org/records/12748982
```

puis les placer dans:

```text
data/raw/STURM-Flood/
```

Inspecter les fichiers locaux:

```bash
python scripts/inspect_sturm_flood.py
```

## DEM alignment for Sen1Floods11

Le DEM brut ne doit pas etre utilise directement par le dataloader. Il doit
d'abord etre reprojete et reechantillonne sur la grille exacte de chaque image
Sen1Floods11: meme CRS, meme transform, meme width, meme height et memes bounds.

Le subset initial de 30 samples est disperse geographiquement. Un DEM global
sur sa bounding box demanderait trop de tuiles SRTM. La strategie prudente est
donc de travailler sur un petit manifest compact, puis de telecharger et aligner
les tuiles SRTM par sample.

Le manifest sans DEM reste conserve:

```text
D:/urban_runoff_data/manifests/sen1floods11_subset_manifest.csv
```

Le manifest avec DEM aligne est separe:

```text
D:/urban_runoff_data/manifests/sen1floods11_subset_with_dem_manifest.csv
```

Inspecter l'emprise du subset:

```powershell
python scripts\inspect_sen1floods11_bounds.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_manifest.csv `
  --output D:\urban_runoff_data\logs\sen1floods11_subset_bounds.json
```

Telecharger un DEM brut SRTM limite a l'emprise du subset:

```powershell
python scripts\download_dem_for_sen1floods11_subset.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_manifest.csv `
  --config configs\local_paths.yaml `
  --source srtm `
  --overwrite
```

Le script SRTM utilise les tuiles publiques AWS Skadi et possede une limite de
securite sur le nombre de tuiles afin d'eviter un telechargement trop large.
Si le subset est trop disperse, preparer un DEM manuellement ou augmenter
`--max-tiles` volontairement.

Aligner le DEM brut sur chaque sample:

```powershell
python scripts\align_dem_to_sen1floods11_manifest.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_manifest.csv `
  --dem D:\urban_runoff_data\raw\DEM\sen1floods11_subset_dem_raw.tif `
  --output-dir D:\urban_runoff_data\processed\aligned_dem\Sen1Floods11 `
  --output-manifest D:\urban_runoff_data\manifests\sen1floods11_subset_with_dem_manifest.csv `
  --overwrite
```

Valider l'alignement:

```powershell
python scripts\validate_dem_alignment_manifest.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_with_dem_manifest.csv `
  --max-samples 30
```

Le smoke test avec DEM valide uniquement l'execution technique du pipeline
`image + mask + valid_mask + DEM`; ce n'est pas encore une evaluation de
performance.

```powershell
python experiments\smoke_tests\smoke_test_geotiff_training.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_with_dem_manifest.csv `
  --use-dem `
  --max-samples 8
```

## Per-sample DEM workflow

Cette variante evite de creer un DEM global. Les tuiles SRTM sont mises en
cache sur:

```text
D:/urban_runoff_data/raw/DEM/srtm_tiles
```

Les DEM alignes par sample sont stockes dans:

```text
D:/urban_runoff_data/processed/aligned_dem/Sen1Floods11
```

Creer un petit manifest compact:

```powershell
python scripts\create_compact_sen1floods11_subset_manifest.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_manifest.csv `
  --bounds-json D:\urban_runoff_data\logs\sen1floods11_subset_bounds.json `
  --output-manifest D:\urban_runoff_data\manifests\sen1floods11_compact_dem_subset_manifest.csv `
  --max-samples 8 `
  --max-unique-srtm-tiles 16
```

Telecharger les tuiles SRTM necessaires et aligner un DEM par sample:

```powershell
python scripts\download_and_align_srtm_per_sample.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_compact_dem_subset_manifest.csv `
  --output-manifest D:\urban_runoff_data\manifests\sen1floods11_compact_with_dem_manifest.csv `
  --tile-cache-dir D:\urban_runoff_data\raw\DEM\srtm_tiles `
  --aligned-dem-dir D:\urban_runoff_data\processed\aligned_dem\Sen1Floods11 `
  --max-samples 8 `
  --max-tiles-total 16 `
  --overwrite
```

Valider puis lancer le smoke test avec DEM:

```powershell
python scripts\validate_dem_alignment_manifest.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_compact_with_dem_manifest.csv `
  --max-samples 8

python experiments\smoke_tests\smoke_test_geotiff_training.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_compact_with_dem_manifest.csv `
  --use-dem `
  --max-samples 8
```

Le manifest avec DEM reste separe du manifest sans DEM. Cette etape ne touche
pas aux batiments, ne cree pas `q_i`, et ne lance pas de gros entrainement.

## Pourquoi Aligner Les DEM ?

Les DEM doivent etre alignes spatialement avec les images et les masques:
resolution, CRS, geotransform, taille raster et emprise. Les recuperer trop tot
risque d'ajouter du bruit avant meme de savoir quel dataset servira de baseline.

## Pourquoi Pas Encore Les Masques Batiments ?

Les masques batiments ou occupation du sol serviront plus tard a construire
`pixel_reliability`, pour ne pas penaliser des zones non hydrauliquement
connectees. Cette etape vient apres la validation des datasets de base et apres
le choix de la baseline.

## Ordre De Travail Recommande

1. Installer les dependances.
2. Telecharger ou inspecter Sen1Floods11.
3. Verifier que les images et masques sont lisibles.
4. Telecharger ou inspecter STURM-Flood.
5. Verifier que les images et masques sont lisibles.
6. Choisir le dataset principal pour la baseline.
7. Seulement ensuite recuperer les DEM alignes.
8. Encore plus tard, ajouter les masques batiments pour la version fiabilisee.
