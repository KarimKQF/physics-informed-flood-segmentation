# Dice+CE Anti-Collapse Baseline — N=50 Seeds 1 & 3

**Generated:** 2026-06-24  
**Model:** TerraMind-L + UPerNet  
**Loss:** `CombinedDiceCELoss` — `smp.DiceLoss + 1.0 × nn.CrossEntropyLoss`  
**N train:** 50  
**Seeds tested here:** 1, 3

---

## 1. Contexte

Seeds 1 et 3 ont collapsed sous les 4 conditions précédemment testées :

| Condition | Seed 1 | Seed 3 |
|---|---|---|
| Dice-only | collapse all-bg | collapse all-bg |
| Physics real DEM (warmup λ=0→0.5) | collapse all-bg | collapse all-bg |
| Physics shuffled DEM (warmup λ=0→0.5) | collapse all-bg | collapse all-bg |
| Physics real DEM no-warmup (λ=0.5 constant) | collapse all-bg | collapse all-bg |

L'audit des manifests a montré que la composition des subsets N=50 n'explique pas ce comportement : seed1 contient plus de pixels eau (1,680,776) que tous les seeds rescués, dont seed42 (724,717 px) qui est rescué par Physics. L'hypothèse retenue est que la perte Dice-only crée une zone de gradient quasi nul quand le modèle prédit tout-background, et que le modèle s'y enfonce dès les premières epochs selon l'initialisation aléatoire (seed). La Cross-Entropy fournit un signal par pixel qui reste non nul même quand les prédictions sont dégénérées.

---

## 2. Résultats Seed 1 — Dice+CE

| Métrique | Valeur |
|---|---|
| **Status** | **Terminé** |
| Stop reason | Early stop — patience=15, no_improve=15 |
| Best epoch | **46** |
| Stop epoch | 61 |
| **val_mIoU** | **0.8396** |
| **val_water_IoU** | **0.7180** |
| val_water_F1 | 0.8359 |
| val_pred_water_px | 2,125,797 |
| pred_water_fraction | 0.1047 (10.5%) |
| val_topo_violation_frac | 0.00155 |
| Collapsed | **NON** |
| Checkpoint | ✓ best_checkpoint.pt |

**Trajectoire epochs 1–3 :**

| Epoch | val_mIoU | val_water_IoU | val_pred_water_px | grad_norm |
|---|---|---|---|---|
| 1 | 0.4489 | 0.0073 | 16,325 | 22,886 |
| 2 | 0.5559 | 0.3348 | 5,929,000 | 4,623 |
| 3 | 0.6778 | 0.4229 | 974,103 | 10,770 |

Grad_norm non nul dès epoch 1 — signature d'absence de collapse. La progression epoch 1→3 est cohérente avec un apprentissage actif.

---

## 3. Résultats Seed 3 — Dice+CE

| Métrique | Valeur |
|---|---|
| **Status** | **En cours au moment du rapport (epoch 40)** |
| Best epoch so far | **31** |
| Current epoch | 40 |
| no_improve so far | 9 |
| **val_mIoU** (best ep 31) | **0.8387** |
| **val_water_IoU** (best ep 31) | **0.7170** |
| val_water_F1 | 0.8352 |
| val_pred_water_px | 2,198,165 |
| pred_water_fraction | 0.1083 (10.8%) |
| val_topo_violation_frac | 0.00173 |
| Collapsed | **NON** |
| Checkpoint | ✓ best_checkpoint.pt |

*Métriques à best_epoch=31. À epoch 40, val_mIoU=0.835 — le modèle est en plateau, pas en collapse.*

---

## 4. Comparaison toutes conditions — Seeds 1 et 3

### Seed 1

| Condition | Collapsed | val_mIoU | val_water_IoU | Collapse type |
|---|---|---|---|---|
| Dice-only | OUI | ~0.50 | <0.01 | all-background |
| Physics real DEM warmup | OUI | ~0.50 | <0.01 | all-background |
| Physics shuffled DEM | OUI | ~0.50 | <0.01 | all-background |
| Physics real DEM no-warmup | OUI | ~0.50 | <0.01 | all-background |
| **Dice+CE** | **NON** | **0.8396** | **0.7180** | — |

### Seed 3

| Condition | Collapsed | val_mIoU | val_water_IoU | Collapse type |
|---|---|---|---|---|
| Dice-only | OUI | ~0.50 | <0.01 | all-background |
| Physics real DEM warmup | OUI | ~0.50 | <0.01 | all-background |
| Physics shuffled DEM | OUI | ~0.50 | <0.01 | all-background |
| Physics real DEM no-warmup | OUI | ~0.50 | <0.01 | all-background |
| **Dice+CE** | **NON** | **0.8387** | **0.7170** | — |

### Récapitulatif collapse rate par condition

| Condition | Seeds testées | Rescued | Collapsed | Collapse rate |
|---|---|---|---|---|
| Dice-only | 5 | 0 | 5 | 5/5 = 100% |
| Physics real DEM warmup | 5 | 3 | 2 | 2/5 = 40% |
| Physics shuffled DEM | 5 | 2 | 3 | 3/5 = 60% |
| Physics real DEM no-warmup | 2 (seeds 1, 3) | 0 | 2 | 2/2 = 100% |
| **Dice+CE** | **2 (seeds 1, 3)** | **2** | **0** | **0/2 = 0%** |

---

## 5. Conclusion scientifique

**Dice+CE rescue les 2 seeds qui étaient systématiquement effondrées.** Seeds 1 et 3 avaient résisté à toutes les interventions précédentes — ajout de loss topographique, modification du schedule lambda, suppression du warmup — sans amélioration. L'ajout d'une Cross-Entropy standard (α=1.0) suffit à casser le cycle de gradient nul et à permettre l'apprentissage.

Ceci confirme que la cause principale du collapse n'est **ni la composition du subset, ni le schedule lambda, ni la topographic loss** : c'est une propriété du paysage de perte Dice seule en régime N=50. La Dice loss pénalise faiblement les prédictions all-background quand une classe est peu représentée et que le modèle atteint un faux optimum local où aucun gradient ne le ramène vers la détection eau. La CE fournit une supervision pixel-wise directe qui échappe à ce phénomène.

**Implication pour la méthode Physics :** les rescues observés (seeds 0, 2, 42) avec Physics real DEM ne doivent pas être interprétés comme une amélioration absolue — ils peuvent simplement refléter des seeds dont l'initialisation ne tombe pas dans l'attracteur all-background. La contribution réelle de la topographic loss ne peut être mesurée qu'en comparant Physics vs Dice+CE (et éventuellement Dice+CE+topo) sur les mêmes seeds, en utilisant Dice+CE comme baseline anti-collapse fiable. Dice+CE est donc une baseline forte, pas un résultat négatif pour la méthode Physics.

---

## 6. Prochaine étape recommandée

**Cas applicable : Dice+CE rescue seeds 1 et 2 (2/2 testées).**

1. **Lancer Dice+CE sur les 5 seeds** (seeds 0, 2, 42 manquent encore) pour établir une baseline complète 5-seeds. Cela permettra de comparer rigoureusement Physics real DEM vs Dice+CE sur une même évaluation multi-seed.

2. **Comparer Physics real DEM vs Dice+CE** : sur les seeds où Physics real DEM est rescue (0, 2, 42), Dice+CE donne-t-il le même mIoU ou mieux ? Sur seeds 1 et 3, Dice+CE est déjà supérieur (val_mIoU ~0.84 vs collapse). Cette comparaison établira si la topographic loss ajoute de la valeur au-dessus d'une baseline stable.

3. **Tester Dice+CE+topo** éventuellement : combiner Dice+CE avec la topographic loss pour vérifier si la topographie améliore le mIoU sur une base non-effondrée.

Ne pas lancer U-Net, STURM, Focal ou Physics avant d'avoir la baseline Dice+CE complète 5-seeds.

---

## 7. Fichiers associés

- `results/dice_ce_n50_seed1_seed3_summary.json` — métriques brutes
- `results/n50_manifest_composition_audit.json` — audit composition
- `reports/N50_MANIFEST_COMPOSITION_AUDIT.md` — rapport audit
- `reports/E1_E2_MULTISEED_N50_FINAL_REPORT.md` — rapport global E1/E2
- `configs/multiseed_n50/n50_seed1_dice_ce.yaml`
- `configs/multiseed_n50/n50_seed3_dice_ce.yaml`
- `src/losses/combined_loss.py` — `CombinedDiceCELoss`
- `scripts/step6c_v3_train.py` — `build_loss` dispatch
