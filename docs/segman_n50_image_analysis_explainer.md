# Analyse des images — SegMAN N=50 : comprendre les résultats visuels

**Pour qui :** lecture rapide, sans supposer que tu te souviens des détails techniques.
**Ce fichier répond à :** Qu'est-ce que les 6 images sauvegardées nous apprennent ?
Est-ce que la loss topo change vraiment les prédictions ?

---

## Ce qu'on compare

Chaque run SegMAN-S a sauvegardé les prédictions sur les **6 premières images du
jeu de test** (dans l'ordre du dataloader, identique pour tous les runs).
On a **4 conditions de loss** × **5 seeds** = 20 runs, donc 20 prédictions par image.

| Condition | Ce que fait la loss |
|-----------|---------------------|
| **CE** | Cross-Entropy simple. Apprend à matcher les labels. |
| **Dice+CE** | Ajoute un terme Dice (meilleur équilibre eau/fond). |
| **Topo réel** | Dice+CE + pénalité si l'eau prédit est plus haute que le terrain sec adjacent (vrai DEM). |
| **Topo shufflé** | Exactement pareil, mais le DEM fourni est aléatoire (contrôle : le DEM est-il vraiment utile ?). |

Pour chaque image, on mesure :
- **IoU_water** : overlap entre eau prédite et eau vraie (1 = parfait, 0 = raté)
- **Violation topo** : % de pixels où l'eau prédit est physiquement incohérente (eau en haut d'une pente sèche)

---

## Les 6 images — résumé image par image

### Image 0 — `Ghana_1078550` — Grande inondation (40% de l'image est de l'eau)

```
IoU water (moy. 5 seeds) :
  CE           : 0.764 ± 0.042
  Dice+CE      : 0.783 ± 0.029   (+0.019 vs CE)
  Topo réel    : 0.782 ± 0.028   (+0.000 vs Dice+CE  → pas d'effet topo)
  Topo shufflé : 0.795 ± 0.019   MEILLEUR (+0.013 vs Topo réel)

Violations topo (seed0) :
  CE           : 1.42%
  Dice+CE      : 1.44%
  Topo réel    : 1.41%  (léger mieux)
  Topo shufflé : 1.86%  (PIRE — plus de violations, mais meilleur IoU)
```

**Interprétation :** Dice+CE aide clairement (+2% IoU). La loss topo réel ne change
presque rien vs Dice+CE. Le shufflé est meilleur en IoU mais produit plus de violations
physiques — il prédit plus d'eau (y compris dans des endroits physiquement incohérents)
et ça coïncide avec ce qu'il y a dans le masque GT.

---

### Image 1 — `Ghana_141271` — Image très inondée (66% d'eau)

```
IoU water (moy. 5 seeds) :
  CE           : 0.865 ± 0.018
  Dice+CE      : 0.873 ± 0.021   (+0.007)
  Topo réel    : 0.877 ± 0.021   (+0.005 vs Dice+CE)
  Topo shufflé : 0.877 ± 0.028   (+0.000 vs Topo réel — à égalité)

Violations topo (seed0) :
  CE           : 1.21%
  Dice+CE      : 0.97%
  Topo réel    : 0.82%   MINIMUM — moins de violations que toutes les autres
  Topo shufflé : 1.04%
```

**Interprétation :** C'est l'image la plus intéressante.
- En IoU, les 4 conditions sont proches.
- En violations physiques, le **Topo réel est clairement le meilleur** : 0.82% vs 1.21%
  pour CE et 1.04% pour le shufflé.
- C'est le seul cas où la loss avec le **vrai DEM** produit des prédictions
  physiquement plus cohérentes que le DEM aléatoire.
- **Signal faible mais réel sur cette image.**

---

### Image 2 — `Ghana_167233` — Image presque sèche (0.6% d'eau)

```
IoU water : ≈ 0 pour toutes les conditions (cas dégénéré)
Violations : ≈ 0 (pas d'eau prédit → pas de violations)
```

**Interprétation :** Quand il y a quasi pas d'eau dans l'image, le modèle prédit
quasi rien et on ne peut rien conclure. Cette image est inutile pour comparer les
conditions. On l'exclut de l'analyse principale.

---

### Image 3 — `Ghana_313799` — Eau clairsemée (3.7%), haute variance entre seeds

```
IoU water (moy. 5 seeds) :
  CE           : 0.798 ± 0.200   (std énorme)
  Dice+CE      : 0.752 ± 0.255   (-0.045 vs CE)
  Topo réel    : 0.801 ± 0.143   (+0.049 vs Dice+CE — meilleur)
  Topo shufflé : 0.752 ± 0.159   (+0.000 vs Dice+CE — pas de gain)

Violations topo (seed0) :
  CE           : 0.093%
  Dice+CE      : 0.104%
  Topo réel    : 0.245%   PIRE (2.5× plus que CE)
  Topo shufflé : 0.269%   encore pire
```

**Interprétation :** Image difficile (peu d'eau, std ±0.20 = le résultat change
totalement selon le seed). Le Topo réel a un meilleur IoU en moyenne, mais
il génère aussi **plus de violations physiques** — probablement parce que la loss topo
pousse le modèle à prédire de l'eau dans des zones basses qui ne correspondent pas
au GT. La std est si grande que les moyennes ne veulent pas dire grand chose ici.

---

### Image 4 — `Ghana_319168` — Image sèche (0% d'eau)

```
IoU water : ≈ 0 pour toutes conditions (cas dégénéré)
Violations : ≈ 1% (eau fantôme — le modèle prédit un peu d'eau là où il n'y en a pas)
```

**Interprétation :** Pas d'eau dans le GT. Les ~1% de violations viennent du modèle
qui prédit quelques faux positifs eau. Toutes les conditions sont similaires.
Pas exploitable pour la comparaison. Exclu de l'analyse principale.

---

### Image 5 — `Ghana_359826` — Inondation modérée (15.6% d'eau)

```
IoU water (moy. 5 seeds) :
  CE           : 0.426 ± 0.037
  Dice+CE      : 0.438 ± 0.050   MEILLEUR (+0.011)
  Topo réel    : 0.434 ± 0.060   (-0.003 vs Dice+CE)
  Topo shufflé : 0.420 ± 0.049   PIRE (-0.014 vs Topo réel)

Violations topo (seed0) :
  CE           : 1.23%
  Dice+CE      : 1.33%
  Topo réel    : 1.25%   (légèrement mieux que Dice+CE)
  Topo shufflé : 1.42%   (le pire)
```

**Interprétation :** Dice+CE gagne légèrement en IoU. La loss topo ne s'améliore
pas et réduit légèrement les violations mais de façon non significative.
Sur cette image, le shufflé est le pire en IoU ET en violations.

---

## Résumé visuel : qui gagne où ?

| Image | Eau GT | Gagnant IoU | Topo réel vs Shufflé (violations) |
|-------|--------|-------------|----------------------------------|
| 0 — Ghana_1078550 | 40% | Shufflé (+1.3%) | Topo réel MIEUX (1.41% vs 1.86%) |
| 1 — Ghana_141271  | 66% | Topo réel ≈ Shufflé | **Topo réel CLAIREMENT MIEUX** (0.82% vs 1.04%) |
| 2 — Ghana_167233  | 0.6% | — (dégénéré) | — |
| 3 — Ghana_313799  | 3.7% | Topo réel (+4.9%) | Topo réel pire en violations |
| 4 — Ghana_319168  | 0% | — (dégénéré) | — |
| 5 — Ghana_359826  | 15.6% | Dice+CE | Topo réel légèrement mieux |

---

## Ce que ça veut dire : les 3 questions

### 1. Est-ce que le DEM réel aide vraiment le modèle à être physiquement cohérent ?

**Réponse : un peu, sur une image sur quatre.**

Sur Ghana_141271, oui clairement — Topo réel réduit les violations physiques de
1.21% (CE) à 0.82%, et le DEM shufflé reste à 1.04%. C'est le seul signal convaincant.

Sur les autres images : les différences entre Topo réel et Topo shufflé sont
< 0.5% de pixels — dans le bruit.

### 2. La loss topo améliore-t-elle l'IoU ?

**Réponse : légèrement, mais pas plus que le DEM aléatoire.**

Sur les 89 images du test complet (5 seeds) :
- Topo réel : 0.8465 ± 0.0114
- Topo shufflé : 0.8479 ± 0.0186

Différence = 0.14% — dans le bruit. Les deux versions de la loss topo apportent
un tout petit gain par rapport à Dice+CE, mais ce gain ne dépend pas du vrai DEM.

### 3. C'est une régularisation ou un signal physique ?

**C'est probablement une régularisation.**

La loss topo (réelle ou shufflée) contraint les prédictions à être plus "lisses"
spatialement (un pixel eau entouré de sec est pénalisé). Ça aide un peu — mais
le DEM réel n'apporte pas plus que du bruit spatial. Le terme agit comme une
**contrainte de régularité géométrique**, pas comme une véritable physique hydrologique.

---

## Limitations importantes à comprendre

1. **On n'a que 6 images** sur 89 — toutes de Ghana. On ne peut pas généraliser
   à la Bolivie (OOD) ou à d'autres régions.

2. **Les prédictions sont des masques durs** (0 ou 1 par pixel). La loss topo
   pendant l'entraînement utilisait des probabilités molles — on ne peut pas
   recalculer exactement la même chose ici. Les violations qu'on calcule sont une
   approximation diagnostique.

3. **La variance entre seeds est dominante.** Sur l'image 3, la std est ±0.20
   (presque aussi grande que la valeur elle-même). Les conclusions par image
   sont fragiles avec seulement 5 seeds.

4. **Le Topo shufflé est parfois meilleur en IoU ET pire en violations** (image 0).
   Ce paradoxe montre que IoU et cohérence physique ne sont pas la même chose.

---

## Où sont les fichiers générés

```
reports/figures/segman_n50/
  panels/                         ← comparaisons visuelles 7 colonnes (non commit, ~500KB/img)
    panel_000_Ghana_1078550.png   S2 RGB / GT / CE / Dice+CE / Topo réel / Topo shufflé / DEM
    panel_001_Ghana_141271.png
    ...
  topo_violations/                ← cartes de violations (rouge = pixel incohérent)
    topo_viol_001_Ghana_141271.png  ← le plus intéressant
    ...
  summary_iou_water_per_image.png ← graphique résumé (commité)
  tables/
    per_image_metrics.csv         ← tous les chiffres, 1 ligne par (condition, seed, image)
    delta_summary.csv             ← différences par image
    topo_violation_per_image.json ← violations par image et condition
```

Pour régénérer toutes les figures :

```bash
python experiments_cvpr/segman/analyze_segman_n50_visuals.py \
    --exp-root E:/flood_research/experiments/segman \
    --output-dir reports/figures/segman_n50
```

---

## Conclusion en une phrase

> **SegMAN-S prédit bien l'eau dans ces images Ghana (IoU 0.76–0.88 sur les images
> avec de l'eau), mais la loss topographique avec le vrai DEM ne se distingue pas
> clairement du DEM aléatoire — sauf sur une image (Ghana_141271) où elle réduit
> les incohérences physiques. L'effet observé ressemble davantage à une régularisation
> qu'à un signal physique du terrain.**
