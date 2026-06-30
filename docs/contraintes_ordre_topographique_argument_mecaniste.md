# Contraintes d'ordre topographique : définition et argument mécaniste
## SegMAN-S · Sen1Floods11 · N = 100 · graine 0

**Objet.** Définir précisément la *classe* des contraintes d'ordre topographique, puis montrer
— par un argument **mécaniste**, indépendant de la formulation — pourquoi cette classe ne dispose
d'aucun headroom net exploitable relativement aux labels pour ce couple modèle–données.

**Invariant :** le DEM $z$ n'est jamais un input du modèle ; il n'apparaît que dans la contrainte.
Tout est relatif aux labels de référence $y$.

---

## 1. La classe des contraintes d'ordre topographique $\mathcal{C}_{\text{topo}}$

**Idée.** Une contrainte d'ordre topographique récompense les prédictions où l'eau se concentre
*en bas* et pénalise l'eau « perchée » au-dessus d'un voisin plus bas et sec. Elle ne dépend du
DEM que par les **relations d'ordre** d'élévation (qui est plus haut / plus bas), non par les
valeurs absolues.

**Définition formelle.** Soit $T(\hat p, z)$ une mesure d'**alignement topographique** entre la
prédiction $\hat p$ et l'ordre d'élévation $-z$ — à quel point la règle « bas = mouillé » est
respectée. La classe est :

$$
\mathcal{C}_{\text{topo}} = \Big\{\, \mathcal{R}_\phi \;:\; \mathcal{R}_\phi \text{ non-croissante en } T,\;\; \text{donc } -\nabla_{\hat p}\mathcal{R}_\phi \;\propto\; +\nabla_{\hat p} T \,\Big\}
$$

En clair : **minimiser n'importe quel $\mathcal{R}_\phi \in \mathcal{C}_{\text{topo}}$ revient à
pousser $\hat p$ vers un alignement topographique $T$ plus élevé.** C'est le dénominateur commun
de toute la classe, indépendamment de la forme exacte $\phi$.

**Deux instances canoniques de $T$ :**

| portée | $T$ | interprétation | diagnostic |
|---|---|---|---|
| **Global** (type-AUC) | $T_{\text{glob}} = A(-z;\hat p)$ | les pixels bas sont-ils plus probablement eau | mesuré par L2 |
| **Local** (type-D8) | $T_{\text{loc}} = -\,\mathrm{VF}(\hat p, z)$ | moins d'eau perchée au-dessus d'un voisin aval sec | pénalisé par D8 |

($A(\cdot)$ = AUROC ; $\mathrm{VF}$ = fraction de violations topographiques.)

---

## 2. L'argument mécaniste

**Le pivot logique** (trois étapes) :

> **(i)** Un régulariseur n'aide (canal du biais inductif) que s'il **tire $\hat p$ vers la cible**
> (les labels $y$, proxy de la vérité).
> **(ii)** Tout $\mathcal{R}_\phi \in \mathcal{C}_{\text{topo}}$ tire dans **une seule direction** :
> $T$ croissant.
> **(iii)** Or la cible définit le « bon » niveau d'alignement, et le modèle non-contraint
> l'atteint déjà.

**L'inégalité-pivot** (le *headroom*) :

$$
H \;=\; T(y, z) \;-\; T(\hat p_0, z) \;\le\; 0
$$

où $\hat p_0$ est la prédiction baseline non-contrainte. **$H \le 0$ signifie : le modèle est déjà
au moins aussi aligné topographiquement que les labels.**

**Conséquence directe.** Pousser $T(\hat p)$ *plus haut* que $T(\hat p_0) \ge T(y)$, c'est
s'éloigner du niveau d'alignement de la cible, non s'en rapprocher. Donc tout
$\mathcal{R}_\phi \in \mathcal{C}_{\text{topo}}$ ne peut faire que :

- **soit rien** — le gradient est négligeable car la contrainte est déjà satisfaite (*slack*) ;
- **soit nuire** — quand on la force, elle *dépasse* (overshoot) au-delà de la cible.

**Rattachement aux chiffres** (rien de théorique en l'air) :

| pièce | quantité | valeur | lecture |
|---|---|---|---|
| $H_{\text{glob}}$ (L2) | $A(-z;y) - A(-z;\hat p_0)$ | test −0,007 · bolivia −0,027 · val +0,008 ($\ll \sigma{=}0{,}18$) | $\le 0$ → cible déjà atteinte |
| $H_R$ natif VF (L1) | $\mathrm{VF}(\hat p_0) - \mathrm{VF}(y)$ | val −0,0015 · test −0,0018 · bolivia −0,0023 (IC entièrement < 0) | $< 0$ → pred 3–4× plus propre que les labels |
| slackness (L1) | $\rho = \lambda\mathcal{R}/\mathcal{L}_{\text{tot}}$ | 0,007 % (λ1) · 0,3 % (λ100) | contrainte quasi-inactive |
| overshoot (L1) | mIoU à λ=100 | dégrade tous les splits | forcer → nuit |
| signature (L1) | precision ↑ / recall ↓ | +0,035 / −0,036 (test) | l'eau « vraie » est supprimée pour gagner de la pureté topographique |

L'argument **n'utilise jamais la forme de $\phi$** — seulement que $\phi$ est monotone en $T$.
**C'est ce qui le fait généraliser à toute la classe**, là où le contrôle real-vs-shuffled (L1)
ne couvre que D8.

---

## 3. Portée exacte (où l'argument s'arrête)

- ✅ **Ferme** : toute contrainte monotone dans le $T$ *mesuré* (ordre d'élévation global, ou
  violations locales de type D8).
- ❌ **Ne ferme pas** : une contrainte sur un $T'$ **orthogonal** — connectivité d'écoulement,
  accumulation de bassin versant, chemins hydrologiques — que $A(-z;\cdot)$ et $\mathrm{VF}$ ne
  capturent pas. Là, $H \le 0$ ne borne rien. **Il faut le nommer explicitement** : « valable
  pour les contraintes d'ordre topographique ponctuel ; les contraintes de connectivité
  d'écoulement ne sont pas testées. »
- ⚠️ **Relatif aux labels** (axiome A3) : « overshoot au-delà de $y$ » pourrait être *vers*
  $y^\star$ (la vérité physique) si les labels sont topographiquement bruités. Caveat séparé.

---

## 4. Les autres façons de le montrer

| méthode | ce qu'elle montre | coût | canal |
|---|---|---|---|
| **$H_R$ natif** ✓ calculé | $\mathrm{VF}(\hat p_0) - \mathrm{VF}(y) < 0$ (IC < 0, val/test/bolivia) : le baseline viole D8 **3–4× moins que les labels** → pas de headroom natif | fait | II, classe |
| **useful violation rate** ✓ calculé | parmi les pixels D8-actifs supprimés : ~59 % utiles (vrais FP) / ~41 % nuisibles (vraie eau) → explique precision↑/recall↓, aucun gain net | fait | II, D8 |
| **λ-sweep dose-réponse** | crank $\lambda$ : $\rho\uparrow$, $T(\hat p)\uparrow$, mais water-IoU $\downarrow$ — signature empirique de $H\le 0$ (contrainte et labels tirent en sens opposés) | 3 graines | II, classe |
| **real-vs-shuffled + TOST** | le contenu topographique vrai $\equiv$ DEM désaligné | multi-graine | II-a, D8 |
| **L4 redondance conditionnelle** | borne informationnelle : 0 signal résiduel DEM donné $\hat p$ | déjà fait | I |

Les deux premières sont **mécanistes / quasi-gratuites et couvrent toute la classe** ; les deux
dernières sont statistiques / informationnelles et couvrent une formulation (real-vs-shuffled) ou
le canal orthogonal (L4).

**Résultat calculé** (inference-only, checkpoint baseline ; `outputs/native_vf_headroom/`).
*$H_R$ natif* : $\mathrm{VF}(\hat p_0) - \mathrm{VF}(y) = $ −0,0015 (val) / −0,0018 (test) /
−0,0023 (bolivia), **IC bootstrap-by-tile entièrement négatif**, prédiction plus propre que les
labels sur 80–90 % des tuiles. Le baseline viole D8 **3–4× moins que les labels** → aucun
headroom natif. *useful violation rate* : parmi les pixels D8-actifs que la contrainte
supprimerait, **~59 % sont de vrais faux positifs** (GT sec, utile à retirer) et **~41 % de la
vraie eau** (GT eau, perte de recall) ; ~7–10 % sont des configurations que **le GT lui-même
viole**. Lecture honnête : les positions de violation sont enrichies ~4× en faux positifs (donc
*signal réel*, pas du bruit), mais le headroom net est nul — retirer 59 % de FP (precision↑) et
41 % de vraie eau (recall↓) laisse l'IoU inchangé, exactement la signature observée. Conclusion :
**« aucun bénéfice net », pas « aucun signal »**.

---

### Synthèse en une phrase

> Une contrainte d'ordre topographique ne fait que pousser $\hat p$ vers plus d'alignement
> élévation–eau. Le modèle baseline atteint déjà — et dépasse — l'alignement des labels :
> globalement ($H_{\text{glob}} \le 0$) et dans les unités mêmes de D8 ($H_R < 0$, IC entièrement
> négatif, le baseline viole 3–4× moins que les labels). La contrainte est *slack*
> ($\rho \le 0{,}3\%$), et ses corrections actives sont un mélange ~59 % utiles / ~41 % nuisibles
> qui explique le compromis precision↑/recall↓ sans gain net d'IoU. **Conclusion calibrée :
> aucun headroom net exploitable** pour cette classe dans cette configuration, relativement aux
> labels — ce qui n'est *pas* « aucun signal » (les violations sont enrichies ~4× en faux
> positifs) mais « aucun bénéfice net ». Preuve mécaniste + empirique, label-relative ; ne couvre
> pas les statistiques orthogonales (connectivité d'écoulement).
