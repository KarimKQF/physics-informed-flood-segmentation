# Headroom des losses d'ordre topographique : une démonstration mécaniste
## SegMAN-S · Sen1Floods11 · N = 100 · graine 0

**Statut :** résultat négatif consolidé, graine unique, relatif aux labels de référence, N=100 mécaniste.
**Invariant respecté tout au long :** le DEM n'est **jamais** un input du modèle. Il n'apparaît
que dans les losses, les diagnostics et les analyses post-hoc.

Ce document énonce la revendication, les axiomes, les définitions, une proposition
théorique-informationnelle, cinq lemmes empiriques (chacun avec les chiffres mesurés),
l'argument composé, les limites, et — explicitement — ce qui n'est *pas* prouvé.

---

## 0. La revendication précise (lire en premier)

> **Revendication (bornée).** Pour le couple *(SegMAN-S entraîné sur Sen1Floods11 N=100 graine 0,
> les labels de référence y)*, nous évaluons le **mécanisme partagé** de la classe des losses
> DEM topographiques monotones locales $\mathcal C_{\mathrm{ord}}$ (voir D5). Dans cette
> configuration, le baseline non contraint Dice+CE satisfait déjà la propriété d'ordre
> topographique local visée — mesurée par la fraction de violations, l'énergie de violation et
> les queues de distribution — **au moins aussi fortement que les labels de référence**. Les
> losses DEM topographiques monotones locales dont le seul mécanisme est de réduire les
> violations $v_{ij}$ n'ont donc **aucun headroom natif relatif aux labels** dans cette
> configuration modèle–données.

Ce que la revendication n'est **pas** :

- Ce n'est **pas** « le DEM est inutile pour la cartographie des inondations en général ».
  (Non — voir L4 : un prédicteur DEM seul atteint AUC 0,77–0,84.)
- Ce n'est **pas** « aucune loss physics-informed ne peut jamais aider un modèle ».
  (Hors périmètre ; voir §10.)
- Ce n'est **pas** une affirmation sur le HAND, l'accumulation de flux, la connectivité
  hydrologique, les contraintes pluie/débit, ou le DEM utilisé comme canal d'entrée. Tous
  sont explicitement hors périmètre (§10).
- Ce n'est **pas** « le modèle comprend l'hydrologie ». (Le contraire — voir L3.)
- Ce n'est **pas** une affirmation sur l'*état physique réel* de l'inondation ; tout est
  **relatif aux labels de référence** (Axiome A3).
- Ce n'est **pas** un résultat sur jeu complet ou multi-architecture. La configuration N=100
  est préliminaire/mécaniste ; §10 liste ce qui reste avant des revendications publiables.

---

## 1. Notation et objets

| symbole | signification |
|---|---|
| $x \in \mathcal X$ | input du modèle : 15 canaux (13 S2L1C + 2 S1GRD) |
| $y \in \{0,1\}^{H\times W}$ | label eau de référence |
| $z \in \mathbb R^{H\times W}$ | MNT (élévation), **jamais** input de $f_\theta$ |
| $f_\theta:\mathcal X\to[0,1]^{H\times W}$ | SegMAN-S ; sortie $\hat p = f_\theta(x) = p_{\text{eau}}$ |
| $\hat y = \mathbb 1[\hat p > \tfrac12]$ | prédiction binarisée |
| $\mathcal L_{\text{seg}}$ | loss de segmentation Dice + CE |
| $\mathcal R_\phi(\hat p, z)$ | régulariseur DEM/physique de formulation $\phi$ (ex. D8) |
| $\lambda$ | poids de la loss physique ; loss totale $\mathcal L_{\text{seg}} + \lambda\,\mathcal R_\phi$ |
| splits | `val` (86 tuiles), `test` (89), `bolivia` (15, OOD) |

Checkpoint de référence : `segman_n100_dice_ce_seed0`, meilleure époque 34, val mIoU 0,8546.

---

## 2. Axiomes (hypothèses rendues explicites)

**A1 — Canal d'action.** Une loss DEM ne peut influencer les paramètres appris *que* via le
gradient $\nabla_\theta \mathcal R_\phi(f_\theta(x), z)$. Elle n'introduit aucun nouveau chemin
d'input (le DEM n'est pas fourni à $f_\theta$).

**A2 — Deux canaux et seulement deux.** Relativement à un ensemble de labels fixé, ajouter
$\lambda\mathcal R_\phi$ ne peut réduire le risque hors-ensemble que via :
- **(I) le canal informationnel** — $\mathcal R_\phi$ injecte une information pertinente pour les
  labels, portée par $z$ mais absente de $f_\theta(x)$ ; ou
- **(II) le canal du biais inductif** — $\mathcal R_\phi$ réduit l'espace d'hypothèses / agit
  comme prior / diminue la variance, améliorant la généralisation *même sans information nouvelle*.

Tout effet réel se décompose en ces deux. (A1–A2 sont le squelette logique ; tout ce qui suit
borne le canal I à $\approx 0$ et teste le canal II pour la classe $\mathcal C_{\mathrm{ord}}$.)

**A3 — Évaluation relative aux labels.** Tous les risques, AUC et quantités d'information
mutuelle sont définis par rapport aux labels de référence $y$, **non** à l'état physique réel
non observé $y^\star$.

**A4 — Deux régimes.** Les diagnostics *post-hoc* (L3, L4) maintiennent $\theta$ **gelé**. Les
expériences de *loss* (L1) **ré-entraînent** $\theta$. Cette distinction est importante : les
tests post-hoc bornent ce que $z$ apporte *étant donné le modèle convergé* ; le test
d'entraînement borne ce que $z$ apporte *quand il est injecté comme contrainte durant la phase
d'apprentissage*.

**A5 — Honnêteté en échantillon fini.** Toute quantité empirique est une estimation. Les
intervalles de confiance bootstrap (L1b, L4) ou les dispersions par tuile (L2) sont reportés et
aucune estimation ponctuelle n'est traitée comme un zéro exact.

---

## 3. Définitions

**D1 — Marge de contrainte au niveau sortie.**
Soit $A(s ; w)$ l'AUROC d'un score réel $s$ pour une cible binaire $w$. En utilisant
l'**élévation négative** $-z$ comme score (l'eau se situe en bas ⇒ $-z$ prédit l'eau),
$$
H_z \;:=\; A(-z;\,y)\;-\;A(-z;\,\hat y).
$$
$H_z>0$ : les *labels* sont plus séparables topographiquement que les *prédictions* — marge
disponible pour une loss topographique. $H_z\le 0$ : les prédictions sont déjà au moins aussi
bien alignées topographiquement que les labels — **aucune marge**.

**D2 — Décodabilité représentationnelle.**
Soit $\phi(x)$ les features gelées et $D(\phi)$ la meilleure décodabilité *linéaire intra-tuile*
(Pearson $r$, z-scoré par tuile) de $z$ à partir de $\phi$. On compare $\phi_{\text{entraîné}}$,
$\phi_{\text{aléatoire}}$ (même architecture, initialisation aléatoire), $\phi_{\text{input}}$
(les 15 canaux bruts). Si $D(\phi_{\text{entraîné}})\approx D(\phi_{\text{input}})$,
l'entraînement n'a induit **aucune** représentation d'élévation privilégiée.

**D3 — Redondance DEM conditionnelle.**
Le DEM est *conditionnellement redondant étant donné la sortie du modèle* ssi
$$
I\!\left(y;\,z \,\middle|\, \hat p\right) \;=\; 0 .
$$
Estimateur : l'écart d'AUC hors-ensemble $\Delta\mathrm{AUC} = \mathrm{AUC}(y\sim \hat p + z) -
\mathrm{AUC}(y\sim \hat p)$ d'un apprenant flexible (gradient boosting). $\Delta\mathrm{AUC}
\approx 0$ (IC englobant 0) ⇒ $I(y;z\mid\hat p)\approx 0$.

**D4 — Contribution effective de la contrainte.**
$\rho := \lambda\,\mathcal R_\phi / (\mathcal L_{\text{seg}} + \lambda\,\mathcal R_\phi)$ à
convergence — la part de l'objectif d'entraînement effectivement fournie par le terme physique.
Une contrainte *slack* (déjà satisfaite) donne $\rho\approx 0$ et n'exerce presque aucun gradient.

**D5 — Classe des losses d'ordre topographique local $\mathcal C_{\mathrm{ord}}$.**
Nous ne réfutons pas D4 et D8 indépendamment. Nous évaluons le **mécanisme partagé** de
leur classe : réduire les violations d'ordre topographique local.
$$
\mathcal C_{\mathrm{ord}} \;=\; \bigl\{\,\mathcal R_\phi \;:\; \mathcal R_\phi \text{ pénalise } p_i > p_j + \tau \text{ quand } h_i > h_j,\; j \in \mathcal{N}(i) \,\bigr\}
$$
où $h_i$ est l'élévation MNT, $j$ est un pixel voisin plus bas, et $\tau$ est une marge de
tolérance. La classe comprend : D4 (4 voisins inférieurs), D8 (descente la plus rapide),
D8 pondéré par la pente, variantes à charnière linéaire/quadratique, variantes à marge, et
toute loss locale dont le mécanisme principal est de réduire ces violations. **Non comprises :**
DEM en input, HAND, accumulation de flux, connectivité hydrologique, et toute contrainte DEM
non locale.

**D6 — Magnitude de violation locale et fraction de violations.**
Pour un arc valide $(i,j)$ avec $h_i > h_j$ :
$$
v_{ij}(\hat p, z) \;=\; \max(0,\, p_i - p_j - \tau)
$$
La **fraction de violations** (VF) pondérée par la pente sur l'ensemble actif D8 $\mathcal E$ :
$$
\mathrm{VF}(\hat p, z) \;=\; \frac{\sum_{(i,j)\in\mathcal E} w_{ij}\,\mathbb 1[p_i > p_j + \tau]}{\sum_{(i,j)\in\mathcal E} w_{ij}}
$$
avec $w_{ij} = \min(1, \mathrm{drop}_{ij}/s_0)$ (paramètres D8 : $s_0=1{,}0$, $\tau=0{,}05$).
Le **headroom natif** est $H_R = \mathrm{VF}(\hat y_0, z) - \mathrm{VF}(y, z)$.
$H_R\le 0$ signifie que le baseline ne viole pas la contrainte plus que les labels.

**D7 — Énergie de violation $E_{\mathrm{topo}}$.**
$$
E_{\mathrm{topo}}(p, z) \;=\; \frac{\sum_{(i,j)\in\mathcal E} w_{ij}\,[\max(0, p_i - p_j - \tau)]^2}{\sum_{(i,j)\in\mathcal E} w_{ij}}
$$
Il s'agit de la violation quadratique pondérée moyenne — correspondant exactement à l'objectif de
la loss D8.

---

## 4. Proposition centrale — la borne informationnelle

> **Proposition 1.** Soit $\ell$ une règle de score propre (ex. log-vraisemblance ou Brier),
> dont le prédicteur de Bayes est l'espérance conditionnelle. Si $I(y;z\mid\hat p)=0$ — i.e.
> $y\perp z\mid\hat p$ — alors pour les risques de Bayes
> $$
> R^\star(\hat p, z) \;=\; R^\star(\hat p),
> $$
> c'est-à-dire **aucune fonction mesurable de $(\hat p, z)$ ne prédit $y$ mieux que la meilleure
> fonction de $\hat p$ seul.**

*Démonstration.* L'indépendance conditionnelle donne $\mathbb E[y\mid \hat p, z] =
\mathbb E[y\mid \hat p]$ p.s. Pour une perte propre, le prédicteur minimisant le risque est
l'espérance conditionnelle, donc les prédicteurs optimaux — et leurs risques — coïncident.
$\;\square$

> **Corollaire 1 (canal informationnel).** Sous A1–A2, une loss DEM agissant via le canal I ne
> peut réduire le risque hors-ensemble que dans la mesure où $I(y;z\mid\hat p) > 0$. Si cette
> information conditionnelle est nulle, le canal I a **exactement zéro** marge exploitable par
> les labels, *pour toute formulation $\phi$* — car toutes les formulations agissent via le même
> $z$ et ne peuvent fournir ce que $z$ ne contient pas.

**Portée de la Proposition 1.** Elle conditionne sur un $\hat p$ *fixé*. Elle borne donc le
canal informationnel du **modèle convergé / post-hoc**. Le chemin qu'elle ne ferme pas par
elle-même est : « la contrainte, appliquée *pendant* l'entraînement, remodèle $\hat p$ en
quelque chose de meilleur. » Ce chemin résiduel est exactement ce que le Lemme L1 (ré-entraînement
D8 réel) et le Lemme L1b (diagnostics mécanistes au niveau de la classe) testent.

---

## 5. Lemmes empiriques

### Lemme L1 — le résultat direct de la loss (canal II, formulation D8)
Loss topographique D8 aval, graine 0, $\lambda\in\{1, 100\}$, DEM réel vs DEM mélangé (contrôle).
mIoU (↑ meilleur) ; précision/rappel eau ; fraction de violations topographiques (VF) du masque
prédit.

| split | métrique | baseline | D8 réel λ1 | D8 mél. λ1 | D8 réel λ100 |
|---|---|---:|---:|---:|---:|
| val | mIoU | 0,8546 | 0,8586 (**+0,004**) | 0,8401 | 0,8398 (−0,015) |
| test | mIoU | **0,8615** | 0,8600 (−0,001) | 0,8460 | 0,8579 (−0,004) |
| bolivia | mIoU | **0,8434** | 0,8419 (−0,001) | 0,8379 | 0,8408 (−0,003) |
| test | précision | 0,8680 | **0,9026** (+0,035) | 0,8927 | 0,8620 |
| test | rappel | **0,8604** | 0,8243 (−0,036) | 0,8036 | 0,8586 |
| test | topo-VF | 0,00110 | 0,00091 | 0,00079 | 0,00094 |

Deux faits décisifs :

1. **La contrainte est slack.** Contribution effective $\rho = 7{,}1\times10^{-5}$ à $\lambda=1$
   et $3{,}0\times10^{-3}$ à $\lambda=100$ — soit **0,007 % / 0,3 %** de l'objectif. Le baseline
   ne viole déjà la contrainte topographique que sur $\approx 0{,}1\%$ des pixels
   (VF $\approx 0{,}0011$). Il n'y a presque rien sur quoi la pénalité puisse agir.
2. **Aucun gain robuste hors-ensemble/OOD.** D8 réel améliore `val` de +0,004 mIoU mais *coûte*
   −0,001 sur `test` et OOD `bolivia`. Mécaniquement, il ne fait qu'échanger **rappel contre
   précision** (test : +0,035 précision, −0,036 rappel) — il rétrécit le masque eau, il ne
   l'améliore pas. $\lambda=100$ dégrade tous les splits.

**Réel vs mélangé n'est pas un signal physique robuste.** À N=100 λ1 le réel devance légèrement
le mélangé, mais le run mélangé a convergé plus tôt (meilleure époque 22 vs 31) et une étude
antérieure N=50 de la loss topographique a trouvé le signe **opposé** (mélangé > réel sur tous
les splits). Sur différents réglages, le signe de (réel − mélangé) **s'inverse** et la mesure
est à graine unique. Combiné à $\rho\approx 0$, le *contenu* topographique ne procure aucun
bénéfice reproductible au-delà d'une régularisation générique. ∎(L1)

---

### Lemme L1b — headroom mécaniste au niveau de la classe : $\mathcal C_{\mathrm{ord}}$ (canal II)

L1 a testé une formulation (D8, graine 0). L1b évalue le **mécanisme partagé** de la classe
entière $\mathcal C_{\mathrm{ord}}$ (D5) : comparer les prédictions baseline et les labels de
référence dans l'espace de violation natif de la contrainte. Toutes les mesures sont
inference-only, checkpoint gelé.

#### 5.1 Headroom de fraction de violations D8

$H_R = \mathrm{VF}(\hat y_0, z) - \mathrm{VF}(y, z)$, pondéré par la pente, $s_0=1{,}0$, $\tau=0{,}05$.

| split | VF(labels) | VF(pred) | $H_R$ (poolé) | moyenne par tuile | IC 95 % (par tuile) | % tuiles pred ≤ labels |
|---|---:|---:|---:|---:|---|---:|
| val | 0,00217 | 0,00062 | **−0,00154** | −0,00228 | [−0,00303, −0,00159] | 88 % |
| test | 0,00241 | 0,00058 | **−0,00182** | −0,00268 | [−0,00344, −0,00120] | 90 % |
| bolivia | 0,00321 | 0,00094 | **−0,00227** | −0,00247 | [−0,00406, −0,00105] | 80 % |

**Note sur les deux estimateurs.** Le $H_R$ *poolé* pondère les tuiles par leur nombre de pixels
actifs (les tuiles plus grandes/actives contribuent davantage) ; la *moyenne par tuile* et son IC
bootstrap donnent un poids égal à chaque tuile. Sur val, la valeur poolée (−0,00154) se trouve
légèrement au-dessus de la borne supérieure de l'IC par tuile (−0,00159), car les grandes tuiles
actives tendent à avoir un $H_R$ de plus faible magnitude, tirant l'estimateur poolé vers zéro.
**Les deux estimateurs sont négatifs ; tous les IC par tuile sont entièrement négatifs.**
La conclusion est inchangée : le baseline viole D8 3–4× moins que les labels.

Interprétation : le $H_R$ négatif indique l'absence de headroom VF natif pour toute loss de style
D8. Les prédictions sont déjà plus cohérentes avec l'ordre topographique local que les masques de
référence.

#### 5.2 Headroom natif D4

D4 utilise tous les voisins inférieurs au sens des 4 connexités (pas seulement la descente la plus
rapide). Il appartient à $\mathcal C_{\mathrm{ord}}$ par le même mécanisme. Résultats (mêmes
paramètres) :

| split | VF$_{\mathrm{D4}}$(labels) | VF$_{\mathrm{D4}}$(pred) | $H_R^{\mathrm{D4}}$ | % tuiles pred ≤ labels |
|---|---:|---:|---:|---:|
| val | 0,00180 | 0,00055 | **−0,00125** | 88 % |
| test | 0,00204 | 0,00051 | **−0,00153** | 90 % |
| bolivia | 0,00284 | 0,00082 | **−0,00202** | 80 % |

Le résultat d'absence de headroom n'est pas un artefact du choix de voisin à descente maximale.
Il tient sous la formulation locale à 4 voisins. $H_R^{\mathrm{D4}} \in [-0{,}00202, -0{,}00125]$
selon les splits. ∎(sous-résultat D4)

#### 5.3 Énergie de violation $E_{\mathrm{topo}}$

| split | $E_{\mathrm{topo}}$(labels) | $E_{\mathrm{topo}}$(pred $\hat y$) | $\Delta E$ | $E_{\mathrm{topo}}$(pred soft $p$) |
|---|---:|---:|---:|---:|
| val | 0,001955 | 0,000564 | **−0,001391** | — |
| test | 0,002174 | 0,000527 | **−0,001647** | — |
| bolivia | 0,002896 | 0,000849 | **−0,002047** | — |

Le baseline a une énergie de violation environ **3× inférieure** à celle des labels. L'absence de
headroom n'est pas seulement visible dans un comptage binaire de violations ; elle tient dans
l'énergie quadratique pondérée que la loss D8 minimise directement. ∎(sous-résultat E_topo)

#### 5.4 Dominance distributionnelle : $P(v > t)$

On pourrait objecter qu'une VF moyenne inférieure est insuffisante — le baseline pourrait quand
même avoir des violations sévères qu'une loss pondérée pourrait cibler. Nous comparons les
queues empiriques $P(v_{\hat p_0}>t)$ vs $P(v_y>t)$ pour $t\in\{0; 0{,}05; 0{,}10; 0{,}20;
0{,}30; 0{,}50; 0{,}70\}$ :

$$
P(v_{\hat p_0} > t) \;\le\; P(v_y > t)
$$

est vérifié à **tous les seuils testés sur les trois splits** (val, test, bolivia/OOD). C'est
plus fort qu'une comparaison de moyennes : le baseline viole la contrainte d'ordre topographique
local moins que les labels sur toute la gamme des sévérités testées, ce qui affaiblit l'objection
selon laquelle une pondération plus forte des violations sévères pourrait récupérer un headroom.
∎(dominance distributionnelle)

#### 5.5 Taux de violations utiles : signal réel, aucun bénéfice net

Parmi les violations D8 actives dans la prédiction baseline — l'ensemble sur lequel la loss D8
agirait — nous classons le pixel centre $i$ selon la vérité terrain :

- **Utile** $(\hat y_i=1,\, y_i=0)$ : supprimer $p_i$ retire un faux positif.
- **Nuisible** $(\hat y_i=1,\, y_i=1)$ : supprimer $p_i$ retire de la vraie eau (perte de rappel).
- **Approuvé par le GT** : le masque GT lui-même a $y_i=1,\, y_{d(i)}=0$ à cet emplacement — la
  loss pénalise une configuration correcte selon les labels.

| split | n violations | utiles | nuisibles | approuvées GT | enrichissement |
|---|---:|---:|---:|---:|---:|
| val | 23 168 | 58,6 % | 41,4 % | 8,9 % | ≈ 3,9× |
| test | 21 950 | 58,6 % | 41,4 % | 10,2 % | ≈ 4,4× |
| bolivia | 5 875 | 60,9 % | 39,1 % | 6,9 % | ≈ 3,7× |

*Enrichissement* = P(GT sec | violation D8-active) / P(GT sec | prédiction eau) ≈ 3,7–4,4×.

**Interprétation.** Le signal D8 est réel mais pas net-utile dans ce contexte relatif aux labels.
Les violations topographiques sont enrichies en faux positifs (≈ 4× au-dessus du taux de FP
baseline). Cependant, environ 41 % des corrections suppriment de la vraie eau. Ceci explique
précisément le pattern observé en L1 : appliquer D8 augmente la **précision** (+0,035) mais coûte
du **rappel** (−0,036), laissant l'IoU essentiellement inchangé. ∎(L1b)

---

### Lemme L2 — la marge au niveau sortie est non positive (canal I, sortie)
$A(-z;\cdot)$ poolé, depuis `elevation_auc_predictions_segman_n100_dice_ce_seed0` :

| split | $A(-z;y)$ | $A(-z;\hat y)$ | $H_z$ | écart-type par tuile |
|---|---:|---:|---:|---:|
| val | 0,7537 | 0,7453 | **+0,008** | ≈ 0,18 |
| test | 0,7615 | 0,7684 | **−0,007** | ≈ 0,18 |
| bolivia | 0,5664 | 0,5929 | **−0,027** | ≈ 0,10 |

Contrôles de cohérence dans le même run : AUC élévation mélangée $= 0{,}500$ (pas de fuite) ;
AUC modèle-vs-GT $= 0{,}985$ (la segmentation est excellente). $H_z\le 0$ sur `test` et OOD
`bolivia` ; la seule valeur positive (`val`, +0,008) est deux ordres de grandeur sous l'écart-type
par tuile (0,18) — indistinguable de zéro. **Les prédictions sont déjà aussi bien alignées
topographiquement que les labels.** ∎(L2)

### Lemme L3 — aucune représentation d'élévation privilégiée (canal I, représentation)
Sonde de décodabilité DEM linéaire intra-tuile, backbone gelé, Pearson $r$ (poolé) :

| split | entraîné | aléatoire | input brut |
|---|---:|---:|---:|
| val | 0,414 | 0,430 | 0,405 |
| test | 0,368 | 0,348 | 0,372 |
| bolivia | 0,337 | 0,367 | **0,474** |

$D(\phi_{\text{entraîné}})\approx D(\phi_{\text{aléatoire}})\approx D(\phi_{\text{input}})$ sur
chaque split ; sur OOD `bolivia`, c'est l'*input brut* qui décode le mieux l'élévation. Un
backbone aléatoire égale le backbone entraîné. **L'entraînement à la segmentation n'a induit
aucune représentation d'élévation privilégiée** que renforcer une contrainte pourrait exploiter ;
la faible décodabilité ($r\approx 0{,}4$) est une corrélation passive optique/SAR↔terrain au
niveau de l'input, que tout réseau transmet. *Limite :* sonde linéaire ; mais
`entraîné ≈ aléatoire ≈ input` ramène la revendication à « l'entraînement n'ajoute rien sur
l'input ». ∎(L3)

### Lemme L4 — redondance conditionnelle : le bras formulation-indépendant (canal I, toutes formulations)
Le gradient boosting prédit $y$ ; $A:\,y\sim\hat p$, $B:\,y\sim\hat p+z_{\text{DEM}}$,
$C:\,y\sim z_{\text{DEM}}$, $D:\,y\sim\hat p+\text{mélange}(z)$. Bootstrap par tuile, IC 95 %.

| split | AUC(A) | AUC(B) | $\Delta$AUC = B−A | IC 95 % | DEM seul (C) |
|---|---:|---:|---:|---|---:|
| val | 0,9840 | 0,9861 | **+0,0021** | [+0,0008, +0,0043] | 0,8366 |
| test | 0,9848 | 0,9840 | **−0,0009** | [−0,0038, +0,0016] | 0,8336 |
| bolivia | 0,9817 | 0,9822 | **+0,0004** | [−0,0014, +0,0029] | 0,7660 |

- **$\Delta$AUC $\approx 0$ partout** tandis que **AUC DEM seul = 0,77–0,84** ⇒ vraie
  *redondance conditionnelle*, pas « le DEM est sans intérêt ». C'est l'estimateur empirique de
  $I(y;z\mid\hat p)\approx 0$.
- **Cohérence** : $D$ (DEM mélangé) $=$ $A$ exactement. **Importance des features** (modèle B) :
  $\hat p=0{,}39$, toute feature DEM $\le 0{,}002$.
- **Vérification anti-plafond** (pixels incertains $0{,}05<\hat p<0{,}95$, où il existe de la
  marge) : $\Delta$AUC = +0,011 (val) / **−0,021 (test)** / +0,004 (bolivia) — les signes
  s'inversent, le DEM *nuit* même sur test. Le résultat nul n'est pas un artefact de plafond.

Le gradient boosting pouvant modéliser des interactions DEM↔$y$ arbitrairement complexes, il
constitue une borne supérieure de la capacité informationnelle de *toute* formulation de loss DEM.
Il trouve $\approx 0$. ∎(L4)

---

## 6. L'argument composé

**Canal informationnel (I).** Par la Proposition 1, le canal I est gouverné par $I(y;z\mid\hat p)$.
- L2 le borne à $\le 0$ au niveau **sortie** (marge non positive).
- L3 le borne à $\approx 0$ au niveau **représentation** (aucun encodage privilégié).
- L4 l'estime directement à $\approx 0$ pour **toutes les formulations** (apprenant flexible,
  avec IC, vérification anti-plafond, contrôle négatif opérationnel).

Ces trois mesures sont indépendantes, à trois niveaux (prédiction, features, information
post-hoc), et elles concordent. ⇒ **Le canal I n'a aucune marge exploitable par les labels dans
ce couple modèle–données.**

**Canal du biais inductif (II), classe $\mathcal C_{\mathrm{ord}}$.** L1 et L1b ferment
conjointement le canal II pour cette classe dans cette configuration :

- L1 : l'entraînement D8 direct montre $\rho\approx 0$, aucun gain robuste test/OOD, seulement
  un échange précision↔rappel.
- L1b §5.1 : $H_R < 0$ sur tous les splits (IC par tuile entièrement négatif) — le baseline
  viole D8 3–4× moins que les labels.
- L1b §5.2 : $H_R^{\mathrm{D4}} < 0$ — même résultat sous la formulation à 4 voisins.
- L1b §5.3 : $E_{\mathrm{topo}}(\hat y_0) < E_{\mathrm{topo}}(y)$ d'un facteur $\approx 3$ —
  tient dans l'énergie pondérée par la sévérité.
- L1b §5.4 : dominance distributionnelle $P(v_{\hat p_0}>t)\le P(v_y>t)$ à tous les seuils.
- L1b §5.5 : 41 % des corrections actives suppriment de la vraie eau, expliquant le compromis
  précision/rappel sans gain net d'IoU.

⇒ **Pour la classe $\mathcal C_{\mathrm{ord}}$, le canal II ne procure aucun bénéfice net
relatif aux labels dans cette configuration. Le signal est réel (enrichissement ≈ 4× en FP)
mais pas net-utile (41 % de collateral).**

> **Théorème (borné, relatif aux labels, au niveau de la classe).** Pour *(SegMAN-S, N=100
> graine 0, labels de référence y)* :
> (i) aucune loss DEM d'aucune formulation n'a de marge exploitable par les labels via le canal
> informationnel (Prop. 1 + L2 + L3 + L4) ; et (ii) la classe $\mathcal C_{\mathrm{ord}}$ des
> losses d'ordre topographique local monotones n'exhibe aucun headroom natif net relatif aux
> labels (L1 + L1b). **Dans cette tâche, une loss DEM topographique locale monotone n'est pas
> justifiée.**

Le résultat nul est donc une **propriété informationnelle et mécaniste du couple modèle–données**,
non un échec à trouver la bonne formule dans cette classe.

---

## 7. Ce qui n'est PAS prouvé (les portes ouvertes — à énoncer explicitement)

1. **Canal II pour d'*autres classes* de formulations.** L1 + L1b ferment le canal II pour
   $\mathcal C_{\mathrm{ord}}$ (D4, D8, variantes VF-monotones). Les losses basées sur un
   **mécanisme différent** — connectivité hydrologique, accumulation de flux, HAND, contraintes
   non locales — sont hors de $\mathcal C_{\mathrm{ord}}$ et **ne sont pas testées**. Le théorème
   ne leur est pas applicable.
2. **État physique réel vs labels (A3).** Chaque lemme est relatif aux labels. Si les labels sont
   topographiquement bruités, une loss DEM pourrait encore améliorer l'accord avec la vérité non
   observée $y^\star$. Les ≈ 8–10 % de violations approuvées par le GT (§5.5) suggèrent qu'un
   certain bruit d'annotation est présent.
3. **DEM comme modalité d'input libre.** Toutes les preuves concernent le DEM comme
   *contrainte/feature*. Le test le plus fort du canal I — ré-entraîner avec le DEM comme canal
   d'entrée, le laissant remodeler la représentation de bout en bout — **n'est pas encore
   effectué**. L4 conditionne sur un $\hat p$ *gelé* et ne peut donc pas exclure qu'un modèle
   entraîné *avec* le DEM ait construit un $\hat p$ différent.
4. **Graine unique / contrôle positif manquant.** Tous les runs d'entraînement sont à graine 0.
   Le contrôle positif définitif (un checkpoint S1-seul ou sous-entraîné où $\Delta$AUC *devrait*
   devenir clairement positif) n'existe pas.

---

## 8. Falsifiabilité (ce qui renverserait cette conclusion)

| observation | quel lemme elle brise | conséquence |
|---|---|---|
| Une loss de $\mathcal C_{\mathrm{ord}}$ donne un gain robuste multi-graine test/OOD | L1 / Théorème(ii) | canal II actif pour cette formulation |
| $H_R > 0$ ou IC par tuile non entièrement négatif | L1b §5.1 | headroom VF existant |
| $E_{\mathrm{topo}}(\hat y_0) > E_{\mathrm{topo}}(y)$ | L1b §5.3 | headroom en énergie existant |
| $P(v_{\hat p_0}>t) > P(v_y>t)$ pour un seuil | L1b §5.4 | headroom en queue existant |
| IC de $\Delta$AUC (B−A) clairement $>0$ sur un split hors-ensemble | L4 / prémisse Prop. 1 | information conditionnelle existante |
| $H_z$ robustement $>0$ au-delà de l'écart-type par tuile sur test/OOD | L2 | marge en sortie existante |
| DEM-en-input bat S1/S2 (et DEM mélangé), multi-graine | porte ouverte n°3 | la modalité porte un signal exploitable |
| Des labels plus propres révèlent un gain DEM vis-à-vis de $y^\star$ | porte ouverte n°2 (A3) | la revendication est un artefact des labels |

Un seul tel résultat falsifie le bras correspondant. À ce jour, aucun n'est observé.

---

## 9. Conclusion

**Revendication défendable précise :**
> « Nous ne réfutons pas D4 et D8 indépendamment. Nous évaluons le mécanisme partagé de leur
> classe : réduire les violations d'ordre topographique local. Dans la configuration N=100
> SegMAN-S/Sen1Floods11, le baseline non contraint Dice+CE présente déjà une fraction de
> violations inférieure, une énergie de violations inférieure et des queues de violations
> inférieures à celles des labels de référence. Les violations D8 actives sont enrichies en faux
> positifs (≈ 4× au-dessus du taux FP baseline), montrant que le signal est réel, mais environ
> 41 % des corrections suppriment de la vraie eau, expliquant le compromis précision–rappel et
> l'absence de gain net d'IoU. Par conséquent, les losses DEM monotones locales d'ordre
> topographique dont le seul mécanisme est de réduire les violations D4/D8/VF n'ont aucun
> headroom natif relatif aux labels dans cette configuration. Il s'agit d'un résultat N=100,
> graine 0, relatif aux labels ; §10 liste ce qui reste avant une revendication publiable. »

**Phrases à éviter (un reviewer les démonterait) :**
- « Le DEM est inutile pour la segmentation d'inondation. » (L4-C : AUC DEM seul 0,77–0,84.)
- « Aucune loss physics-informed ne peut jamais aider. » (Canal II, autres classes — porte ouverte n°1.)
- « Le modèle a appris / compris l'élévation. » (L3 : le contraire.)
- « DEM-en-input inutile ⇒ loss DEM inutile. » (Saute le canal II ; le test input n'est pas encore effectué.)
- « Aucune méthode basée sur le DEM ne peut jamais fonctionner. » (Le périmètre est $\mathcal C_{\mathrm{ord}}$ uniquement.)

**Conséquence opérationnelle.** Arrêter d'itérer sur des formulations dans $\mathcal C_{\mathrm{ord}}$
pour ce couple SegMAN N=100. Les prochains mouvements à haute valeur sont : (a) l'ablation
DEM-en-input + contrôle positif S1-seul pour fermer les portes ouvertes n°3 et n°4 ; (b) porter
le diagnostic headroom/redondance sur la prochaine architecture (EoMT) pour montrer qu'il
*généralise* comme test a priori de l'utilité d'une loss physique — transformant ce résultat
négatif en méthode.

---

## 10. Limites

1. **N=100 est préliminaire/mécaniste.** Ces expériences démontrent le mécanisme dans une
   configuration à données restreintes contrôlée. Les revendications publiables nécessitent des
   expériences Sen1Floods11 complètes avec confirmation multi-graine sur plusieurs architectures.

2. **Graine unique pour les expériences d'entraînement.** L1 (résultat direct de la loss) est à
   graine 0 uniquement. Une confirmation multi-graine (cible : 5 graines × 3 conditions, test
   TOST d'équivalence avec $\delta\approx\sigma_{\text{graine}}\approx 0{,}003$–$0{,}005$
   IoU-eau) est une étape suivante requise. Les diagnostics de L1b sont inference-only et ne
   dépendent pas de la graine d'entraînement.

3. **Relatif aux labels (Axiome A3).** Tous les résultats sont relatifs aux labels de référence
   disponibles. Si les labels sont systématiquement bruités dans les zones topographiquement
   complexes, le DEM pourrait améliorer l'accord avec l'état physique vrai $y^\star$ sans
   améliorer l'accord avec $y$. Les ≈ 8–10 % de violations approuvées par le GT (§5.5)
   suggèrent la présence d'un certain bruit d'annotation.

4. **Périmètre : losses d'ordre topographique local monotones uniquement.** La conclusion
   s'applique à $\mathcal C_{\mathrm{ord}}$ — losses pénalisant $p_i>p_j+\tau$ quand $h_i>h_j$
   pour des pixels voisins. Elle ne s'applique **pas** à :
   DEM en input · HAND · accumulation de flux · losses de connectivité hydrologique ·
   contraintes pluie/débit · contraintes hydrodynamiques temporelles · méthodes DEM non locales
   ou orthogonales.

5. **DEM en input non testé.** Cette étude porte sur le DEM comme contrainte uniquement. Un
   modèle entraîné avec le DEM comme canal d'entrée pourrait construire un $\hat p$ différent,
   pour lequel L4 devrait être réévalué.

6. **Résolution spatiale et limites d'annotation.** Le MNT Copernicus GLO-30 est un MNS (résolution
   30 m, inclut bâtiments/canopée, non terrain nu). Les règles locales D8/D4 à 30 m peuvent
   signaler des configurations d'inondation légitimes comme violations — eau de retenue, eau
   contenue par des digues, drainage urbain, accumulation d'eau de pluie, ambiguïté de plaine
   inondable plate — contribuant aux 41 % de violations nuisibles et 7–10 % de violations
   approuvées par le GT.

7. **Architecture unique.** Les résultats portent sur SegMAN-S. Si le diagnostic mécaniste se
   généralise à d'autres architectures (EoMT, modèles ViT) est une étape suivante explicite.

---

## 11. Environnement et reproductibilité

Nous utilisons l'architecture complète SegMAN-S adaptée à des inputs Sentinel-1/Sentinel-2
à 15 canaux. Dans l'environnement Windows/PyTorch, les extensions CUDA optimisées sont remplacées
par des opérateurs PyTorch de substitution pour la compatibilité et la reproductibilité.
**Le DEM n'est jamais utilisé comme input du modèle.**

| Vérification | Résultat |
|---|---|
| OS | Windows 10 Pro for Workstations, build 26200 |
| Python | 3.11.9 (MSC v.1938 64 bits) |
| PyTorch / CUDA | 2.5.1+cu121 / CUDA 12.1 |
| GPU | NVIDIA RTX 5000 Ada Generation, 32 Go |
| Paramètres SegMAN-S | 33 447 272 (33,45 M) |
| Forme d'entrée | (1, 15, 512, 512) |
| Forme de sortie | (1, 2, 512, 512) |
| Extensions CUDA officielles | non installées (mamba-ssm, natten, mmcv-full) |
| Shims PyTorch | actifs : selective scan (exact), cross-scan (exact), NATTEN (exact sur pixels intérieurs) |
| DEM comme input modèle | jamais |
| Checkpoints N=100 | 41 runs, tous avec `best_checkpoint.pt` (≈ 767 Mo chacun) |
| Rapport de reproductibilité | `E:\flood_research\setup_logs\windows_segman_env\windows_segman_reproducibility_report.md` |

L'équivalence NATTEN a été vérifiée pour les pixels intérieurs ; le comportement aux bords
(anneau de ≈ 1 pixel) est documenté dans le code source des shims (`segman_kernels/compat.py`)
et est négligeable à 512×512.

**Dépôt principal :** commit `8017b883` sur `https://github.com/KarimKQF/physics-informed-flood-segmentation.git`  
**Source SegMAN externe :** commit `9ced66ab` (officiel CVPR 2025)

---

## 12. Provenance (artefacts sous-tendant chaque chiffre)

- **L1 :** `reports/segman_n100_d8_seed0_results.json`, `reports/segman_n100_d8_lambda100_seed0_results.json`
- **L1b §5.1–5.2 :** `outputs/native_vf_headroom/native_vf_headroom_results.json`
- **L1b §5.2–5.4 (D4/E_topo/queues) :** `outputs/topographic_order_headroom/topographic_order_headroom_results.json`
- **L2 :** `reports/elevation_auc_predictions_segman_n100_dice_ce_seed0.json`
- **L3 :** `docs/dem_decodability_segman_n100_dice_ce_seed0.md`, `reports/dem_decodability_segman_n100_dice_ce_seed0.json`
- **L4 :** `outputs/conditional_dem_redundancy/conditional_dem_redundancy_report.md` (+ `.json`, `.csv`)
- **Contrôle N=50 renversement de signe réel-vs-mélangé :** `reports/segman_seed0_loss_comparison_summary.json`
- **Rapport de reproductibilité :** `E:\flood_research\setup_logs\windows_segman_env\windows_segman_reproducibility_report.md`
