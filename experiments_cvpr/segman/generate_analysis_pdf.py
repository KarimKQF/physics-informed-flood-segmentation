"""
Generate a self-contained PDF report of the SegMAN N=50 visual/physical analysis.
Embeds all comparison panels, topo violation maps, and the summary chart.
Uses only matplotlib + PIL (no external tools needed).

Output: reports/segman_n50_visual_analysis_report.pdf
Usage:  python experiments_cvpr/segman/generate_analysis_pdf.py
"""
from __future__ import annotations

import csv
import json
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image

# ── Paths ───────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT  = REPO_ROOT / "reports" / "figures" / "segman_n50"
OUT_PDF   = REPO_ROOT / "reports" / "segman_n50_visual_analysis_report.pdf"

# ── Data constants ───────────────────────────────────────────────────────────
TILE_IDS = [
    "Ghana_1078550", "Ghana_141271", "Ghana_167233",
    "Ghana_313799",  "Ghana_319168", "Ghana_359826",
]
WATER_GT = [0.401, 0.659, 0.006, 0.037, 0.000, 0.156]

CONDITIONS  = ["ce", "dice_ce", "dice_ce_topo", "dice_ce_topo_dem_shuffled"]
LABELS      = {"ce": "CE", "dice_ce": "Dice+CE",
               "dice_ce_topo": "Topo réel",
               "dice_ce_topo_dem_shuffled": "Topo shufflé"}
COLORS      = {"ce": "#94a3b8", "dice_ce": "#38bdf8",
               "dice_ce_topo": "#22d3ee",
               "dice_ce_topo_dem_shuffled": "#fbbf24"}

# ── Colour palette ───────────────────────────────────────────────────────────
BG   = "#0d1117"
CARD = "#161b22"
TXT  = "#e5e7eb"
DIM  = "#64748b"
ACC  = "#22d3ee"
WARN = "#fbbf24"
GRN  = "#22c55e"
RED  = "#ef4444"
TAN  = "#d4a373"

A4P = (8.27, 11.69)    # portrait inches
A4L = (11.69, 8.27)    # landscape inches
LH  = 0.038            # normal line-height in figure coords (portrait)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_metrics() -> dict:
    rows = list(csv.DictReader(
        (FIG_ROOT / "tables" / "per_image_metrics.csv").open(encoding="utf-8")))
    topo = json.loads(
        (FIG_ROOT / "tables" / "topo_violation_per_image.json").read_text(encoding="utf-8"))
    deltas = list(csv.DictReader(
        (FIG_ROOT / "tables" / "delta_summary.csv").open(encoding="utf-8")))

    iou: dict[int, dict[str, tuple]] = {i: {} for i in range(6)}
    for idx in range(6):
        for cond in CONDITIONS:
            vals = [float(r["iou_water"]) for r in rows
                    if r["condition"] == cond and int(r["img_idx"]) == idx and r["iou_water"]]
            if vals:
                mu = sum(vals) / len(vals)
                sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1))
                iou[idx][cond] = (mu, sd)
            else:
                iou[idx][cond] = (float("nan"), 0.0)

    viol: dict[int, dict[str, float]] = {}
    for k, v in topo.items():
        viol[int(k)] = v.get("violations", {})

    return {"iou": iou, "viol": viol, "deltas": deltas}


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_png(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("RGB"))


def show_png(ax, path: Path, title: str | None = None, title_color: str = TXT,
             title_size: int = 8):
    ax.set_facecolor(CARD)
    img = load_png(path)
    if img is not None:
        ax.imshow(img, aspect="auto")
    else:
        ax.text(0.5, 0.5, "Fichier\nnon trouvé", transform=ax.transAxes,
                ha="center", va="center", color=DIM, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
        spine.set_linewidth(0.5)
    if title:
        ax.set_title(title, color=title_color, fontsize=title_size, pad=3)


# ── fig.text helper ───────────────────────────────────────────────────────────

def ft(fig, x: float, y: float, text: str, size: int = 10,
       color: str = TXT, weight: str = "normal",
       ha: str = "left", va: str = "top", **kw) -> float:
    """Write text and return y minus consumed height (for tracking cursor)."""
    fig.text(x, y, text, fontsize=size, color=color, weight=weight,
             ha=ha, va=va, **kw)
    return y


def ft_wrap(fig, x: float, y: float, text: str, width: int = 90,
            size: int = 9, color: str = TXT, indent: float = 0.0,
            line_h: float = LH) -> float:
    """Write wrapped text, return final y position after all lines."""
    for line in textwrap.wrap(text, width):
        fig.text(x + indent, y, line, fontsize=size, color=color, va="top")
        y -= line_h
    return y


def hline(fig, y: float, x0: float = 0.05, x1: float = 0.95,
          color: str = "#1e293b", lw: float = 0.8):
    """Draw a horizontal line in figure coordinates."""
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=color, lw=lw,
                              transform=fig.transFigure, clip_on=False))


# ────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Titre et introduction
# ────────────────────────────────────────────────────────────────────────────

def page_title(pdf: PdfPages):
    fig = plt.figure(figsize=A4P, facecolor=BG)

    y = 0.95
    ft(fig, 0.5, y, "SegMAN-S N=50", size=20, color=ACC, weight="bold",
       ha="center")
    y -= 0.05
    ft(fig, 0.5, y, "Analyse Visuelle et Physique des Prédictions",
       size=14, color=TXT, weight="bold", ha="center")
    y -= 0.035
    ft(fig, 0.5, y, "Loss ablation : CE / Dice+CE / Topo réel / Topo shufflé  ·  N_train = 50  ·  Seeds 0,1,2,3,42",
       size=9, color=DIM, ha="center")
    y -= 0.025
    hline(fig, y, color=ACC, lw=1.2)
    y -= 0.030

    # ── Introduction ──────────────────────────────────────────────────────
    ft(fig, 0.06, y, "Objectif", size=11, color=ACC, weight="bold")
    y -= 0.028
    intro = (
        "Ce rapport compare quatre fonctions de perte sur SegMAN-S entraîné sur 50 images "
        "Sen1Floods11 (15 canaux S1+S2). La question centrale : est-ce que la loss "
        "topographique avec le vrai DEM (modèle numérique de terrain) produit des "
        "prédictions physiquement plus cohérentes que le même terme avec un DEM aléatoire ?"
    )
    y = ft_wrap(fig, 0.06, y, intro, width=88, size=9.5, line_h=0.030)
    y -= 0.018
    hline(fig, y, color="#1e293b")
    y -= 0.028

    # ── Conditions table ──────────────────────────────────────────────────
    ft(fig, 0.06, y, "Les 4 conditions de loss", size=11, color=ACC, weight="bold")
    y -= 0.032

    cond_info = [
        ("ce",                        "Cross-Entropy (CE)",
         "Référence de base. Apprend à matcher les labels pixel par pixel."),
        ("dice_ce",                   "Dice + Cross-Entropy",
         "Meilleur équilibre eau/fond. Réduit l'effet de déséquilibre de classe."),
        ("dice_ce_topo",              "Dice+CE + Topo DEM réel",
         "Ajoute une pénalité si l'eau prédit est plus haute que le terrain sec "
         "adjacent (vrai DEM Copernicus GLO-30 aligné)."),
        ("dice_ce_topo_dem_shuffled", "Dice+CE + Topo DEM shufflé (contrôle)",
         "Identique à Topo réel, mais DEM tiré aléatoirement d'une autre tuile. "
         "Si ce variant ≈ Topo réel, le DEM réel n'apporte rien de spécifique."),
    ]
    for cond, name, desc in cond_info:
        clr = COLORS[cond]
        # Colored bullet
        fig.add_artist(Rectangle(
            (0.06, y - 0.009), 0.025, 0.022,
            facecolor=clr, transform=fig.transFigure, clip_on=False, zorder=3
        ))
        ft(fig, 0.093, y, name, size=10, color=clr, weight="bold")
        y -= 0.026
        y = ft_wrap(fig, 0.093, y, desc, width=82, size=8.5,
                    color="#cbd5e1", line_h=0.026)
        y -= 0.014
    y -= 0.008
    hline(fig, y, color="#1e293b")
    y -= 0.028

    # ── Métriques ─────────────────────────────────────────────────────────
    ft(fig, 0.06, y, "Métriques utilisées", size=11, color=ACC, weight="bold")
    y -= 0.032
    metrics_info = [
        ("IoU_water",        ACC,
         "Intersection / Union pour la classe eau. De 0 (raté) à 1 (parfait). "
         "Moyenne sur 5 seeds d'initialisation par image."),
        ("Violation topo",   TAN,
         "% de pixels où l'eau prédit est physiquement incohérente : pixel eau "
         "plus haut qu'un voisin sec adjacent (margin = 0 m). "
         "Calculé sur les masques durs du seed 0 — approximation diagnostique."),
        ("ΔIoU",             WARN,
         "Différence de IoU_water entre deux conditions. Positif = amélioration. "
         "Barres vertes = gain > 0.005, rouges = perte < -0.005."),
    ]
    for mname, mclr, mdesc in metrics_info:
        ft(fig, 0.08, y, f"• {mname} :", size=9.5, color=mclr, weight="bold")
        y -= 0.026
        y = ft_wrap(fig, 0.10, y, mdesc, width=82, size=8.5,
                    color="#cbd5e1", line_h=0.026)
        y -= 0.010

    # ── Limitation box ────────────────────────────────────────────────────
    y -= 0.015
    hline(fig, y, color="#1e293b")
    y -= 0.025
    ft(fig, 0.06, y, "Limitation importante", size=10, color=WARN, weight="bold")
    y -= 0.028
    lim = (
        "Seules les 6 premières images du jeu de test (dataloader déterministe) ont "
        "été sauvegardées comme prédictions NPZ — toutes issues de Ghana. L'analyse "
        "visuelle ne couvre pas la Bolivie (OOD) ni les autres régions géographiques. "
        "Le jeu de test complet (89 images) a été évalué via les métriques agrégées "
        "du summary JSON, mais sans prédictions NPZ individuelles pour chaque image."
    )
    y = ft_wrap(fig, 0.06, y, lim, width=88, size=8.5, color="#cbd5e1", line_h=0.026)

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Graphique résumé IoU
# ────────────────────────────────────────────────────────────────────────────

def page_summary_chart(pdf: PdfPages):
    fig = plt.figure(figsize=A4L, facecolor=BG)

    # Title strip
    ft(fig, 0.5, 0.97,
       "Résumé — IoU_water par image et condition (moyenne ± std, 5 seeds)",
       size=13, color=ACC, weight="bold", ha="center")

    # Chart image taking most of the page
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.88])
    show_png(ax, FIG_ROOT / "summary_iou_water_per_image.png")

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Per-image notes
# ────────────────────────────────────────────────────────────────────────────

IMG_NOTES = {
    0: {
        "titre":   "Image 0 — Ghana_1078550 — Grande inondation (40% eau)",
        "verdict": "Pas d'avantage DEM réel sur cette image",
        "vclr":    DIM,
        "obs": (
            "Dice+CE améliore clairement CE (+2% IoU). La loss Topo réel n'apporte "
            "rien de plus que Dice+CE en IoU. Le Topo shufflé est le meilleur en IoU "
            "(+1.3%) mais produit paradoxalement plus de violations physiques (1.86% "
            "vs 1.41% pour Topo réel). Le shufflé prédit plus d'eau, y compris dans "
            "des zones incohérentes, et ça coïncide accidentellement avec le GT."
        ),
    },
    1: {
        "titre":   "Image 1 — Ghana_141271 — Très inondée (66% eau) — IMAGE CLÉ",
        "verdict": "Signal physique du DEM réel visible ici",
        "vclr":    GRN,
        "obs": (
            "Image la plus importante. En IoU, les 4 conditions sont quasi égales "
            "(0.865–0.877). En violations physiques, le Topo réel est clairement "
            "le meilleur : 0.82% vs 1.21% pour CE et 1.04% pour le shufflé. "
            "Seul cas où le vrai DEM produit des prédictions physiquement plus "
            "cohérentes que le DEM aléatoire de manière mesurable."
        ),
    },
    2: {
        "titre":   "Image 2 — Ghana_167233 — Quasi sèche (0.6% eau) — CAS DÉGÉNÉRÉ",
        "verdict": "Non exploitable — image trop sèche",
        "vclr":    DIM,
        "obs": (
            "Quand il n'y a presque pas d'eau, le modèle ne prédit rien et IoU ≈ 0 "
            "pour toutes les conditions. Les violations sont nulles. Cette image ne "
            "permet aucune comparaison entre les losses. Elle montre simplement que "
            "le modèle ne génère pas de faux positifs massifs sur terrain sec."
        ),
    },
    3: {
        "titre":   "Image 3 — Ghana_313799 — Eau clairsemée (3.7%), haute variance",
        "verdict": "Haute variance — pas de conclusion fiable",
        "vclr":    WARN,
        "obs": (
            "Résultats très instables selon le seed (std ±0.20, presque aussi grande "
            "que la valeur). Topo réel a un meilleur IoU moyen (+4.9% vs Dice+CE) "
            "mais génère aussi plus de violations (0.245% vs 0.093%). La loss topo "
            "pousse probablement le modèle à prédire de l'eau dans des zones basses "
            "ambiguës ne correspondant pas au GT."
        ),
    },
    4: {
        "titre":   "Image 4 — Ghana_319168 — Complètement sèche (0% eau) — CAS DÉGÉNÉRÉ",
        "verdict": "Non exploitable — image sans eau",
        "vclr":    DIM,
        "obs": (
            "Aucune eau dans le masque vrai. ~1% de violations viennent de quelques "
            "faux positifs eau produits par toutes les conditions de manière identique. "
            "Exclue de l'analyse principale."
        ),
    },
    5: {
        "titre":   "Image 5 — Ghana_359826 — Inondation modérée (15.6% eau)",
        "verdict": "Dice+CE optimal ici — Topo sans effet net",
        "vclr":    DIM,
        "obs": (
            "Dice+CE améliore légèrement CE (+1.1% IoU). Topo réel perd légèrement "
            "vs Dice+CE (-0.3%) mais réduit un peu les violations (1.25% vs 1.33%). "
            "Topo shufflé est le moins bon en IoU ET en violations. Aucun signal fort."
        ),
    },
}


# ────────────────────────────────────────────────────────────────────────────
# PAGE 3-8 — Une page par image
# ────────────────────────────────────────────────────────────────────────────

def page_image(pdf: PdfPages, idx: int, metrics: dict):
    tile  = TILE_IDS[idx]
    info  = IMG_NOTES[idx]
    iou   = metrics["iou"][idx]
    viol  = metrics["viol"].get(idx, {})

    fig = plt.figure(figsize=A4L, facecolor=BG)

    # ── GridSpec: 3 rows × 3 cols ────────────────────────────────────────
    # Row 0: title (thin)
    # Row 1: panel image (wide, tall)
    # Row 2: [topo map | bar charts (nested) | observation text]
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        height_ratios=[0.06, 0.54, 0.40],
        width_ratios=[0.41, 0.27, 0.32],
        hspace=0.06, wspace=0.04,
        left=0.01, right=0.99,
        top=0.99, bottom=0.01,
    )

    # ── Row 0: Title ─────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor("#0f172a")
    ax_title.axis("off")
    ax_title.text(0.5, 0.65, info["titre"],
                  transform=ax_title.transAxes, fontsize=11,
                  color=ACC, weight="bold", ha="center", va="center")
    ax_title.text(0.5, 0.15, f"Eau GT : {WATER_GT[idx]:.1%}   |   Seed représentatif : seed0   |   Métriques IoU : moy. ± std sur 5 seeds",
                  transform=ax_title.transAxes, fontsize=7.5,
                  color=DIM, ha="center", va="center")

    # ── Row 1: Comparison panel (full width) ─────────────────────────────
    ax_panel = fig.add_subplot(gs[1, :])
    show_png(ax_panel,
             FIG_ROOT / "panels" / f"panel_{idx:03d}_{tile}.png",
             title="S2 RGB  |  GT  |  CE  |  Dice+CE  |  Topo réel  |  Topo shufflé  |  DEM",
             title_color=DIM, title_size=7.5)

    # ── Row 2 left: Topo violation map ───────────────────────────────────
    ax_topo = fig.add_subplot(gs[2, 0])
    show_png(ax_topo,
             FIG_ROOT / "topo_violations" / f"topo_viol_{idx:03d}_{tile}.png",
             title="Violations topographiques  (rouge = eau physiquement incohérente)",
             title_color=TAN, title_size=7.5)

    # ── Row 2 middle: Two stacked bar charts ─────────────────────────────
    gs_mid = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[2, 1], hspace=0.55
    )
    ax_iou  = fig.add_subplot(gs_mid[0])
    ax_viol = fig.add_subplot(gs_mid[1])

    cond_list  = CONDITIONS
    label_list = [LABELS[c] for c in cond_list]
    color_list = [COLORS[c] for c in cond_list]
    x = np.arange(len(cond_list))

    # IoU bars
    mus = [iou[c][0] if not math.isnan(iou[c][0]) else 0 for c in cond_list]
    sds = [iou[c][1] for c in cond_list]
    ax_iou.bar(x, mus, color=color_list, alpha=0.85, width=0.6, zorder=2)
    ax_iou.errorbar(x, mus, yerr=sds, fmt="none", ecolor="white",
                    capsize=2, lw=0.8, zorder=3)
    ax_iou.set_ylim(0, 1.0)
    ax_iou.set_xticks(x)
    ax_iou.set_xticklabels(label_list, rotation=30, ha="right",
                            fontsize=6.5, color=TXT)
    ax_iou.tick_params(axis="y", labelcolor=TXT, labelsize=6.5, length=2)
    ax_iou.set_facecolor(CARD)
    ax_iou.set_title("IoU_water\n(moy.±std, 5 seeds)", color=TXT,
                     fontsize=7, pad=2)
    for sp in ax_iou.spines.values():
        sp.set_edgecolor("#334155"); sp.set_linewidth(0.5)
    ax_iou.yaxis.grid(True, color="#1e293b", lw=0.5, zorder=0)
    ax_iou.set_axisbelow(True)

    # Violation bars
    viol_vals = [viol.get(c, 0.0) * 100 for c in cond_list]
    ax_viol.bar(x, viol_vals, color=color_list, alpha=0.85, width=0.6, zorder=2)
    ax_viol.set_xticks(x)
    ax_viol.set_xticklabels(label_list, rotation=30, ha="right",
                             fontsize=6.5, color=TXT)
    ax_viol.tick_params(axis="y", labelcolor=TXT, labelsize=6.5, length=2)
    ax_viol.set_facecolor(CARD)
    ax_viol.set_title("Violations topo %\n(seed0)", color=TAN,
                      fontsize=7, pad=2)
    for sp in ax_viol.spines.values():
        sp.set_edgecolor("#334155"); sp.set_linewidth(0.5)
    ax_viol.yaxis.grid(True, color="#1e293b", lw=0.5, zorder=0)
    ax_viol.set_axisbelow(True)

    # ── Row 2 right: Text observation ────────────────────────────────────
    ax_txt = fig.add_subplot(gs[2, 2])
    ax_txt.set_facecolor(CARD)
    ax_txt.axis("off")

    # Numbers summary at top
    lines_num = []
    for cond in cond_list:
        mu, sd = iou[cond]
        v = viol.get(cond, float("nan")) * 100
        mu_s = f"{mu:.3f}±{sd:.3f}" if not math.isnan(mu) else "N/A"
        v_s  = f"{v:.3f}%" if not math.isnan(v) else "N/A"
        lines_num.append(f"{LABELS[cond][:12]:<12}  IoU {mu_s}  viol {v_s}")

    y_t = 0.98
    ax_txt.text(0.04, y_t, "Métriques numériques (seed0 + 5 seeds)",
                transform=ax_txt.transAxes, fontsize=7, color=WARN,
                weight="bold", va="top")
    y_t -= 0.14
    for line in lines_num:
        ax_txt.text(0.04, y_t, line,
                    transform=ax_txt.transAxes, fontsize=6.5,
                    color=TXT, va="top", family="monospace")
        y_t -= 0.13

    # Divider
    ax_txt.axhline(y_t + 0.04, color="#1e293b", lw=0.6)

    # Observation text
    y_t -= 0.06
    ax_txt.text(0.04, y_t, "Observation :", transform=ax_txt.transAxes,
                fontsize=7, color=ACC, weight="bold", va="top")
    y_t -= 0.12
    obs_lines = textwrap.wrap(info["obs"], width=48)
    for line in obs_lines[:8]:
        ax_txt.text(0.04, y_t, line, transform=ax_txt.transAxes,
                    fontsize=6.5, color="#cbd5e1", va="top")
        y_t -= 0.11

    # Verdict
    ax_txt.axhline(0.07, color="#1e293b", lw=0.6)
    ax_txt.text(0.04, 0.05, f"▶  {info['verdict']}",
                transform=ax_txt.transAxes, fontsize=7.5,
                color=info["vclr"], weight="bold", va="bottom")

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 9 — Grille violations topo (2×3)
# ────────────────────────────────────────────────────────────────────────────

def page_topo_grid(pdf: PdfPages):
    fig = plt.figure(figsize=A4L, facecolor=BG)

    ft(fig, 0.5, 0.975,
       "Cartes de violations topographiques — toutes les images (seed 0)",
       size=12, color=TAN, weight="bold", ha="center")
    ft(fig, 0.5, 0.945,
       "Rouge = pixel eau physiquement incohérent (eau plus haute que voisin sec adjacent)",
       size=8.5, color=DIM, ha="center")

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.12, wspace=0.04,
        left=0.01, right=0.99,
        top=0.93, bottom=0.01,
    )
    for idx in range(6):
        row, col = idx // 3, idx % 3
        ax = fig.add_subplot(gs[row, col])
        topo_path = FIG_ROOT / "topo_violations" / f"topo_viol_{idx:03d}_{TILE_IDS[idx]}.png"
        show_png(ax, topo_path,
                 title=f"img{idx} — {TILE_IDS[idx]}  ({WATER_GT[idx]:.0%} eau GT)",
                 title_color=TXT, title_size=8)

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 10 — Tableau des deltas
# ────────────────────────────────────────────────────────────────────────────

def _fmt_delta(v: float) -> tuple[str, str]:
    if math.isnan(v):
        return "N/A", DIM
    color = GRN if v > 0.005 else (RED if v < -0.005 else TXT)
    return f"{v:+.4f}", color


def page_delta_summary(pdf: PdfPages, metrics: dict):
    fig = plt.figure(figsize=A4P, facecolor=BG)

    y = 0.95
    ft(fig, 0.5, y, "Synthèse quantitative — Deltas IoU_water",
       size=14, color=ACC, weight="bold", ha="center")
    y -= 0.035
    ft(fig, 0.5, y,
       "Différence de IoU_water entre conditions (moy. sur 5 seeds par image)  |  Vert > +0.005  ·  Rouge < −0.005",
       size=8, color=DIM, ha="center")
    y -= 0.025
    hline(fig, y, color=ACC, lw=1)
    y -= 0.032

    # ── Column headers ────────────────────────────────────────────────────
    col_x = [0.05, 0.10, 0.30, 0.43, 0.58, 0.73, 0.87]
    headers = ["#", "Tile", "Eau GT", "Dice+CE − CE", "Topo − D+CE", "Shuf − Topo", "Shuf − D+CE"]
    for hdr, x in zip(headers, col_x):
        ft(fig, x, y, hdr, size=9, color=WARN, weight="bold")
    y -= 0.026
    hline(fig, y, color="#334155")
    y -= 0.008

    # ── Data rows ─────────────────────────────────────────────────────────
    deltas_data = metrics["deltas"]
    row_h = 0.048

    for i, dr in enumerate(deltas_data):
        idx  = int(dr["img_idx"])
        tile = dr["tile_id"]
        wgt  = WATER_GT[idx]

        d = {k: float(dr[k]) if dr.get(k) else float("nan") for k in
             ["delta_dice_ce_minus_ce", "delta_topo_minus_dice_ce",
              "delta_shuffled_minus_topo", "delta_shuffled_minus_dce"]}

        # Row background
        bg_clr = "#161b22" if i % 2 == 0 else "#0d1117"
        fig.add_artist(Rectangle(
            (0.04, y - row_h + 0.006), 0.92, row_h - 0.002,
            facecolor=bg_clr, transform=fig.transFigure,
            clip_on=False, zorder=0
        ))

        mid_y = y - row_h / 2 + 0.010
        ft(fig, col_x[0], mid_y, str(idx),   size=9, color=TXT, va="center")
        ft(fig, col_x[1], mid_y, tile[:20],  size=8, color=TXT, va="center")
        ft(fig, col_x[2], mid_y, f"{wgt:.0%}", size=9, color=DIM, va="center")

        for key, cx in [("delta_dice_ce_minus_ce",    col_x[3]),
                         ("delta_topo_minus_dice_ce",   col_x[4]),
                         ("delta_shuffled_minus_topo",  col_x[5]),
                         ("delta_shuffled_minus_dce",   col_x[6])]:
            txt, clr = _fmt_delta(d[key])
            ft(fig, cx, mid_y, txt, size=9, color=clr, weight="bold", va="center",
               family="monospace")

        y -= row_h

    y -= 0.010
    hline(fig, y, color="#334155")
    y -= 0.032

    # ── Global means ──────────────────────────────────────────────────────
    ft(fig, 0.05, y, "Moyennes globales (sur les 6 images, tous seeds) :",
       size=10.5, color=WARN, weight="bold")
    y -= 0.035

    all_d = {
        "Dice+CE − CE":        [float(dr["delta_dice_ce_minus_ce"])    for dr in deltas_data if dr.get("delta_dice_ce_minus_ce")],
        "Topo réel − Dice+CE": [float(dr["delta_topo_minus_dice_ce"])  for dr in deltas_data if dr.get("delta_topo_minus_dice_ce")],
        "Shufflé − Topo réel": [float(dr["delta_shuffled_minus_topo"]) for dr in deltas_data if dr.get("delta_shuffled_minus_topo")],
        "Shufflé − Dice+CE":   [float(dr["delta_shuffled_minus_dce"])  for dr in deltas_data if dr.get("delta_shuffled_minus_dce")],
    }
    for label, vals in all_d.items():
        if vals:
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu)**2 for v in vals) / max(len(vals)-1, 1))
            txt = f"{mu:+.4f} ± {sd:.4f}"
            clr = GRN if mu > 0.005 else (RED if mu < -0.005 else TXT)
        else:
            txt, clr = "N/A", DIM
        row_txt = f"  {label:<26}  {txt}"
        ft(fig, 0.06, y, row_txt, size=10, color=clr, weight="bold",
           family="monospace")
        y -= 0.038

    y -= 0.015
    hline(fig, y, color="#1e293b")
    y -= 0.030

    # ── Interprétation ────────────────────────────────────────────────────
    ft(fig, 0.05, y, "Interprétation :", size=10.5, color=ACC, weight="bold")
    y -= 0.032

    interp_lines = [
        ("Dice+CE améliore CE en moyenne (+0.0011, std=0.023) — dans le bruit.", TXT),
        ("Topo réel améliore Dice+CE en moyenne (+0.0081, std=0.020) — signal positif mais inconclant.", TXT),
        ("Sur ces 6 images, Topo réel est légèrement au-dessus du shufflé (delta = +0.0077).", TXT),
        ("Sur les 89 images test complètes (agrégé), shufflé dépasse légèrement Topo réel (+0.14%).", DIM),
        ("→ Les deux estimations sont dans le bruit. Pas de différence robuste réel vs shufflé.", WARN),
    ]
    for line, clr in interp_lines:
        y = ft_wrap(fig, 0.06, y, line, width=85, size=9, color=clr, line_h=0.026)
        y -= 0.008

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 11 — Conclusion
# ────────────────────────────────────────────────────────────────────────────

def page_conclusion(pdf: PdfPages):
    fig = plt.figure(figsize=A4P, facecolor=BG)

    y = 0.95
    ft(fig, 0.5, y, "Conclusion scientifique", size=15, color=ACC,
       weight="bold", ha="center")
    y -= 0.030
    hline(fig, y, color=ACC, lw=1.2)
    y -= 0.035

    # 3 findings
    findings = [
        ("1 — Régularisation faible : probable", WARN,
         "Les deux variantes topo (réel et shufflé) apportent un gain marginal "
         "sur Dice+CE en moyenne (+0.6–0.8% mIoU sur 89 images de test, 5 seeds). "
         "Ce gain est cohérent entre les deux variantes, ce qui suggère un effet "
         "de régularisation géométrique (le terme pénalise les prédictions "
         "spatialement incohérentes) plutôt qu'un signal physique spécifique au DEM."),
        ("2 — Effet physique du DEM réel : non établi", RED,
         "La différence Topo réel vs Topo shufflé est de +0.14% mIoU sur le test "
         "complet (89 images) — dans le bruit avec 5 seeds. Sur les 6 images "
         "sauvegardées, seule Ghana_141271 montre un signal clair : Topo réel "
         "réduit les violations de 1.21% (CE) à 0.82%, vs 1.04% pour le shufflé. "
         "Ce signal isolé ne suffit pas à conclure à un effet physique robuste."),
        ("3 — Variance seed dominante", DIM,
         "La variabilité entre seeds (std 0.010–0.019) est 3 à 10× plus grande "
         "que les deltas inter-conditions (0.003–0.008). Sur l'image 3, la std "
         "atteint ±0.20. À N=5 seeds, la variabilité masque le signal de loss."),
    ]

    for title, clr, body in findings:
        # Titre avec barre colorée
        fig.add_artist(Rectangle(
            (0.04, y - 0.025), 0.005, 0.028,
            facecolor=clr, transform=fig.transFigure, clip_on=False, zorder=3
        ))
        ft(fig, 0.055, y, title, size=11, color=clr, weight="bold")
        y -= 0.032
        y = ft_wrap(fig, 0.065, y, body, width=84, size=9,
                    color="#cbd5e1", line_h=0.028)
        y -= 0.025
        hline(fig, y + 0.012, color="#1e293b")
        y -= 0.012

    y -= 0.012
    # Prochaines étapes
    ft(fig, 0.05, y, "Prochaines étapes recommandées :", size=11,
       color=WARN, weight="bold")
    y -= 0.035

    steps = [
        ("A — Court terme (recommandé)", GRN,
         "Passer à TerraMind (étape 2 roadmap). Traiter le résultat SegMAN comme "
         "«régularisation plausible, effet physique DEM non établi à N=50/5 seeds»."),
        ("B — Si l'effet physique est la question centrale", WARN,
         "Lancer une ré-inférence sur les 89 images test (CPU, pas de ré-entraînement) "
         "pour avoir la comparaison complète par image avec métriques topo. "
         "Nécessite approbation explicite."),
        ("C — Si reformulation de la contrainte topo", ACC,
         "Augmenter lambda_topo > 0.5, utiliser une pondération par élévation, "
         "ou tester à N=100+ pour réduire la variance seed."),
    ]
    for step_title, clr, desc in steps:
        ft(fig, 0.06, y, f"Option {step_title} :", size=9.5, color=clr, weight="bold")
        y -= 0.028
        y = ft_wrap(fig, 0.07, y, desc, width=83, size=8.5,
                    color="#cbd5e1", line_h=0.026)
        y -= 0.014

    # Quote finale
    y -= 0.020
    hline(fig, y + 0.010, color=ACC, lw=0.8)
    y -= 0.010

    quote = (
        "SegMAN-S est validé comme backbone de segmentation 15-canaux robuste à "
        "faible volume de données (mIoU 0.83–0.85 sur le test Sen1Floods11). "
        "La loss topographique n'apporte pas encore de preuve claire d'un effet "
        "physique spécifique au DEM. Les versions réel et shufflé se comportent "
        "de manière similaire, suggérant que le terme agit principalement comme "
        "un régulariseur géométrique faible."
    )
    box_top = y
    obs_lines = textwrap.wrap(quote, 84)
    box_h = len(obs_lines) * 0.028 + 0.030
    fig.add_artist(Rectangle(
        (0.04, box_top - box_h), 0.92, box_h,
        facecolor="#0f172a", edgecolor=ACC, lw=0.8,
        transform=fig.transFigure, clip_on=False, zorder=0
    ))
    y -= 0.018
    for line in obs_lines:
        ft(fig, 0.06, y, line, size=9, color=TXT, style="italic")
        y -= 0.028

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading metrics ...")
    metrics = load_metrics()

    print(f"Generating: {OUT_PDF}")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(OUT_PDF)) as pdf:
        print("  p1  Titre + introduction")
        page_title(pdf)

        print("  p2  Graphique résumé IoU")
        page_summary_chart(pdf)

        for idx in range(6):
            print(f"  p{3+idx}  Image {idx}: {TILE_IDS[idx]}")
            page_image(pdf, idx, metrics)

        print("  p9  Grille violations topo")
        page_topo_grid(pdf)

        print("  p10 Tableau deltas")
        page_delta_summary(pdf, metrics)

        print("  p11 Conclusion")
        page_conclusion(pdf)

    size_mb = OUT_PDF.stat().st_size / 1_048_576
    print(f"\nDone. {OUT_PDF}  ({size_mb:.1f} MB, 11 pages)")


if __name__ == "__main__":
    main()
