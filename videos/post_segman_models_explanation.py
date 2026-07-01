"""
After SegMAN: VFMNet, EoMT, ViT-P and TerraMind -- pedagogical Manim video.

A dark-background explainer that situates SegMAN among four other model
families we may test under the *same* physics-informed loss protocol:

    1.  Why look beyond SegMAN (architecture is not the contribution)
    2.  The shared protocol  (fixed loss comparison; DEM never a model input)
    3.  VFMNet / VFMSeg   -- AAAI 2025, Visual Foundation Models
    4.  EoMT              -- CVPR 2025 Highlight, Encoder-only Mask Transformer
    5.  ViT-P             -- CVPR 2026, SOTA-level universal segmentation
    6.  Our data are not RGB  (pivot: Sentinel-1 SAR + Sentinel-2 multispectral)
    7.  TerraMind         -- Earth Observation foundation model  (NEW)
    8.  TerraMind risks / limitations
    9.  Comparison table  (SegMAN + four model families)
   10.  Physics-informed transfer  (DEM invariant preserved)
   11.  Roadmap  (SegMAN, TerraMind, VFMNet, EoMT, ViT-P)
   12.  Conclusion

Render (CPU-only, no GPU needed):
    # full video, 1080p60:
    manim -pqh videos/post_segman_models_explanation.py PostSegMANModelsVideo

    # quick preview, 480p15:
    manim -pql videos/post_segman_models_explanation.py PostSegMANModelsVideo

    # render the TerraMind chapter on its own:
    manim -pqh videos/post_segman_models_explanation.py S07_TerraMind

Requires Manim Community Edition:
    pip install manim
    # plus a LaTeX distribution (MiKTeX on Windows) for the equations.

Voiceover script lives in videos/post_segman_models_voiceover.md.
Storyboard lives in videos/post_segman_models_storyboard.md.

NOTE: This file only *describes* an animation. It does not load any model,
read any checkpoint, touch any experiment directory, or use the GPU.

SCIENTIFIC INVARIANT (must stay true in every scene):
    The DEM is never given as a model input in our current controlled protocol.
    It remains a supervision signal through the topographic loss and a physical
    consistency metric only.
"""

from __future__ import annotations

import random

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Arrow,
    Create,
    Cross,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    Indicate,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    Text,
    VGroup,
    Write,
    config,
)

# ---------------------------------------------------------------------------
# Palette  (dark, high-contrast, colour-blind friendly) -- shared with the
# SegMAN video so the two explainers feel like one series.
# ---------------------------------------------------------------------------
BG_COLOR   = "#0b0f1a"   # near-black navy background
TXT        = "#e5e7eb"   # primary light text
MUTED      = "#94a3b8"   # secondary grey text
WATER      = "#22d3ee"   # cyan   -> water / SegMAN
WATER_DEEP = "#0ea5e9"   # blue   -> deep water / S2
LAND       = "#243042"   # slate  -> background / land
SAR        = "#a78bfa"   # violet -> Sentinel-1 / SAR
DEM_CLR    = "#d4a373"   # tan    -> elevation / DEM
PENALTY    = "#ef4444"   # red    -> penalty / risk / "no input"
WARN       = "#f97316"   # orange -> caution
VALID      = "#22c55e"   # green  -> physically coherent
ACCENT     = "#fbbf24"   # amber  -> equation / key callout

# Per-model accent colours
SEGMAN_CLR = WATER          # cyan    -> SegMAN (current main model)
VFM_CLR    = "#38bdf8"      # sky     -> VFMNet / VFMSeg
EOMT_CLR   = "#a78bfa"      # violet  -> EoMT
VITP_CLR   = "#fbbf24"      # amber   -> ViT-P
TERRA_CLR  = "#34d399"      # emerald -> TerraMind (Earth Observation)

config.background_color = BG_COLOR
Text.set_default(font="Arial")   # DejaVu Sans not installed on this Windows host

random.seed(11)


# ===========================================================================
# Shared helpers (mixin used by the master scene and the per-section scenes)
# ===========================================================================
class PostSegMANBase(Scene):
    """Common construction helpers and the 12 section methods."""

    # -- small utilities ----------------------------------------------------
    def fade_all(self, run_time: float = 0.6) -> None:
        mobs = [m for m in self.mobjects]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def section_banner(self, number: str, title: str):
        tag = VGroup(
            Text(number, color=ACCENT, weight="BOLD").scale(0.45),
            Text(title, color=MUTED).scale(0.40),
        ).arrange(RIGHT, buff=0.25)
        tag.to_corner(UP + LEFT, buff=0.4)
        return tag

    def pixel_grid(self, data, cell=0.30, stroke_w=0.6):
        grid = VGroup()
        for r, row in enumerate(data):
            for c, col in enumerate(row):
                sq = Rectangle(width=cell, height=cell)
                sq.set_fill(col, opacity=1.0)
                sq.set_stroke(BG_COLOR, width=stroke_w)
                sq.move_to([c * cell, -r * cell, 0])
                grid.add(sq)
        grid.move_to(ORIGIN)
        return grid

    @staticmethod
    def _water_mask(rows, cols):
        data = [[LAND for _ in range(cols)] for _ in range(rows)]
        path_c = 1.5
        for r in range(rows):
            path_c += random.choice([-0.6, 0.3, 0.9, 1.2])
            width = 1 + (r // 3)
            for w in range(-width, width + 1):
                c = int(round(path_c)) + w
                if 0 <= c < cols:
                    data[r][c] = WATER
        return data

    def block(self, label, colour, w=1.9, h=0.95, scale=0.34):
        box = RoundedRectangle(corner_radius=0.12, width=w, height=h)
        box.set_fill(colour, opacity=0.22).set_stroke(colour, width=2.5)
        txt = Text(label, color=TXT).scale(scale)
        txt.move_to(box.get_center())
        return VGroup(box, txt)

    def _bullet(self, colour, text, scale=0.44):
        dot = Dot(color=colour, radius=0.08)
        t = Text(text, color=TXT).scale(scale)
        return VGroup(dot, t).arrange(RIGHT, buff=0.28)

    def _chip(self, colour, label, scale=0.4):
        sq = Rectangle(width=0.30, height=0.30).set_fill(colour, opacity=1.0)
        sq.set_stroke(colour, width=1.5)
        txt = Text(label, color=TXT).scale(scale)
        return VGroup(sq, txt).arrange(RIGHT, buff=0.2)

    def _model_header(self, name, venue, idea, colour):
        """Big coloured model name + venue tag + one-line idea."""
        title = Text(name, color=colour, weight="BOLD").scale(0.95)
        tag = Text(venue, color=MUTED).scale(0.45)
        tag.next_to(title, RIGHT, buff=0.4, aligned_edge=DOWN)
        head = VGroup(title, tag)
        sub = Text(idea, color=TXT).scale(0.5)
        group = VGroup(head, sub).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        return group

    # =======================================================================
    # SECTION 0 -- Title
    # =======================================================================
    def section_title(self):
        # VOICEOVER: "SegMAN is our current main model. But the architecture is
        # not the contribution. So the real question is whether our physics-
        # informed loss survives across very different model families. After
        # SegMAN, four of them: VFMNet, EoMT, ViT-P, and TerraMind."
        top = Text("After SegMAN", color=WATER, weight="BOLD").scale(1.25)
        models = Text("VFMNet  -  EoMT  -  ViT-P  -  TerraMind",
                      color=TXT).scale(0.6)
        sub = Text("for Physics-Informed Flood Segmentation",
                   color=MUTED).scale(0.5)
        group = VGroup(top, models, sub).arrange(DOWN, buff=0.4)

        underline = Line(LEFT, RIGHT, color=ACCENT).set_width(models.width + 0.6)
        underline.next_to(models, DOWN, buff=0.18)

        self.play(Write(top), run_time=1.3)
        self.play(FadeIn(models, shift=UP * 0.3), Create(underline), run_time=1.0)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.wait(2.0)
        self.play(FadeOut(underline))
        self.fade_all()

    # =======================================================================
    # SECTION 1 -- Why look beyond SegMAN
    # =======================================================================
    def section_motivation(self):
        # VOICEOVER: "Architecture is not the contribution. We hold the loss
        # protocol fixed and swap the model. If the topographic loss only helps
        # one architecture, it is an artefact. If it helps across families, it
        # is a property worth reporting."
        banner = self.section_banner("01", "Why look beyond SegMAN")
        self.play(FadeIn(banner))

        thesis = Text("The architecture is NOT the contribution.",
                      color=ACCENT, weight="BOLD").scale(0.62)
        thesis.to_edge(UP, buff=1.2)
        self.play(Write(thesis))

        q = Text("Does a physics-informed loss help across model families?",
                 color=TXT).scale(0.52)
        q.next_to(thesis, DOWN, buff=0.5)
        self.play(FadeIn(q, shift=UP * 0.2))

        bullets = VGroup(
            self._bullet(WATER, "Hold the loss protocol fixed."),
            self._bullet(SAR,   "Swap the model: SegMAN, VFMNet, EoMT, ViT-P, TerraMind."),
            self._bullet(WARN,  "Helps only one model  ->  artefact."),
            self._bullet(VALID, "Helps across families  ->  a property worth reporting."),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        bullets.next_to(q, DOWN, buff=0.6)
        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.25), run_time=0.55)
        self.wait(2.0)
        self.fade_all()

    # =======================================================================
    # SECTION 2 -- The shared protocol (DEM invariant)
    # =======================================================================
    def section_protocol(self):
        # VOICEOVER: "The protocol is the same for every model. Sentinel-1 and
        # Sentinel-2 go in. The model emits two logits per pixel and a softmax
        # gives the water mask. The DEM is never a model input -- it lives only
        # in the topographic loss and the physical consistency metric."
        banner = self.section_banner("02", "The shared protocol")
        self.play(FadeIn(banner))

        x_in = self.block("S1 + S2\n(15ch)", WATER_DEEP)
        model = self.block("any model\n(SegMAN / ... )", MUTED, w=2.3)
        logits = self.block("logits\n[B,2,H,W]", VALID, w=1.8)
        chain = VGroup(x_in, model, logits).arrange(RIGHT, buff=0.7)
        chain.shift(UP * 1.1)
        arrows = VGroup()
        for a, b in zip(chain[:-1], chain[1:]):
            arrows.add(Arrow(a.get_right(), b.get_left(), color=MUTED,
                             buff=0.1, stroke_width=3,
                             max_tip_length_to_length_ratio=0.25))
        self.play(FadeIn(x_in))
        for box, arr in zip(chain[1:], arrows):
            self.play(GrowArrow(arr), FadeIn(box, shift=RIGHT * 0.2), run_time=0.5)

        # DEM kept aside, with a crossed-out arrow into the model
        dem = self.block("DEM", DEM_CLR, w=1.4, h=0.8)
        dem.next_to(model, DOWN, buff=1.6)
        bad_arrow = Arrow(dem.get_top(), model.get_bottom(), color=PENALTY,
                          buff=0.12, stroke_width=4)
        cross = Cross(bad_arrow, stroke_color=PENALTY, stroke_width=6).scale(0.5)
        self.play(FadeIn(dem), GrowArrow(bad_arrow))
        self.play(Create(cross))

        note = Text("The DEM is NOT a model input.",
                    color=PENALTY, weight="BOLD").scale(0.5)
        sub = Text("It is used only in the topographic loss & physical consistency metrics.",
                   color=MUTED).scale(0.4)
        cap = VGroup(note, sub).arrange(DOWN, buff=0.18)
        cap.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(cap, shift=UP * 0.2))
        self.wait(2.2)
        self.fade_all()

    # =======================================================================
    # SECTION 3 -- VFMNet / VFMSeg
    # =======================================================================
    def section_vfmnet(self):
        # VOICEOVER: "First family: VFMNet, or VFMSeg, from AAAI 2025. It builds
        # segmentation on top of large visual foundation models, aiming for
        # generalization and out-of-distribution robustness."
        banner = self.section_banner("03", "VFMNet / VFMSeg")
        self.play(FadeIn(banner))

        head = self._model_header(
            "VFMNet / VFMSeg", "AAAI 2025",
            "Visual Foundation Models for generalizable segmentation", VFM_CLR)
        head.to_edge(UP, buff=1.1).to_edge(LEFT, buff=0.8)
        self.play(FadeIn(head[0]), run_time=0.8)
        self.play(FadeIn(head[1], shift=UP * 0.2))

        bullets = VGroup(
            self._bullet(VFM_CLR, "Reuses large pretrained visual foundation models."),
            self._bullet(VALID,   "Strength: out-of-distribution generalization."),
            self._bullet(WARN,    "Mostly RGB-centric pretraining."),
            self._bullet(MUTED,   "Our line: foundation-model generalization (OOD)."),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        bullets.next_to(head, DOWN, buff=0.7).to_edge(LEFT, buff=0.9)
        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.25), run_time=0.55)
        self.wait(2.0)
        self.fade_all()

    # =======================================================================
    # SECTION 3b -- VFMNet: how it works
    # =======================================================================
    def section_vfmnet_how(self):
        # VOICEOVER: "How does VFMNet work? It does not learn features from
        # scratch. It takes one or more large, pretrained visual foundation
        # models -- think DINOv2 or SAM -- and keeps the backbone frozen. A light
        # adapter fuses those rich features, and only a small segmentation head is
        # trained. The bet: foundation-model features already generalize, so they
        # transfer to new, out-of-distribution scenes."
        banner = self.section_banner("03", "VFMNet -- how it works")
        self.play(FadeIn(banner))

        b1 = self.block("image", WATER_DEEP, w=1.4)
        b2 = self.block("frozen VFM\nbackbone(s)", VFM_CLR, w=2.1)
        b3 = self.block("rich\nfeatures", SAR, w=1.5)
        b4 = self.block("light\nadapter", MUTED, w=1.5)
        b5 = self.block("seg head", ACCENT, w=1.5)
        chain = VGroup(b1, b2, b3, b4, b5).arrange(RIGHT, buff=0.4)
        chain.scale(0.92).shift(UP * 1.1)
        arrows = VGroup()
        for a, b in zip(chain[:-1], chain[1:]):
            arrows.add(Arrow(a.get_right(), b.get_left(), color=MUTED, buff=0.08,
                             stroke_width=2.8, max_tip_length_to_length_ratio=0.22))
        self.play(FadeIn(b1))
        for box, arr in zip(chain[1:], arrows):
            self.play(GrowArrow(arr), FadeIn(box, shift=RIGHT * 0.2), run_time=0.4)

        lock = Text("frozen -- not trained", color=VFM_CLR).scale(0.36)
        lock.next_to(b2, UP, buff=0.22)
        train = Text("trained", color=ACCENT).scale(0.36)
        train.next_to(b5, UP, buff=0.22)
        self.play(FadeIn(lock), FadeIn(train))

        bullets = VGroup(
            self._bullet(VFM_CLR, "Reuse a large pretrained Visual Foundation Model (DINOv2 / SAM-like)."),
            self._bullet(SAR,     "Optionally fuse features from several VFMs."),
            self._bullet(ACCENT,  "Train only a light adapter + segmentation head."),
            self._bullet(VALID,   "Pretrained features transfer better out-of-distribution."),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.34)
        bullets.next_to(chain, DOWN, buff=0.9).to_edge(LEFT, buff=0.9)
        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.5)
        self.wait(2.2)
        self.fade_all()

    # =======================================================================
    # SECTION 4 -- EoMT
    # =======================================================================
    def section_eomt(self):
        # VOICEOVER: "Second: EoMT, a CVPR 2025 Highlight. Encoder-only Mask
        # Transformer. A plain ViT encoder does the segmentation directly, with
        # mask classification instead of a separate heavy decoder."
        banner = self.section_banner("04", "EoMT")
        self.play(FadeIn(banner))

        head = self._model_header(
            "EoMT", "CVPR 2025 Highlight",
            "Encoder-only Mask Transformer", EOMT_CLR)
        head.to_edge(UP, buff=1.1).to_edge(LEFT, buff=0.8)
        self.play(FadeIn(head[0]), run_time=0.8)
        self.play(FadeIn(head[1], shift=UP * 0.2))

        bullets = VGroup(
            self._bullet(EOMT_CLR, "A plain ViT encoder segments directly."),
            self._bullet(VALID,    "Strength: simple, fast, strong mask-classification."),
            self._bullet(WARN,     "ViT-scale compute; mask-cls head, not dense logits."),
            self._bullet(MUTED,    "Our line: second CVPR architecture (ViT encoder-only)."),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        bullets.next_to(head, DOWN, buff=0.7).to_edge(LEFT, buff=0.9)
        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.25), run_time=0.55)
        self.wait(2.0)
        self.fade_all()

    # =======================================================================
    # SECTION 4b -- EoMT: how it works
    # =======================================================================
    def section_eomt_how(self):
        # VOICEOVER: "How does EoMT work? Take the patch tokens of a plain ViT and
        # append a few learnable query tokens. The encoder's own self-attention
        # lets those queries read the patches -- there is no separate mask
        # decoder. At the output, each query predicts a class and a mask
        # embedding; the mask itself is the dot product of that embedding with the
        # patch features. Encoder-only, yet it does full mask classification."
        banner = self.section_banner("04", "EoMT -- how it works")
        self.play(FadeIn(banner))

        # token row: patches (slate) + query tokens (violet)
        patches = VGroup()
        for i in range(8):
            sq = Rectangle(width=0.34, height=0.34).set_fill(LAND, opacity=1.0)
            sq.set_stroke(MUTED, width=1.0)
            patches.add(sq)
        patches.arrange(RIGHT, buff=0.08)
        queries = VGroup()
        for i in range(2):
            sq = Rectangle(width=0.34, height=0.34).set_fill(EOMT_CLR, opacity=0.9)
            sq.set_stroke(EOMT_CLR, width=1.5)
            queries.add(sq)
        queries.arrange(RIGHT, buff=0.08)
        tokens = VGroup(patches, queries).arrange(RIGHT, buff=0.4)
        tokens.to_edge(UP, buff=1.3)
        p_lbl = Text("patch tokens", color=MUTED).scale(0.34).next_to(patches, DOWN, buff=0.15)
        q_lbl = Text("query tokens", color=EOMT_CLR).scale(0.34).next_to(queries, DOWN, buff=0.15)
        self.play(FadeIn(patches), FadeIn(p_lbl))
        self.play(FadeIn(queries), FadeIn(q_lbl))

        vit = self.block("shared ViT self-attention  (no separate decoder)",
                         EOMT_CLR, w=6.4, h=0.9, scale=0.36)
        vit.next_to(tokens, DOWN, buff=0.7)
        arr_in = Arrow(tokens.get_bottom(), vit.get_top(), color=MUTED, buff=0.12, stroke_width=3)
        self.play(GrowArrow(arr_in), FadeIn(vit))

        out_cls = self.block("class", VALID, w=1.6, h=0.7, scale=0.34)
        out_mask = self.block("mask = query . patch", ACCENT, w=3.0, h=0.7, scale=0.32)
        outs = VGroup(out_cls, out_mask).arrange(RIGHT, buff=0.8)
        outs.next_to(vit, DOWN, buff=0.7)
        arr_o1 = Arrow(vit.get_bottom(), out_cls.get_top(), color=MUTED, buff=0.12, stroke_width=2.6)
        arr_o2 = Arrow(vit.get_bottom(), out_mask.get_top(), color=MUTED, buff=0.12, stroke_width=2.6)
        self.play(GrowArrow(arr_o1), GrowArrow(arr_o2), FadeIn(outs))

        cap = Text("Encoder-only -- queries read patches via the ViT's own attention.",
                   color=TXT).scale(0.44)
        cap.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(cap, shift=UP * 0.2))
        self.wait(2.2)
        self.fade_all()

    # =======================================================================
    # SECTION 5 -- ViT-P
    # =======================================================================
    def section_vitp(self):
        # VOICEOVER: "Third: ViT-P, a CVPR 2026, SOTA-level universal image
        # segmentation reference. Top accuracy -- but the newest and the
        # hardest to integrate."
        banner = self.section_banner("05", "ViT-P")
        self.play(FadeIn(banner))

        head = self._model_header(
            "ViT-P", "CVPR 2026",
            "SOTA-level universal image segmentation", VITP_CLR)
        head.to_edge(UP, buff=1.1).to_edge(LEFT, buff=0.8)
        self.play(FadeIn(head[0]), run_time=0.8)
        self.play(FadeIn(head[1], shift=UP * 0.2))

        bullets = VGroup(
            self._bullet(VITP_CLR, "Universal segmentation, state-of-the-art accuracy."),
            self._bullet(VALID,    "Strength: top reported performance."),
            self._bullet(PENALTY,  "Highest integration risk; newest, least battle-tested."),
            self._bullet(MUTED,    "Our line: future SOTA reference."),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        bullets.next_to(head, DOWN, buff=0.7).to_edge(LEFT, buff=0.9)
        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.25), run_time=0.55)
        self.wait(2.0)
        self.fade_all()

    # =======================================================================
    # SECTION 5b -- ViT-P: how it works
    # =======================================================================
    def section_vitp_how(self):
        # VOICEOVER: "How does ViT-P work? A plain ViT backbone produces dense
        # features, and the model predicts a SET of pairs -- a class and a mask
        # for each query. That single mask-classification view covers semantic,
        # instance and panoptic segmentation with one model: that is what
        # 'universal' means. It is the newest of the four, so treat the exact
        # internals as a moving state-of-the-art reference rather than a fixed
        # recipe."
        banner = self.section_banner("05", "ViT-P -- how it works")
        self.play(FadeIn(banner))

        b1 = self.block("image", WATER_DEEP, w=1.4)
        b2 = self.block("ViT\nbackbone", VITP_CLR, w=1.8)
        b3 = self.block("N queries\n(class, mask)", ACCENT, w=2.1)
        chain = VGroup(b1, b2, b3).arrange(RIGHT, buff=0.55)
        chain.shift(UP * 1.2)
        arrows = VGroup()
        for a, b in zip(chain[:-1], chain[1:]):
            arrows.add(Arrow(a.get_right(), b.get_left(), color=MUTED, buff=0.1,
                             stroke_width=3, max_tip_length_to_length_ratio=0.25))
        self.play(FadeIn(b1))
        for box, arr in zip(chain[1:], arrows):
            self.play(GrowArrow(arr), FadeIn(box, shift=RIGHT * 0.2), run_time=0.45)

        tasks = VGroup(
            self._chip(WATER, "semantic"),
            self._chip(SAR,   "instance"),
            self._chip(VALID, "panoptic"),
        ).arrange(RIGHT, buff=0.6)
        tasks.next_to(chain, DOWN, buff=0.7)
        one = Text("one model -- universal segmentation", color=VITP_CLR).scale(0.42)
        one.next_to(tasks, DOWN, buff=0.3)
        link = Arrow(b3.get_bottom(), tasks.get_top(), color=MUTED, buff=0.15, stroke_width=2.6)
        self.play(GrowArrow(link), FadeIn(tasks), FadeIn(one))

        bullets = VGroup(
            self._bullet(VITP_CLR, "Plain ViT backbone -> dense features."),
            self._bullet(ACCENT,   "Predict a SET of (class, mask) pairs = mask classification."),
            self._bullet(VALID,    "One model covers semantic / instance / panoptic."),
            self._bullet(PENALTY,  "Newest -- exact internals are a moving SOTA target."),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        bullets.next_to(one, DOWN, buff=0.5).to_edge(LEFT, buff=0.9)
        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.45)
        self.wait(2.2)
        self.fade_all()

    # =======================================================================
    # SECTION 6 -- Our data are not RGB  (pivot to EO)
    # =======================================================================
    def section_not_rgb(self):
        # VOICEOVER: "Notice something. VFMNet, EoMT and ViT-P are general-vision
        # models, trained mostly on natural RGB images. But our data are not RGB.
        # We use Sentinel-1 SAR and Sentinel-2 multispectral. That distribution
        # gap is exactly why an Earth Observation foundation model matters."
        banner = self.section_banner("06", "Our data are not RGB")
        self.play(FadeIn(banner))

        left = VGroup(
            Text("General-vision models", color=MUTED, weight="BOLD").scale(0.5),
            self._chip(VFM_CLR, "VFMNet / VFMSeg"),
            self._chip(EOMT_CLR, "EoMT"),
            self._chip(VITP_CLR, "ViT-P"),
            Text("mostly trained on natural RGB", color=WARN).scale(0.42),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        left.to_edge(LEFT, buff=1.0).shift(UP * 0.2)

        right = VGroup(
            Text("Our inputs", color=MUTED, weight="BOLD").scale(0.5),
            self._chip(SAR, "Sentinel-1 SAR (2 ch)"),
            self._chip(WATER_DEEP, "Sentinel-2 multispectral (13 ch)"),
            Text("a different input distribution", color=VALID).scale(0.42),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        right.to_edge(RIGHT, buff=1.0).shift(UP * 0.2)

        self.play(FadeIn(left, shift=RIGHT * 0.2))
        self.play(FadeIn(right, shift=LEFT * 0.2))

        gap = Text("domain gap", color=PENALTY, weight="BOLD").scale(0.5)
        gap.move_to(ORIGIN).shift(UP * 0.2)
        arr = Arrow(left.get_right(), right.get_left(), color=PENALTY, buff=0.4)
        self.play(GrowArrow(arr), FadeIn(gap, shift=UP * 0.2))

        cap = Text("=> an Earth Observation foundation model is naturally aligned.",
                   color=ACCENT).scale(0.5)
        cap.to_edge(DOWN, buff=0.6)
        self.play(Write(cap))
        self.wait(2.2)
        self.fade_all()

    # =======================================================================
    # SECTION 7 -- TerraMind  (NEW)
    # =======================================================================
    def section_terramind(self):
        # VOICEOVER: "TerraMind is an Earth Observation foundation model -- not a
        # generic RGB segmentation model. It is built around satellite and
        # geospatial modalities, so it is naturally closer to our domain.
        # Sentinel-1 and Sentinel-2 go into TerraMind; it produces EO features;
        # a segmentation head turns those into two logits per pixel and a water
        # mask. TerraMind is the most domain-aligned model for Earth Observation,
        # but SegMAN is currently the main architecture for clean loss-comparison
        # experiments, because it is already integrated and produces dense logits
        # directly. TerraMind is not abandoned: it remains our Earth Observation,
        # Sen1Floods11 baseline, and it preserves continuity with our previous
        # TerraMind experiments."
        banner = self.section_banner("07", "TerraMind  (Earth Observation)")
        self.play(FadeIn(banner))

        title = Text("TerraMind", color=TERRA_CLR, weight="BOLD").scale(0.95)
        tag = Text("Earth Observation foundation model", color=MUTED).scale(0.46)
        tag.next_to(title, RIGHT, buff=0.4, aligned_edge=DOWN)
        head = VGroup(title, tag).to_edge(UP, buff=0.9).to_edge(LEFT, buff=0.8)
        self.play(FadeIn(title), FadeIn(tag, shift=UP * 0.2))

        idea = Text("Built for satellite / geospatial data -- not generic RGB.",
                    color=TXT).scale(0.5)
        idea.next_to(head, DOWN, buff=0.4).to_edge(LEFT, buff=0.9)
        self.play(FadeIn(idea, shift=UP * 0.2))

        # EO pipeline
        b1 = self.block("S1 / S2\nmodalities", SAR, w=1.9)
        b2 = self.block("TerraMind\nEO foundation", TERRA_CLR, w=2.1)
        b3 = self.block("EO\nfeatures", WATER_DEEP, w=1.6)
        b4 = self.block("seg head /\ndecoder", ACCENT, w=1.7)
        b5 = self.block("logits\n[B,2,H,W]", VALID, w=1.7)
        chain = VGroup(b1, b2, b3, b4, b5).arrange(RIGHT, buff=0.42)
        chain.scale(0.9).next_to(idea, DOWN, buff=0.6)
        arrows = VGroup()
        for a, b in zip(chain[:-1], chain[1:]):
            arrows.add(Arrow(a.get_right(), b.get_left(), color=MUTED,
                             buff=0.08, stroke_width=2.8,
                             max_tip_length_to_length_ratio=0.22))
        self.play(FadeIn(b1))
        for box, arr in zip(chain[1:], arrows):
            self.play(GrowArrow(arr), FadeIn(box, shift=RIGHT * 0.2), run_time=0.4)

        mask = self.pixel_grid(self._water_mask(5, 5), cell=0.2).scale(0.9)
        mask.next_to(b5, DOWN, buff=0.9)
        link = Arrow(b5.get_bottom(), mask.get_top(), color=MUTED, buff=0.15)
        mlbl = Text("water mask", color=WATER).scale(0.38)
        mlbl.next_to(mask, RIGHT, buff=0.2)
        self.play(GrowArrow(link), FadeIn(mask), FadeIn(mlbl))

        key = Text(
            "Most domain-aligned for EO -- but SegMAN stays the main\n"
            "loss-comparison model (integrated, dense logits directly).",
            color=ACCENT).scale(0.44)
        key.to_edge(DOWN, buff=0.5)
        self.play(Write(key))
        self.wait(2.4)
        self.fade_all()

    # =======================================================================
    # SECTION 7a -- TerraMind: how it works
    # =======================================================================
    def section_terramind_how(self):
        # VOICEOVER: "How does TerraMind work? Unlike an RGB model, it is
        # pretrained across many Earth-Observation modalities -- radar, optical,
        # and more -- each tokenized and fused inside a shared transformer with
        # cross-modal attention. That pretraining teaches it EO-specific
        # statistics: SAR speckle, multispectral reflectance. We then fine-tune a
        # segmentation head for water. Important: TerraMind CAN ingest a DEM as a
        # modality, but in our controlled protocol we feed only Sentinel-1 and
        # Sentinel-2 -- the DEM stays out of the model and lives only in the loss."
        banner = self.section_banner("07", "TerraMind -- how it works")
        self.play(FadeIn(banner))

        # several EO modalities tokenized into a shared transformer
        mods = VGroup(
            self._chip(SAR,        "S1 SAR"),
            self._chip(WATER_DEEP, "S2 optical"),
            self._chip(MUTED,      "other EO modalities"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        mods.to_edge(LEFT, buff=0.9).shift(UP * 0.6)

        trans = self.block("shared transformer\n(cross-modal attention)", TERRA_CLR,
                           w=3.0, h=1.3, scale=0.34)
        trans.move_to(ORIGIN).shift(UP * 0.6)
        head = self.block("fine-tuned\nseg head", ACCENT, w=1.7, h=1.0, scale=0.34)
        head.next_to(trans, RIGHT, buff=1.1)

        a_in = Arrow(mods.get_right(), trans.get_left(), color=MUTED, buff=0.15, stroke_width=3)
        a_out = Arrow(trans.get_right(), head.get_left(), color=MUTED, buff=0.12, stroke_width=3)
        self.play(FadeIn(mods))
        self.play(GrowArrow(a_in), FadeIn(trans))
        self.play(GrowArrow(a_out), FadeIn(head))

        pre = Text("pretrained on EO modalities  ->  learns SAR speckle & multispectral reflectance",
                   color=TERRA_CLR).scale(0.4)
        pre.next_to(trans, UP, buff=0.4)
        self.play(FadeIn(pre, shift=UP * 0.2))

        guard = Text(
            "In our protocol we feed only S1 / S2.  DEM stays out of the model -- loss only.",
            color=ACCENT, weight="BOLD").scale(0.46)
        guard.to_edge(DOWN, buff=0.6)
        self.play(Write(guard))
        self.wait(2.4)
        self.fade_all()

    # =======================================================================
    # SECTION 7b -- TerraMind role + previous context
    # =======================================================================
    def section_terramind_role(self):
        # VOICEOVER: "Its role in the project: a specialized satellite reference
        # that lets us compare a recent general architecture, SegMAN, against a
        # model designed for EO data. We already ran TerraMind and TerraTorch
        # with Dice, Dice plus CE, topo on the real DEM, and topo on the shuffled
        # DEM. It showed strong full-data performance and interesting low-data
        # behaviour, including collapse and rescue patterns. SegMAN does not
        # replace TerraMind; it complements it. The DEM modality caution still
        # holds: be explicit about which modalities the model is given."
        banner = self.section_banner("07", "TerraMind -- role & continuity")
        self.play(FadeIn(banner))

        col1_title = Text("Role in the project", color=TERRA_CLR, weight="BOLD").scale(0.52)
        col1 = VGroup(
            col1_title,
            self._bullet(VALID, "Not abandoned.", scale=0.42),
            self._bullet(VALID, "Remains our EO / Sen1Floods11 baseline.", scale=0.42),
            self._bullet(VALID, "A specialized satellite reference.", scale=0.42),
            self._bullet(VALID, "Compare SegMAN (general) vs an EO-designed model.", scale=0.42),
            self._bullet(VALID, "Continuity with previous TerraMind experiments.", scale=0.42),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        col1.to_edge(LEFT, buff=0.8).shift(UP * 0.3)
        self.play(FadeIn(col1_title))
        for b in col1[1:]:
            self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.45)

        prev = VGroup(
            Text("Previously, with TerraMind / TerraTorch:", color=MUTED, weight="BOLD").scale(0.46),
            self._chip(MUTED, "Dice"),
            self._chip(WATER, "Dice + CE"),
            self._chip(VALID, "Topo (real DEM)"),
            self._chip(WARN,  "Topo (shuffled DEM)"),
            Text("strong full-data; low-data collapse/rescue.", color=TXT).scale(0.4),
            Text("SegMAN complements TerraMind -- it does not replace it.",
                 color=ACCENT).scale(0.42),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.26)
        prev.to_edge(RIGHT, buff=0.8).shift(UP * 0.3)
        self.play(FadeIn(prev, shift=LEFT * 0.2))
        self.wait(2.4)

        caution = Text(
            "Modality caution: be explicit about which modalities TerraMind is given.",
            color=WARN).scale(0.46)
        caution.to_edge(DOWN, buff=0.5)
        self.play(Write(caution))
        self.wait(2.0)
        self.fade_all()

    # =======================================================================
    # SECTION 8 -- TerraMind risks / limitations
    # =======================================================================
    def section_terramind_risks(self):
        # VOICEOVER: "But TerraMind carries real risks. The TerraTorch pipeline
        # is heavier. Modality handling must be controlled carefully. And
        # critically: if the DEM is ever given as an input, the experiment no
        # longer tests the same hypothesis as our loss-only DEM setup. Plus,
        # integration and reproducibility are harder than with SegMAN."
        banner = self.section_banner("08", "TerraMind -- risks & limitations")
        self.play(FadeIn(banner))

        title = Text("Risks to control", color=WARN, weight="BOLD").scale(0.6)
        title.to_edge(UP, buff=1.1)
        self.play(Write(title))

        risks = VGroup(
            self._bullet(WARN,    "Heavier TerraTorch pipeline."),
            self._bullet(WARN,    "Modality handling must be controlled carefully."),
            self._bullet(PENALTY, "DEM as input  ->  no longer the same hypothesis"),
            self._bullet(PENALTY, "   as our loss-only DEM setup."),
            self._bullet(WARN,    "Integration & reproducibility harder than SegMAN."),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        risks.next_to(title, DOWN, buff=0.7)
        for b in risks:
            self.play(FadeIn(b, shift=RIGHT * 0.25), run_time=0.55)
        self.wait(1.0)

        guard = Text("Invariant: DEM stays a loss/metric signal -- never a model input.",
                     color=ACCENT, weight="BOLD").scale(0.48)
        guard.to_edge(DOWN, buff=0.6)
        self.play(Write(guard))
        self.wait(2.0)
        self.fade_all()

    # =======================================================================
    # SECTION 9 -- Comparison table
    # =======================================================================
    def section_comparison(self):
        # VOICEOVER: "Side by side. SegMAN: integrated, dense logits now.
        # VFMNet: foundation-model generalization. EoMT: a ViT encoder-only
        # design. ViT-P: future SOTA. And TerraMind: the Earth Observation
        # foundation model, most aligned with Sentinel data and Sen1Floods11 --
        # at the cost of a heavier pipeline and careful modality control."
        banner = self.section_banner("09", "Comparison")
        self.play(FadeIn(banner))

        # Column x-centres
        cx = [-5.6, -2.6, 0.9, 4.7]
        headers = ["Model", "Strength", "Risk", "Role in project"]
        top_y = 2.7

        def cell(text, colour, x, y, scale=0.30, w=None):
            t = Text(text, color=colour).scale(scale)
            if w is not None and t.width > w:
                t.scale(w / t.width)
            t.move_to([x, y, 0])
            return t

        head_row = VGroup(*[
            cell(h, ACCENT, cx[i], top_y, scale=0.34) for i, h in enumerate(headers)
        ])
        rule = Line([-6.7, top_y - 0.3, 0], [6.7, top_y - 0.3, 0], color=MUTED)
        self.play(FadeIn(head_row), Create(rule))

        # rows: (name, name_colour, idea, strength, risk, role)
        rows = [
            ("SegMAN", SEGMAN_CLR, "dense seg (attn+SSM)",
             "integrated, dense logits now", "general-vision, not EO", "current main model"),
            ("VFMNet/VFMSeg", VFM_CLR, "AAAI'25 VFM seg",
             "OOD generalization", "RGB-centric pretraining", "FM generalization line"),
            ("EoMT", EOMT_CLR, "CVPR'25 enc-only",
             "simple, fast mask-cls", "ViT compute; mask-cls head", "2nd CVPR architecture"),
            ("ViT-P", VITP_CLR, "CVPR'26 universal",
             "SOTA accuracy", "highest integration risk", "future SOTA reference"),
            ("TerraMind", TERRA_CLR, "EO foundation (S1/S2)",
             "aligned w/ S1/S2 + Sen1Floods11",
             "heavy pipeline; DEM-leak if misconfigured",
             "specialized EO baseline + continuity"),
        ]

        row_y0 = top_y - 0.85
        dy = 0.95
        row_groups = []
        for ri, (name, col, idea, strength, risk, role) in enumerate(rows):
            y = row_y0 - ri * dy
            name_t = cell(name, col, cx[0], y + 0.13, scale=0.32, w=2.6)
            idea_t = cell(idea, MUTED, cx[0], y - 0.18, scale=0.24, w=2.7)
            strength_t = cell(strength, TXT, cx[1], y, scale=0.28, w=3.2)
            risk_t = cell(risk, TXT, cx[2], y, scale=0.27, w=3.3)
            role_t = cell(role, TXT, cx[3], y, scale=0.27, w=3.5)
            rg = VGroup(name_t, idea_t, strength_t, risk_t, role_t)
            row_groups.append(rg)

        for ri, rg in enumerate(row_groups):
            self.play(FadeIn(rg, shift=UP * 0.15), run_time=0.5)

        # Highlight the TerraMind row (last)
        hl = SurroundingRectangle(row_groups[-1], color=TERRA_CLR, buff=0.14)
        self.play(Create(hl), Indicate(row_groups[-1][0], color=TERRA_CLR))
        self.wait(2.4)
        self.fade_all()

    # =======================================================================
    # SECTION 10 -- Physics-informed transfer (DEM invariant)
    # =======================================================================
    def section_transfer(self):
        # VOICEOVER: "Whichever model we pick, the physics-informed transfer is
        # identical. Sentinel-1 and Sentinel-2 are the inputs. The model emits
        # logits. The topographic loss reads the DEM only to penalise physically
        # inconsistent predictions. The DEM is never given as a model input in
        # our controlled protocol -- it remains a supervision signal and a
        # physical consistency metric."
        banner = self.section_banner("10", "Physics-informed transfer")
        self.play(FadeIn(banner))

        x_in = self.block("S1 + S2", WATER_DEEP, w=1.6)
        model = self.block("model\n(any family)", MUTED, w=2.1)
        logits = self.block("logits", VALID, w=1.5)
        loss = self.block("Dice + CE\n+ Topo loss", ACCENT, w=2.0)
        chain = VGroup(x_in, model, logits, loss).arrange(RIGHT, buff=0.6)
        chain.shift(UP * 1.0)
        arrows = VGroup()
        for a, b in zip(chain[:-1], chain[1:]):
            arrows.add(Arrow(a.get_right(), b.get_left(), color=MUTED, buff=0.1,
                             stroke_width=3, max_tip_length_to_length_ratio=0.25))
        self.play(FadeIn(x_in))
        for box, arr in zip(chain[1:], arrows):
            self.play(GrowArrow(arr), FadeIn(box, shift=RIGHT * 0.2), run_time=0.5)

        dem = self.block("DEM", DEM_CLR, w=1.3, h=0.75)
        dem.next_to(loss, DOWN, buff=1.4)
        ok_arrow = Arrow(dem.get_top(), loss.get_bottom(), color=VALID, buff=0.12,
                         stroke_width=3.5)
        ok_lbl = Text("topographic loss only", color=VALID).scale(0.4)
        ok_lbl.next_to(ok_arrow, RIGHT, buff=0.2)
        self.play(FadeIn(dem), GrowArrow(ok_arrow), FadeIn(ok_lbl))

        # crossed link from DEM to model
        bad = Arrow(dem.get_left(), model.get_bottom(), color=PENALTY, buff=0.15,
                    stroke_width=3)
        cross = Cross(bad, stroke_color=PENALTY, stroke_width=5).scale(0.4)
        self.play(GrowArrow(bad), Create(cross))

        inv = Text(
            "The DEM is never a model input -- only a loss / physical-consistency signal.",
            color=ACCENT, weight="BOLD").scale(0.46)
        inv.to_edge(DOWN, buff=0.55)
        self.play(Write(inv))
        self.wait(2.2)
        self.fade_all()

    # =======================================================================
    # SECTION 11 -- Roadmap
    # =======================================================================
    def section_roadmap(self):
        # VOICEOVER: "The roadmap, in order. SegMAN, the current main model,
        # already running multi-seed. TerraMind, the EO and Sen1Floods11 baseline
        # to preserve and complete. VFMNet, the AAAI line for foundation-model
        # generalization. EoMT, the second CVPR architecture. And ViT-P, a SOTA
        # reference for later, with higher integration risk."
        banner = self.section_banner("11", "Roadmap")
        self.play(FadeIn(banner))

        title = Text("Order of attack", color=TXT, weight="BOLD").scale(0.6)
        title.to_edge(UP, buff=1.0)
        self.play(Write(title))

        items = [
            (SEGMAN_CLR, "1.  SegMAN", "current main model -- already running multi-seed"),
            (TERRA_CLR,  "2.  TerraMind", "EO / Sen1Floods11 baseline to preserve & complete"),
            (VFM_CLR,    "3.  VFMNet / VFMSeg", "AAAI line for OOD / FM generalization"),
            (EOMT_CLR,   "4.  EoMT", "second CVPR architecture, ViT encoder-only"),
            (VITP_CLR,   "5.  ViT-P", "SOTA-level future reference, higher integration risk"),
        ]
        rows = VGroup()
        for col, name, desc in items:
            n = Text(name, color=col, weight="BOLD").scale(0.5)
            d = Text(desc, color=MUTED).scale(0.4)
            row = VGroup(n, d).arrange(RIGHT, buff=0.4, aligned_edge=DOWN)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        rows.next_to(title, DOWN, buff=0.7)
        for r in rows:
            self.play(FadeIn(r, shift=RIGHT * 0.25), run_time=0.55)
        self.wait(2.4)
        self.fade_all()

    # =======================================================================
    # SECTION 12 -- Conclusion
    # =======================================================================
    def section_conclusion(self):
        # VOICEOVER: "After SegMAN, the goal is not to chase scores blindly. The
        # goal is to test whether our topographic physics-informed loss remains
        # useful across different model families: dense segmentation
        # architectures, Earth Observation foundation models, foundation-model
        # generalization systems, ViT encoder-only segmentation, and universal
        # mask-classification frameworks. And throughout, the DEM is never given
        # as a model input -- it stays a supervision signal and a physical
        # consistency metric."
        banner = self.section_banner("12", "Conclusion")
        self.play(FadeIn(banner))

        head = Text("Not chasing scores -- testing a property.",
                    color=ACCENT, weight="BOLD").scale(0.6)
        head.to_edge(UP, buff=1.1)
        self.play(Write(head))

        q = Text("Does the topographic physics-informed loss stay useful across:",
                 color=TXT).scale(0.5)
        q.next_to(head, DOWN, buff=0.5)
        self.play(FadeIn(q, shift=UP * 0.2))

        families = VGroup(
            self._bullet(WATER,   "dense segmentation architectures"),
            self._bullet(TERRA_CLR, "Earth Observation foundation models"),
            self._bullet(VFM_CLR, "foundation-model generalization systems"),
            self._bullet(EOMT_CLR, "ViT encoder-only segmentation"),
            self._bullet(VITP_CLR, "universal mask-classification frameworks"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.34)
        families.next_to(q, DOWN, buff=0.5)
        for f in families:
            self.play(FadeIn(f, shift=RIGHT * 0.2), run_time=0.45)
        self.wait(1.0)

        inv = Text(
            "The DEM is never a model input -- only a loss / physical-consistency signal.",
            color=ACCENT).scale(0.46)
        inv.to_edge(DOWN, buff=0.8)
        self.play(Write(inv))
        self.wait(1.6)

        self.play(FadeOut(head), FadeOut(q), FadeOut(families),
                  FadeOut(banner), FadeOut(inv))
        outro = VGroup(
            Text("After SegMAN", color=WATER, weight="BOLD").scale(1.0),
            Text("VFMNet  -  EoMT  -  ViT-P  -  TerraMind", color=MUTED).scale(0.5),
            Text("physics-informed flood segmentation", color=MUTED).scale(0.42),
        ).arrange(DOWN, buff=0.3)
        self.play(FadeIn(outro))
        self.wait(2.2)
        self.fade_all()


# ===========================================================================
# Master scene -- the full video
# ===========================================================================
class PostSegMANModelsVideo(PostSegMANBase):
    def construct(self):
        self.section_title()
        self.section_motivation()
        self.section_protocol()
        self.section_vfmnet()
        self.section_vfmnet_how()
        self.section_eomt()
        self.section_eomt_how()
        self.section_vitp()
        self.section_vitp_how()
        self.section_not_rgb()
        self.section_terramind()
        self.section_terramind_how()
        self.section_terramind_role()
        self.section_terramind_risks()
        self.section_comparison()
        self.section_transfer()
        self.section_roadmap()
        self.section_conclusion()


# ===========================================================================
# Per-section scenes -- render any chapter on its own while iterating
# ===========================================================================
class S00_Title(PostSegMANBase):
    def construct(self): self.section_title()


class S01_Motivation(PostSegMANBase):
    def construct(self): self.section_motivation()


class S02_Protocol(PostSegMANBase):
    def construct(self): self.section_protocol()


class S03_VFMNet(PostSegMANBase):
    def construct(self): self.section_vfmnet()


class S03b_VFMNetHow(PostSegMANBase):
    def construct(self): self.section_vfmnet_how()


class S04_EoMT(PostSegMANBase):
    def construct(self): self.section_eomt()


class S04b_EoMTHow(PostSegMANBase):
    def construct(self): self.section_eomt_how()


class S05_ViTP(PostSegMANBase):
    def construct(self): self.section_vitp()


class S05b_ViTPHow(PostSegMANBase):
    def construct(self): self.section_vitp_how()


class S06_NotRGB(PostSegMANBase):
    def construct(self): self.section_not_rgb()


class S07_TerraMind(PostSegMANBase):
    def construct(self): self.section_terramind()


class S07a_TerraMindHow(PostSegMANBase):
    def construct(self): self.section_terramind_how()


class S07b_TerraMindRole(PostSegMANBase):
    def construct(self): self.section_terramind_role()


class S08_TerraMindRisks(PostSegMANBase):
    def construct(self): self.section_terramind_risks()


class S09_Comparison(PostSegMANBase):
    def construct(self): self.section_comparison()


class S10_Transfer(PostSegMANBase):
    def construct(self): self.section_transfer()


class S11_Roadmap(PostSegMANBase):
    def construct(self): self.section_roadmap()


class S12_Conclusion(PostSegMANBase):
    def construct(self): self.section_conclusion()
