"""
AnySat and Panopticon for physics-informed flood segmentation.

A pedagogical Manim explainer for the current flood-segmentation research line:
we have a 15-channel Sentinel-2 + Sentinel-1 input, binary water/background
labels, and a topographic loss where the DEM is deliberately NOT a model input.

The video explains why RGB-only models are an awkward fit, then introduces two
2025 Earth Observation foundation-model directions:

    * AnySat: multimodal EO across resolutions, scales, and modalities, using a
      JEPA-style feature prediction objective and scale-adaptive encoders.
    * Panopticon: an any-sensor DINOv2-style EO foundation model that encodes
      channel identities and fuses arbitrary optical/SAR channels.

Render examples (CPU render; do not use the GPU):

    # quick preview
    manim -pql videos/anysat_panopticon_explanation.py AnySatPanopticonVideo

    # high quality
    manim -pqh videos/anysat_panopticon_explanation.py AnySatPanopticonVideo

    # one section only
    manim -pql videos/anysat_panopticon_explanation.py S04_AnySatForOurProject

Voiceover: videos/anysat_panopticon_voiceover.md
Storyboard: videos/anysat_panopticon_storyboard.md

NOTE: This file only describes an animation. It does not load models, read
checkpoints, touch experiment directories, launch training, or use the GPU.

SCIENTIFIC INVARIANT:
    DEM is never passed to AnySat, Panopticon, SegMAN, or TerraMind as an input
    in this controlled protocol. DEM is used only after prediction, inside the
    topographic loss and physical consistency metrics.
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
    Brace,
    Circle,
    Create,
    Cross,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    Indicate,
    Line,
    Rectangle,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    Text,
    Transform,
    VGroup,
    Write,
    config,
)


# ---------------------------------------------------------------------------
# Palette: dark, clean, and consistent with the existing project videos.
# ---------------------------------------------------------------------------
BG_COLOR = "#0b0f1a"
TXT = "#e5e7eb"
MUTED = "#94a3b8"
WATER = "#22d3ee"
S2 = "#34d399"
S1 = "#a78bfa"
DEM = "#d4a373"
PAN = "#fbbf24"
ANY = "#38bdf8"
SEG = "#fb7185"
LAND = "#243042"
PENALTY = "#ef4444"
VALID = "#22c55e"
WARN = "#f97316"
WHITE_SOFT = "#f8fafc"

config.background_color = BG_COLOR
Text.set_default(font="Arial")
random.seed(23)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
class AnySatPanopticonBase(Scene):
    """Shared drawing helpers and section methods."""

    def fade_all(self, run_time: float = 0.45) -> None:
        mobs = [m for m in self.mobjects]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def title(self, text: str, subtitle: str | None = None) -> VGroup:
        title = Text(text, color=TXT, weight="BOLD").scale(0.46).to_edge(UP)
        group = VGroup(title)
        if subtitle:
            sub = Text(subtitle, color=MUTED).scale(0.26).next_to(title, DOWN, buff=0.12)
            group.add(sub)
        self.play(FadeIn(group, shift=0.15 * DOWN), run_time=0.5)
        return group

    def section_banner(self, number: str, text: str) -> VGroup:
        badge = RoundedRectangle(width=0.68, height=0.42, corner_radius=0.08)
        badge.set_fill(ANY, opacity=0.16).set_stroke(ANY, width=1.5)
        label = Text(number, color=ANY, weight="BOLD").scale(0.25).move_to(badge)
        title = Text(text, color=TXT, weight="BOLD").scale(0.32).next_to(badge, RIGHT, buff=0.18)
        group = VGroup(badge, label, title).to_corner(UP + LEFT, buff=0.35)
        self.play(FadeIn(group, shift=0.15 * RIGHT), run_time=0.4)
        return group

    def pill(self, text: str, color: str, width: float = 2.3, height: float = 0.52) -> VGroup:
        box = RoundedRectangle(width=width, height=height, corner_radius=0.12)
        box.set_fill(color, opacity=0.16).set_stroke(color, width=1.6)
        label = Text(text, color=TXT).scale(0.22).move_to(box)
        return VGroup(box, label)

    def block(
        self,
        text: str,
        color: str,
        width: float = 2.5,
        height: float = 0.78,
        scale: float = 0.25,
    ) -> VGroup:
        box = RoundedRectangle(width=width, height=height, corner_radius=0.08)
        box.set_fill(color, opacity=0.14).set_stroke(color, width=1.7)
        label = Text(text, color=TXT, line_spacing=0.86).scale(scale).move_to(box)
        return VGroup(box, label)

    def arrow_between(self, left_obj, right_obj, color: str = MUTED) -> Arrow:
        return Arrow(
            left_obj.get_right() + 0.08 * RIGHT,
            right_obj.get_left() + 0.08 * LEFT,
            buff=0.05,
            color=color,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.14,
        )

    def channel_stack(
        self,
        n: int,
        color: str,
        label: str,
        width: float = 1.55,
        height: float = 1.15,
        dx: float = 0.05,
        dy: float = 0.04,
    ) -> VGroup:
        rects = VGroup()
        for i in range(n):
            rect = Rectangle(width=width, height=height)
            rect.set_fill(color, opacity=0.12 + 0.018 * min(i, 5))
            rect.set_stroke(color, width=1.2)
            rect.shift(i * dx * RIGHT + i * dy * UP)
            rects.add(rect)
        text = Text(label, color=TXT, line_spacing=0.85).scale(0.22).next_to(rects, DOWN, buff=0.16)
        return VGroup(rects, text)

    def mini_satellite_tile(self, width: float = 2.0, height: float = 1.25) -> VGroup:
        tile = Rectangle(width=width, height=height)
        tile.set_fill(LAND, opacity=1.0).set_stroke(MUTED, width=1.2)
        river = Line(tile.get_left() + 0.18 * RIGHT + 0.34 * UP, tile.get_right() + 0.14 * LEFT + 0.30 * DOWN)
        river.set_color(WATER).set_stroke(width=8, opacity=0.78)
        river2 = Line(tile.get_left() + 0.30 * RIGHT + 0.52 * DOWN, tile.get_center() + 0.10 * UP)
        river2.set_color(WATER).set_stroke(width=5, opacity=0.52)
        fields = VGroup()
        for _ in range(8):
            x = random.uniform(-width / 2 + 0.18, width / 2 - 0.18)
            y = random.uniform(-height / 2 + 0.16, height / 2 - 0.16)
            f = Rectangle(width=random.uniform(0.18, 0.34), height=random.uniform(0.08, 0.18))
            f.set_fill(S2, opacity=random.uniform(0.16, 0.34)).set_stroke(S2, width=0.4, opacity=0.5)
            f.move_to(tile.get_center() + x * RIGHT + y * UP)
            fields.add(f)
        return VGroup(tile, fields, river, river2)

    def token_grid(self, rows: int, cols: int, color: str, cell: float = 0.24) -> VGroup:
        grid = VGroup()
        for r in range(rows):
            for c in range(cols):
                dot = RoundedRectangle(width=cell, height=cell, corner_radius=0.03)
                dot.set_fill(color, opacity=0.22).set_stroke(color, width=0.7)
                dot.move_to((c - (cols - 1) / 2) * (cell + 0.05) * RIGHT + ((rows - 1) / 2 - r) * (cell + 0.05) * UP)
                grid.add(dot)
        return grid

    def small_note(self, text: str, color: str = MUTED) -> Text:
        return Text(text, color=color, line_spacing=0.88).scale(0.22)

    # ------------------------------------------------------------------
    # Section 0 - Opening
    # ------------------------------------------------------------------
    def section_00_opening(self) -> None:
        # VOICEOVER: We begin with our actual problem: binary flood segmentation from Sentinel-2 and Sentinel-1.
        self.title(
            "AnySat and Panopticon",
            "SOTA 2025 multimodal EO models for physics-informed flood segmentation",
        )

        s2 = self.channel_stack(13, S2, "Sentinel-2\n13 optical bands").shift(3.9 * LEFT + 0.45 * UP)
        s1 = self.channel_stack(2, S1, "Sentinel-1\n2 SAR channels").shift(1.8 * LEFT + 0.45 * UP)
        tensor = Text("X in R^(15 x 512 x 512)", color=TXT, weight="BOLD").scale(0.32)
        tensor.shift(1.25 * RIGHT + 0.55 * UP)
        mask = self.block("flood segmentation\n0 background | 1 water\n-1 ignore", WATER, width=2.55, height=1.2, scale=0.22)
        mask.shift(4.15 * RIGHT + 0.55 * UP)
        a1 = self.arrow_between(s2, tensor, S2)
        a2 = self.arrow_between(s1, tensor, S1)
        a3 = self.arrow_between(tensor, mask, WATER)

        rgb = self.block("RGB-only\nmodels", PENALTY, width=1.95, height=0.86, scale=0.25).shift(0.35 * DOWN)
        cross = Cross(rgb, stroke_color=PENALTY, stroke_width=7)
        why = self.small_note(
            "Our data are multispectral + radar.\nCompressing them to fake RGB can erase useful sensor physics.",
            MUTED,
        ).next_to(rgb, DOWN, buff=0.42)

        self.play(FadeIn(s2), FadeIn(s1), run_time=0.7)
        self.play(GrowArrow(a1), GrowArrow(a2), Write(tensor), run_time=0.8)
        self.play(GrowArrow(a3), FadeIn(mask), run_time=0.6)
        self.play(FadeIn(rgb), Create(cross), FadeIn(why), run_time=0.7)
        self.wait(2.2)
        self.fade_all()

    # ------------------------------------------------------------------
    # Section 1 - Why RGB is a problem
    # ------------------------------------------------------------------
    def section_01_why_rgb(self) -> None:
        # VOICEOVER: A satellite band is not simply a color channel. Sentinel-2 and Sentinel-1 measure different physical quantities.
        self.section_banner("01", "Why RGB is a problem")

        left = VGroup(
            self.block("RGB image", SEG, width=1.7),
            self.block("3 channels", SEG, width=1.7),
            self.block("classical\nvision backbone", SEG, width=2.0),
        ).arrange(RIGHT, buff=0.5).shift(1.45 * UP)
        l_arrows = VGroup(self.arrow_between(left[0], left[1], SEG), self.arrow_between(left[1], left[2], SEG))

        right = VGroup(
            self.block("S2 bands\n+ S1 SAR", ANY, width=1.95),
            self.block("15 channels", ANY, width=1.75),
            self.block("not RGB", PENALTY, width=1.65),
        ).arrange(RIGHT, buff=0.5).shift(0.05 * UP)
        r_arrows = VGroup(self.arrow_between(right[0], right[1], ANY), self.arrow_between(right[1], right[2], PENALTY))

        options = VGroup(
            self.block("1. select RGB-like bands", WARN, width=2.35, height=0.62, scale=0.2),
            self.block("2. project 15 -> 3", WARN, width=2.35, height=0.62, scale=0.2),
            self.block("3. inflate conv 3 -> 15", WARN, width=2.35, height=0.62, scale=0.2),
        ).arrange(RIGHT, buff=0.32).shift(1.7 * DOWN)
        note = self.small_note(
            "These are workable engineering shortcuts, but not ideal scientific priors.",
            MUTED,
        ).next_to(options, DOWN, buff=0.25)

        self.play(FadeIn(left), *[GrowArrow(a) for a in l_arrows], run_time=0.8)
        self.play(FadeIn(right), *[GrowArrow(a) for a in r_arrows], run_time=0.8)
        self.play(FadeIn(options, shift=0.15 * UP), FadeIn(note), run_time=0.8)
        self.wait(2.1)
        self.fade_all()

    # ------------------------------------------------------------------
    # Section 2 - AnySat concept
    # ------------------------------------------------------------------
    def section_02_anysat_concept(self) -> None:
        # VOICEOVER: AnySat is interesting because it is built for heterogeneous Earth Observation data, not for one fixed RGB camera.
        self.section_banner("02", "AnySat intuition")

        sensors = VGroup(
            self.pill("Sentinel-2", S2),
            self.pill("Sentinel-1", S1),
            self.pill("aerial", ANY),
            self.pill("Landsat", VALID),
            self.pill("MODIS", PAN),
            self.pill("other sensors", WARN),
        ).arrange(DOWN, buff=0.16).scale(0.94).shift(4.1 * LEFT)

        model = self.block("AnySat", ANY, width=2.2, height=1.2, scale=0.34).shift(0.55 * LEFT)
        rep = self.block("shared EO\nrepresentation", VALID, width=2.35, height=1.0, scale=0.24).shift(2.05 * RIGHT)
        tasks = VGroup(
            self.pill("land cover", WATER, width=1.6, height=0.42),
            self.pill("crop", S2, width=1.6, height=0.42),
            self.pill("change", PAN, width=1.6, height=0.42),
            self.pill("flood", WATER, width=1.6, height=0.42),
            self.pill("burn scar", WARN, width=1.6, height=0.42),
            self.pill("deforestation", VALID, width=1.6, height=0.42),
        ).arrange(DOWN, buff=0.12).shift(4.35 * RIGHT)

        arrows_in = VGroup(*[self.arrow_between(s, model, ANY) for s in sensors])
        arrow_rep = self.arrow_between(model, rep, VALID)
        arrows_out = VGroup(*[self.arrow_between(rep, t, WATER) for t in tasks])
        core = self.small_note(
            "One EO model across many resolutions, scales, and modalities.",
            TXT,
        ).next_to(model, DOWN, buff=0.6)

        self.play(FadeIn(sensors), run_time=0.7)
        self.play(FadeIn(model), *[GrowArrow(a) for a in arrows_in], run_time=1.0)
        self.play(GrowArrow(arrow_rep), FadeIn(rep), run_time=0.65)
        self.play(FadeIn(tasks), *[GrowArrow(a) for a in arrows_out], FadeIn(core), run_time=1.0)
        self.wait(2.1)
        self.fade_all()

    # ------------------------------------------------------------------
    # Section 3 - AnySat architecture
    # ------------------------------------------------------------------
    def section_03_anysat_architecture(self) -> None:
        # VOICEOVER: AnySat uses a JEPA-like idea: predict representations in feature space rather than reconstructing every pixel.
        self.section_banner("03", "AnySat architecture intuition")

        tile = self.mini_satellite_tile().shift(4.6 * LEFT + 0.8 * UP)
        masked = self.mini_satellite_tile().shift(4.6 * LEFT + 1.25 * DOWN)
        mask_patch = Rectangle(width=0.72, height=0.42).set_fill(BG_COLOR, opacity=0.82).set_stroke(PENALTY, width=2)
        mask_patch.move_to(masked[0].get_center() + 0.25 * RIGHT)
        masked.add(mask_patch)

        context = self.block("context view", ANY, width=1.9).next_to(tile, RIGHT, buff=0.55)
        encoder = self.block("encoder", ANY, width=1.65).next_to(context, RIGHT, buff=0.48)
        pred = self.block("predicted\nembedding", VALID, width=1.85).next_to(encoder, RIGHT, buff=0.48)
        target = self.block("target\nembedding", PAN, width=1.85).move_to(pred.get_center() + 1.95 * DOWN)
        loss = self.block("feature-space\nprediction loss", PENALTY, width=2.2, scale=0.22)
        loss.next_to(VGroup(pred, target), RIGHT, buff=0.5)

        arrows = VGroup(
            self.arrow_between(tile, context, ANY),
            self.arrow_between(context, encoder, ANY),
            self.arrow_between(encoder, pred, VALID),
            Arrow(masked.get_right(), target.get_left(), buff=0.08, color=PAN, stroke_width=3),
            self.arrow_between(pred, loss, PENALTY),
            self.arrow_between(target, loss, PENALTY),
        )

        scale_note = self.block(
            "scale-adaptive\nspatial encoders",
            S2,
            width=2.2,
            height=0.9,
            scale=0.23,
        ).to_edge(DOWN).shift(1.3 * RIGHT)
        note = self.small_note(
            "Different EO sensors have different resolutions.\nAnySat adapts the spatial encoding instead of forcing one fixed image recipe.",
            MUTED,
        ).next_to(scale_note, LEFT, buff=0.7)

        self.play(FadeIn(tile), FadeIn(masked), run_time=0.7)
        self.play(FadeIn(context), FadeIn(encoder), FadeIn(pred), FadeIn(target), run_time=0.75)
        self.play(*[GrowArrow(a) for a in arrows], FadeIn(loss), run_time=1.0)
        self.play(FadeIn(scale_note), FadeIn(note), run_time=0.7)
        self.wait(2.2)
        self.fade_all()

    # ------------------------------------------------------------------
    # Section 4 - AnySat for our project
    # ------------------------------------------------------------------
    def section_04_anysat_project(self) -> None:
        # VOICEOVER: In our protocol, AnySat would replace the backbone, not the scientific loss comparison.
        self.section_banner("04", "AnySat for our flood project")

        s2 = self.channel_stack(13, S2, "S2 optical").shift(4.6 * LEFT + 0.9 * UP)
        s1 = self.channel_stack(2, S1, "S1 SAR").shift(4.6 * LEFT + 1.05 * DOWN)
        wrapper = self.block("AnySat\nwrapper", ANY, width=1.9, height=1.0, scale=0.25).shift(2.1 * LEFT)
        head = self.block("segmentation\nhead", WATER, width=2.0, height=1.0, scale=0.25).shift(0.25 * RIGHT)
        logits = Text("logits [B,2,H,W]", color=TXT, weight="BOLD").scale(0.28).shift(2.55 * RIGHT + 0.58 * UP)
        pwater = Text("p_water", color=WATER, weight="BOLD").scale(0.34).shift(2.55 * RIGHT + 0.65 * DOWN)

        dem = self.block("DEM", DEM, width=1.1, height=0.58, scale=0.24).shift(0.3 * RIGHT + 2.1 * DOWN)
        ce = self.block("GT mask\nCE / DiceCE", VALID, width=2.0, height=0.78, scale=0.22).shift(3.15 * RIGHT + 1.75 * DOWN)
        topo = self.block("Topo loss\n+ metrics", DEM, width=2.0, height=0.78, scale=0.22).shift(5.3 * RIGHT + 1.75 * DOWN)

        arrows = VGroup(
            self.arrow_between(s2, wrapper, S2),
            self.arrow_between(s1, wrapper, S1),
            self.arrow_between(wrapper, head, ANY),
            self.arrow_between(head, logits, WATER),
            Arrow(logits.get_bottom(), pwater.get_top(), buff=0.08, color=WATER, stroke_width=3),
            Arrow(pwater.get_bottom(), ce.get_top(), buff=0.08, color=VALID, stroke_width=3),
            Arrow(pwater.get_bottom(), topo.get_top(), buff=0.08, color=DEM, stroke_width=3),
            self.arrow_between(dem, topo, DEM),
        )

        no_input = self.small_note("DEM is NOT passed into AnySat.", PENALTY).next_to(dem, DOWN, buff=0.25)
        risks = VGroup(
            self.pill("audit expected S2 bands", WARN, width=2.35, height=0.42),
            self.pill("map 13 S2 + 2 S1 channels", WARN, width=2.8, height=0.42),
            self.pill("ensure dense logits", WARN, width=2.15, height=0.42),
        ).arrange(DOWN, buff=0.12).to_corner(DOWN + LEFT, buff=0.45)

        self.play(FadeIn(s2), FadeIn(s1), FadeIn(wrapper), run_time=0.8)
        self.play(*[GrowArrow(a) for a in arrows[:3]], FadeIn(head), run_time=0.8)
        self.play(GrowArrow(arrows[3]), Write(logits), FadeIn(pwater), run_time=0.7)
        self.play(FadeIn(dem), FadeIn(ce), FadeIn(topo), *[GrowArrow(a) for a in arrows[4:]], FadeIn(no_input), run_time=1.0)
        self.play(FadeIn(risks), run_time=0.7)
        self.wait(2.1)
        self.fade_all()

    # ------------------------------------------------------------------
    # Section 5 - Panopticon concept
    # ------------------------------------------------------------------
    def section_05_panopticon_concept(self) -> None:
        # VOICEOVER: Panopticon asks the model to know what each channel physically means.
        self.section_banner("05", "Panopticon intuition")

        names = ["S2 B2", "S2 B3", "S2 B4", "S2 NIR", "S2 SWIR", "S1 VV", "S1 VH"]
        colors = [S2, S2, S2, S2, S2, S1, S1]
        rows = VGroup()
        for name, color in zip(names, colors):
            channel = self.pill(name, color, width=1.35, height=0.38)
            ident = self.pill("wavelength" if "S2" in name else "SAR mode", color, width=1.6, height=0.38)
            row = VGroup(channel, ident).arrange(RIGHT, buff=0.22)
            rows.add(row)
        rows.arrange(DOWN, buff=0.1).shift(4.15 * LEFT)

        embed = self.block("channels\n+\nchannel identities", PAN, width=2.3, height=1.45, scale=0.22).shift(0.25 * LEFT)
        patch = self.block("spectral/channel-aware\npatch embedding", PAN, width=2.7, height=1.0, scale=0.22).shift(2.55 * RIGHT + 0.55 * UP)
        backbone = self.block("DINOv2-like\nbackbone", VALID, width=2.25, height=0.95, scale=0.24).shift(2.55 * RIGHT + 0.85 * DOWN)

        arrows = VGroup(
            *[self.arrow_between(row, embed, PAN) for row in rows],
            self.arrow_between(embed, patch, PAN),
            Arrow(patch.get_bottom(), backbone.get_top(), buff=0.08, color=VALID, stroke_width=3),
        )
        note = self.small_note(
            "Pixel values alone are not enough.\nA B8 NIR channel and a VV radar channel should not be treated as anonymous slots.",
            MUTED,
        ).to_edge(DOWN)

        self.play(FadeIn(rows), run_time=0.8)
        self.play(FadeIn(embed), *[GrowArrow(a) for a in arrows[: len(rows)]], run_time=1.0)
        self.play(FadeIn(patch), FadeIn(backbone), *[GrowArrow(a) for a in arrows[len(rows) :]], FadeIn(note), run_time=0.9)
        self.wait(2.1)
        self.fade_all()

    # ------------------------------------------------------------------
    # Section 6 - Panopticon architecture
    # ------------------------------------------------------------------
    def section_06_panopticon_architecture(self) -> None:
        # VOICEOVER: Compared with standard DINOv2, Panopticon adds flexible channel fusion before patch tokens enter the ViT.
        self.section_banner("06", "Panopticon architecture intuition")

        rgb = VGroup(
            self.block("RGB patches", SEG, width=1.85),
            self.block("patch\nembedding", SEG, width=1.7),
            self.block("ViT\nDINOv2", SEG, width=1.7),
        ).arrange(RIGHT, buff=0.45).shift(1.55 * UP)
        rgb_arrows = VGroup(self.arrow_between(rgb[0], rgb[1], SEG), self.arrow_between(rgb[1], rgb[2], SEG))
        rgb_label = self.small_note("standard image foundation model", MUTED).next_to(rgb, UP, buff=0.18)

        channel_tokens = VGroup()
        for i, (label, color) in enumerate([("B2", S2), ("B3", S2), ("B4", S2), ("NIR", S2), ("SWIR", S2), ("VV", S1), ("VH", S1)]):
            token = self.pill(label, color, width=0.72, height=0.34)
            token.shift((i - 3) * 0.55 * RIGHT)
            channel_tokens.add(token)
        channel_tokens.shift(4.1 * LEFT + 0.55 * DOWN)
        cross_attn = self.block("cross-attention\nover channels", PAN, width=2.2, height=0.88, scale=0.22).shift(0.35 * LEFT + 0.55 * DOWN)
        patch_tokens = self.token_grid(3, 5, PAN).shift(2.1 * RIGHT + 0.55 * DOWN)
        vit = self.block("DINOv2-like\ntransformer", VALID, width=2.05, height=0.88, scale=0.22).shift(4.65 * RIGHT + 0.55 * DOWN)

        pan_arrows = VGroup(
            Arrow(channel_tokens.get_right(), cross_attn.get_left(), buff=0.08, color=PAN, stroke_width=3),
            Arrow(cross_attn.get_right(), patch_tokens.get_left(), buff=0.08, color=PAN, stroke_width=3),
            Arrow(patch_tokens.get_right(), vit.get_left(), buff=0.08, color=VALID, stroke_width=3),
        )

        ideas = VGroup(
            self.pill("same geolocation = sensor views", ANY, width=2.85, height=0.42),
            self.pill("channel subsampling = robustness", ANY, width=2.8, height=0.42),
            self.pill("cross-attention = flexible fusion", ANY, width=2.85, height=0.42),
        ).arrange(DOWN, buff=0.12).to_edge(DOWN)

        self.play(FadeIn(rgb_label), FadeIn(rgb), *[GrowArrow(a) for a in rgb_arrows], run_time=0.9)
        self.play(FadeIn(channel_tokens), FadeIn(cross_attn), FadeIn(patch_tokens), FadeIn(vit), run_time=0.9)
        self.play(*[GrowArrow(a) for a in pan_arrows], FadeIn(ideas), run_time=1.0)
        self.wait(2.1)
        self.fade_all()

    # ------------------------------------------------------------------
    # Section 7 - Panopticon for our project
    # ------------------------------------------------------------------
    def section_07_panopticon_project(self) -> None:
        # VOICEOVER: Panopticon is especially attractive for our 15-channel problem because channel identity is part of the input contract.
        self.section_banner("07", "Panopticon for our flood project")

        s2 = self.block("13 S2 bands\nwith wavelengths", S2, width=2.2, height=0.9, scale=0.22).shift(4.55 * LEFT + 0.85 * UP)
        s1 = self.block("2 S1 SAR\nVV / VH mode IDs", S1, width=2.2, height=0.9, scale=0.22).shift(4.55 * LEFT + 0.85 * DOWN)
        pan = self.block("Panopticon\nbackbone", PAN, width=2.15, height=1.05, scale=0.25).shift(1.8 * LEFT)
        feat = self.block("patch\nfeatures", VALID, width=1.6, height=0.85, scale=0.24).shift(0.45 * RIGHT)
        dec = self.block("dense decoder\n/ head", WATER, width=1.95, height=0.9, scale=0.23).shift(2.55 * RIGHT)
        logits = Text("[B,2,H,W]", color=TXT, weight="BOLD").scale(0.34).shift(4.5 * RIGHT)

        arrows = VGroup(
            self.arrow_between(s2, pan, S2),
            self.arrow_between(s1, pan, S1),
            self.arrow_between(pan, feat, PAN),
            self.arrow_between(feat, dec, WATER),
            self.arrow_between(dec, logits, WATER),
        )
        benefits = VGroup(
            self.pill("no RGB compression", VALID, width=2.2, height=0.42),
            self.pill("arbitrary channel handling", VALID, width=2.55, height=0.42),
            self.pill("good Sentinel-1 + Sentinel-2 fit", VALID, width=3.0, height=0.42),
        ).arrange(DOWN, buff=0.12).to_corner(DOWN + LEFT, buff=0.48)
        risks = VGroup(
            self.pill("needs dense head", WARN, width=2.0, height=0.42),
            self.pill("patch upsampling", WARN, width=2.0, height=0.42),
            self.pill("DEM stays out", PENALTY, width=1.9, height=0.42),
        ).arrange(DOWN, buff=0.12).to_corner(DOWN + RIGHT, buff=0.48)

        self.play(FadeIn(s2), FadeIn(s1), FadeIn(pan), run_time=0.8)
        self.play(*[GrowArrow(a) for a in arrows], FadeIn(feat), FadeIn(dec), Write(logits), run_time=1.0)
        self.play(FadeIn(benefits), FadeIn(risks), run_time=0.8)
        self.wait(2.1)
        self.fade_all()

    # ------------------------------------------------------------------
    # Section 8 - Comparison
    # ------------------------------------------------------------------
    def section_08_comparison(self) -> None:
        # VOICEOVER: The model candidates differ, but their role is to plug into the same dense-logit protocol.
        self.section_banner("08", "AnySat vs Panopticon vs current baselines")

        headers = ["Model", "Main idea", "Strength", "Risk", "Role"]
        rows = [
            ["AnySat", "many EO scales\n+ modalities", "EO downstream\nincluding segmentation", "input formatting\nband mapping", "EO multimodal\nSOTA candidate"],
            ["Panopticon", "any-sensor\nDINOv2-like", "arbitrary optical\n+ SAR channels", "needs dense\nsegmentation head", "best fit for\n15-channel issue"],
            ["SegMAN", "dense segmentation\narchitecture", "already integrated\nlogits direct", "not EO-native", "validated\nbaseline"],
            ["TerraMind", "EO foundation\nbaseline", "domain aligned", "heavier pipeline", "continuity with\nprevious work"],
        ]

        table = VGroup()
        col_widths = [1.35, 2.2, 2.2, 1.85, 2.15]
        x0 = -5.25
        y0 = 1.95
        row_h = 0.72
        for r, row in enumerate([headers] + rows):
            y = y0 - r * row_h
            for c, text in enumerate(row):
                x = x0 + sum(col_widths[:c]) + col_widths[c] / 2
                color = MUTED if r == 0 else [ANY, PAN, SEG, VALID][r - 1]
                cell = RoundedRectangle(width=col_widths[c] - 0.05, height=row_h - 0.06, corner_radius=0.04)
                cell.set_fill(color, opacity=0.18 if r == 0 else 0.09).set_stroke(color, width=1.0)
                cell.move_to(x * RIGHT + y * UP)
                label = Text(text, color=TXT if r > 0 else WHITE_SOFT, line_spacing=0.8)
                label.scale(0.145 if r > 0 else 0.16).move_to(cell)
                table.add(VGroup(cell, label))

        callout = self.small_note(
            "For the science question, the important output is always dense logits [B,2,H,W].",
            WATER,
        ).to_edge(DOWN)

        self.play(FadeIn(table, shift=0.1 * UP), run_time=1.1)
        self.play(FadeIn(callout), run_time=0.5)
        self.wait(2.5)
        self.fade_all()

    # ------------------------------------------------------------------
    # Section 9 - Physics-informed pipeline
    # ------------------------------------------------------------------
    def section_09_physics_pipeline(self) -> None:
        # VOICEOVER: The backbone changes, but the loss protocol stays fixed. That is how we isolate the physical question.
        self.section_banner("09", "One physics-informed protocol")

        model = self.block("Any model", ANY, width=1.75).shift(4.8 * LEFT + 1.0 * UP)
        logits = self.block("dense logits\n[B,2,H,W]", WATER, width=2.0).shift(2.3 * LEFT + 1.0 * UP)
        softmax = self.block("softmax", WATER, width=1.5).shift(0.0 * RIGHT + 1.0 * UP)
        pwater = Text("p_water", color=WATER, weight="BOLD").scale(0.34).shift(1.95 * RIGHT + 1.0 * UP)
        arrows_top = VGroup(
            self.arrow_between(model, logits, WATER),
            self.arrow_between(logits, softmax, WATER),
            self.arrow_between(softmax, pwater, WATER),
        )

        losses = VGroup(
            self.block("1. CE", VALID, width=2.0, height=0.58, scale=0.22),
            self.block("2. Dice + CE", VALID, width=2.0, height=0.58, scale=0.22),
            self.block("3. Dice + CE + Topo\nreal DEM", DEM, width=2.6, height=0.72, scale=0.2),
            self.block("4. Dice + CE + Topo\nshuffled DEM", WARN, width=2.8, height=0.72, scale=0.2),
        ).arrange(DOWN, buff=0.18).shift(1.4 * DOWN)
        dem = self.block("DEM", DEM, width=1.25, height=0.58, scale=0.24).shift(4.0 * RIGHT + 0.2 * DOWN)
        dem_no = Cross(dem, stroke_color=PENALTY, stroke_width=5)
        no_in = self.small_note("not a model input", PENALTY).next_to(dem, DOWN, buff=0.14)
        p_to_losses = VGroup(*[Arrow(pwater.get_bottom(), l.get_left(), buff=0.1, color=WATER, stroke_width=2.6) for l in losses])
        dem_to_topo = VGroup(Arrow(dem.get_left(), losses[2].get_right(), buff=0.08, color=DEM, stroke_width=2.6), Arrow(dem.get_left(), losses[3].get_right(), buff=0.08, color=WARN, stroke_width=2.6))

        note = self.small_note(
            "The DEM-shuffled control separates real physical information from generic regularization.",
            TXT,
        ).to_edge(DOWN)

        self.play(FadeIn(model), FadeIn(logits), FadeIn(softmax), Write(pwater), *[GrowArrow(a) for a in arrows_top], run_time=1.0)
        self.play(FadeIn(losses), *[GrowArrow(a) for a in p_to_losses], run_time=0.9)
        self.play(FadeIn(dem), Create(dem_no), FadeIn(no_in), *[GrowArrow(a) for a in dem_to_topo], FadeIn(note), run_time=0.9)
        self.wait(2.2)
        self.fade_all()

    # ------------------------------------------------------------------
    # Section 10 - Final roadmap
    # ------------------------------------------------------------------
    def section_10_final_roadmap(self) -> None:
        # VOICEOVER: The roadmap is deliberately conservative: finish the current diagnostic, then audit foundation models before integrating them.
        self.section_banner("10", "Roadmap")

        steps = [
            ("1", "Finish SegMAN N=100 diagnostic", WATER),
            ("2", "Complete TerraMind baseline", VALID),
            ("3", "Audit AnySat", ANY),
            ("4", "Audit Panopticon", PAN),
            ("5", "Integrate only if dense logits are clean", WARN),
            ("6", "Compare real DEM vs shuffled DEM", DEM),
        ]
        nodes = VGroup()
        for number, text, color in steps:
            dot = Circle(radius=0.18).set_fill(color, opacity=0.35).set_stroke(color, width=2)
            label = Text(number, color=TXT, weight="BOLD").scale(0.18).move_to(dot)
            body = Text(text, color=TXT).scale(0.23).next_to(dot, RIGHT, buff=0.18)
            node = VGroup(dot, label, body)
            nodes.add(node)
        nodes.arrange(DOWN, buff=0.35).shift(2.2 * LEFT + 0.25 * UP)
        connectors = VGroup()
        for i in range(len(nodes) - 1):
            connectors.add(Line(nodes[i][0].get_bottom(), nodes[i + 1][0].get_top(), color=MUTED, stroke_width=2))

        final = Text(
            "Not RGB-only. Not flood-only.\nMultimodal and sensor-aware.",
            color=WHITE_SOFT,
            weight="BOLD",
            line_spacing=0.9,
        ).scale(0.36).shift(2.6 * RIGHT + 0.65 * UP)
        invariant = self.block(
            "DEM remains loss-only\nand metric-only",
            PENALTY,
            width=2.9,
            height=0.9,
            scale=0.22,
        ).next_to(final, DOWN, buff=0.65)
        surround = SurroundingRectangle(VGroup(final, invariant), color=ANY, buff=0.35, stroke_width=1.5)

        self.play(FadeIn(nodes), Create(connectors), run_time=1.0)
        self.play(FadeIn(final), FadeIn(invariant), Create(surround), run_time=0.9)
        self.play(Indicate(invariant, color=PENALTY), run_time=0.8)
        self.wait(3.0)
        self.fade_all()

    def construct_sections(self) -> None:
        self.section_00_opening()
        self.section_01_why_rgb()
        self.section_02_anysat_concept()
        self.section_03_anysat_architecture()
        self.section_04_anysat_project()
        self.section_05_panopticon_concept()
        self.section_06_panopticon_architecture()
        self.section_07_panopticon_project()
        self.section_08_comparison()
        self.section_09_physics_pipeline()
        self.section_10_final_roadmap()


class AnySatPanopticonVideo(AnySatPanopticonBase):
    """Master scene containing the complete 8-12 minute explainer."""

    def construct(self) -> None:
        self.construct_sections()


class S00_Opening(AnySatPanopticonBase):
    def construct(self) -> None:
        self.section_00_opening()


class S01_WhyRGBIsProblem(AnySatPanopticonBase):
    def construct(self) -> None:
        self.section_01_why_rgb()


class S02_AnySatConcept(AnySatPanopticonBase):
    def construct(self) -> None:
        self.section_02_anysat_concept()


class S03_AnySatArchitecture(AnySatPanopticonBase):
    def construct(self) -> None:
        self.section_03_anysat_architecture()


class S04_AnySatForOurProject(AnySatPanopticonBase):
    def construct(self) -> None:
        self.section_04_anysat_project()


class S05_PanopticonConcept(AnySatPanopticonBase):
    def construct(self) -> None:
        self.section_05_panopticon_concept()


class S06_PanopticonArchitecture(AnySatPanopticonBase):
    def construct(self) -> None:
        self.section_06_panopticon_architecture()


class S07_PanopticonForOurProject(AnySatPanopticonBase):
    def construct(self) -> None:
        self.section_07_panopticon_project()


class S08_Comparison(AnySatPanopticonBase):
    def construct(self) -> None:
        self.section_08_comparison()


class S09_PhysicsPipeline(AnySatPanopticonBase):
    def construct(self) -> None:
        self.section_09_physics_pipeline()


class S10_FinalRoadmap(AnySatPanopticonBase):
    def construct(self) -> None:
        self.section_10_final_roadmap()
