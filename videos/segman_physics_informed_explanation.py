"""
SegMAN in Physics-Informed Flood Segmentation -- pedagogical Manim video.

A clean, dark-background mathematical explainer covering:
    1.  The flood segmentation problem
    2.  The 15-channel input tensor (DEM is NOT an input)
    3.  The SegMAN-S global pipeline
    4.  Local attention (Q/K/V over a neighborhood)
    5.  State-space / Mamba-like directional scan
    6.  Multi-scale feature fusion
    7.  The segmentation head (logits -> softmax)
    8.  The four loss variants
    9.  Topographic-loss intuition
   10.  The DEM-shuffled ablation
   11.  Transfer to our experiment
   12.  Current (cautious) experimental status

Render (CPU-only, no GPU needed):
    # full video, 1080p60:
    manim -pqh videos/segman_physics_informed_explanation.py SegMANPhysicsVideo

    # quick preview, 480p15 (much faster while iterating):
    manim -pql videos/segman_physics_informed_explanation.py SegMANPhysicsVideo

    # render a single section, e.g. the topographic-loss intuition:
    manim -pqh videos/segman_physics_informed_explanation.py S09_TopoLoss

Requires Manim Community Edition:
    pip install manim
    # plus a LaTeX distribution (MiKTeX on Windows) for the equations.

Voiceover script lives in videos/segman_physics_informed_voiceover.md.
Storyboard lives in videos/segman_physics_informed_storyboard.md.

NOTE: This file only *describes* an animation. It does not load any model,
read any checkpoint, touch any experiment directory, or use the GPU.
"""

from __future__ import annotations

import random

from manim import (
    BLACK,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Arrow,
    Brace,
    Circle,
    Create,
    CurvedArrow,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromCenter,
    Indicate,
    Line,
    MathTex,
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
# Palette  (dark, high-contrast, colour-blind friendly)
# ---------------------------------------------------------------------------
BG_COLOR   = "#0b0f1a"   # near-black navy background
TXT        = "#e5e7eb"   # primary light text
MUTED      = "#94a3b8"   # secondary grey text
WATER      = "#22d3ee"   # cyan  -> water / flood
WATER_DEEP = "#0ea5e9"   # blue  -> deep water / S2
LAND       = "#243042"   # slate -> background / land
SAR        = "#a78bfa"   # violet-> Sentinel-1 / SAR
DEM_CLR    = "#d4a373"   # tan   -> elevation / DEM
PENALTY    = "#ef4444"   # red   -> penalty / inconsistency
WARN       = "#f97316"   # orange-> warning / caution
VALID      = "#22c55e"   # green -> physically coherent
ACCENT     = "#fbbf24"   # amber -> equation highlight

config.background_color = BG_COLOR
Text.set_default(font="Arial")   # DejaVu Sans not installed on this Windows host

random.seed(7)


# ===========================================================================
# Shared helpers (mixin used by the master scene and the per-section scenes)
# ===========================================================================
class SegMANBase(Scene):
    """Common construction helpers and the 12 section methods."""

    # -- small utilities ----------------------------------------------------
    def fade_all(self, run_time: float = 0.6) -> None:
        """Fade out everything currently on screen."""
        mobs = [m for m in self.mobjects]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def section_banner(self, number: str, title: str):
        """Top-left numbered section label that persists during a section."""
        tag = VGroup(
            Text(number, font="DejaVu Sans Mono", color=ACCENT, weight="BOLD").scale(0.45),
            Text(title, font="DejaVu Sans", color=MUTED).scale(0.40),
        ).arrange(RIGHT, buff=0.25)
        tag.to_corner(UP + LEFT, buff=0.4)
        return tag

    def pixel_grid(self, data, cell=0.34, stroke_w=0.6):
        """Build a VGroup grid of coloured squares from a 2D list of colours."""
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
    def _sat_colours(rows, cols):
        """Pseudo-satellite RGB-ish texture (blues/greens/tans)."""
        palette = ["#1e3a5f", "#2a4a6b", "#356985", "#3f7a5a", "#4a6b3f",
                   "#6b5a3f", "#7a6b4a", "#2a4a6b", "#1f2d40"]
        return [[random.choice(palette) for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def _water_mask(rows, cols):
        """A meandering 'river' mask: 1 = water (cyan), 0 = background (slate)."""
        data = [[LAND for _ in range(cols)] for _ in range(rows)]
        # a diagonal-ish river that widens
        path_c = 2.0
        for r in range(rows):
            path_c += random.choice([-0.6, 0.3, 0.9, 1.2])
            width = 1 + (r // 3)
            for w in range(-width, width + 1):
                c = int(round(path_c)) + w
                if 0 <= c < cols:
                    data[r][c] = WATER
        return data

    # =======================================================================
    # SECTION 0 -- Title
    # =======================================================================
    def section_title(self):
        # VOICEOVER: "How does SegMAN see a flood? And can the laws of
        # topography make it see better? Let's build the picture from pixels up."
        title = Text("SegMAN", color=WATER, weight="BOLD").scale(1.5)
        sub = Text("for Physics-Informed Flood Segmentation",
                   color=TXT).scale(0.6)
        group = VGroup(title, sub).arrange(DOWN, buff=0.4)

        underline = Line(LEFT, RIGHT, color=ACCENT).set_width(title.width + 1.0)
        underline.next_to(title, DOWN, buff=0.15)

        self.play(Write(title), run_time=1.4)
        self.play(Create(underline), FadeIn(sub, shift=UP * 0.3), run_time=1.0)
        self.wait(14)
        self.play(FadeOut(underline))
        self.fade_all()

    # =======================================================================
    # SECTION 1 -- The flood segmentation problem
    # =======================================================================
    def section_problem(self):
        # VOICEOVER: "Start with a satellite image. Each pixel is either water
        # or it is not. Our task is binary semantic segmentation: assign every
        # pixel a label -- water, or background."
        banner = self.section_banner("01", "The flood segmentation problem")
        self.play(FadeIn(banner))

        rows, cols = 10, 12
        sat = self.pixel_grid(self._sat_colours(rows, cols))
        sat.scale(0.9).shift(LEFT * 3.2)
        sat_label = Text("Satellite image", color=MUTED).scale(0.45)
        sat_label.next_to(sat, DOWN, buff=0.3)

        mask = self.pixel_grid(self._water_mask(rows, cols))
        mask.scale(0.9).shift(RIGHT * 3.2)
        mask_label = Text("Binary water mask", color=MUTED).scale(0.45)
        mask_label.next_to(mask, DOWN, buff=0.3)

        arrow = Arrow(sat.get_right(), mask.get_left(), color=ACCENT, buff=0.3)
        arrow_lbl = Text("segment", color=ACCENT).scale(0.4)
        arrow_lbl.next_to(arrow, UP, buff=0.1)

        self.play(FadeIn(sat), FadeIn(sat_label), run_time=0.9)
        self.play(GrowArrow(arrow), FadeIn(arrow_lbl))
        self.play(FadeIn(mask), FadeIn(mask_label), run_time=0.9)

        legend = VGroup(
            self._legend_chip(WATER, "1 = water / flood"),
            self._legend_chip(LAND,  "0 = background"),
            self._legend_chip("#000000", "-1 = ignore_index", outline=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        legend.scale(0.85).to_edge(DOWN, buff=0.45)

        self.play(FadeIn(legend, shift=UP * 0.3))
        self.wait(33)
        self.fade_all()

    def _legend_chip(self, colour, label, outline=None):
        sq = Rectangle(width=0.32, height=0.32).set_fill(colour, opacity=1.0)
        sq.set_stroke(outline if outline else colour, width=1.5)
        txt = Text(label, color=TXT).scale(0.42)
        return VGroup(sq, txt).arrange(RIGHT, buff=0.2)

    # =======================================================================
    # SECTION 2 -- The 15-channel input tensor
    # =======================================================================
    def section_input(self):
        # VOICEOVER: "What actually goes in? A stack of fifteen channels:
        # thirteen Sentinel-2 optical bands, and two Sentinel-1 radar channels.
        # Crucially, the elevation map -- the DEM -- is NOT one of them."
        banner = self.section_banner("02", "The input tensor")
        self.play(FadeIn(banner))

        # Build a stacked cube of 15 planes (13 cyan-ish S2 + 2 violet SAR)
        planes = VGroup()
        n = 15
        for i in range(n):
            colour = WATER_DEEP if i < 13 else SAR
            p = Rectangle(width=2.2, height=1.5)
            p.set_fill(colour, opacity=0.85)
            p.set_stroke(BG_COLOR, width=1.2)
            p.shift(RIGHT * 0.13 * i + UP * 0.10 * i)
            planes.add(p)
        planes.move_to(LEFT * 3.0 + UP * 0.3)

        eq = MathTex(r"X \in \mathbb{R}^{15 \times H \times W}", color=TXT).scale(0.9)
        eq.next_to(planes, DOWN, buff=0.7)

        b_s2 = Brace(planes[:13], direction=LEFT, color=WATER_DEEP)
        s2_lbl = Text("13 Sentinel-2 bands", color=WATER_DEEP).scale(0.42)
        s2_lbl.next_to(b_s2, LEFT, buff=0.15)

        b_sar = Brace(planes[13:], direction=UP, color=SAR)
        sar_lbl = Text("2 Sentinel-1 / SAR", color=SAR).scale(0.42)
        sar_lbl.next_to(b_sar, UP, buff=0.1)

        self.play(FadeIn(planes, lag_ratio=0.06, shift=LEFT * 0.2), run_time=1.6)
        self.play(Write(eq))
        self.play(GrowFromCenter(b_s2), FadeIn(s2_lbl),
                  GrowFromCenter(b_sar), FadeIn(sar_lbl))
        self.wait(0.8)

        # DEM shown separately, with a clear "not an input" mark
        dem = Rectangle(width=2.0, height=1.4).set_fill(DEM_CLR, opacity=0.9)
        dem.set_stroke(DEM_CLR, width=1.5)
        dem.shift(RIGHT * 3.6 + UP * 0.4)
        dem_lbl = Text("DEM (elevation)", color=DEM_CLR).scale(0.45)
        dem_lbl.next_to(dem, UP, buff=0.2)

        self.play(FadeIn(dem, shift=RIGHT * 0.2), FadeIn(dem_lbl))

        # red crossed-out arrow from DEM into the cube
        no_arrow = Arrow(dem.get_left(), planes.get_right() + RIGHT * 0.1,
                         color=PENALTY, buff=0.2)
        cross = VGroup(
            Line(UP * 0.25 + LEFT * 0.25, DOWN * 0.25 + RIGHT * 0.25, color=PENALTY),
            Line(UP * 0.25 + RIGHT * 0.25, DOWN * 0.25 + LEFT * 0.25, color=PENALTY),
        ).set_stroke(width=5).move_to(no_arrow.get_center())
        note = Text("DEM is NOT fed to SegMAN", color=PENALTY, weight="BOLD").scale(0.5)
        note.next_to(dem, DOWN, buff=0.5)

        self.play(GrowArrow(no_arrow))
        self.play(Create(cross), Write(note))
        self.wait(0.5)
        sub = Text("(DEM is used only in the topographic loss & metrics)",
                   color=MUTED).scale(0.4)
        sub.next_to(note, DOWN, buff=0.2)
        self.play(FadeIn(sub))
        self.wait(40)
        self.fade_all()

    # =======================================================================
    # SECTION 3 -- The SegMAN global pipeline
    # =======================================================================
    def section_pipeline(self):
        # VOICEOVER: "Here is the whole pipeline. The fifteen-channel input
        # passes through a stem, then the SegMAN encoder, producing features at
        # several scales. A multi-scale decoder fuses them into two logits per
        # pixel, and a softmax turns those into a water probability map."
        banner = self.section_banner("03", "SegMAN-S pipeline")
        self.play(FadeIn(banner))

        def block(label, colour, w=1.7, h=0.9):
            box = RoundedRectangle(corner_radius=0.12, width=w, height=h)
            box.set_fill(colour, opacity=0.25).set_stroke(colour, width=2.5)
            txt = Text(label, color=TXT).scale(0.36)
            txt.move_to(box.get_center())
            return VGroup(box, txt)

        b1 = block("X  (15ch)", WATER_DEEP)
        b2 = block("Stem", MUTED)
        b3 = block("SegMAN\nEncoder", WATER)
        b4 = block("Multi-scale\nfeatures", SAR, w=1.9)
        b5 = block("MMSCopE\nDecoder", ACCENT, w=1.9)
        b6 = block("logits\n[B,2,H,W]", VALID)

        chain = VGroup(b1, b2, b3, b4, b5, b6).arrange(RIGHT, buff=0.45)
        chain.scale(0.92).shift(UP * 0.3)

        arrows = VGroup()
        for a, b in zip(chain[:-1], chain[1:]):
            arrows.add(Arrow(a.get_right(), b.get_left(), color=MUTED,
                             buff=0.08, stroke_width=3, max_tip_length_to_length_ratio=0.25))

        self.play(FadeIn(b1))
        for box, arr in zip(chain[1:], arrows):
            self.play(GrowArrow(arr), FadeIn(box, shift=RIGHT * 0.2), run_time=0.45)

        # softmax tail -> water mask
        softmax = MathTex(r"\xrightarrow{\ \text{softmax}\ }", color=ACCENT).scale(0.7)
        mini_mask = self.pixel_grid(self._water_mask(6, 6), cell=0.22).scale(0.9)
        tail = VGroup(softmax, mini_mask).arrange(RIGHT, buff=0.2)
        tail.next_to(chain, DOWN, buff=0.9).shift(RIGHT * 2.0)
        link = Arrow(b6.get_bottom(), tail.get_top(), color=MUTED, buff=0.15)
        mask_lbl = Text("water mask", color=WATER).scale(0.4)
        mask_lbl.next_to(mini_mask, DOWN, buff=0.15)

        self.play(GrowArrow(link), Write(softmax), FadeIn(mini_mask), FadeIn(mask_lbl))
        self.wait(1.4)

        # Highlight the two SegMAN ingredients we'll zoom into next
        hl = SurroundingRectangle(VGroup(b3, b4), color=ACCENT, buff=0.12)
        hl_lbl = Text("local attention  +  state-space scan", color=ACCENT).scale(0.42)
        hl_lbl.next_to(hl, UP, buff=0.2)
        self.play(Create(hl), FadeIn(hl_lbl))
        self.wait(35)
        self.fade_all()

    # =======================================================================
    # SECTION 4 -- Local attention
    # =======================================================================
    def section_attention(self):
        # VOICEOVER: "Inside the encoder, local attention lets each pixel look
        # at its neighbours. For a centre pixel i, we form a query; each
        # neighbour j contributes a key and a value. The dot-product of query
        # and key, scaled and soft-maxed, decides how much each neighbour
        # matters -- so the model learns which nearby pixels signal water."
        banner = self.section_banner("04", "Local attention")
        self.play(FadeIn(banner))

        # 5x5 neighbourhood grid, centre highlighted
        cell = 0.6
        grid = VGroup()
        centre = None
        for r in range(5):
            for c in range(5):
                sq = Rectangle(width=cell, height=cell)
                is_centre = (r == 2 and c == 2)
                sq.set_fill(WATER if is_centre else LAND, opacity=0.9)
                sq.set_stroke(MUTED, width=1.0)
                sq.move_to([c * cell, -r * cell, 0])
                grid.add(sq)
                if is_centre:
                    centre = sq
        grid.move_to(LEFT * 3.4)

        i_lbl = MathTex("i", color=WATER).scale(0.7).move_to(centre.get_center())
        nbhd = Text("neighborhood  N(i)", color=MUTED).scale(0.42)
        nbhd.next_to(grid, DOWN, buff=0.35)

        self.play(FadeIn(grid), FadeIn(i_lbl))
        self.play(FadeIn(nbhd))

        # arrows from the 8 neighbours to centre with varying weight
        neighbours = [grid[r * 5 + c] for (r, c) in
                      [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]]
        weights = [0.25, 0.9, 0.3, 0.55, 0.8, 0.2, 0.95, 0.4]
        arrs = VGroup()
        for nb, w in zip(neighbours, weights):
            a = Arrow(nb.get_center(), centre.get_center(),
                      color=ACCENT, buff=0.18,
                      stroke_width=1 + 6 * w,
                      max_tip_length_to_length_ratio=0.2)
            a.set_opacity(0.35 + 0.65 * w)
            arrs.add(a)
        self.play(*[GrowArrow(a) for a in arrs], run_time=1.2)

        # Q / K / V equations
        qkv = VGroup(
            MathTex(r"q_i = W_Q\, x_i", color=WATER),
            MathTex(r"k_j = W_K\, x_j", color=ACCENT),
            MathTex(r"v_j = W_V\, x_j", color=SAR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).scale(0.8)
        qkv.shift(RIGHT * 3.3 + UP * 1.0)

        attn = MathTex(
            r"\mathrm{Attention}(i) = \sum_{j \in N(i)}",
            r"\mathrm{softmax}\!\left(\frac{q_i^{\top} k_j}{\sqrt{d}}\right)",
            r"v_j",
            color=TXT,
        ).scale(0.62)
        attn[1].set_color(ACCENT)
        attn[2].set_color(SAR)
        attn.next_to(qkv, DOWN, buff=0.7).shift(LEFT * 0.2)

        self.play(Write(qkv), run_time=1.4)
        self.play(Write(attn), run_time=1.6)
        self.wait(0.5)
        intuition = Text("which nearby pixels matter for 'is i water?'",
                         color=MUTED).scale(0.42)
        intuition.next_to(attn, DOWN, buff=0.5)
        self.play(FadeIn(intuition), Indicate(centre, color=WATER))
        self.wait(40)
        self.fade_all()

    # =======================================================================
    # SECTION 5 -- State-space / Mamba-like scan
    # =======================================================================
    def section_statespace(self):
        # VOICEOVER: "Attention is local. To carry information across the whole
        # image cheaply, SegMAN also uses a state-space scan -- a Mamba-like
        # recurrence. A hidden state marches along the tokens, updated at each
        # step, propagating long-range context in linear time."
        banner = self.section_banner("05", "State-space scan")
        self.play(FadeIn(banner))

        # a row of tokens
        n = 9
        tokens = VGroup()
        for t in range(n):
            sq = Rectangle(width=0.7, height=0.7)
            sq.set_fill(LAND, opacity=0.9).set_stroke(MUTED, width=1.0)
            tokens.add(sq)
        tokens.arrange(RIGHT, buff=0.12).shift(UP * 1.4)
        tok_lbl = Text("tokens  x_t", color=MUTED).scale(0.42).next_to(tokens, UP, buff=0.25)

        self.play(FadeIn(tokens, lag_ratio=0.1), FadeIn(tok_lbl))

        # hidden-state dot sweeping left -> right with a trailing bar
        state = Circle(radius=0.22, color=WATER).set_fill(WATER, opacity=0.8)
        state.move_to(tokens[0].get_center() + DOWN * 1.1)
        h_lbl = MathTex("h_t", color=WATER).scale(0.6).next_to(state, DOWN, buff=0.15)
        hstate = VGroup(state, h_lbl)
        self.play(FadeIn(hstate))

        eqs = VGroup(
            MathTex(r"h_t = f(h_{t-1},\, x_t)", color=WATER),
            MathTex(r"y_t = g(h_t)", color=VALID),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.8)
        eqs.shift(DOWN * 1.5)
        self.play(Write(eqs))

        for t in range(n):
            target = tokens[t].get_center() + DOWN * 1.1
            link = Arrow(tokens[t].get_bottom(), target + UP * 0.22,
                         color=ACCENT, buff=0.05, stroke_width=2.5,
                         max_tip_length_to_length_ratio=0.3)
            self.play(
                hstate.animate.move_to(target).shift(DOWN * 0.0),
                tokens[t].animate.set_fill(WATER_DEEP, opacity=0.9),
                GrowArrow(link),
                run_time=0.35,
            )
            self.remove(link)
        # keep h_lbl under the dot
        self.wait(0.4)

        # multi-directional scans
        dirs = VGroup(
            Text("left -> right", color=ACCENT).scale(0.4),
            Text("right -> left", color=ACCENT).scale(0.4),
            Text("top -> bottom", color=ACCENT).scale(0.4),
        ).arrange(RIGHT, buff=0.6)
        dirs.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(dirs, shift=UP * 0.2))
        caption = Text("long-range spatial context, linear cost",
                       color=MUTED).scale(0.42)
        caption.next_to(eqs, RIGHT, buff=0.8)
        self.play(FadeIn(caption))
        self.wait(35)
        self.fade_all()

    # =======================================================================
    # SECTION 6 -- Multi-scale fusion
    # =======================================================================
    def section_multiscale(self):
        # VOICEOVER: "The encoder produces features at decreasing resolution --
        # a quarter, an eighth, a sixteenth, a thirty-second of the image. The
        # decoder fuses these scales: coarse maps give context, fine maps give
        # sharp boundaries, together yielding one dense prediction."
        banner = self.section_banner("06", "Multi-scale fusion")
        self.play(FadeIn(banner))

        scales = [("H/4", 1.8, WATER), ("H/8", 1.4, WATER_DEEP),
                  ("H/16", 1.0, SAR), ("H/32", 0.7, ACCENT)]
        maps = VGroup()
        for name, size, colour in scales:
            sq = Rectangle(width=size, height=size)
            sq.set_fill(colour, opacity=0.25).set_stroke(colour, width=2.2)
            lbl = Text(name, color=colour).scale(0.4).next_to(sq, DOWN, buff=0.15)
            maps.add(VGroup(sq, lbl))
        maps.arrange(RIGHT, buff=0.7, aligned_edge=UP).shift(LEFT * 1.5 + UP * 0.4)

        self.play(FadeIn(maps, lag_ratio=0.2, shift=UP * 0.2), run_time=1.4)

        out = Rectangle(width=1.6, height=1.6)
        out.set_fill(VALID, opacity=0.25).set_stroke(VALID, width=2.5)
        out_lbl = Text("dense prediction", color=VALID).scale(0.42)
        out_lbl.next_to(out, DOWN, buff=0.15)
        out_grp = VGroup(out, out_lbl).shift(RIGHT * 4.2 + UP * 0.4)

        merge = VGroup()
        for m in maps:
            merge.add(Arrow(m[0].get_right(), out.get_left(), color=MUTED,
                            buff=0.15, stroke_width=2.5,
                            max_tip_length_to_length_ratio=0.15))
        self.play(*[GrowArrow(a) for a in merge], FadeIn(out_grp), run_time=1.3)
        self.wait(26)
        self.fade_all()

    # =======================================================================
    # SECTION 7 -- Segmentation head
    # =======================================================================
    def section_head(self):
        # VOICEOVER: "At the head, the network emits two logit maps per pixel:
        # one for background, one for water. A softmax converts them into the
        # water probability -- a number between zero and one at every pixel."
        banner = self.section_banner("07", "Segmentation head")
        self.play(FadeIn(banner))

        z_bg = Rectangle(width=2.0, height=2.0).set_fill(LAND, opacity=0.9)
        z_bg.set_stroke(MUTED, width=1.5)
        z_bg_lbl = MathTex(r"Z_{\text{background}}", color=MUTED).scale(0.55)
        z_bg_lbl.next_to(z_bg, UP, buff=0.2)
        bg_grp = VGroup(z_bg, z_bg_lbl).shift(LEFT * 4.0 + UP * 0.3)

        z_w = Rectangle(width=2.0, height=2.0).set_fill(WATER_DEEP, opacity=0.6)
        z_w.set_stroke(WATER, width=1.5)
        z_w_lbl = MathTex(r"Z_{\text{water}}", color=WATER).scale(0.55)
        z_w_lbl.next_to(z_w, UP, buff=0.2)
        w_grp = VGroup(z_w, z_w_lbl).shift(LEFT * 1.4 + UP * 0.3)

        self.play(FadeIn(bg_grp), FadeIn(w_grp))

        softmax_eq = MathTex(
            r"p_{\text{water}}(i) = ",
            r"\frac{e^{\,z_{\text{water}}(i)}}"
            r"{e^{\,z_{\text{background}}(i)} + e^{\,z_{\text{water}}(i)}}",
            color=TXT,
        ).scale(0.7)
        softmax_eq[1].set_color(WATER)
        softmax_eq.shift(RIGHT * 2.2 + UP * 0.6)

        # probability heatmap (cyan intensity)
        prob_rows = [[WATER if random.random() > 0.55 else LAND for _ in range(6)]
                     for _ in range(6)]
        prob = self.pixel_grid(prob_rows, cell=0.32)
        prob.shift(RIGHT * 3.0 + DOWN * 1.4)
        prob_lbl = Text("p_water  in [0,1]", color=WATER).scale(0.42)
        prob_lbl.next_to(prob, DOWN, buff=0.15)

        arr = Arrow(VGroup(bg_grp, w_grp).get_right(), softmax_eq.get_left(),
                    color=ACCENT, buff=0.3)
        self.play(GrowArrow(arr), Write(softmax_eq), run_time=1.6)
        self.play(FadeIn(prob), FadeIn(prob_lbl))
        self.wait(22)
        self.fade_all()

    # =======================================================================
    # SECTION 8 -- The four loss variants
    # =======================================================================
    def section_losses(self):
        # VOICEOVER: "Now the heart of the study. We hold the architecture fixed
        # and change only the loss. Four variants: cross-entropy; Dice plus
        # cross-entropy; that plus a topographic term using the real DEM; and a
        # control where the DEM is shuffled. Lambda-topo is fixed at one half."
        banner = self.section_banner("08", "Four loss variants")
        self.play(FadeIn(banner))

        def loss_card(title, eq, colour):
            box = RoundedRectangle(corner_radius=0.12, width=5.6, height=1.15)
            box.set_fill(colour, opacity=0.12).set_stroke(colour, width=2.0)
            t = Text(title, color=colour, weight="BOLD").scale(0.4)
            e = MathTex(eq, color=TXT).scale(0.6)
            content = VGroup(t, e).arrange(DOWN, buff=0.18)
            content.move_to(box.get_center())
            return VGroup(box, content)

        c1 = loss_card("CE", r"\mathcal{L} = \mathcal{L}_{CE}", MUTED)
        c2 = loss_card("Dice + CE",
                       r"\mathcal{L}_{DiceCE} = \mathcal{L}_{Dice} + \alpha\,\mathcal{L}_{CE}",
                       WATER)
        c3 = loss_card("Dice + CE + Topo  (real DEM)",
                       r"\mathcal{L}_{total} = \mathcal{L}_{DiceCE} "
                       r"+ \lambda_{topo}\,\mathcal{L}_{topo}(\mathrm{DEM}_{real})",
                       VALID)
        c4 = loss_card("Dice + CE + Topo  (shuffled)",
                       r"\mathcal{L}_{shuffled} = \mathcal{L}_{DiceCE} "
                       r"+ \lambda_{topo}\,\mathcal{L}_{topo}(\mathrm{DEM}_{shuffled})",
                       WARN)

        cards = VGroup(c1, c2, c3, c4).arrange(DOWN, buff=0.28).scale(0.92)
        cards.shift(UP * 0.1)

        for c in cards:
            self.play(FadeIn(c, shift=RIGHT * 0.2), run_time=0.5)

        lam = MathTex(r"\lambda_{topo} = 0.5", color=ACCENT).scale(0.8)
        lam.to_edge(DOWN, buff=0.35)
        self.play(Write(lam), Indicate(c3, color=VALID), Indicate(c4, color=WARN))
        self.wait(35)
        self.fade_all()

    # =======================================================================
    # SECTION 9 -- Topographic-loss intuition
    # =======================================================================
    def section_topo(self):
        # VOICEOVER: "What does the topographic term mean? Take two neighbours.
        # If pixel i sits higher than pixel j by more than a margin, yet the
        # model calls the HIGH pixel water and the LOW pixel dry -- that is
        # physically suspicious. Water does not perch above dry ground. We
        # penalise exactly that pattern: p_i times one-minus-p_j."
        banner = self.section_banner("09", "Topographic-loss intuition")
        self.play(FadeIn(banner))

        # elevation bars
        ground = Line(LEFT * 5, RIGHT * 5, color=MUTED).shift(DOWN * 2.2)
        bar_i = Rectangle(width=1.4, height=2.6).set_fill(DEM_CLR, opacity=0.6)
        bar_i.set_stroke(DEM_CLR, width=2).next_to(ground, UP, buff=0).shift(LEFT * 2.6)
        bar_j = Rectangle(width=1.4, height=1.2).set_fill(DEM_CLR, opacity=0.6)
        bar_j.set_stroke(DEM_CLR, width=2).next_to(ground, UP, buff=0).shift(RIGHT * 2.6)

        hi_lbl = MathTex("h_i", color=DEM_CLR).scale(0.7).next_to(bar_i, LEFT, buff=0.2)
        hj_lbl = MathTex("h_j", color=DEM_CLR).scale(0.7).next_to(bar_j, RIGHT, buff=0.2)

        self.play(Create(ground))
        self.play(FadeIn(bar_i), FadeIn(bar_j), Write(hi_lbl), Write(hj_lbl))

        cond = MathTex(r"h_i > h_j + \text{margin}", color=DEM_CLR).scale(0.7)
        cond.to_edge(UP, buff=1.2)
        self.play(Write(cond))

        # pixel i = water (blue) on top of tall bar, pixel j = dry on short bar
        pix_i = Rectangle(width=1.0, height=1.0).set_fill(WATER, opacity=1.0)
        pix_i.set_stroke(WATER, width=2).next_to(bar_i, UP, buff=0.1)
        pix_j = Rectangle(width=1.0, height=1.0).set_fill(LAND, opacity=1.0)
        pix_j.set_stroke(MUTED, width=2).next_to(bar_j, UP, buff=0.1)
        pi_lbl = MathTex(r"p_i \approx 1", color=WATER).scale(0.55).next_to(pix_i, UP, buff=0.15)
        pj_lbl = MathTex(r"p_j \approx 0", color=MUTED).scale(0.55).next_to(pix_j, UP, buff=0.15)

        self.play(FadeIn(pix_i), FadeIn(pix_j), Write(pi_lbl), Write(pj_lbl))
        self.wait(0.5)

        # penalty flash
        penalty_eq = MathTex(r"\text{penalty} \;\propto\; p_i\,(1 - p_j)",
                             color=PENALTY).scale(0.8)
        penalty_eq.shift(DOWN * 3.0)
        warn = Text("water perched above dry ground  =  inconsistent",
                    color=PENALTY).scale(0.45)
        warn.next_to(cond, DOWN, buff=0.3)
        flash = SurroundingRectangle(pix_i, color=PENALTY, buff=0.08)
        self.play(Create(flash), Write(penalty_eq), FadeIn(warn))
        self.play(Indicate(penalty_eq, color=PENALTY), Indicate(flash, color=PENALTY))
        self.wait(1.2)

        # coherent case: swap -> water low, dry high -> green, penalty ~ 0
        coherent = Text("physically coherent  ->  no penalty",
                        color=VALID).scale(0.45).next_to(cond, DOWN, buff=0.3)
        self.play(
            pix_i.animate.set_fill(LAND, opacity=1.0).set_stroke(MUTED),
            pix_j.animate.set_fill(WATER, opacity=1.0).set_stroke(WATER),
            Transform(pi_lbl, MathTex(r"p_i \approx 0", color=MUTED).scale(0.55).move_to(pi_lbl)),
            Transform(pj_lbl, MathTex(r"p_j \approx 1", color=WATER).scale(0.55).move_to(pj_lbl)),
            FadeOut(warn), FadeOut(flash),
            FadeIn(coherent),
        )
        ok = SurroundingRectangle(pix_j, color=VALID, buff=0.08)
        zero = MathTex(r"p_i\,(1 - p_j) \approx 0", color=VALID).scale(0.8)
        zero.move_to(penalty_eq)
        self.play(Create(ok), Transform(penalty_eq, zero))
        self.wait(43)
        self.fade_all()

    # =======================================================================
    # SECTION 10 -- The DEM-shuffled ablation
    # =======================================================================
    def section_ablation(self):
        # VOICEOVER: "But does the network actually use the physics? We run a
        # control. In one branch the topographic loss sees the REAL elevation;
        # in the other, the elevation map is shuffled across samples. If the
        # real DEM helps more, the model is exploiting topography. If shuffling
        # does just as well, the term is mostly a regulariser."
        banner = self.section_banner("10", "DEM-shuffled ablation")
        self.play(FadeIn(banner))

        def branch(title, colour, shift):
            box = RoundedRectangle(corner_radius=0.14, width=4.6, height=3.2)
            box.set_fill(colour, opacity=0.10).set_stroke(colour, width=2.2)
            t = Text(title, color=colour, weight="BOLD").scale(0.46)
            t.next_to(box.get_top(), DOWN, buff=0.25)
            box_grp = VGroup(box, t).shift(shift)
            return box_grp

        real = branch("Real DEM", VALID, LEFT * 3.2)
        shuf = branch("Shuffled DEM", WARN, RIGHT * 3.2)
        self.play(FadeIn(real), FadeIn(shuf))

        # real: aligned terrain + mask; shuffled: mismatched
        real_eq = MathTex(r"\mathcal{L}_{topo}(\mathrm{DEM}_{real})",
                          color=VALID).scale(0.6).move_to(real).shift(DOWN * 0.4)
        shuf_eq = MathTex(r"\mathcal{L}_{topo}(\mathrm{DEM}_{shuffled})",
                          color=WARN).scale(0.6).move_to(shuf).shift(DOWN * 0.4)
        real_note = Text("elevation matches the scene", color=MUTED).scale(0.38)
        real_note.move_to(real).shift(DOWN * 1.1)
        shuf_note = Text("elevation from a different tile", color=MUTED).scale(0.38)
        shuf_note.move_to(shuf).shift(DOWN * 1.1)
        self.play(Write(real_eq), Write(shuf_eq))
        self.play(FadeIn(real_note), FadeIn(shuf_note))

        # the decision logic
        logic = VGroup(
            Text("real  >  shuffled   ->   model exploits physical topography",
                 color=VALID).scale(0.42),
            Text("real  ~=  shuffled   ->   topo term acts as regularization",
                 color=WARN).scale(0.42),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        logic.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(logic, shift=UP * 0.2))
        self.wait(25)
        self.fade_all()

    # =======================================================================
    # SECTION 11 -- Transfer to our experiment
    # =======================================================================
    def section_transfer(self):
        # VOICEOVER: "Putting it together: SegMAN-S, fifteen-channel Sentinel
        # input, Dice plus cross-entropy, and the topographic loss with the DEM
        # kept strictly out of the model. The architecture is not the
        # contribution. The question is whether a physics-informed loss
        # improves or stabilises segmentation when the architecture is fixed."
        banner = self.section_banner("11", "Transfer to our experiment")
        self.play(FadeIn(banner))

        def node(label, colour):
            box = RoundedRectangle(corner_radius=0.12, width=3.0, height=0.9)
            box.set_fill(colour, opacity=0.18).set_stroke(colour, width=2.2)
            t = Text(label, color=TXT).scale(0.38).move_to(box.get_center())
            return VGroup(box, t)

        n1 = node("SegMAN-S  (fixed)", WATER)
        n2 = node("15-channel  S1 + S2", WATER_DEEP)
        n3 = node("Dice + CE  loss", VALID)
        n4 = node("Topographic loss", DEM_CLR)
        col = VGroup(n1, n2, n3, n4).arrange(DOWN, buff=0.4).shift(LEFT * 3.0)

        plus = MathTex("+", color=MUTED).scale(0.8)
        self.play(FadeIn(n1), FadeIn(n2))
        self.play(FadeIn(n3), FadeIn(n4))

        # converge arrows into a result
        result = node("flood mask", WATER)
        result[0].set_height(1.4)
        result.shift(RIGHT * 3.4)
        arrs = VGroup(*[Arrow(n.get_right(), result.get_left(), color=MUTED,
                              buff=0.2, stroke_width=2.2,
                              max_tip_length_to_length_ratio=0.12) for n in col])
        self.play(*[GrowArrow(a) for a in arrs], FadeIn(result))

        thesis = VGroup(
            Text("Architecture is NOT the contribution.", color=MUTED).scale(0.46),
            Text("Question: does a physics-informed loss help at fixed architecture?",
                 color=ACCENT).scale(0.46),
        ).arrange(DOWN, buff=0.2)
        thesis.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(thesis, shift=UP * 0.2))
        self.wait(22)
        self.fade_all()

    # =======================================================================
    # SECTION 12 -- Current experimental status (cautious)
    # =======================================================================
    def section_status(self):
        # VOICEOVER: "Where do we stand? On seed zero, every variant trained
        # stably -- no collapse. But the shuffled-DEM control actually beat the
        # real DEM. So we cannot yet claim a physical effect. Multi-seed
        # experiments at fifty training samples are running now to test whether
        # any difference survives across random seeds."
        banner = self.section_banner("12", "Current status")
        self.play(FadeIn(banner))

        title = Text("Where we stand", color=TXT, weight="BOLD").scale(0.6)
        title.to_edge(UP, buff=1.0)
        self.play(Write(title))

        bullets = VGroup(
            self._status_line(VALID, "Seed 0: all variants converged, no collapse."),
            self._status_line(WARN,  "Seed 0: DEM-shuffled OUTPERFORMED real DEM."),
            self._status_line(PENALTY, "=> No strong physical claim can be made yet."),
            self._status_line(WATER, "Multi-seed N=50 runs in progress (robustness test)."),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        bullets.shift(UP * 0.1)

        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.25), run_time=0.6)
        self.wait(1.0)

        caution = Text("Regularization vs. physics: still an open question.",
                       color=ACCENT).scale(0.5)
        caution.to_edge(DOWN, buff=0.7)
        self.play(Write(caution))
        self.wait(2.0)

        # outro
        self.play(FadeOut(bullets), FadeOut(title), FadeOut(banner), FadeOut(caution))
        outro = VGroup(
            Text("SegMAN", color=WATER, weight="BOLD").scale(1.1),
            Text("physics-informed flood segmentation", color=MUTED).scale(0.5),
        ).arrange(DOWN, buff=0.3)
        self.play(FadeIn(outro))
        self.wait(28)
        self.fade_all()

    def _status_line(self, colour, text):
        dot = Dot(color=colour, radius=0.09)
        t = Text(text, color=TXT).scale(0.46)
        return VGroup(dot, t).arrange(RIGHT, buff=0.3)


# ===========================================================================
# Master scene -- the full 6-8 minute video
# ===========================================================================
class SegMANPhysicsVideo(SegMANBase):
    def construct(self):
        self.section_title()
        self.section_problem()
        self.section_input()
        self.section_pipeline()
        self.section_attention()
        self.section_statespace()
        self.section_multiscale()
        self.section_head()
        self.section_losses()
        self.section_topo()
        self.section_ablation()
        self.section_transfer()
        self.section_status()


# ===========================================================================
# Per-section scenes -- render any chapter on its own while iterating
# ===========================================================================
class S00_Title(SegMANBase):
    def construct(self): self.section_title()


class S01_Problem(SegMANBase):
    def construct(self): self.section_problem()


class S02_Input(SegMANBase):
    def construct(self): self.section_input()


class S03_Pipeline(SegMANBase):
    def construct(self): self.section_pipeline()


class S04_Attention(SegMANBase):
    def construct(self): self.section_attention()


class S05_StateSpace(SegMANBase):
    def construct(self): self.section_statespace()


class S06_MultiScale(SegMANBase):
    def construct(self): self.section_multiscale()


class S07_Head(SegMANBase):
    def construct(self): self.section_head()


class S08_Losses(SegMANBase):
    def construct(self): self.section_losses()


class S09_TopoLoss(SegMANBase):
    def construct(self): self.section_topo()


class S10_Ablation(SegMANBase):
    def construct(self): self.section_ablation()


class S11_Transfer(SegMANBase):
    def construct(self): self.section_transfer()


class S12_Status(SegMANBase):
    def construct(self): self.section_status()
