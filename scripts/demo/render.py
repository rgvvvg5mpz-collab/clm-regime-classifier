"""Frame renderer for the CLASP walkthrough: camera moves, spotlights, crossfades, lower-thirds.

Each shot is a 3840x2160 still plus an optional spotlight rectangle (in source pixels). Frames are
rendered at the output size and piped to ffmpeg as raw RGB:

  * camera   UI shots push in on the spotlight (eased, ~1.1 s), continuing from the previous shot's
             camera within a scene; slides get a slow 3.5 % Ken Burns drift.
  * spotlight the rest of the frame dims to navy at 36 %, with a rounded gold ring and a soft glow;
             eases in after the camera starts moving.
  * crossfade 0.45 s dissolve between consecutive shots (dip from black at the very start / to black at the end).
  * chapter card a 1.5 s navy title card ("1 · Train") between the UI chapters, so no overlay ever
             covers the interface.

Static stretches (camera settled, no overlay animating) reuse the previous frame's bytes, so a
nine-minute video renders in a few minutes.
"""
from __future__ import annotations

import math
import subprocess

from PIL import Image, ImageDraw, ImageFilter, ImageFont

NAVY, GOLD, BRAND = (11, 42, 74), (200, 162, 77), (150, 21, 29)
SF = "/System/Library/Fonts/SFNS.ttf"


def ease(t: float) -> float:   # easeInOutCubic on [0, 1]
    t = min(1.0, max(0.0, t))
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def font(size: int, weight: str = "Semibold") -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(SF, size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


class Renderer:
    def __init__(self, ffmpeg: str, out_path: str, width: int = 1920, height: int = 1080, fps: int = 30, crf: int = 24):
        self.W, self.H, self.fps = width, height, fps
        self.s = width / 1920                      # overlay scale factor
        self.proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-tune", "animation", "-profile:v", "high",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
        self.prev_last: Image.Image | None = None   # last rendered frame of the previous shot
        self.cam = None                              # (cx, cy, zoom) at the end of the previous shot
        self.frames = 0
        self.f_title = font(round(34 * self.s), "Semibold")
        self.f_num = font(round(34 * self.s), "Bold")
        self.f_card = font(round(76 * self.s), "Bold")
        self.f_card_num = font(round(58 * self.s), "Heavy")
        self.f_card_sub = font(round(28 * self.s), "Medium")

    # ------------------------------------------------------------------ geometry
    def _target_cam(self, size, rect):
        sw, sh = size
        if not rect:
            return (sw / 2, sh / 2, 1.0)
        x0, y0, x1, y1 = rect
        rw, rh = max(1, x1 - x0), max(1, y1 - y0)
        z = min(sw / (rw * 1.45), sh / (rh * 1.45))
        z = max(1.0, min(1.55, z))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return self._clamp(size, cx, cy, z)

    @staticmethod
    def _clamp(size, cx, cy, z):
        sw, sh = size
        hw, hh = sw / z / 2, sh / z / 2
        return (min(max(cx, hw), sw - hw), min(max(cy, hh), sh - hh), z)

    def _frame(self, img: Image.Image, cam, rect, spot_a: float) -> Image.Image:
        sw, sh = img.size
        cx, cy, z = cam
        cw, ch = sw / z, sh / z
        left, top = cx - cw / 2, cy - ch / 2
        fr = img.resize((self.W, self.H), Image.BICUBIC, box=(left, top, left + cw, top + ch), reducing_gap=2.0)
        if rect and spot_a > 0.01:
            k = self.W / cw
            x0, y0, x1, y1 = [(rect[0] - left) * k, (rect[1] - top) * k, (rect[2] - left) * k, (rect[3] - top) * k]
            pad, r = 10 * self.s, 16 * self.s
            box = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
            dim = Image.new("L", (self.W, self.H), int(255 * 0.36 * spot_a))
            ImageDraw.Draw(dim).rounded_rectangle(box, radius=r, fill=0)
            fr = Image.composite(Image.new("RGB", fr.size, NAVY), fr, dim)
            glow = Image.new("L", (self.W, self.H), 0)
            ImageDraw.Draw(glow).rounded_rectangle(box, radius=r, outline=int(150 * spot_a), width=round(10 * self.s))
            glow = glow.filter(ImageFilter.GaussianBlur(9 * self.s))
            fr = Image.composite(Image.new("RGB", fr.size, GOLD), fr, glow)
            ring = Image.new("L", (self.W, self.H), 0)
            ImageDraw.Draw(ring).rounded_rectangle(box, radius=r, outline=int(255 * spot_a), width=max(2, round(4 * self.s)))
            fr = Image.composite(Image.new("RGB", fr.size, GOLD), fr, ring)
        return fr

    def _lower_third(self, fr: Image.Image, title: str, t: float) -> Image.Image:
        show = ease((t - 0.25) / 0.4) * (1 - ease((t - 3.6) / 0.4))
        if show <= 0.01:
            return fr
        num, _, rest = title.partition(" · ")
        has_num = bool(rest)
        txt = rest if has_num else title
        d = ImageDraw.Draw(fr)
        tw = d.textlength(txt, font=self.f_title)
        nw = d.textlength(num, font=self.f_num) if has_num else 0
        padx, h = 26 * self.s, 66 * self.s
        circle = h * 0.62
        w = padx * 2 + tw + (circle + 16 * self.s if has_num else 0)
        x = 64 * self.s - (1 - show) * 40 * self.s
        y = self.H - 86 * self.s - h
        over = Image.new("RGBA", fr.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(over)
        a = int(235 * show)
        od.rounded_rectangle((x, y, x + w, y + h), radius=h / 2, fill=NAVY + (a,))
        od.rounded_rectangle((x, y + h - 5 * self.s, x + w * show, y + h), radius=3 * self.s, fill=GOLD + (a,))
        tx = x + padx
        if has_num:
            cy = y + h / 2
            od.ellipse((tx, cy - circle / 2, tx + circle, cy + circle / 2), fill=BRAND + (int(255 * show),))
            od.text((tx + circle / 2 - nw / 2, cy), num, font=self.f_num, fill=(255, 255, 255, int(255 * show)), anchor="lm")
            tx += circle + 16 * self.s
        od.text((tx, y + h / 2), txt, font=self.f_title, fill=(255, 255, 255, int(255 * show)), anchor="lm")
        return Image.alpha_composite(fr.convert("RGBA"), over).convert("RGB")

    def card(self, title: str, frames: int, subtitle: str = "", first: bool = False) -> None:
        """Navy chapter card: numbered circle + title, gold rule drawing in, dissolve in from the previous shot."""
        num, _, rest = title.partition(" · ")
        has_num = bool(rest)
        txt = rest if has_num else title
        nx = int(0.45 * self.fps)
        for f in range(frames):
            t = f / self.fps
            a = ease(t / 0.5)
            fr = Image.new("RGB", (self.W, self.H), NAVY)
            d = ImageDraw.Draw(fr)
            tw = d.textlength(txt, font=self.f_card)
            circ = 112 * self.s if has_num else 0
            gap = 40 * self.s if has_num else 0
            total = circ + gap + tw
            x = (self.W - total) / 2 + (1 - a) * 30 * self.s
            cy = self.H / 2 - 18 * self.s
            over = Image.new("RGBA", fr.size, (0, 0, 0, 0)); od = ImageDraw.Draw(over)
            if has_num:
                od.ellipse((x, cy - circ / 2, x + circ, cy + circ / 2), fill=BRAND + (int(255 * a),))
                od.text((x + circ / 2, cy), num, font=self.f_card_num, fill=(255, 255, 255, int(255 * a)), anchor="mm")
            od.text((x + circ + gap, cy), txt, font=self.f_card, fill=(255, 255, 255, int(255 * a)), anchor="lm")
            rule = ease((t - 0.25) / 0.6) * total
            od.rounded_rectangle((x, cy + 84 * self.s, x + rule, cy + 90 * self.s), radius=3 * self.s, fill=GOLD + (int(255 * a),))
            if subtitle:
                od.text((x, cy + 128 * self.s), subtitle, font=self.f_card_sub, fill=(185, 196, 210, int(255 * a)), anchor="lm")
            fr = Image.alpha_composite(fr.convert("RGBA"), over).convert("RGB")
            if f < nx and self.prev_last is not None and not first:
                fr = Image.blend(self.prev_last, fr, ease((f + 1) / nx))
            self.proc.stdin.write(fr.tobytes()); self.frames += 1
            last = fr
        self.prev_last = last
        self.cam = None

    # ------------------------------------------------------------------ shots
    def shot(self, png: str, frames: int, rect=None, kind: str = "ui", new_scene: bool = False,
             title: str | None = None, first: bool = False, last: bool = False, xfade: float = 0.45) -> None:
        img = Image.open(png).convert("RGB")
        size = img.size
        if kind == "slide":
            start, end = (size[0] / 2, size[1] / 2, 1.0), (size[0] / 2, size[1] / 2, 1.035)
            move_t0, move_dur = 0.0, frames / self.fps
            rect = None
        else:
            end = self._target_cam(size, rect)
            start = self.cam if (self.cam and not new_scene and self.cam[2] >= 1.0) else (size[0] / 2, size[1] / 2, 1.0)
            start = self._clamp(size, *start)
            move_t0, move_dur = 0.15, 1.1
        nx = 0 if first else int(xfade * self.fps)
        fade_out = int(0.5 * self.fps) if last else 0
        cached, cached_key = None, None
        for f in range(frames):
            t = f / self.fps
            p = ease((t - move_t0) / move_dur) if kind != "slide" else t / max(move_dur, 1e-6)
            cam = tuple(start[i] + (end[i] - start[i]) * p for i in range(3))
            spot = ease((t - 0.55) / 0.45) if rect else 0.0
            animating = (kind == "slide") or p < 1 or (rect and spot < 1) or (title and t < 4.1) or f < nx or (first and f < 15) \
                or (fade_out and f >= frames - fade_out)
            if not animating and cached is not None:
                self.proc.stdin.write(cached); self.frames += 1
                continue
            fr = self._frame(img, cam, rect, spot)
            if title:
                fr = self._lower_third(fr, title, t)
            if f < nx and self.prev_last is not None:
                fr = Image.blend(self.prev_last, fr, ease((f + 1) / nx))
            if first and f < 15:
                fr = Image.blend(Image.new("RGB", fr.size, (0, 0, 0)), fr, (f + 1) / 15)
            if fade_out and f >= frames - fade_out:
                fr = Image.blend(fr, Image.new("RGB", fr.size, (0, 0, 0)), (f - (frames - fade_out) + 1) / fade_out)
            data = fr.tobytes()
            self.proc.stdin.write(data); self.frames += 1
            if not animating:
                cached = data
            last_frame = fr
        self.prev_last = last_frame if frames else self.prev_last
        self.cam = end if kind != "slide" else None

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
