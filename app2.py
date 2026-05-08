"""
Cookie Inspector Pro — YOLO11 + Feret Metrology
A professional-grade desktop inspection tool.

Run:
    python cookie_inspector_app.py

Dependencies:
    pip install pillow ultralytics opencv-python
    sudo apt install python3-tk  (Linux/Raspberry Pi)
"""

import os
import json
import math
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# Picamera2 is only available on Raspberry Pi OS with the camera stack
# installed. Import it optionally so the app still runs on dev machines.
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except Exception:
    Picamera2 = None
    PICAMERA_AVAILABLE = False


# ─────────────────────────────────────────────
# Colour palette  (dark industrial / precision)
# ─────────────────────────────────────────────
C = {
    "bg":           "#0d0f12",
    "bg2":          "#13161b",
    "bg3":          "#1a1e25",
    "panel":        "#1e2229",
    "border":       "#2a2f3a",
    "border2":      "#353c4a",
    "text":         "#e8ecf4",
    "text2":        "#8a93a6",
    "text3":        "#545d70",
    "accent":       "#00d4ff",
    "accent2":      "#0099cc",
    "green":        "#00e5a0",
    "green_dim":    "#00a06f",
    "red":          "#ff4560",
    "red_dim":      "#b03040",
    "yellow":       "#ffd166",
    "yellow_dim":   "#a08040",
    "magenta":      "#d166ff",
    "cyan":         "#00e5ff",
    "separator":    "#252a35",
}

FONT_TITLE  = ("Courier New", 13, "bold")
FONT_LABEL  = ("Courier New", 9)
FONT_SMALL  = ("Courier New", 8)
FONT_MONO   = ("Courier New", 10)
FONT_LARGE  = ("Courier New", 18, "bold")
FONT_VALUE  = ("Courier New", 22, "bold")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEFAULT_MODEL_PATH = "runs/segment/cookie_seg_y11/weights/best.pt"
DEFAULT_PIXEL_TO_MM = 5.206
NUM_ANGLES         = 180
LOPSIDED_THRESHOLD = 0.15
DEFAULT_CONF       = 0.50

# ─────────────────────────────────────────────
# Persistent settings
# ─────────────────────────────────────────────
SETTINGS_PATH = Path.home() / ".cookie_inspector_settings.json"


class Settings:
    """Tiny JSON-backed settings store. Currently holds only pixel_to_mm."""

    def __init__(self):
        self.pixel_to_mm = DEFAULT_PIXEL_TO_MM
        self.calibrated_at = None  # ISO timestamp of last calibration
        self.load()

    def load(self):
        try:
            if SETTINGS_PATH.exists():
                data = json.loads(SETTINGS_PATH.read_text())
                self.pixel_to_mm   = float(data.get("pixel_to_mm",
                                                   DEFAULT_PIXEL_TO_MM))
                self.calibrated_at = data.get("calibrated_at")
        except Exception as e:
            print(f"[Settings] could not read {SETTINGS_PATH}: {e}")

    def save(self):
        try:
            SETTINGS_PATH.write_text(json.dumps({
                "pixel_to_mm":   self.pixel_to_mm,
                "calibrated_at": self.calibrated_at,
            }, indent=2))
        except Exception as e:
            print(f"[Settings] could not write {SETTINGS_PATH}: {e}")


# Global instance — analyze_cookie reads from this rather than a constant
SETTINGS = Settings()


# ─────────────────────────────────────────────
# Camera wrapper (picamera2)
# ─────────────────────────────────────────────
class Camera:
    """
    Thin wrapper around picamera2 that hides start/stop lifecycle
    and lets the GUI poll the latest BGR frame.

    Frames returned from `get_frame()` are in BGR order (OpenCV-style),
    matching the rest of the pipeline (cv2.imread, YOLO results, etc.).
    """

    def __init__(self, size=(1280, 720)):
        if not PICAMERA_AVAILABLE:
            raise RuntimeError(
                "picamera2 is not installed. "
                "Run: sudo apt install -y python3-picamera2")
        self._size = size
        self._cam  = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._cam = Picamera2()
        config = self._cam.create_preview_configuration(
            main={"format": "RGB888", "size": self._size})
        self._cam.configure(config)
        self._cam.start()
        # Small warm-up so auto-exposure has time to settle
        time.sleep(0.3)
        self._running = True

    def stop(self):
        if not self._running:
            return
        try:
            self._cam.stop()
        except Exception:
            pass
        try:
            self._cam.close()
        except Exception:
            pass
        self._cam = None
        self._running = False

    def get_frame(self):
        """Return the latest frame as a BGR numpy array, or None."""
        if not self._running or self._cam is None:
            return None
        rgb = self._cam.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @property
    def is_running(self):
        return self._running


# ═════════════════════════════════════════════
# Metrology core  (pure logic)
# ═════════════════════════════════════════════
def ray_contour_intersections(contour, center, angle_rad):
    cx, cy = center
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)
    hits = []
    n = len(contour)
    for i in range(n):
        x1, y1 = float(contour[i][0]),          float(contour[i][1])
        x2, y2 = float(contour[(i+1)%n][0]),    float(contour[(i+1)%n][1])
        ex, ey = x2 - x1, y2 - y1
        denom  = dx * ey - dy * ex
        if abs(denom) < 1e-9:
            continue
        u = (dx*(y1-cy) - dy*(x1-cx)) / denom
        if 0.0 <= u <= 1.0:
            hits.append((x1 + u*ex, y1 + u*ey))
    return hits


def longest_through_center_chord(contour, center, num_angles):
    best_len, best_chord, chords = 0.0, None, []
    for angle in np.linspace(0.0, math.pi, num_angles, endpoint=False):
        pts = ray_contour_intersections(contour, center, angle)
        if len(pts) < 2:
            continue
        dx, dy = math.cos(angle), math.sin(angle)
        proj = sorted([(p[0]*dx + p[1]*dy, p) for p in pts], key=lambda x: x[0])
        p_min, p_max = proj[0][1], proj[-1][1]
        chords.append((p_min, p_max))
        length = math.hypot(p_max[0]-p_min[0], p_max[1]-p_min[1])
        if length > best_len:
            best_len, best_chord = length, (p_min, p_max)
    return best_chord, best_len, chords


def max_feret_diameter(contour):
    """Longest distance between any two convex-hull vertices."""
    hull = cv2.convexHull(contour.reshape(-1,1,2)).reshape(-1,2)
    n = len(hull)
    if n < 2:
        return None, 0.0
    best_len, best_pair = 0.0, None
    for i in range(n):
        for j in range(i+1, n):
            d = math.hypot(float(hull[i][0])-float(hull[j][0]),
                           float(hull[i][1])-float(hull[j][1]))
            if d > best_len:
                best_len = d
                best_pair = (tuple(hull[i].astype(int)), tuple(hull[j].astype(int)))
    return best_pair, best_len


def min_feret_diameter(contour):
    """
    Minimum Feret diameter -- the narrowest dimension of the shape.

    Algorithm (rotating calipers on convex hull):
      For each edge of the convex hull, project all other hull vertices onto
      the edge perpendicular direction. The width at that orientation equals
      max_projection - min_projection. The smallest such width across all
      edge orientations is the minimum Feret diameter.
    """
    hull = cv2.convexHull(contour.reshape(-1,1,2)).reshape(-1,2).astype(float)
    n = len(hull)
    if n < 2:
        return None, 0.0

    best_len  = float('inf')
    best_pair = None

    for i in range(n):
        p1 = hull[i]
        p2 = hull[(i + 1) % n]
        edge = p2 - p1
        edge_len = math.hypot(edge[0], edge[1])
        if edge_len < 1e-9:
            continue
        ux, uy = edge[0] / edge_len, edge[1] / edge_len
        # Perpendicular unit vector (rotated 90 deg)
        px, py = -uy, ux

        # Project all hull points onto perpendicular axis
        projs = [v[0] * px + v[1] * py for v in hull]
        lo, hi = min(projs), max(projs)
        width = hi - lo

        if width < best_len:
            best_len = width
            idx_lo = projs.index(lo)
            idx_hi = projs.index(hi)
            # Anchor both endpoints at the same position along the edge direction
            # so the drawn segment is purely perpendicular (visually clean)
            edge_proj_lo = hull[idx_lo][0]*ux + hull[idx_lo][1]*uy
            edge_proj_hi = hull[idx_hi][0]*ux + hull[idx_hi][1]*uy
            edge_mid = (edge_proj_lo + edge_proj_hi) / 2
            pt_a = (int(round(edge_mid*ux + lo*px)),
                    int(round(edge_mid*uy + lo*py)))
            pt_b = (int(round(edge_mid*ux + hi*px)),
                    int(round(edge_mid*uy + hi*py)))
            best_pair = (pt_a, pt_b)

    if best_len == float('inf'):
        return None, 0.0
    return best_pair, best_len


def lopsidedness_score(centroid, contour):
    (mx, my), r = cv2.minEnclosingCircle(contour.reshape(-1,1,2))
    drift = math.hypot(centroid[0]-mx, centroid[1]-my)
    ratio = drift / r if r > 1e-6 else 0.0
    return drift, ratio, (int(mx), int(my)), int(r)


def analyze_cookie(mask_coords, raw_mask):
    M = cv2.moments(mask_coords)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    tc_chord, tc_len_px, all_chords = longest_through_center_chord(
        mask_coords, (cx, cy), NUM_ANGLES)
    feret_pair,     feret_len_px     = max_feret_diameter(mask_coords)
    min_feret_pair, min_feret_len_px = min_feret_diameter(mask_coords)
    drift_px, drift_ratio, mec_center, mec_radius = lopsidedness_score(
        (cx, cy), mask_coords)
    return {
        'mask':           raw_mask,
        'center':         (cx, cy),
        'mec_center':     mec_center,
        'mec_radius':     mec_radius,
        'drift_ratio':    drift_ratio,
        'lopsided':       drift_ratio > LOPSIDED_THRESHOLD,
        'tc_chord':       tc_chord,
        'tc_mm':          tc_len_px / SETTINGS.pixel_to_mm if tc_chord else 0.0,
        'feret_pair':     feret_pair,
        'feret_mm':       feret_len_px     / SETTINGS.pixel_to_mm if feret_pair     else 0.0,
        'min_feret_pair': min_feret_pair,
        'min_feret_mm':   min_feret_len_px / SETTINGS.pixel_to_mm if min_feret_pair else 0.0,
        'elongation':     (feret_len_px / min_feret_len_px
                           if min_feret_len_px > 1e-6 else 1.0),
        'grid':           all_chords,
    }


def to_int_pt(p):
    return (int(round(p[0])), int(round(p[1])))


def render_overlays(base_img, cookies, opts):
    display = base_img.copy()
    selected_idx = opts.get('selected_idx')
    for i, d in enumerate(cookies):
        is_selected = (i == selected_idx)

        # Highlight the selected cookie's mask with a brighter overlay
        # and a thick ring around its enclosing circle for emphasis
        if is_selected and d['mask'] is not None:
            overlay = display.copy()
            overlay[d['mask'] > 0.5] = (0, 212, 255)   # accent cyan
            cv2.addWeighted(overlay, 0.45, display, 0.55, 0, display)
            cv2.circle(display, d['mec_center'],
                       d['mec_radius'] + 8, (0, 212, 255), 3)

        if opts['mask'] and not opts['yolo_overlay'] and d['mask'] is not None and not is_selected:
            overlay = display.copy()
            overlay[d['mask'] > 0.5] = (0, 255, 0)
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        if opts['grid']:
            for c in d['grid']:
                cv2.line(display, to_int_pt(c[0]), to_int_pt(c[1]),
                         (60, 70, 90), 1)
        if opts['diameter'] and opts['through_center'] and d['tc_chord']:
            p1, p2 = d['tc_chord']
            cv2.line(display, to_int_pt(p1), to_int_pt(p2), (200, 80, 255), 2)
        if opts['diameter'] and opts['feret'] and d['feret_pair']:
            cv2.line(display, d['feret_pair'][0], d['feret_pair'][1],
                     (0, 220, 255), 2)
        if opts['diameter'] and opts['min_feret'] and d['min_feret_pair']:
            cv2.line(display, d['min_feret_pair'][0], d['min_feret_pair'][1],
                     (0, 255, 140), 2)
        cv2.circle(display, d['center'], 5, (50, 80, 255), -1)
        if d['lopsided']:
            cv2.circle(display, d['mec_center'], 5, (0, 140, 255), -1)
            cv2.line(display, d['center'], d['mec_center'], (0, 140, 255), 1)
        cx, cy = d['center']
        y = cy - 12
        if opts['feret']:
            cv2.putText(display, f"Max {d['feret_mm']:.1f}mm",
                        (cx-55, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 220, 255), 2)
            y -= 20
        if opts['min_feret']:
            cv2.putText(display, f"Min {d['min_feret_mm']:.1f}mm",
                        (cx-55, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 140), 2)
            y -= 20
        if opts['through_center']:
            cv2.putText(display, f"Ctr {d['tc_mm']:.1f}mm",
                        (cx-55, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 80, 255), 2)
            y -= 20
        if d['lopsided']:
            cv2.putText(display, f"LOPSIDED {d['drift_ratio']*100:.0f}%",
                        (cx-55, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 100, 255), 2)
    return display


# ═════════════════════════════════════════════
# Custom Tkinter widgets
# ═════════════════════════════════════════════
class FlatButton(tk.Label):
    """Flat styled button using Label for full color control."""
    def __init__(self, parent, text, command=None,
                 bg=C["accent"], fg=C["bg"], font=FONT_LABEL,
                 pad_x=16, pad_y=7, **kwargs):
        super().__init__(parent, text=text, bg=bg, fg=fg, font=font,
                         cursor="hand2", padx=pad_x, pady=pad_y, **kwargs)
        self._bg   = bg
        self._fg   = fg
        self._cmd  = command
        self._dim  = self._darken(bg)
        self.bind("<Button-1>",  self._on_click)
        self.bind("<Enter>",     lambda e: self.config(bg=self._dim))
        self.bind("<Leave>",     lambda e: self.config(bg=self._bg))

    @staticmethod
    def _darken(hex_color):
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return "#{:02x}{:02x}{:02x}".format(
            max(0, int(r*0.75)), max(0, int(g*0.75)), max(0, int(b*0.75)))

    def _on_click(self, _):
        if self._cmd:
            self._cmd()

    def set_state(self, enabled: bool):
        if enabled:
            self.config(bg=self._bg, cursor="hand2")
            self.bind("<Button-1>", self._on_click)
        else:
            dim = self._darken(self._darken(self._bg))
            self.config(bg=dim, cursor="")
            self.unbind("<Button-1>")


class Toggle(tk.Frame):
    """Pill-shaped toggle switch."""
    W, H = 42, 22

    def __init__(self, parent, variable, command=None, **kwargs):
        super().__init__(parent, bg=C["bg2"], **kwargs)
        self._var = variable
        self._cmd = command
        self._canvas = tk.Canvas(self, width=self.W, height=self.H,
                                 bg=C["bg2"], highlightthickness=0,
                                 cursor="hand2")
        self._canvas.pack()
        self._canvas.bind("<Button-1>", self._toggle)
        self._draw()

    def _draw(self):
        c  = self._canvas
        on = self._var.get()
        c.delete("all")
        track_col = C["accent"] if on else C["border2"]
        knob_x    = self.W - 11 if on else 11
        r = self.H // 2
        c.create_oval(0, 0, self.H, self.H, fill=track_col, outline="")
        c.create_oval(self.W-self.H, 0, self.W, self.H,
                      fill=track_col, outline="")
        c.create_rectangle(r, 0, self.W-r, self.H,
                           fill=track_col, outline="")
        c.create_oval(knob_x-9, 2, knob_x+9, self.H-2,
                      fill=C["text"] if on else C["text3"], outline="")

    def _toggle(self, _):
        self._var.set(not self._var.get())
        self._draw()
        if self._cmd:
            self._cmd()


class Separator(tk.Frame):
    def __init__(self, parent, orient="h", **kwargs):
        super().__init__(parent,
                         bg=C["separator"],
                         height=1 if orient == "h" else 0,
                         width=0 if orient == "h" else 1,
                         **kwargs)


class StatCard(tk.Frame):
    """A metric display card: label + large value."""
    def __init__(self, parent, label, unit="", color=C["accent"], **kwargs):
        super().__init__(parent, bg=C["panel"],
                         highlightbackground=C["border"],
                         highlightthickness=1, **kwargs)
        tk.Label(self, text=label.upper(), bg=C["panel"],
                 fg=C["text3"], font=FONT_SMALL).pack(
            anchor=tk.W, padx=10, pady=(8, 0))
        self._val = tk.Label(self, text="—", bg=C["panel"],
                             fg=color, font=FONT_VALUE)
        self._val.pack(anchor=tk.W, padx=10)
        if unit:
            tk.Label(self, text=unit, bg=C["panel"],
                     fg=C["text3"], font=FONT_SMALL).pack(
                anchor=tk.W, padx=10, pady=(0, 8))

    def set(self, value):
        self._val.config(text=str(value))


# ═════════════════════════════════════════════
# Main Application
# ═════════════════════════════════════════════
class CookieInspectorApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Cookie Inspector Pro")
        self.root.geometry("1380x860")
        self.root.minsize(1100, 720)
        self.root.configure(bg=C["bg"])

        self.model        = None
        self.model_path   = tk.StringVar(value=DEFAULT_MODEL_PATH)
        self.image_path   = None
        self.orig_img     = None
        self.yolo_plotted = None
        self.cookies      = []
        self.last_result  = None
        self._photo       = None
        self._busy        = False

        self.v_yolo     = tk.BooleanVar(value=True)
        self.v_mask     = tk.BooleanVar(value=True)
        self.v_grid     = tk.BooleanVar(value=False)
        self.v_diameter = tk.BooleanVar(value=True)
        self.v_feret    = tk.BooleanVar(value=True)
        self.v_min_feret = tk.BooleanVar(value=True)
        self.v_through  = tk.BooleanVar(value=True)
        self.v_conf     = tk.DoubleVar(value=DEFAULT_CONF)

        # ── Zoom / pan state ──
        self._zoom        = 1.0
        self._zoom_min    = 0.1
        self._zoom_max    = 20.0
        self._pan_x       = 0.0
        self._pan_y       = 0.0
        self._pan_start   = None
        self._pan_origin  = None
        self._was_dragged = False       # True if mouse moved during press
        self._selected_idx = None       # index of cookie shown in detail panel
        self._detail      = None        # lazily created detail panel widget

        # ── Live camera state ──
        self._camera        = None       # Camera instance (lazy)
        self._live_mode     = False      # True when displaying live feed
        self._live_after_id = None       # Tk after() handle for the live loop
        self._calibrating   = False      # blocks re-entry of calibrate flow

        self._build()
        self._load_model_bg(self.model_path.get(), silent=True)

    # ──────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────
    def _build(self):
        # Header
        hdr = tk.Frame(self.root, bg=C["bg2"],
                       highlightbackground=C["border"],
                       highlightthickness=1)
        hdr.pack(fill=tk.X)

        logo = tk.Frame(hdr, bg=C["bg2"])
        logo.pack(side=tk.LEFT, padx=18, pady=10)
        tk.Label(logo, text="◈", bg=C["bg2"],
                 fg=C["accent"], font=("Courier New", 20)).pack(side=tk.LEFT)
        tk.Label(logo, text=" COOKIE INSPECTOR",
                 bg=C["bg2"], fg=C["text"],
                 font=FONT_TITLE).pack(side=tk.LEFT)
        tk.Label(logo, text="  PRO",
                 bg=C["bg2"], fg=C["accent"],
                 font=FONT_TITLE).pack(side=tk.LEFT)

        btns = tk.Frame(hdr, bg=C["bg2"])
        btns.pack(side=tk.RIGHT, padx=18, pady=10)
        self.btn_save = FlatButton(btns, "⬇  EXPORT",
                                   command=self.on_save,
                                   bg=C["bg3"], fg=C["text2"],
                                   font=FONT_SMALL)
        self.btn_save.pack(side=tk.RIGHT, padx=4)
        self.btn_run = FlatButton(btns, "▶  RUN INSPECTION",
                                  command=self.on_run,
                                  bg=C["accent"], fg=C["bg"],
                                  font=("Courier New", 9, "bold"))
        self.btn_run.pack(side=tk.RIGHT, padx=4)
        self.btn_upload = FlatButton(btns, "↑  UPLOAD IMAGE",
                                     command=self.on_upload,
                                     bg=C["bg3"], fg=C["text"],
                                     font=FONT_SMALL)
        self.btn_upload.pack(side=tk.RIGHT, padx=4)
        self.btn_live = FlatButton(btns, "📷  LIVE",
                                   command=self.on_toggle_live,
                                   bg=C["bg3"], fg=C["text"],
                                   font=FONT_SMALL)
        self.btn_live.pack(side=tk.RIGHT, padx=4)
        if not PICAMERA_AVAILABLE:
            self.btn_live.set_state(False)

        # Body
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill=tk.BOTH, expand=True)
        self._build_sidebar(body)
        self._build_canvas(body)

        # Status bar
        status_bar = tk.Frame(self.root, bg=C["bg2"],
                              highlightbackground=C["border"],
                              highlightthickness=1, height=28)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)
        self._status_dot = tk.Label(status_bar, text="●",
                                    bg=C["bg2"], fg=C["green"],
                                    font=FONT_SMALL)
        self._status_dot.pack(side=tk.LEFT, padx=(12, 4), pady=4)
        self._status_var = tk.StringVar(
            value="Ready — load a model and upload an image.")
        tk.Label(status_bar, textvariable=self._status_var,
                 bg=C["bg2"], fg=C["text2"],
                 font=FONT_SMALL).pack(side=tk.LEFT)
        tk.Label(status_bar, text="YOLO11 · Feret Metrology v2",
                 bg=C["bg2"], fg=C["text3"],
                 font=FONT_SMALL).pack(side=tk.RIGHT, padx=12)

    # ── Sidebar ──────────────────────────────
    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=C["bg2"], width=300,
                      highlightbackground=C["border"],
                      highlightthickness=1)
        sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(1, 0))
        sb.pack_propagate(False)

        canvas = tk.Canvas(sb, bg=C["bg2"], highlightthickness=0, width=298)
        scroll = tk.Scrollbar(sb, orient=tk.VERTICAL, command=canvas.yview)
        inner  = tk.Frame(canvas, bg=C["bg2"])
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        iid = canvas.create_window((0, 0), window=inner, anchor=tk.NW)
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(
            iid, width=e.width))

        # ── Model section ──
        self._section_hdr(inner, "MODEL")
        mf = tk.Frame(inner, bg=C["bg2"], padx=12, pady=6)
        mf.pack(fill=tk.X)
        tk.Label(mf, text="WEIGHTS PATH", bg=C["bg2"],
                 fg=C["text3"], font=FONT_SMALL).pack(
            anchor=tk.W, pady=(0, 3))
        ef = tk.Frame(mf, bg=C["border"])
        ef.pack(fill=tk.X, pady=(0, 6))
        tk.Entry(ef, textvariable=self.model_path,
                 bg=C["bg3"], fg=C["text2"],
                 insertbackground=C["accent"],
                 relief=tk.FLAT, font=FONT_SMALL,
                 width=28).pack(fill=tk.X, padx=1, pady=1, ipady=5)
        br = tk.Frame(mf, bg=C["bg2"])
        br.pack(fill=tk.X)
        FlatButton(br, "BROWSE", command=self.on_browse_model,
                   bg=C["bg3"], fg=C["text2"], font=FONT_SMALL,
                   pad_x=10, pad_y=5).pack(side=tk.LEFT, padx=(0, 4))
        FlatButton(br, "LOAD",
                   command=lambda: self._load_model_bg(self.model_path.get()),
                   bg=C["accent2"], fg=C["bg"], font=FONT_SMALL,
                   pad_x=10, pad_y=5).pack(side=tk.LEFT)
        self._model_lbl = tk.Label(mf, text="● Not loaded",
                                   bg=C["bg2"], fg=C["red"],
                                   font=FONT_SMALL)
        self._model_lbl.pack(anchor=tk.W, pady=(6, 0))

        Separator(inner).pack(fill=tk.X, padx=12, pady=8)

        # ── Inference section ──
        self._section_hdr(inner, "INFERENCE")
        inf = tk.Frame(inner, bg=C["bg2"], padx=12, pady=6)
        inf.pack(fill=tk.X)
        conf_row = tk.Frame(inf, bg=C["bg2"])
        conf_row.pack(fill=tk.X)
        tk.Label(conf_row, text="CONFIDENCE THRESHOLD",
                 bg=C["bg2"], fg=C["text3"], font=FONT_SMALL).pack(
            side=tk.LEFT)
        self._conf_lbl = tk.Label(conf_row, bg=C["bg2"],
                                  fg=C["accent"], font=FONT_MONO,
                                  text=f"{self.v_conf.get():.2f}")
        self._conf_lbl.pack(side=tk.RIGHT)
        tk.Scale(inf, from_=0.05, to=0.95,
                 variable=self.v_conf, orient=tk.HORIZONTAL,
                 resolution=0.01,
                 bg=C["bg2"], fg=C["text2"],
                 troughcolor=C["bg3"],
                 activebackground=C["accent"],
                 highlightthickness=0,
                 sliderrelief=tk.FLAT,
                 showvalue=False,
                 command=lambda v: self._conf_lbl.config(
                     text=f"{float(v):.2f}")).pack(fill=tk.X)

        Separator(inner).pack(fill=tk.X, padx=12, pady=8)

        # ── Calibration section ──
        self._section_hdr(inner, "CALIBRATION")
        cal = tk.Frame(inner, bg=C["bg2"], padx=12, pady=6)
        cal.pack(fill=tk.X)

        # Current px/mm value display
        cal_row = tk.Frame(cal, bg=C["bg2"])
        cal_row.pack(fill=tk.X)
        tk.Label(cal_row, text="PIXELS PER MM",
                 bg=C["bg2"], fg=C["text3"], font=FONT_SMALL).pack(
            side=tk.LEFT)
        self._cal_value_lbl = tk.Label(
            cal_row, bg=C["bg2"], fg=C["accent"], font=FONT_MONO,
            text=f"{SETTINGS.pixel_to_mm:.3f}")
        self._cal_value_lbl.pack(side=tk.RIGHT)

        self._cal_when_lbl = tk.Label(
            cal, bg=C["bg2"], fg=C["text3"], font=FONT_SMALL,
            text=self._format_calibration_when())
        self._cal_when_lbl.pack(anchor=tk.W, pady=(4, 8))

        # Calibrate button
        self.btn_calibrate = FlatButton(
            cal, "◎  CALIBRATE  WITH  CURRENT  FRAME",
            command=self.on_calibrate,
            bg=C["accent2"], fg=C["bg"],
            font=("Courier New", 9, "bold"),
            pad_x=10, pad_y=7)
        self.btn_calibrate.pack(fill=tk.X)

        # Reset to default
        FlatButton(cal, "RESET TO DEFAULT",
                   command=self.on_reset_calibration,
                   bg=C["bg3"], fg=C["text2"], font=FONT_SMALL,
                   pad_x=10, pad_y=4).pack(fill=tk.X, pady=(4, 0))

        Separator(inner).pack(fill=tk.X, padx=12, pady=8)

        # ── Overlay toggles ──
        self._section_hdr(inner, "OVERLAYS")
        ov = tk.Frame(inner, bg=C["bg2"], padx=12, pady=6)
        ov.pack(fill=tk.X)
        for lbl, var, color in [
            ("YOLO DETECTION",   self.v_yolo,     C["yellow"]),
            ("MASK FILL",        self.v_mask,      C["green"]),
            ("ANGLE GRID",       self.v_grid,      C["text2"]),
            ("DIAMETER LINES",   self.v_diameter,  C["text"]),
            ("MAX FERET DIAM.",  self.v_feret,     C["cyan"]),
            ("MIN FERET DIAM.",  self.v_min_feret, C["green"]),
            ("THROUGH-CENTER",   self.v_through,   C["magenta"]),
        ]:
            row = tk.Frame(ov, bg=C["bg2"])
            row.pack(fill=tk.X, pady=4)
            tk.Label(row, text="▌", bg=C["bg2"],
                     fg=color, font=FONT_SMALL).pack(
                side=tk.LEFT, padx=(0, 6))
            tk.Label(row, text=lbl, bg=C["bg2"],
                     fg=C["text2"], font=FONT_SMALL).pack(side=tk.LEFT)
            Toggle(row, variable=var,
                   command=self._refresh_display).pack(side=tk.RIGHT)

        Separator(inner).pack(fill=tk.X, padx=12, pady=8)

        # ── Statistics cards ──
        self._section_hdr(inner, "STATISTICS")
        sf = tk.Frame(inner, bg=C["bg2"], padx=12, pady=6)
        sf.pack(fill=tk.X)
        top = tk.Frame(sf, bg=C["bg2"])
        top.pack(fill=tk.X, pady=(0, 4))
        self._card_det = StatCard(top, "Detected", color=C["accent"])
        self._card_det.pack(side=tk.LEFT, expand=True,
                            fill=tk.BOTH, padx=(0, 4))
        self._card_lop = StatCard(top, "Lopsided", color=C["red"])
        self._card_lop.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        bot = tk.Frame(sf, bg=C["bg2"])
        bot.pack(fill=tk.X, pady=4)
        self._card_fer = StatCard(bot, "Avg MaxFeret", unit="mm",
                                  color=C["cyan"])
        self._card_fer.pack(side=tk.LEFT, expand=True,
                            fill=tk.BOTH, padx=(0, 4))
        self._card_mfr = StatCard(bot, "Avg MinFeret", unit="mm",
                                  color=C["green"])
        self._card_mfr.pack(side=tk.LEFT, expand=True,
                            fill=tk.BOTH, padx=(0, 4))
        self._card_ctr = StatCard(bot, "Avg Center", unit="mm",
                                  color=C["magenta"])
        self._card_ctr.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        Separator(inner).pack(fill=tk.X, padx=12, pady=8)

        # ── Results table ──
        self._section_hdr(inner, "PER-COOKIE RESULTS")
        tf = tk.Frame(inner, bg=C["bg2"], padx=12, pady=6)
        tf.pack(fill=tk.X)
        hdr_row = tk.Frame(tf, bg=C["bg3"])
        hdr_row.pack(fill=tk.X)
        for col, w in [("#", 3), ("MAX-F", 6), ("MIN-F", 6),
                       ("CTR", 6), ("ELONG", 5), ("STATUS", 7)]:
            tk.Label(hdr_row, text=col, bg=C["bg3"],
                     fg=C["text3"], font=FONT_SMALL,
                     width=w, anchor=tk.CENTER,
                     padx=4, pady=5).pack(side=tk.LEFT)
        self._tbl = tk.Frame(tf, bg=C["bg2"])
        self._tbl.pack(fill=tk.X)

    def _section_hdr(self, parent, title):
        row = tk.Frame(parent, bg=C["bg2"], padx=12, pady=0)
        row.pack(fill=tk.X, pady=(10, 2))
        tk.Label(row, text=title, bg=C["bg2"],
                 fg=C["accent"], font=FONT_SMALL).pack(side=tk.LEFT)
        tk.Frame(row, bg=C["border"], height=1).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

    # ── Canvas area ──────────────────────────
    def _build_canvas(self, parent):
        outer = tk.Frame(parent, bg=C["bg"],
                         highlightbackground=C["border"],
                         highlightthickness=1)
        outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 1))

        ctool = tk.Frame(outer, bg=C["bg2"], height=32,
                         highlightbackground=C["border"],
                         highlightthickness=1)
        ctool.pack(fill=tk.X)
        ctool.pack_propagate(False)
        self._file_lbl = tk.Label(ctool, text="No image loaded",
                                  bg=C["bg2"], fg=C["text3"],
                                  font=FONT_SMALL)
        self._file_lbl.pack(side=tk.LEFT, padx=12, pady=6)

        # ── Zoom controls (right side of toolbar) ──
        zoom_frame = tk.Frame(ctool, bg=C["bg2"])
        zoom_frame.pack(side=tk.RIGHT, padx=8, pady=4)
        FlatButton(zoom_frame, "⟳", command=self._zoom_reset,
                   bg=C["bg3"], fg=C["text2"], font=FONT_SMALL,
                   pad_x=7, pad_y=3).pack(side=tk.RIGHT, padx=(2, 0))
        FlatButton(zoom_frame, "+", command=self._zoom_in,
                   bg=C["bg3"], fg=C["text"],
                   font=("Courier New", 10, "bold"),
                   pad_x=8, pad_y=3).pack(side=tk.RIGHT, padx=2)
        FlatButton(zoom_frame, "−", command=self._zoom_out,
                   bg=C["bg3"], fg=C["text"],
                   font=("Courier New", 10, "bold"),
                   pad_x=8, pad_y=3).pack(side=tk.RIGHT, padx=2)
        self._zoom_lbl = tk.Label(zoom_frame, text="100%",
                                  bg=C["bg2"], fg=C["accent"],
                                  font=FONT_MONO, width=5, anchor=tk.E)
        self._zoom_lbl.pack(side=tk.RIGHT, padx=(0, 4))
        tk.Label(zoom_frame, text="ZOOM", bg=C["bg2"],
                 fg=C["text3"], font=FONT_SMALL).pack(side=tk.RIGHT)

        self._dim_lbl = tk.Label(ctool, text="",
                                 bg=C["bg2"], fg=C["text3"],
                                 font=FONT_SMALL)
        self._dim_lbl.pack(side=tk.RIGHT, padx=12, pady=6)

        self._canvas = tk.Canvas(outer, bg=C["bg"], highlightthickness=0,
                                 cursor="crosshair")
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>",       lambda e: self._refresh_display())
        # Scroll-wheel zoom (Linux: Button-4/5; Windows/Mac: MouseWheel)
        self._canvas.bind("<MouseWheel>",      self._on_mousewheel)
        self._canvas.bind("<Button-4>",        self._on_mousewheel)
        self._canvas.bind("<Button-5>",        self._on_mousewheel)
        # Middle-button pan
        self._canvas.bind("<ButtonPress-2>",   self._on_pan_start)
        self._canvas.bind("<B2-Motion>",       self._on_pan_move)
        self._canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        # Left-button drag pan / click-to-inspect
        self._canvas.bind("<ButtonPress-1>",   self._on_pan_start)
        self._canvas.bind("<B1-Motion>",       self._on_pan_move)
        self._canvas.bind("<ButtonRelease-1>", self._on_pan_end)
        # Hover: change cursor to hand when over a cookie
        self._canvas.bind("<Motion>",          self._on_hover)
        self._draw_placeholder()

    def _draw_placeholder(self):
        self._canvas.delete("all")
        cx, cy = 500, 320
        for i in range(0, 300, 16):
            self._canvas.create_line(
                cx-150+i, cy-100, cx-150+i+8, cy-100,
                fill=C["border2"], width=1, tags="ph")
            self._canvas.create_line(
                cx-150+i, cy+100, cx-150+i+8, cy+100,
                fill=C["border2"], width=1, tags="ph")
        for i in range(0, 200, 16):
            self._canvas.create_line(
                cx-150, cy-100+i, cx-150, cy-100+i+8,
                fill=C["border2"], width=1, tags="ph")
            self._canvas.create_line(
                cx+150, cy-100+i, cx+150, cy-100+i+8,
                fill=C["border2"], width=1, tags="ph")
        self._canvas.create_text(
            cx, cy-20, text="↑", fill=C["border2"],
            font=("Courier New", 36), tags="ph")
        self._canvas.create_text(
            cx, cy+20, text="Upload an image to begin",
            fill=C["text3"], font=("Courier New", 12), tags="ph")
        self._canvas.create_text(
            cx, cy+44, text="Supports: JPG · PNG · BMP · TIFF",
            fill=C["text3"], font=FONT_SMALL, tags="ph")

    # ──────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────
    def _load_model_bg(self, path, silent=False):
        def worker():
            try:
                self._set_status("Loading model…", "yellow")
                self.root.after(0, lambda: self._model_lbl.config(
                    text="● Loading…", fg=C["yellow"]))
                m = YOLO(path)
                self.model = m
                self._set_status(
                    f"Model ready — {os.path.basename(path)}", "green")
                self.root.after(0, lambda: self._model_lbl.config(
                    text=f"● {os.path.basename(path)}", fg=C["green"]))
            except Exception as e:
                self.model = None
                self._set_status(f"Model error: {e}", "red")
                self.root.after(0, lambda: self._model_lbl.config(
                    text="● Load failed", fg=C["red"]))
                if not silent:
                    self.root.after(
                        0, lambda: messagebox.showerror("Model Error", str(e)))
        threading.Thread(target=worker, daemon=True).start()

    def on_browse_model(self):
        p = filedialog.askopenfilename(
            title="Select YOLO model weights",
            filetypes=[("PyTorch weights", "*.pt"),
                       ("All files", "*.*")])
        if p:
            self.model_path.set(p)

    # ──────────────────────────────────────────
    # Image upload
    # ──────────────────────────────────────────
    def on_upload(self):
        # Always exit live mode before loading a static image
        if self._live_mode:
            self._stop_live()
        path = filedialog.askopenfilename(
            title="Select inspection image",
            filetypes=[("Images",
                        "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Cannot read: {path}")
            return
        self.image_path   = path
        self.orig_img     = img
        self.yolo_plotted = None
        self.cookies      = []
        self.last_result  = None
        name = os.path.basename(path)
        self._file_lbl.config(text=name, fg=C["text2"])
        self._dim_lbl.config(text=f"{img.shape[1]} × {img.shape[0]} px")
        self._clear_table()
        self._reset_cards()
        self._zoom  = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._selected_idx = None
        if self._detail is not None:
            self._detail.place_forget()
        self._set_status(
            f"Loaded: {name}  —  click  ▶ RUN INSPECTION  to analyse",
            "green")
        self._refresh_display()

    # ──────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────
    def on_run(self):
        if self.orig_img is None:
            messagebox.showinfo("No image", "Upload an image first.")
            return
        if self.model is None:
            messagebox.showwarning("No model",
                                   "Model not loaded — set path and click LOAD.")
            return
        if self._busy:
            return
        self._busy = True
        self.btn_run.set_state(False)
        threading.Thread(target=self._inference_worker, daemon=True).start()

    def _inference_worker(self):
        try:
            self._set_status("Running YOLO inference…", "yellow")
            # Pass numpy array directly so this works for both uploaded
            # files and live camera frames (no path required).
            results = self.model.predict(
                source=self.orig_img,
                conf=float(self.v_conf.get()),
                retina_masks=True,
                verbose=False)
            if not results:
                self._set_status("No results returned.", "red")
                return
            r = results[0]
            self.last_result  = r
            self.yolo_plotted = r.plot(boxes=True, labels=True, conf=True)
            cookies = []
            if r.masks is not None:
                n = len(r.masks)
                self._set_status(
                    f"Computing metrology for {n} mask(s)…", "yellow")
                for mi in r.masks:
                    mc = mi.xy[0].astype(np.int32)
                    if len(mc) < 3:
                        continue
                    rm = mi.data[0].cpu().numpy()
                    rm = cv2.resize(rm, (self.orig_img.shape[1],
                                        self.orig_img.shape[0]))
                    d = analyze_cookie(mc, rm)
                    if d:
                        cookies.append(d)
            self.cookies = cookies
            self.root.after(0, self._on_done)
        except Exception as e:
            self._set_status(f"Error: {e}", "red")
            self.root.after(
                0, lambda: messagebox.showerror("Inference Error", str(e)))
        finally:
            self._busy = False
            self.root.after(0, lambda: self.btn_run.set_state(True))

    def _on_done(self):
        n        = len(self.cookies)
        n_lop    = sum(1 for d in self.cookies if d['lopsided'])
        ferets   = [d['feret_mm']     for d in self.cookies if d['feret_mm']     > 0]
        minfers  = [d['min_feret_mm'] for d in self.cookies if d['min_feret_mm'] > 0]
        centers  = [d['tc_mm']        for d in self.cookies if d['tc_mm']        > 0]
        self._card_det.set(str(n))
        self._card_lop.set(str(n_lop))
        self._card_fer.set(
            f"{sum(ferets)/len(ferets):.1f}"  if ferets  else "—")
        self._card_mfr.set(
            f"{sum(minfers)/len(minfers):.1f}" if minfers else "—")
        self._card_ctr.set(
            f"{sum(centers)/len(centers):.1f}" if centers else "—")
        self._populate_table()
        suffix = f"  ·  {n_lop} lopsided" if n_lop else ""
        self._set_status(
            f"Inspection complete — {n} cookie(s) detected{suffix}", "green")
        self._refresh_display()

    def _reset_cards(self):
        for c in (self._card_det, self._card_lop,
                  self._card_fer, self._card_mfr, self._card_ctr):
            c.set("—")

    # ──────────────────────────────────────────
    # Table
    # ──────────────────────────────────────────
    def _clear_table(self):
        for w in self._tbl.winfo_children():
            w.destroy()

    def _populate_table(self):
        self._clear_table()
        for i, d in enumerate(self.cookies):
            bg      = C["bg3"] if i % 2 == 0 else C["bg2"]
            flag    = "LOPSIDED" if d['lopsided'] else "OK"
            fg_flag = C["red"] if d['lopsided'] else C["green"]
            row = tk.Frame(self._tbl, bg=bg)
            row.pack(fill=tk.X)
            for val, w, fg in [
                (str(i),                        3, C["text3"]),
                (f"{d['feret_mm']:.1f}",        6, C["cyan"]),
                (f"{d['min_feret_mm']:.1f}",    6, C["green"]),
                (f"{d['tc_mm']:.1f}",           6, C["magenta"]),
                (f"{d['elongation']:.2f}",      5, C["yellow"]),
                (flag,                          7, fg_flag),
            ]:
                tk.Label(row, text=val, bg=bg, fg=fg,
                         font=FONT_SMALL, width=w,
                         anchor=tk.CENTER, padx=4,
                         pady=5).pack(side=tk.LEFT)

    # ──────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────
    def _current_frame(self):
        if self.orig_img is None:
            return None
        opts = {
            'yolo_overlay':   self.v_yolo.get(),
            'mask':           self.v_mask.get(),
            'grid':           self.v_grid.get(),
            'diameter':       self.v_diameter.get(),
            'feret':          self.v_feret.get(),
            'min_feret':      self.v_min_feret.get(),
            'through_center': self.v_through.get(),
            'selected_idx':   self._selected_idx,
        }
        base = (self.yolo_plotted
                if opts['yolo_overlay'] and self.yolo_plotted is not None
                else self.orig_img)
        return render_overlays(base, self.cookies, opts) if self.cookies \
            else base.copy()

    def _refresh_display(self):
        img = self._current_frame()
        if img is None:
            return
        self._canvas.delete("ph")
        cw = max(self._canvas.winfo_width(),  1)
        ch = max(self._canvas.winfo_height(), 1)
        h, w = img.shape[:2]

        # Fit-to-canvas base scale, then apply user zoom on top
        fit = min(cw / w, ch / h)
        scale = fit * self._zoom

        nw = max(int(w * scale), 1)
        nh = max(int(h * scale), 1)

        # Center offset + pan
        ox = (cw - nw) // 2 + int(self._pan_x)
        oy = (ch - nh) // 2 + int(self._pan_y)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        interp = (cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        rgb = cv2.resize(rgb, (nw, nh), interpolation=interp)

        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._canvas.delete("img")
        self._canvas.create_image(ox, oy, anchor=tk.NW,
                                  image=self._photo, tags="img")
        # Update zoom label
        self._zoom_lbl.config(text=f"{int(self._zoom * 100)}%")

    # ──────────────────────────────────────────
    # Zoom helpers
    # ──────────────────────────────────────────
    def _zoom_in(self):
        self._apply_zoom(1.25)

    def _zoom_out(self):
        self._apply_zoom(1 / 1.25)

    def _zoom_reset(self):
        self._zoom  = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._refresh_display()

    def _apply_zoom(self, factor, pivot_x=None, pivot_y=None):
        """
        Zoom by `factor` keeping the canvas point (pivot_x, pivot_y) fixed.
        If no pivot is given, zooms around the canvas centre.
        """
        old_zoom = self._zoom
        new_zoom = max(self._zoom_min,
                       min(self._zoom_max, self._zoom * factor))
        if new_zoom == old_zoom:
            return

        cw = max(self._canvas.winfo_width(),  1)
        ch = max(self._canvas.winfo_height(), 1)

        if pivot_x is None:
            pivot_x = cw / 2
        if pivot_y is None:
            pivot_y = ch / 2

        # The point under the cursor should stay fixed:
        # new_pan = pivot - (pivot - old_pan) * (new_zoom / old_zoom)
        ratio = new_zoom / old_zoom
        self._pan_x = pivot_x - (pivot_x - self._pan_x) * ratio  \
                      + (pivot_x - cw / 2) * (ratio - 1)
        self._pan_y = pivot_y - (pivot_y - self._pan_y) * ratio  \
                      + (pivot_y - ch / 2) * (ratio - 1)

        # Simpler, correct version: keep image point under cursor stationary
        img = self._current_frame()
        if img is not None:
            h, w = img.shape[:2]
            fit  = min(cw / w, ch / h)
            old_nw = w * fit * old_zoom
            old_nh = h * fit * old_zoom
            old_ox = (cw - old_nw) / 2 + self._pan_x / ratio * ratio
            # Recalculate properly:
            # cursor in image coords before zoom
            old_ox = (cw - w * fit * old_zoom) / 2 + self._pan_x
            old_oy = (ch - h * fit * old_zoom) / 2 + self._pan_y
            img_x  = (pivot_x - old_ox) / (fit * old_zoom)
            img_y  = (pivot_y - old_oy) / (fit * old_zoom)
            # After zoom, same image point should sit at pivot
            self._pan_x = pivot_x - (cw - w * fit * new_zoom) / 2 \
                          - img_x * fit * new_zoom
            self._pan_y = pivot_y - (ch - h * fit * new_zoom) / 2 \
                          - img_y * fit * new_zoom

        self._zoom = new_zoom
        self._refresh_display()

    # ──────────────────────────────────────────
    # Mouse-wheel zoom
    # ──────────────────────────────────────────
    def _on_mousewheel(self, event):
        # Determine scroll direction (cross-platform)
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = 1.15
        else:
            factor = 1 / 1.15
        self._apply_zoom(factor, pivot_x=event.x, pivot_y=event.y)

    # ──────────────────────────────────────────
    # Pan (drag)
    # ──────────────────────────────────────────
    # Distance (in pixels) the mouse must move during a press to be
    # treated as a drag rather than a click
    CLICK_THRESHOLD = 5

    def _on_pan_start(self, event):
        self._pan_start    = (event.x, event.y)
        self._pan_origin   = (self._pan_x, self._pan_y)
        self._was_dragged  = False
        # Only switch to drag cursor once the user has actually moved
        # (avoids a flicker on every simple click)

    def _on_pan_move(self, event):
        if self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        if not self._was_dragged and (abs(dx) > self.CLICK_THRESHOLD
                                      or abs(dy) > self.CLICK_THRESHOLD):
            self._was_dragged = True
            self._canvas.config(cursor="fleur")
        if self._was_dragged:
            self._pan_x = self._pan_origin[0] + dx
            self._pan_y = self._pan_origin[1] + dy
            self._refresh_display()

    def _on_pan_end(self, event):
        was_drag = self._was_dragged
        self._pan_start  = None
        self._pan_origin = None
        self._was_dragged = False
        self._canvas.config(cursor="crosshair")
        # If the user didn't drag, treat as a click
        if not was_drag:
            self._on_canvas_click(event)

    # ──────────────────────────────────────────
    # Click-to-inspect
    # ──────────────────────────────────────────
    def _canvas_to_image_coords(self, cx_canvas, cy_canvas):
        """
        Convert a click in canvas (widget) coordinates into the
        corresponding (x, y) in the original image. Returns None if
        the click landed outside the displayed image.
        """
        if self.orig_img is None:
            return None
        cw = max(self._canvas.winfo_width(),  1)
        ch = max(self._canvas.winfo_height(), 1)
        h, w = self.orig_img.shape[:2]
        fit   = min(cw / w, ch / h)
        scale = fit * self._zoom
        nw, nh = w * scale, h * scale
        ox = (cw - nw) / 2 + self._pan_x
        oy = (ch - nh) / 2 + self._pan_y
        ix = (cx_canvas - ox) / scale
        iy = (cy_canvas - oy) / scale
        if 0 <= ix < w and 0 <= iy < h:
            return int(ix), int(iy)
        return None

    def _hit_test(self, ix, iy):
        """Return the index of the cookie whose mask contains (ix, iy)."""
        for i, d in enumerate(self.cookies):
            mask = d.get('mask')
            if mask is None:
                continue
            if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]:
                if mask[iy, ix] > 0.5:
                    return i
        return None

    def _on_canvas_click(self, event):
        if not self.cookies:
            return
        pt = self._canvas_to_image_coords(event.x, event.y)
        if pt is None:
            return
        ix, iy = pt
        idx = self._hit_test(ix, iy)
        if idx is not None:
            self._show_detail_panel(idx)
        else:
            self._hide_detail_panel()

    def _on_hover(self, event):
        # Skip while a drag is in progress to avoid cursor jitter
        if self._pan_start is not None:
            return
        if not self.cookies:
            return
        pt = self._canvas_to_image_coords(event.x, event.y)
        if pt is None:
            self._canvas.config(cursor="crosshair")
            return
        ix, iy = pt
        idx = self._hit_test(ix, iy)
        self._canvas.config(cursor="hand2" if idx is not None else "crosshair")

    # ──────────────────────────────────────────
    # Detail panel (slides over canvas)
    # ──────────────────────────────────────────
    def _show_detail_panel(self, idx):
        d = self.cookies[idx]
        # Build / rebuild panel
        if getattr(self, '_detail', None) is None:
            self._build_detail_panel()
        self._detail.lift()
        self._detail.place(relx=1.0, rely=0.0, x=-12, y=12, anchor='ne')
        self._populate_detail(idx, d)
        # Highlight the selected cookie on the canvas
        self._selected_idx = idx
        self._refresh_display()

    def _hide_detail_panel(self):
        if getattr(self, '_detail', None) is not None:
            self._detail.place_forget()
        self._selected_idx = None
        self._refresh_display()

    def _build_detail_panel(self):
        self._detail = tk.Frame(
            self._canvas.master,
            bg=C["panel"],
            highlightbackground=C["accent"],
            highlightthickness=1,
            width=320,
        )
        self._detail.pack_propagate(False)

        # Header bar with title + close
        hdr = tk.Frame(self._detail, bg=C["bg3"], height=36)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        self._detail_title = tk.Label(
            hdr, text="COOKIE #—", bg=C["bg3"], fg=C["accent"],
            font=("Courier New", 11, "bold"))
        self._detail_title.pack(side=tk.LEFT, padx=12, pady=8)
        FlatButton(hdr, "✕", command=self._hide_detail_panel,
                   bg=C["bg3"], fg=C["text2"],
                   font=("Courier New", 11, "bold"),
                   pad_x=10, pad_y=4).pack(side=tk.RIGHT, padx=4, pady=4)

        # Body — populated dynamically
        self._detail_body = tk.Frame(self._detail, bg=C["panel"])
        self._detail_body.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

    def _populate_detail(self, idx, d):
        # Set height based on content (will fix at ~430)
        self._detail.config(height=460)

        self._detail_title.config(text=f"COOKIE  #{idx:02d}")
        for w in self._detail_body.winfo_children():
            w.destroy()

        # Status pill
        is_lop = d['lopsided']
        pill_bg = C["red"] if is_lop else C["green"]
        pill_fg = C["bg"]
        status_txt = "LOPSIDED" if is_lop else "WITHIN SPEC"
        pill = tk.Label(self._detail_body, text=f"  {status_txt}  ",
                        bg=pill_bg, fg=pill_fg,
                        font=("Courier New", 9, "bold"), padx=8, pady=4)
        pill.pack(anchor=tk.W, pady=(0, 12))

        # Helper to render a key/value row
        def kv_row(label, value, color=C["text"], unit="", note=""):
            row = tk.Frame(self._detail_body, bg=C["panel"])
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label, bg=C["panel"], fg=C["text3"],
                     font=FONT_SMALL, anchor=tk.W,
                     width=18).pack(side=tk.LEFT)
            val = tk.Label(row, text=value, bg=C["panel"], fg=color,
                           font=("Courier New", 11, "bold"))
            val.pack(side=tk.LEFT)
            if unit:
                tk.Label(row, text=f" {unit}", bg=C["panel"],
                         fg=C["text3"], font=FONT_SMALL).pack(side=tk.LEFT)
            if note:
                tk.Label(row, text=f"   {note}", bg=C["panel"],
                         fg=C["text3"],
                         font=FONT_SMALL).pack(side=tk.LEFT)

        # Section: Dimensions
        tk.Label(self._detail_body, text="DIMENSIONS",
                 bg=C["panel"], fg=C["accent"],
                 font=FONT_SMALL).pack(anchor=tk.W, pady=(4, 6))
        kv_row("Max Feret",        f"{d['feret_mm']:.2f}",      C["cyan"],    "mm",
               "longest span")
        kv_row("Min Feret",        f"{d['min_feret_mm']:.2f}",  C["green"],   "mm",
               "narrowest span")
        kv_row("Through-center",   f"{d['tc_mm']:.2f}",         C["magenta"], "mm",
               "longest chord through centroid")

        # Spacer
        tk.Frame(self._detail_body, bg=C["border"], height=1).pack(
            fill=tk.X, pady=10)

        # Section: Shape analysis
        tk.Label(self._detail_body, text="SHAPE ANALYSIS",
                 bg=C["panel"], fg=C["accent"],
                 font=FONT_SMALL).pack(anchor=tk.W, pady=(0, 6))

        elong = d['elongation']
        if elong < 1.1:
            elong_note = "near circular"
        elif elong < 1.3:
            elong_note = "slightly oval"
        elif elong < 1.6:
            elong_note = "oval"
        else:
            elong_note = "elongated"
        elong_color = C["green"] if elong < 1.3 else (
            C["yellow"] if elong < 1.6 else C["red"])
        kv_row("Elongation",   f"{elong:.2f}", elong_color, "ratio", elong_note)

        drift_pct = d['drift_ratio'] * 100
        drift_color = C["red"] if is_lop else C["green"]
        kv_row("Centroid drift", f"{drift_pct:.1f}", drift_color, "%",
               f"thr {LOPSIDED_THRESHOLD*100:.0f}%")

        # Spacer
        tk.Frame(self._detail_body, bg=C["border"], height=1).pack(
            fill=tk.X, pady=10)

        # Section: Geometry
        tk.Label(self._detail_body, text="GEOMETRY",
                 bg=C["panel"], fg=C["accent"],
                 font=FONT_SMALL).pack(anchor=tk.W, pady=(0, 6))

        cx, cy = d['center']
        kv_row("Centroid",     f"({cx}, {cy})",  C["text"], "px")
        kv_row("Encl. radius", f"{d['mec_radius']}", C["text"], "px",
               f"= {d['mec_radius']/SETTINGS.pixel_to_mm:.2f} mm")

        # Hint footer
        tk.Frame(self._detail_body, bg=C["border"], height=1).pack(
            fill=tk.X, pady=10)
        tk.Label(self._detail_body,
                 text="Click another cookie to inspect it,\n"
                      "or click empty space to close.",
                 bg=C["panel"], fg=C["text3"], font=FONT_SMALL,
                 justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

    # ──────────────────────────────────────────
    # Save / Export
    # ──────────────────────────────────────────
    # ──────────────────────────────────────────
    # Live camera mode
    # ──────────────────────────────────────────
    def on_toggle_live(self):
        if self._live_mode:
            self._stop_live()
        else:
            self._start_live()

    def _start_live(self):
        if not PICAMERA_AVAILABLE:
            messagebox.showwarning(
                "Camera unavailable",
                "picamera2 is not installed. "
                "Run on a Raspberry Pi with the camera stack:\n\n"
                "    sudo apt install -y python3-picamera2")
            return
        try:
            if self._camera is None:
                self._camera = Camera(size=(1280, 720))
            self._camera.start()
        except Exception as e:
            messagebox.showerror("Camera error",
                                 f"Could not start camera:\n{e}")
            self._camera = None
            return

        # Switch button to "STOP" state and update visuals
        self._live_mode = True
        self.btn_live.config(text="■  STOP LIVE", bg=C["red"])
        self.btn_live._bg  = C["red"]
        self.btn_live._dim = FlatButton._darken(C["red"])
        self._file_lbl.config(text="● LIVE FEED", fg=C["red"])
        self._set_status("Live feed active — press CALIBRATE or RUN INSPECTION",
                         "green")
        # Clear any prior detection state
        self.cookies      = []
        self.yolo_plotted = None
        self._selected_idx = None
        if self._detail is not None:
            self._detail.place_forget()
        # Kick off the frame pump
        self._tick_live()

    def _stop_live(self):
        self._live_mode = False
        if self._live_after_id is not None:
            try:
                self.root.after_cancel(self._live_after_id)
            except Exception:
                pass
            self._live_after_id = None
        if self._camera is not None:
            self._camera.stop()
        # Restore button
        self.btn_live.config(text="📷  LIVE", bg=C["bg3"])
        self.btn_live._bg  = C["bg3"]
        self.btn_live._dim = FlatButton._darken(C["bg3"])
        self._set_status("Live feed stopped.", "green")

    def _tick_live(self):
        """Grab one frame from the camera and refresh the display."""
        if not self._live_mode or self._camera is None:
            return
        frame = self._camera.get_frame()
        if frame is not None:
            self.orig_img = frame
            # Live mode: don't carry over stale YOLO overlay between frames
            if not self._busy:
                self.yolo_plotted = None
                self.cookies      = []
            h, w = frame.shape[:2]
            self._dim_lbl.config(text=f"{w} × {h} px · LIVE")
            self._refresh_display()
        # Aim for ~30 FPS; the inference itself runs separately on demand
        self._live_after_id = self.root.after(33, self._tick_live)

    # ──────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────
    def on_calibrate(self):
        if self._calibrating:
            return
        if self.orig_img is None:
            messagebox.showinfo(
                "No frame",
                "Capture a frame first — start LIVE feed or upload an image.")
            return
        if self.model is None:
            messagebox.showwarning(
                "No model",
                "Model not loaded — set path and click LOAD.")
            return
        # Run in background so the GUI doesn't block
        self._calibrating = True
        self.btn_calibrate.set_state(False)
        threading.Thread(target=self._calibration_worker,
                         daemon=True).start()

    def _calibration_worker(self):
        try:
            # Snapshot current frame so it can't change mid-process
            frame = self.orig_img.copy()
            self._set_status("Calibrating — segmenting cookie…", "yellow")
            results = self.model.predict(
                source=frame,
                conf=max(0.4, float(self.v_conf.get())),
                retina_masks=True,
                verbose=False)
            if not results:
                self.root.after(0, lambda: self._calibration_failed(
                    "No model output."))
                return
            r = results[0]
            if r.masks is None or len(r.masks) == 0:
                self.root.after(0, lambda: self._calibration_failed(
                    "No cookie detected.\n"
                    "Place a single cookie clearly in view and try again."))
                return
            n = len(r.masks)
            if n > 1:
                # Pick the largest mask (by area). For calibration the user
                # is supposed to put just one cookie, but if extras sneak in
                # we still want to do the right thing.
                areas = []
                for mi in r.masks:
                    mc = mi.xy[0].astype(np.int32)
                    if len(mc) < 3:
                        areas.append(0.0)
                        continue
                    areas.append(cv2.contourArea(mc))
                best_i = int(np.argmax(areas))
                mask_inst = r.masks[best_i]
                msg_extra = (f" ({n} detected — used the largest)" )
            else:
                mask_inst = r.masks[0]
                msg_extra = ""

            mc = mask_inst.xy[0].astype(np.int32)
            if len(mc) < 3:
                self.root.after(0, lambda: self._calibration_failed(
                    "Detected mask is degenerate."))
                return

            # Compute max Feret in pixels
            _, feret_px = max_feret_diameter(mc)
            if feret_px <= 0:
                self.root.after(0, lambda: self._calibration_failed(
                    "Could not compute Feret diameter."))
                return

            # Keep the YOLO-plotted preview so the user can see what was matched
            self.yolo_plotted = r.plot(boxes=True, labels=True, conf=True)
            self.root.after(0,
                lambda: self._prompt_real_diameter(feret_px, msg_extra))
        except Exception as e:
            self.root.after(0, lambda: self._calibration_failed(str(e)))
        finally:
            self._calibrating = False
            self.root.after(0, lambda: self.btn_calibrate.set_state(True))

    def _prompt_real_diameter(self, feret_px, msg_extra=""):
        """Ask the user for the cookie's real-world diameter (mm)."""
        # Show preview while asking
        self._refresh_display()
        # Default suggestion based on current calibration so common cookies
        # only need a quick Enter to confirm
        suggested = feret_px / SETTINGS.pixel_to_mm
        msg = (
            f"Detected max Feret: {feret_px:.1f} px{msg_extra}\n"
            f"(approx {suggested:.1f} mm at current calibration)\n\n"
            "Enter the cookie's actual diameter in mm:")
        try:
            value = simpledialog.askfloat(
                "Calibration",
                msg,
                parent=self.root,
                minvalue=0.1, maxvalue=10000.0,
                initialvalue=round(suggested, 1))
        except Exception:
            value = None
        if value is None or value <= 0:
            self._set_status("Calibration cancelled.", "yellow")
            return
        new_ratio = feret_px / value
        SETTINGS.pixel_to_mm   = new_ratio
        SETTINGS.calibrated_at = time.strftime("%Y-%m-%d %H:%M")
        SETTINGS.save()
        # Refresh the labels in the calibration section
        self._cal_value_lbl.config(text=f"{new_ratio:.3f}")
        self._cal_when_lbl.config(text=self._format_calibration_when())
        self._set_status(
            f"Calibration saved — {new_ratio:.3f} px/mm "
            f"(from {feret_px:.1f} px = {value:.1f} mm)", "green")

    def _calibration_failed(self, msg):
        self._set_status(f"Calibration failed: {msg}", "red")
        messagebox.showwarning("Calibration", msg)

    def on_reset_calibration(self):
        if not messagebox.askyesno(
                "Reset calibration",
                "Reset pixel/mm ratio to the factory default "
                f"({DEFAULT_PIXEL_TO_MM:.3f})?"):
            return
        SETTINGS.pixel_to_mm   = DEFAULT_PIXEL_TO_MM
        SETTINGS.calibrated_at = None
        SETTINGS.save()
        self._cal_value_lbl.config(text=f"{SETTINGS.pixel_to_mm:.3f}")
        self._cal_when_lbl.config(text=self._format_calibration_when())
        self._set_status("Calibration reset to default.", "green")

    def _format_calibration_when(self):
        if SETTINGS.calibrated_at:
            return f"last calibrated  {SETTINGS.calibrated_at}"
        return "factory default (uncalibrated)"

    # ──────────────────────────────────────────
    # Save / Export
    # ──────────────────────────────────────────
    def on_save(self):
        img = self._current_frame()
        if img is None:
            messagebox.showinfo("Nothing to export",
                                "Run inspection first.")
            return
        default = ("inspection_result.jpg" if not self.image_path
                   else f"{Path(self.image_path).stem}_inspected.jpg")
        path = filedialog.asksaveasfilename(
            title="Export annotated image",
            defaultextension=".jpg",
            initialfile=default,
            filetypes=[("JPEG", "*.jpg *.jpeg"),
                       ("PNG",  "*.png"),
                       ("All files", "*.*")])
        if not path:
            return
        if cv2.imwrite(path, img):
            self._set_status(f"Exported: {path}", "green")
        else:
            messagebox.showerror("Export failed",
                                 f"Cannot write: {path}")

    # ──────────────────────────────────────────
    # Status helper
    # ──────────────────────────────────────────
    def _set_status(self, msg, level="green"):
        color = {"green": C["green"],
                 "yellow": C["yellow"],
                 "red": C["red"]}.get(level, C["text2"])
        self.root.after(0, lambda: (
            self._status_var.set(msg),
            self._status_dot.config(fg=color)
        ))


# ═════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════
def main():
    root = tk.Tk()
    app = CookieInspectorApp(root)

    def _on_close():
        try:
            if app._live_mode:
                app._stop_live()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
