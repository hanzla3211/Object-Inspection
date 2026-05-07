import os
import math
import json
import threading
import tkinter as tk

from tkinter import (
    filedialog,
    messagebox,
    simpledialog,
)

from pathlib import Path

import cv2
import numpy as np

from PIL import Image, ImageTk
from ultralytics import YOLO
from picamera2 import Picamera2


# ─────────────────────────────────────────────
# Colour palette
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
FONT_VALUE  = ("Courier New", 22, "bold")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEFAULT_MODEL_PATH = "runs/segment/cookie_seg_y11/weights/best.pt"

SETTINGS_FILE = "cookie_inspector_settings.json"

DEFAULT_SETTINGS = {
    "pixel_to_mm": 5.206
}

NUM_ANGLES = 180
LOPSIDED_THRESHOLD = 0.15
DEFAULT_CONF = 0.50


# ═════════════════════════════════════════════
# Geometry / Metrology
# ═════════════════════════════════════════════
def ray_contour_intersections(contour, center, angle_rad):

    cx, cy = center
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)

    hits = []

    n = len(contour)

    for i in range(n):

        x1, y1 = float(contour[i][0]), float(contour[i][1])
        x2, y2 = float(contour[(i+1)%n][0]), float(contour[(i+1)%n][1])

        ex, ey = x2 - x1, y2 - y1

        denom = dx * ey - dy * ex

        if abs(denom) < 1e-9:
            continue

        u = (dx*(y1-cy) - dy*(x1-cx)) / denom

        if 0.0 <= u <= 1.0:
            hits.append((x1 + u*ex, y1 + u*ey))

    return hits


def longest_through_center_chord(contour, center, num_angles):

    best_len = 0.0
    best_chord = None

    for angle in np.linspace(0.0, math.pi, num_angles, endpoint=False):

        pts = ray_contour_intersections(contour, center, angle)

        if len(pts) < 2:
            continue

        dx, dy = math.cos(angle), math.sin(angle)

        proj = sorted(
            [(p[0]*dx + p[1]*dy, p) for p in pts],
            key=lambda x: x[0]
        )

        p_min = proj[0][1]
        p_max = proj[-1][1]

        length = math.hypot(
            p_max[0]-p_min[0],
            p_max[1]-p_min[1]
        )

        if length > best_len:
            best_len = length
            best_chord = (p_min, p_max)

    return best_chord, best_len


def max_feret_diameter(contour):

    hull = cv2.convexHull(
        contour.reshape(-1,1,2)
    ).reshape(-1,2)

    n = len(hull)

    best_len = 0.0
    best_pair = None

    for i in range(n):

        for j in range(i+1, n):

            d = math.hypot(
                float(hull[i][0]) - float(hull[j][0]),
                float(hull[i][1]) - float(hull[j][1])
            )

            if d > best_len:

                best_len = d

                best_pair = (
                    tuple(hull[i].astype(int)),
                    tuple(hull[j].astype(int))
                )

    return best_pair, best_len


def min_feret_diameter(contour):

    hull = cv2.convexHull(
        contour.reshape(-1,1,2)
    ).reshape(-1,2).astype(float)

    n = len(hull)

    best_len = float('inf')
    best_pair = None

    for i in range(n):

        p1 = hull[i]
        p2 = hull[(i+1) % n]

        edge = p2 - p1

        edge_len = math.hypot(edge[0], edge[1])

        if edge_len < 1e-9:
            continue

        ux, uy = edge[0]/edge_len, edge[1]/edge_len

        px, py = -uy, ux

        projs = [v[0]*px + v[1]*py for v in hull]

        lo, hi = min(projs), max(projs)

        width = hi - lo

        if width < best_len:

            best_len = width

            idx_lo = projs.index(lo)
            idx_hi = projs.index(hi)

            edge_proj_lo = hull[idx_lo][0]*ux + hull[idx_lo][1]*uy
            edge_proj_hi = hull[idx_hi][0]*ux + hull[idx_hi][1]*uy

            edge_mid = (edge_proj_lo + edge_proj_hi)/2

            pt_a = (
                int(round(edge_mid*ux + lo*px)),
                int(round(edge_mid*uy + lo*py))
            )

            pt_b = (
                int(round(edge_mid*ux + hi*px)),
                int(round(edge_mid*uy + hi*py))
            )

            best_pair = (pt_a, pt_b)

    return best_pair, best_len


def lopsidedness_score(centroid, contour):

    (mx, my), r = cv2.minEnclosingCircle(
        contour.reshape(-1,1,2)
    )

    drift = math.hypot(
        centroid[0]-mx,
        centroid[1]-my
    )

    ratio = drift / r if r > 1e-6 else 0.0

    return ratio


def analyze_cookie(mask_coords, raw_mask, pixel_to_mm):

    M = cv2.moments(mask_coords)

    if M['m00'] == 0:
        return None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    tc_chord, tc_len_px = longest_through_center_chord(
        mask_coords,
        (cx, cy),
        NUM_ANGLES
    )

    feret_pair, feret_len_px = max_feret_diameter(mask_coords)

    min_feret_pair, min_feret_len_px = min_feret_diameter(mask_coords)

    drift_ratio = lopsidedness_score(
        (cx, cy),
        mask_coords
    )

    return {
        'mask': raw_mask,
        'center': (cx, cy),

        'tc_chord': tc_chord,
        'tc_mm': tc_len_px / pixel_to_mm if tc_len_px else 0.0,

        'feret_pair': feret_pair,
        'feret_mm': feret_len_px / pixel_to_mm if feret_len_px else 0.0,

        'min_feret_pair': min_feret_pair,
        'min_feret_mm': min_feret_len_px / pixel_to_mm if min_feret_len_px else 0.0,

        'elongation': (
            feret_len_px / min_feret_len_px
            if min_feret_len_px > 1e-6 else 1.0
        ),

        'lopsided': drift_ratio > LOPSIDED_THRESHOLD,
        'drift_ratio': drift_ratio,
    }


# ═════════════════════════════════════════════
# Flat Button
# ═════════════════════════════════════════════
class FlatButton(tk.Label):

    def __init__(
        self,
        parent,
        text,
        command=None,
        bg=C["accent"],
        fg=C["bg"],
        font=FONT_LABEL,
        pad_x=16,
        pad_y=7,
        **kwargs
    ):

        super().__init__(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            font=font,
            cursor="hand2",
            padx=pad_x,
            pady=pad_y,
            **kwargs
        )

        self._bg = bg
        self._cmd = command

        self.bind("<Button-1>", self._on_click)

    def _on_click(self, _):

        if self._cmd:
            self._cmd()

    def set_state(self, enabled):

        if enabled:
            self.bind("<Button-1>", self._on_click)
        else:
            self.unbind("<Button-1>")


# ═════════════════════════════════════════════
# Main App
# ═════════════════════════════════════════════
class CookieInspectorApp:

    def __init__(self, root):

        self.root = root

        self.root.title("Cookie Inspector Pro")
        self.root.geometry("1400x900")
        self.root.configure(bg=C["bg"])

        self._load_settings()

        self.model = None

        self.model_path = tk.StringVar(
            value=DEFAULT_MODEL_PATH
        )

        self.orig_img = None
        self.image_path = None

        self.cookies = []

        self.picam2 = None
        self.camera_running = False
        self.live_frame = None

        self.v_conf = tk.DoubleVar(value=DEFAULT_CONF)

        self._photo = None

        self._build()

        self._load_model_bg(
            self.model_path.get(),
            silent=True
        )

    # ──────────────────────────────────────────
    # Settings
    # ──────────────────────────────────────────
    def _load_settings(self):

        if os.path.exists(SETTINGS_FILE):

            try:
                with open(SETTINGS_FILE, "r") as f:
                    self.settings = json.load(f)

            except Exception:
                self.settings = DEFAULT_SETTINGS.copy()

        else:
            self.settings = DEFAULT_SETTINGS.copy()

    def _save_settings(self):

        with open(SETTINGS_FILE, "w") as f:
            json.dump(self.settings, f, indent=4)

    # ──────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────
    def _build(self):

        hdr = tk.Frame(
            self.root,
            bg=C["bg2"]
        )

        hdr.pack(fill=tk.X)

        tk.Label(
            hdr,
            text="COOKIE INSPECTOR PRO",
            bg=C["bg2"],
            fg=C["accent"],
            font=FONT_TITLE
        ).pack(side=tk.LEFT, padx=20, pady=12)

        btns = tk.Frame(hdr, bg=C["bg2"])
        btns.pack(side=tk.RIGHT, padx=12)

        FlatButton(
            btns,
            "UPLOAD IMAGE",
            command=self.on_upload,
            bg=C["bg3"],
            fg=C["text"]
        ).pack(side=tk.RIGHT, padx=4)

        FlatButton(
            btns,
            "RUN INSPECTION",
            command=self.on_run,
            bg=C["accent"],
            fg=C["bg"]
        ).pack(side=tk.RIGHT, padx=4)

        FlatButton(
            btns,
            "LIVE CAMERA",
            command=self.toggle_camera,
            bg=C["bg3"],
            fg=C["text"]
        ).pack(side=tk.RIGHT, padx=4)

        FlatButton(
            btns,
            "CALIBRATE",
            command=self.calibrate_camera,
            bg=C["yellow_dim"],
            fg=C["bg"]
        ).pack(side=tk.RIGHT, padx=4)

        side = tk.Frame(
            self.root,
            bg=C["bg2"],
            width=320
        )

        side.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(
            side,
            text="Calibration",
            bg=C["bg2"],
            fg=C["text3"],
            font=FONT_SMALL
        ).pack(anchor=tk.W, padx=12, pady=(20,0))

        self._cal_lbl = tk.Label(
            side,
            text=f"{self.settings['pixel_to_mm']:.4f} px/mm",
            bg=C["bg2"],
            fg=C["yellow"],
            font=FONT_MONO
        )

        self._cal_lbl.pack(anchor=tk.W, padx=12)

        self._status = tk.Label(
            side,
            text="Ready",
            bg=C["bg2"],
            fg=C["green"],
            font=FONT_SMALL
        )

        self._status.pack(anchor=tk.W, padx=12, pady=(20,0))

        self._canvas = tk.Canvas(
            self.root,
            bg=C["bg"],
            highlightthickness=0
        )

        self._canvas.pack(
            fill=tk.BOTH,
            expand=True
        )

    # ──────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────
    def _set_status(self, msg, level="green"):

        color = {
            "green": C["green"],
            "yellow": C["yellow"],
            "red": C["red"]
        }.get(level, C["text"])

        self.root.after(
            0,
            lambda: self._status.config(
                text=msg,
                fg=color
            )
        )

    # ──────────────────────────────────────────
    # Camera
    # ──────────────────────────────────────────
    def toggle_camera(self):

        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):

        try:

            self.picam2 = Picamera2()

            config = self.picam2.create_preview_configuration(
                main={
                    "format": "RGB888",
                    "size": (1280, 720)
                }
            )

            self.picam2.configure(config)

            self.picam2.start()

            self.camera_running = True

            self._set_status(
                "Camera started",
                "green"
            )

            self.update_camera_feed()

        except Exception as e:

            messagebox.showerror(
                "Camera Error",
                str(e)
            )

    def stop_camera(self):

        self.camera_running = False

        if self.picam2:
            self.picam2.stop()
            self.picam2 = None

        self._set_status(
            "Camera stopped",
            "yellow"
        )

    def update_camera_feed(self):

        if not self.camera_running:
            return

        frame = self.picam2.capture_array()

        frame_bgr = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        self.live_frame = frame_bgr

        self.orig_img = frame_bgr

        self._refresh_display()

        self.root.after(
            30,
            self.update_camera_feed
        )

    # ──────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────
    def calibrate_camera(self):

        if self.live_frame is None:

            messagebox.showwarning(
                "No Camera",
                "Start camera first."
            )

            return

        if self.model is None:

            messagebox.showwarning(
                "No Model",
                "Load YOLO model first."
            )

            return

        try:

            frame = self.live_frame.copy()

            results = self.model.predict(
                source=frame,
                conf=float(self.v_conf.get()),
                retina_masks=True,
                verbose=False
            )

            if not results:

                messagebox.showerror(
                    "Calibration Error",
                    "No detection result."
                )

                return

            r = results[0]

            if r.masks is None:

                messagebox.showerror(
                    "Calibration Error",
                    "No cookie detected."
                )

                return

            mi = r.masks[0]

            mc = mi.xy[0].astype(np.int32)

            pair, feret_px = max_feret_diameter(mc)

            if feret_px <= 0:

                messagebox.showerror(
                    "Calibration Error",
                    "Could not compute Feret."
                )

                return

            real_mm = simpledialog.askfloat(
                "Calibration",
                f"Detected Max Feret:\n"
                f"{feret_px:.2f} pixels\n\n"
                f"Enter REAL cookie diameter (mm):",
                minvalue=1.0
            )

            if real_mm is None:
                return

            new_pixel_to_mm = feret_px / real_mm

            self.settings["pixel_to_mm"] = new_pixel_to_mm

            self._save_settings()

            self._cal_lbl.config(
                text=f"{new_pixel_to_mm:.4f} px/mm"
            )

            self._set_status(
                "Calibration updated",
                "green"
            )

            messagebox.showinfo(
                "Calibration Complete",
                f"New value:\n\n"
                f"{new_pixel_to_mm:.4f} px/mm"
            )

        except Exception as e:

            messagebox.showerror(
                "Calibration Error",
                str(e)
            )

    # ──────────────────────────────────────────
    # Upload
    # ──────────────────────────────────────────
    def on_upload(self):

        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp")
            ]
        )

        if not path:
            return

        img = cv2.imread(path)

        if img is None:

            messagebox.showerror(
                "Error",
                "Could not load image."
            )

            return

        self.image_path = path
        self.orig_img = img

        self._refresh_display()

    # ──────────────────────────────────────────
    # Model
    # ──────────────────────────────────────────
    def _load_model_bg(self, path, silent=False):

        def worker():

            try:

                self._set_status(
                    "Loading model...",
                    "yellow"
                )

                self.model = YOLO(path)

                self._set_status(
                    "Model loaded",
                    "green"
                )

            except Exception as e:

                self.model = None

                self._set_status(
                    "Model load failed",
                    "red"
                )

                if not silent:
                    messagebox.showerror(
                        "Model Error",
                        str(e)
                    )

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    # ──────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────
    def on_run(self):

        if self.orig_img is None:

            messagebox.showinfo(
                "No image",
                "Load image or start camera."
            )

            return

        if self.model is None:

            messagebox.showwarning(
                "No model",
                "Model not loaded."
            )

            return

        threading.Thread(
            target=self._run_worker,
            daemon=True
        ).start()

    def _run_worker(self):

        try:

            self._set_status(
                "Running inference...",
                "yellow"
            )

            results = self.model.predict(
                source=self.orig_img,
                conf=float(self.v_conf.get()),
                retina_masks=True,
                verbose=False
            )

            r = results[0]

            display = self.orig_img.copy()

            self.cookies = []

            if r.masks is not None:

                for mi in r.masks:

                    mc = mi.xy[0].astype(np.int32)

                    raw_mask = mi.data[0].cpu().numpy()

                    raw_mask = cv2.resize(
                        raw_mask,
                        (
                            self.orig_img.shape[1],
                            self.orig_img.shape[0]
                        )
                    )

                    d = analyze_cookie(
                        mc,
                        raw_mask,
                        self.settings["pixel_to_mm"]
                    )

                    if d:

                        self.cookies.append(d)

                        if d['feret_pair']:

                            cv2.line(
                                display,
                                d['feret_pair'][0],
                                d['feret_pair'][1],
                                (0,255,255),
                                2
                            )

                            cx, cy = d['center']

                            cv2.putText(
                                display,
                                f"{d['feret_mm']:.1f}mm",
                                (cx-40, cy-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0,255,255),
                                2
                            )

                        overlay = display.copy()

                        overlay[raw_mask > 0.5] = (0,255,0)

                        cv2.addWeighted(
                            overlay,
                            0.3,
                            display,
                            0.7,
                            0,
                            display
                        )

            self.orig_img = display

            self.root.after(
                0,
                self._refresh_display
            )

            self._set_status(
                f"{len(self.cookies)} cookies detected",
                "green"
            )

        except Exception as e:

            self._set_status(
                "Inference failed",
                "red"
            )

            messagebox.showerror(
                "Inference Error",
                str(e)
            )

    # ──────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────
    def _refresh_display(self):

        if self.orig_img is None:
            return

        img = self.orig_img.copy()

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        h, w = rgb.shape[:2]

        cw = max(
            self._canvas.winfo_width(),
            1
        )

        ch = max(
            self._canvas.winfo_height(),
            1
        )

        scale = min(cw/w, ch/h)

        nw = int(w * scale)
        nh = int(h * scale)

        rgb = cv2.resize(
            rgb,
            (nw, nh)
        )

        self._photo = ImageTk.PhotoImage(
            Image.fromarray(rgb)
        )

        self._canvas.delete("all")

        self._canvas.create_image(
            cw//2,
            ch//2,
            image=self._photo,
            anchor=tk.CENTER
        )


# ═════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════
def main():

    root = tk.Tk()

    CookieInspectorApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()
