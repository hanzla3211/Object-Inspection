"""
Cookie Inspector Pro â€” YOLO11 + Feret Metrology
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
import traceback
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
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

# matplotlib (optional, for embedded texture/histogram plots)
try:
    import matplotlib
    matplotlib.use("Agg")  # headless drawing; we embed via FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    from matplotlib.figure   import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

# scikit-image (optional, for LBP texture)
try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

# RPi.GPIO is only available on a Raspberry Pi. Imported optionally so
# the app still runs on dev machines without it.
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception:
    GPIO = None
    GPIO_AVAILABLE = False

# GPIO pin assignments for the hardware quick-capture button + buzzer.
GPIO_BUTTON_PIN = 23
GPIO_BUZZER_PIN = 24


class GPIOQuickCaptureTrigger:
    """Hardware button trigger for quick capture.

    Polls a momentary push-button on ``GPIO_BUTTON_PIN`` (active LOW with
    internal pull-up) on a background thread, beeps a buzzer on
    ``GPIO_BUZZER_PIN``, and dispatches the callback onto the Tk main
    thread via ``root.after`` (Tkinter is not thread-safe).

    Press detection:
        Edge-triggered, not level-triggered. A press only fires on the
        HIGH -> LOW transition. The loop refuses to fire until it has
        observed the pin sitting HIGH (i.e. button released) for at
        least one poll cycle. This prevents a spurious "press" at
        startup when:
          * the pin is briefly read LOW before the internal pull-up
            settles after GPIO.setup, or
          * the systemd-launched process opens GPIO while the line is
            transiently floating low from boot.
        It also means holding the button down on power-on does NOT
        trigger inspection -- the user must actually release and press.

    Buzzer:
        Driven as a steady HIGH for ``BEEP_DURATION`` seconds. Correct
        waveform for an active buzzer (built-in oscillator) -- gets
        full supply voltage and plays at rated loudness. Beeps run on
        a dedicated daemon thread so the polling loop and Tk UI never
        block, lock-guarded so overlapping beeps don't clip each other.
    """

    POLL_INTERVAL    = 0.02   # seconds between button reads
    DEBOUNCE_DELAY   = 0.20   # seconds after a press before next is accepted
    BEEP_DURATION    = 0.18   # seconds the buzzer stays high per beep
    STARTUP_SETTLE   = 0.30   # seconds to wait before the loop arms itself

    def __init__(self, root, on_press,
                 button_pin=GPIO_BUTTON_PIN,
                 buzzer_pin=GPIO_BUZZER_PIN):
        self._root       = root
        self._on_press   = on_press
        self._button_pin = button_pin
        self._buzzer_pin = buzzer_pin
        self._stop       = threading.Event()
        self._thread     = None
        self._ok         = False
        # Serialise beep threads so overlapping calls don't fight over
        # the pin state (one drops the pin LOW while another is
        # mid-beep, which would clip the buzz short).
        self._beep_lock  = threading.Lock()

        if not GPIO_AVAILABLE:
            print("[GPIO] RPi.GPIO not available â€” hardware trigger disabled.")
            return
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self._buzzer_pin, GPIO.OUT)
            GPIO.output(self._buzzer_pin, GPIO.LOW)
            self._ok = True
        except Exception as e:
            print(f"[GPIO] setup failed: {e} â€” hardware trigger disabled.")
            self._ok = False

    def start(self):
        if not self._ok or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="GPIOTrigger", daemon=True)
        self._thread.start()
        print(f"[GPIO] watching pin {self._button_pin} for quick capture.")

    def _beep_blocking(self):
        """Actually drive the buzzer. Runs on a dedicated thread so it
        never holds up the GPIO polling loop or the Tk main loop.

        Lock-guarded so overlapping beep() calls queue up rather than
        racing each other to drop the pin LOW mid-pulse.
        """
        if not self._ok:
            return
        with self._beep_lock:
            try:
                GPIO.output(self._buzzer_pin, GPIO.HIGH)
                time.sleep(self.BEEP_DURATION)
                GPIO.output(self._buzzer_pin, GPIO.LOW)
            except Exception as e:
                print(f"[GPIO] beep failed: {e}")
                try:
                    GPIO.output(self._buzzer_pin, GPIO.LOW)
                except Exception:
                    pass

    def beep(self):
        """Public, non-blocking buzzer trigger.

        Safe to call from any thread (Tk main loop included): the
        actual GPIO work runs on a short-lived daemon thread, so the
        caller returns immediately. No-op if GPIO isn't available.

        Called from ``on_quick_capture`` so that EVERY inspection
        trigger -- Q hotkey, on-screen QUICK CAPTURE button, hardware
        GPIO button -- produces an audible beep.
        """
        if not self._ok:
            return
        threading.Thread(
            target=self._beep_blocking,
            name="GPIOBeep", daemon=True).start()

    def _run(self):
        """Button polling loop with edge detection.

        Two startup guards prevent the auto-fire-on-boot problem:
          1. A short settle delay (``STARTUP_SETTLE``) before the first
             read, giving the GPIO subsystem time to engage the
             internal pull-up after ``GPIO.setup``.
          2. A ``primed`` flag: the loop will not treat a LOW as a
             press until it has first seen the pin HIGH. So if the
             line is already LOW when the loop starts (transient or
             held button), no inspection fires -- the operator must
             release and re-press for a clean HIGH -> LOW edge.
        """
        # Settle: let the pull-up stabilise before we look at the pin.
        if self._stop.wait(self.STARTUP_SETTLE):
            return

        # The loop only fires on a HIGH -> LOW edge, and it only arms
        # itself ('primed') after observing at least one HIGH read.
        primed    = False
        last_high = False

        while not self._stop.is_set():
            try:
                level = GPIO.input(self._button_pin)
                if level == GPIO.HIGH:
                    last_high = True
                    primed    = True
                elif primed and last_high and level == GPIO.LOW:
                    # Clean falling edge -- the button was released and
                    # is now pressed. Fire the callback.
                    last_high = False
                    try:
                        self._root.after(0, self._on_press)
                    except Exception:
                        pass
                    # Debounce: ignore further reads briefly so a
                    # single press doesn't fire multiple captures.
                    if self._stop.wait(self.DEBOUNCE_DELAY):
                        break
                    continue
                else:
                    # LOW but not yet primed (held at boot, or noise
                    # before first HIGH). Track it but do not fire.
                    last_high = False
            except Exception as e:
                print(f"[GPIO] read failed: {e}")
                if self._stop.wait(0.5):
                    break
                continue
            if self._stop.wait(self.POLL_INTERVAL):
                break

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._ok:
            try:
                GPIO.output(self._buzzer_pin, GPIO.LOW)
                GPIO.cleanup()
            except Exception as e:
                print(f"[GPIO] cleanup failed: {e}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Colour palette  (dark industrial / precision)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Config
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Resolved relative to this script's directory so the app finds the
# weights regardless of the working directory the user launched it from.
# Layout expected on disk:
#     <script dir>/
#       final_app.py
#       640/
#         best.pt
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "640" / "best.pt")
DEFAULT_PIXEL_TO_MM = 5.206
NUM_ANGLES         = 180
LOPSIDED_THRESHOLD = 0.15
DEFAULT_CONF       = 0.50

# Class-name override applied to YOLO results after the model loads.
# Keys MUST be ints (Ultralytics uses int keys internally) and must match
# the class indices the model was trained with.
CLASS_NAMES = {
    0: "midnight",
    1: "choc_walnut",
    2: "cinnamon_roll",
    3: "peanut_butter",
    4: "red_velvet",
    5: "classic_choc",
    6: "lotus_lava",
    7: "kunafa",
    8: "normal",
}


def class_label(cls_id):
    """Return the human-readable name for a class id, falling back to the id."""
    if cls_id is None:
        return "unknown"
    return CLASS_NAMES.get(int(cls_id), f"cls_{int(cls_id)}")


# Per-class info â€” single source of truth for both the marketing
# display name AND the production-line spec targets. Each entry:
#   "full_name":      marketing name used in the walkthrough popup
#                     dropdown / title.
#   "ideal_diameter": target diameter in mm.
#   "ideal_min":      lower acceptable texture bound (from the
#                     histogram calculator's batch flow).
#   "ideal_max":      upper acceptable texture bound.
# Texture bounds below are placeholders and should be replaced with
# the actual per-class targets measured on the production line. Add
# a new entry here whenever CLASS_NAMES gains a class.
COOKIE_INFO = {
    "midnight":      {"full_name": "Midnight Cookies & Cream Cookie",
                      "ideal_diameter":  95.0, "ideal_min": -2.0, "ideal_max": 2.0},
    "choc_walnut":   {"full_name": "Chocolate Chip Walnut Cookie",
                      "ideal_diameter": 100.0, "ideal_min": -1.5, "ideal_max": 2.5},
    "cinnamon_roll": {"full_name": "Cinnamon Roll Cookie",
                      "ideal_diameter":  98.0, "ideal_min": -2.0, "ideal_max": 2.0},
    "peanut_butter": {"full_name": "Peanut Butter Cookie",
                      "ideal_diameter":  92.0, "ideal_min": -1.0, "ideal_max": 3.0},
    "red_velvet":    {"full_name": "Red Velvet Cookie",
                      "ideal_diameter":  97.0, "ideal_min": -2.0, "ideal_max": 2.0},
    "classic_choc":  {"full_name": "Classic Chocolate Chip Cookie",
                      "ideal_diameter": 100.0, "ideal_min": -1.5, "ideal_max": 2.5},
    "lotus_lava":    {"full_name": "Lotus Lava Cookie",
                      "ideal_diameter":  95.0, "ideal_min": -2.0, "ideal_max": 2.5},
    "kunafa":        {"full_name": "Kunafa Double Chocolate Cookie",
                      "ideal_diameter":  90.0, "ideal_min": -1.0, "ideal_max": 3.0},
    "normal":        {"full_name": "Normal Cookie",
                      "ideal_diameter":  95.0, "ideal_min": -2.0, "ideal_max": 2.0},
}


def class_full_name(short):
    """Map a short class label to its full marketing name (fallback to
    the prettified short label if no mapping exists)."""
    if short is None or short == "unknown":
        return "Unknown"
    info = COOKIE_INFO.get(short)
    if info is not None:
        return info["full_name"]
    return short.replace("_", " ").title()


def ideal_info(label):
    """Return (ideal_diameter, ideal_min, ideal_max) for a class label,
    or (None, None, None) if the label isn't in COOKIE_INFO."""
    if label is None:
        return (None, None, None)
    info = COOKIE_INFO.get(label)
    if info is None:
        return (None, None, None)
    return (info["ideal_diameter"], info["ideal_min"], info["ideal_max"])

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Persistent settings
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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


# Global instance â€” analyze_cookie reads from this rather than a constant
SETTINGS = Settings()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Camera wrapper (picamera2)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class Camera:
    """
    Thin wrapper around picamera2 that hides start/stop lifecycle
    and lets the GUI poll the latest BGR frame.

    Frames returned from `get_frame()` are in BGR order (OpenCV-style),
    matching the rest of the pipeline (cv2.imread, YOLO results, etc.).

    Resolution strategy (tuned for IMX219 â€” 3280x2464 max):
      1. Try the camera's max sensor size (full FOV, full resolution).
      2. If that fails, fall back to a 1920x1080 preview.
      3. If THAT fails, fall back to picamera2's default config.
    """

    # Preferred max resolution to try first. For IMX219 this is 3280x2464.
    # The code will auto-detect the camera's actual max and clamp to it.
    PREFERRED_W = 3280
    PREFERRED_H = 2464

    # Mid-tier fallback if max-res preview is rejected
    FALLBACK_W = 1920
    FALLBACK_H = 1080

    def __init__(self):
        if not PICAMERA_AVAILABLE:
            raise RuntimeError(
                "picamera2 is not installed. "
                "Run: sudo apt install -y python3-picamera2")
        self._cam     = None
        self._running = False

    @staticmethod
    def _detect_max_size(cam):
        """Return (w, h) of the camera's largest available sensor mode."""
        try:
            modes = cam.sensor_modes or []
            if modes:
                largest = max(modes, key=lambda m: m["size"][0] * m["size"][1])
                return tuple(largest["size"])
        except Exception:
            pass
        try:
            props = cam.camera_properties
            return tuple(props.get("PixelArraySize", (1920, 1080)))
        except Exception:
            return (1920, 1080)

    def _try_configure(self, main_size, sensor_size=None):
        """Try a single config; return True on success."""
        try:
            kwargs = {"main": {"format": "RGB888", "size": main_size}}
            if sensor_size is not None:
                kwargs["sensor"] = {"output_size": sensor_size}
            config = self._cam.create_preview_configuration(**kwargs)
            self._cam.configure(config)
            self._cam.start()
            return True
        except Exception as e:
            print(f"[Camera] config main={main_size} sensor={sensor_size} "
                  f"failed: {e}")
            return False

    def start(self):
        if self._running:
            return
        self._cam = Picamera2()

        # Detect what the camera actually supports
        max_size = self._detect_max_size(self._cam)
        print(f"[Camera] detected max sensor size: {max_size}")

        # Choose the highest size we're willing to try (clamped to actual max)
        target_w = min(self.PREFERRED_W, max_size[0])
        target_h = min(self.PREFERRED_H, max_size[1])
        target = (target_w, target_h)

        # Attempt 1: max sensor + matching main output (full FOV, full res)
        if self._try_configure(target, sensor_size=target):
            print(f"[Camera] streaming at MAX RES {target}")
            time.sleep(0.5)
            self._running = True
            return

        # Attempt 2: max sensor for full FOV, but downscale main output
        # (some IMX219 setups can't deliver 3280x2464 as a main stream
        #  but can read the full sensor and ISP-downscale)
        if self._try_configure((self.FALLBACK_W, self.FALLBACK_H),
                               sensor_size=target):
            print(f"[Camera] streaming at {self.FALLBACK_W}x"
                  f"{self.FALLBACK_H} (full FOV via sensor={target})")
            time.sleep(0.5)
            self._running = True
            return

        # Attempt 3: plain preview at fallback size, no sensor override
        if self._try_configure((self.FALLBACK_W, self.FALLBACK_H)):
            print(f"[Camera] streaming at {self.FALLBACK_W}x"
                  f"{self.FALLBACK_H} (no sensor override)")
            time.sleep(0.5)
            self._running = True
            return

        # Attempt 4: bare default â€” should always work if camera works at all
        print("[Camera] falling back to picamera2 default config")
        config = self._cam.create_preview_configuration()
        self._cam.configure(config)
        self._cam.start()
        time.sleep(0.5)
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
        try:
            arr = self._cam.capture_array()
        except Exception as e:
            print(f"[Camera] capture_array failed: {e}")
            return None
        if arr is None:
            return None
        # picamera2 quirk: when configured with format="RGB888", the
        # numpy array is laid out as BGR (matches OpenCV) â€” no swap needed.
        # If we got a 4-channel buffer (XBGR8888 fallback) drop the alpha.
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    @property
    def is_running(self):
        return self._running


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Metrology core  (pure logic)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Texture analysis
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LBP parameters â€” should match across calls so per-cookie scores stay
# comparable to the batch distribution
LBP_RADIUS   = 3
LBP_N_POINTS = 24
# Hue range for "golden" / properly baked color (HSV)
GOLD_HSV_LOW  = np.array([10, 80, 50])
GOLD_HSV_HIGH = np.array([25, 255, 255])


def compute_image_texture(bgr_img):
    """
    Compute per-pixel texture-variance map for an entire BGR image.

    Pipeline (matches the reference notebook):
      gray  -> CLAHE                  -> contrast-equalised gray
            -> local_binary_pattern   -> texture pattern code per pixel
            -> local mean / variance  -> "roughness" score per pixel

    Returns:
        variance: float32 ndarray (H, W) â€” texture roughness per pixel
        hsv:      ndarray (H, W, 3)      â€” HSV image for color analysis
    Returns (None, None) if scikit-image isn't installed.
    """
    if not SKIMAGE_AVAILABLE:
        return None, cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    eq    = clahe.apply(gray)
    lbp   = local_binary_pattern(eq, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    # Local variance via two box filters: Var(X) = E[X^2] - E[X]^2
    mean_lbp  = cv2.blur(lbp,        (15, 15))
    mean_lbp2 = cv2.blur(lbp * lbp,  (15, 15))
    variance  = mean_lbp2 - mean_lbp * mean_lbp
    return variance.astype(np.float32), hsv


def analyze_cookie(mask_coords, raw_mask, variance_map=None, hsv_img=None,
                   cls_id=None):
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

    # â”€â”€ Texture & colour stats (only if maps were precomputed) â”€â”€
    binary_mask = (raw_mask > 0.5)
    tex = {}
    if variance_map is not None and binary_mask.any():
        # Pull texture-variance values for pixels inside this cookie
        vals = variance_map[binary_mask]
        # P5/P95 are robust min/max â€” they ignore single bright crumbs
        tex['tex_values'] = vals
        tex['tex_median'] = float(np.median(vals))
        tex['tex_p5']     = float(np.percentile(vals, 5))
        tex['tex_p95']    = float(np.percentile(vals, 95))

    color = {}
    if hsv_img is not None and binary_mask.any():
        hsv_pixels = hsv_img[binary_mask]
        # Existing fields (kept for back-compat with other code paths).
        color['hue_median']        = float(np.median(hsv_pixels[:, 0]))
        # New: full per-channel medians + a subsampled HSV pixel cloud
        # for the walkthrough popup's color-region picker. Subsampling
        # keeps Canvas plotting fast â€” ~3000 points is plenty to read
        # the cluster shape and renders in well under a frame on a Pi.
        color['sat_median']        = float(np.median(hsv_pixels[:, 1]))
        color['val_median']        = float(np.median(hsv_pixels[:, 2]))
        WT_HSV_SAMPLE_N = 3000
        if hsv_pixels.shape[0] > WT_HSV_SAMPLE_N:
            sel = np.random.choice(hsv_pixels.shape[0],
                                   WT_HSV_SAMPLE_N, replace=False)
            color['hsv_sample']    = hsv_pixels[sel].astype(np.uint8)
        else:
            color['hsv_sample']    = hsv_pixels.astype(np.uint8)
        # Average BGR â€” the "what the camera actually saw" swatch color.
        # Use the median (per channel) of BGR rather than mean: more
        # robust against specular highlights and edge bleed.
        bgr_pixels = cv2.cvtColor(
            hsv_pixels.reshape(-1, 1, 3),
            cv2.COLOR_HSV2BGR).reshape(-1, 3)
        color['bgr_median']        = np.median(bgr_pixels, axis=0).astype(int).tolist()
        # Bake-quality: % of cookie pixels falling in the "golden" hue range
        gold_mask = cv2.inRange(hsv_img, GOLD_HSV_LOW, GOLD_HSV_HIGH)
        n_gold    = int(np.count_nonzero(gold_mask & binary_mask.astype(np.uint8) * 255))
        n_total   = int(np.count_nonzero(binary_mask))
        color['quality_pct'] = (n_gold / n_total) * 100.0 if n_total else 0.0

    return {
        'mask':           raw_mask,
        'cls_id':         cls_id,                          # â† class id from YOLO
        'label':          class_label(cls_id),             # â† human-readable name
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
        # Texture / colour additions
        **tex,
        **color,
    }


def to_int_pt(p):
    return (int(round(p[0])), int(round(p[1])))


def render_overlays(base_img, cookies, opts):
    display = base_img.copy()
    selected_idx = opts.get('selected_idx')
    orig_img     = opts.get('orig_img')   # raw frame, used to "punch through"
                                          # the YOLO mask on the selected cookie

    for i, d in enumerate(cookies):
        is_selected = (i == selected_idx)

        # When the YOLO overlay is on it paints a coloured fill across the
        # cookie which hides its surface. For the SELECTED cookie we want
        # to see the texture, so locally restore the original pixels.
        if (is_selected and opts.get('yolo_overlay')
                and orig_img is not None and d.get('mask') is not None):
            mask_bool = d['mask'] > 0.5
            display[mask_bool] = orig_img[mask_bool]

        # Highlight the selected cookie with a non-occluding outline so
        # surface texture stays fully visible. Three layers, drawn outside
        # â†’ inside, give a "glow" effect:
        #   1. soft outer halo on a copy, alpha-blended (no fill on the cookie)
        #   2. a clean accent ring exactly on the cookie boundary
        #   3. a faint reference circle around the enclosing circle
        if is_selected and d['mask'] is not None:
            # Find the boundary contour of the YOLO mask
            mask_u8 = (d['mask'] > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # 1. Soft halo (drawn on a separate canvas, then alpha-blended
            #    so we don't permanently brighten pixels under the halo)
            if contours:
                halo = display.copy()
                cv2.drawContours(halo, contours, -1, (180, 105, 255), 20)
                cv2.addWeighted(halo, 0.45, display, 0.55, 0, display)

                # 2. Crisp accent ring on the boundary
                cv2.drawContours(display, contours, -1, (180, 105, 255), 8)

            # 3. Faint reference circle around the enclosing circle
            cv2.circle(display, d['mec_center'],
                       d['mec_radius'] + 12, (180, 105, 255), 2)

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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Custom Tkinter widgets
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
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


class RangeSlider(tk.Frame):
    """Horizontal slider with two draggable handles for setting a (lo, hi) range.

    Used in the cookie walkthrough popup to let the user define a custom
    "acceptable hue" range per cookie class. Values snap to integers.
    Fires ``on_change(lo, hi)`` whenever a handle moves (during drag),
    and ``on_release(lo, hi)`` when the user lets go â€” handy for cheaper
    redraws on release vs. live updates.
    """
    H        = 56       # total widget height
    TRACK_Y  = 32       # vertical centre of the track
    TRACK_H  = 6        # track thickness
    HANDLE_R = 9        # handle radius

    def __init__(self, parent, vmin=0, vmax=179, lo=10, hi=25,
                 width=380, on_change=None, on_release=None,
                 track_color=None, fill_color=None, **kwargs):
        # Pull our own bg out of kwargs before forwarding to tk.Frame,
        # so we can also reuse it for the inner canvas without calling
        # cget() (which has been observed to misbehave on some tk/python
        # combos when other kwargs are present).
        bg_color = kwargs.pop("bg", C["panel"])
        # Drop any kwargs that would conflict with the underlying Frame
        # (e.g. width/height â€” RangeSlider sizes itself via the canvas).
        kwargs.pop("width",  None)
        kwargs.pop("height", None)
        super().__init__(parent, bg=bg_color, **kwargs)

        self._vmin       = int(vmin)
        self._vmax       = int(vmax)
        self._lo         = int(max(vmin, min(vmax, lo)))
        self._hi         = int(max(self._lo, min(vmax, hi)))
        # NB: don't call this `self._w` â€” tk widgets store their internal
        # tcl path name in `self._w`, so shadowing it produces a cryptic
        # "invalid command name '<int>'" error from cget/configure later.
        self._slider_w   = int(width)
        self._bg         = bg_color
        self._on_change  = on_change
        self._on_release = on_release
        self._track_col  = track_color or C["border2"]
        self._fill_col   = fill_color  or C["accent"]
        self._drag       = None   # 'lo' | 'hi' | None

        self._canvas = tk.Canvas(self, width=self._slider_w, height=self.H,
                                 bg=self._bg,
                                 highlightthickness=0)
        self._canvas.pack(fill=tk.X)
        self._canvas.bind("<Button-1>",        self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_release_evt)
        self._redraw()

    # â”€â”€ public API â”€â”€
    def get(self):
        return (self._lo, self._hi)

    def set_range(self, lo, hi):
        self._lo = int(max(self._vmin, min(self._vmax, lo)))
        self._hi = int(max(self._lo,   min(self._vmax, hi)))
        self._redraw()

    def set_bounds(self, vmin, vmax):
        """Change the underlying value range (e.g. hue 0..179)."""
        self._vmin, self._vmax = int(vmin), int(vmax)
        self._lo = max(self._vmin, min(self._vmax, self._lo))
        self._hi = max(self._lo,   min(self._vmax, self._hi))
        self._redraw()

    # â”€â”€ coordinate helpers â”€â”€
    def _val_to_x(self, v):
        if self._vmax == self._vmin:
            return self.HANDLE_R
        usable = self._slider_w - 2 * self.HANDLE_R
        return (self.HANDLE_R
                + (v - self._vmin) / (self._vmax - self._vmin) * usable)

    def _x_to_val(self, x):
        usable = self._slider_w - 2 * self.HANDLE_R
        if usable <= 0:
            return self._vmin
        v = (x - self.HANDLE_R) / usable * (self._vmax - self._vmin) \
            + self._vmin
        return int(round(max(self._vmin, min(self._vmax, v))))

    # â”€â”€ drawing â”€â”€
    def _redraw(self):
        c = self._canvas
        c.delete("all")
        y      = self.TRACK_Y
        th2    = self.TRACK_H // 2
        x_lo   = self._val_to_x(self._lo)
        x_hi   = self._val_to_x(self._hi)

        # Inactive track (full width).
        c.create_rectangle(self.HANDLE_R, y - th2,
                           self._slider_w - self.HANDLE_R, y + th2,
                           fill=self._track_col, outline="")
        # Active (selected) span between the two handles.
        c.create_rectangle(x_lo, y - th2, x_hi, y + th2,
                           fill=self._fill_col, outline="")

        # Handles.
        for x, tag in ((x_lo, "lo"), (x_hi, "hi")):
            c.create_oval(x - self.HANDLE_R, y - self.HANDLE_R,
                          x + self.HANDLE_R, y + self.HANDLE_R,
                          fill=C["text"], outline=self._fill_col,
                          width=2, tags=("handle", tag))

        # Numeric labels above each handle.
        c.create_text(x_lo, y - self.HANDLE_R - 10,
                      text=str(self._lo), fill=C["text"],
                      font=FONT_SMALL)
        c.create_text(x_hi, y - self.HANDLE_R - 10,
                      text=str(self._hi), fill=C["text"],
                      font=FONT_SMALL)

        # End-stop labels (min/max).
        c.create_text(self.HANDLE_R, y + self.HANDLE_R + 10,
                      text=str(self._vmin), fill=C["text3"],
                      font=FONT_SMALL)
        c.create_text(self._slider_w - self.HANDLE_R, y + self.HANDLE_R + 10,
                      text=str(self._vmax), fill=C["text3"],
                      font=FONT_SMALL)

    # â”€â”€ mouse handlers â”€â”€
    def _nearest_handle(self, x):
        d_lo = abs(x - self._val_to_x(self._lo))
        d_hi = abs(x - self._val_to_x(self._hi))
        # Tiebreak: if exactly equal, prefer whichever side x is on.
        if d_lo == d_hi:
            return 'hi' if x > self._val_to_x(self._lo) else 'lo'
        return 'lo' if d_lo < d_hi else 'hi'

    def _on_press(self, ev):
        self._drag = self._nearest_handle(ev.x)
        self._on_motion(ev)

    def _on_motion(self, ev):
        if self._drag is None:
            return
        v = self._x_to_val(ev.x)
        if self._drag == 'lo':
            # Don't let lo overshoot hi.
            self._lo = min(v, self._hi)
        else:
            self._hi = max(v, self._lo)
        self._redraw()
        if self._on_change:
            try:
                self._on_change(self._lo, self._hi)
            except Exception as e:
                print(f"[RangeSlider] on_change error: {e}")

    def _on_release_evt(self, _ev):
        if self._drag is None:
            return
        self._drag = None
        if self._on_release:
            try:
                self._on_release(self._lo, self._hi)
            except Exception as e:
                print(f"[RangeSlider] on_release error: {e}")


class Slider(tk.Frame):
    """Horizontal slider with a single draggable handle. Integer values."""
    H        = 56
    TRACK_Y  = 32
    TRACK_H  = 6
    HANDLE_R = 9

    def __init__(self, parent, vmin=0, vmax=10, value=0,
                 width=380, on_change=None, on_release=None,
                 track_color=None, fill_color=None,
                 value_format=None, **kwargs):
        bg_color = kwargs.pop("bg", C["panel"])
        kwargs.pop("width",  None)
        kwargs.pop("height", None)
        super().__init__(parent, bg=bg_color, **kwargs)
        self._vmin = int(vmin)
        self._vmax = int(vmax)
        self._val  = int(max(self._vmin, min(self._vmax, value)))
        self._slider_w = int(width)
        self._bg = bg_color
        self._on_change  = on_change
        self._on_release = on_release
        self._track_col  = track_color or C["border2"]
        self._fill_col   = fill_color  or C["accent"]
        self._fmt        = value_format or str
        self._drag = False

        self._canvas = tk.Canvas(self, width=self._slider_w, height=self.H,
                                 bg=self._bg, highlightthickness=0)
        self._canvas.pack(fill=tk.X)
        self._canvas.bind("<Button-1>",        self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_release_evt)
        self._redraw()

    def get(self):
        return self._val

    def set_value(self, v):
        self._val = int(max(self._vmin, min(self._vmax, v)))
        self._redraw()

    def _val_to_x(self, v):
        if self._vmax == self._vmin:
            return self.HANDLE_R
        usable = self._slider_w - 2 * self.HANDLE_R
        return (self.HANDLE_R
                + (v - self._vmin) / (self._vmax - self._vmin) * usable)

    def _x_to_val(self, x):
        usable = self._slider_w - 2 * self.HANDLE_R
        if usable <= 0:
            return self._vmin
        v = (x - self.HANDLE_R) / usable * (self._vmax - self._vmin) \
            + self._vmin
        return int(round(max(self._vmin, min(self._vmax, v))))

    def _redraw(self):
        c = self._canvas
        c.delete("all")
        y   = self.TRACK_Y
        th2 = self.TRACK_H // 2
        x   = self._val_to_x(self._val)
        c.create_rectangle(self.HANDLE_R, y - th2,
                           self._slider_w - self.HANDLE_R, y + th2,
                           fill=self._track_col, outline="")
        c.create_rectangle(self.HANDLE_R, y - th2, x, y + th2,
                           fill=self._fill_col, outline="")
        c.create_oval(x - self.HANDLE_R, y - self.HANDLE_R,
                      x + self.HANDLE_R, y + self.HANDLE_R,
                      fill=C["text"], outline=self._fill_col, width=2)
        c.create_text(x, y - self.HANDLE_R - 10,
                      text=self._fmt(self._val),
                      fill=C["text"], font=FONT_SMALL)
        c.create_text(self.HANDLE_R, y + self.HANDLE_R + 10,
                      text=str(self._vmin), fill=C["text3"],
                      font=FONT_SMALL)
        c.create_text(self._slider_w - self.HANDLE_R, y + self.HANDLE_R + 10,
                      text=str(self._vmax), fill=C["text3"],
                      font=FONT_SMALL)

    def _on_press(self, ev):
        self._drag = True
        self._on_motion(ev)

    def _on_motion(self, ev):
        if not self._drag:
            return
        new_val = self._x_to_val(ev.x)
        if new_val == self._val:
            return
        self._val = new_val
        self._redraw()
        if self._on_change:
            try:
                self._on_change(self._val)
            except Exception as e:
                print(f"[Slider] on_change error: {e}")

    def _on_release_evt(self, _ev):
        if not self._drag:
            return
        self._drag = False
        if self._on_release:
            try:
                self._on_release(self._val)
            except Exception as e:
                print(f"[Slider] on_release error: {e}")


class HueStripPicker(tk.Frame):
    """Stacked histogram + draggable rainbow strip for the hue axis.

    Layout (top â†’ bottom):
      1. Histogram canvas â€” cyan bars showing the cookie's hue distribution.
      2. Rainbow strip canvas â€” solid HSV rainbow with the active band
         highlighted in green, two draggable edge handles, and a marker
         for the cookie's median hue.

    Interactions:
      - Click on an edge handle (left/right of the green band) and drag
        to resize that edge.
      - Click inside the band and drag to slide the whole band.
      - Click outside the band to start a new band at that point (drag
        to set the other edge).

    Fires ``on_change(h_lo, h_hi)`` continuously while dragging.
    """
    HIST_H   = 50
    STRIP_H  = 30
    PAD      = 6     # so end-handles aren't clipped
    EDGE_GRAB = 7    # px tolerance for grabbing an edge

    def __init__(self, parent, width=380, on_change=None,
                 read_only=False, **kwargs):
        bg = kwargs.pop("bg", C["bg2"])
        kwargs.pop("width",  None)
        kwargs.pop("height", None)
        super().__init__(parent, bg=bg, **kwargs)
        self._w_px      = int(width)
        self._on_change = on_change
        self._read_only = bool(read_only)
        self._h_lo      = 0
        self._h_hi      = 179
        self._hue_data  = None     # 1D uint8 array of cookie hues (or None)
        self._cookie_hue = None

        # Histogram on top.
        self._hist = tk.Canvas(self, width=self._w_px, height=self.HIST_H,
                               bg=C["panel"], highlightthickness=0)
        self._hist.pack(fill=tk.X)
        # Strip below.
        strip_cursor = "" if self._read_only else "hand2"
        self._strip = tk.Canvas(self, width=self._w_px, height=self.STRIP_H,
                                bg=bg, highlightthickness=0,
                                cursor=strip_cursor)
        self._strip.pack(fill=tk.X, pady=(2, 0))

        # Pre-render the rainbow once as a PhotoImage â€” much faster than
        # painting individual lines on every redraw.
        self._rainbow_photo = self._build_rainbow_photo()
        self._strip_drag = None    # 'lo' | 'hi' | 'band' | 'new'
        self._drag_start_x = 0
        self._drag_start_lo = 0
        self._drag_start_hi = 0

        # Skip event bindings in read-only mode so the strip is purely
        # informational (no drag handles, no clickable cursor).
        if not self._read_only:
            self._strip.bind("<Button-1>",        self._on_press)
            self._strip.bind("<B1-Motion>",       self._on_motion)
            self._strip.bind("<ButtonRelease-1>", self._on_release)

        self._redraw_strip()

    # â”€â”€ public API â”€â”€
    def set_band(self, lo, hi, fire=True):
        self._h_lo = int(max(0,   min(179, lo)))
        self._h_hi = int(max(self._h_lo, min(179, hi)))
        self._redraw_strip()
        if fire and self._on_change:
            try:
                self._on_change(self._h_lo, self._h_hi)
            except Exception as e:
                print(f"[HueStripPicker] on_change error: {e}")

    def set_data(self, hue_array, lo, hi, cookie_hue=None):
        """Update the histogram and the band; doesn't fire callback."""
        self._hue_data   = hue_array
        self._cookie_hue = cookie_hue
        self._h_lo = int(max(0, min(179, lo)))
        self._h_hi = int(max(self._h_lo, min(179, hi)))
        self._redraw_hist()
        self._redraw_strip()

    # â”€â”€ helpers â”€â”€
    def _val_to_x(self, v):
        usable = self._w_px - 2 * self.PAD
        return self.PAD + (v / 179.0) * usable

    def _x_to_val(self, x):
        usable = self._w_px - 2 * self.PAD
        if usable <= 0:
            return 0
        return int(round(max(0, min(179,
                    (x - self.PAD) / usable * 179.0))))

    def _build_rainbow_photo(self):
        """Make a PhotoImage of the full HSV rainbow for the strip."""
        # Build a 1Ã—N RGB array, then upscale to STRIP_H rows with PIL.
        grad = np.linspace(0, 179, self._w_px).astype(np.uint8)
        hsv  = np.zeros((1, self._w_px, 3), dtype=np.uint8)
        hsv[0, :, 0] = grad
        hsv[0, :, 1] = 220
        hsv[0, :, 2] = 235
        rgb  = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        pil  = Image.fromarray(rgb).resize(
            (self._w_px, self.STRIP_H), Image.NEAREST)
        return ImageTk.PhotoImage(pil)

    # â”€â”€ drawing â”€â”€
    def _redraw_hist(self):
        c = self._hist
        c.delete("all")
        if self._hue_data is None or len(self._hue_data) == 0:
            c.create_text(self._w_px // 2, self.HIST_H // 2,
                          text="(no hue data)",
                          fill=C["text3"], font=FONT_SMALL)
            return
        # 60 bins across hue 0..179.
        n_bins = 60
        hist, _edges = np.histogram(self._hue_data, bins=n_bins,
                                    range=(0, 180))
        if hist.max() == 0:
            return
        usable = self._w_px - 2 * self.PAD
        bw     = usable / n_bins
        ymax   = self.HIST_H - 2
        for i, count in enumerate(hist):
            if count == 0:
                continue
            h = int(count / hist.max() * ymax)
            x0 = self.PAD + i * bw
            x1 = x0 + bw - 1
            c.create_rectangle(x0, self.HIST_H - h,
                               x1, self.HIST_H,
                               fill=C["accent2"], outline="")

    def _redraw_strip(self):
        c = self._strip
        c.delete("all")
        c.create_image(0, 0, image=self._rainbow_photo, anchor="nw")

        # In read-only mode the strip is purely an informational
        # reference â€” no band, no handles, just the rainbow + cookie
        # marker.
        if not self._read_only:
            x_lo = self._val_to_x(self._h_lo)
            x_hi = self._val_to_x(self._h_hi)
            # Dim the region outside the band.
            if x_lo > self.PAD:
                c.create_rectangle(0, 0, x_lo, self.STRIP_H,
                                   fill=C["bg"], stipple="gray50",
                                   outline="")
            if x_hi < self._w_px - self.PAD:
                c.create_rectangle(x_hi, 0, self._w_px, self.STRIP_H,
                                   fill=C["bg"], stipple="gray50",
                                   outline="")
            # Highlight band outline.
            c.create_rectangle(x_lo, 0, x_hi, self.STRIP_H - 1,
                               outline=C["green"], width=2)
            # Edge handles (small vertical bars).
            for x in (x_lo, x_hi):
                c.create_rectangle(x - 2, 0, x + 2, self.STRIP_H,
                                   fill=C["green"], outline=C["bg"])

        # Cookie-hue marker.
        if self._cookie_hue is not None:
            xm = self._val_to_x(float(self._cookie_hue))
            c.create_line(xm, 0, xm, self.STRIP_H,
                          fill="black", width=3)
            c.create_line(xm, 0, xm, self.STRIP_H,
                          fill=C["text"], width=1)

    # â”€â”€ mouse handling â”€â”€
    def _on_press(self, ev):
        x_lo = self._val_to_x(self._h_lo)
        x_hi = self._val_to_x(self._h_hi)
        if abs(ev.x - x_lo) <= self.EDGE_GRAB:
            self._strip_drag = 'lo'
        elif abs(ev.x - x_hi) <= self.EDGE_GRAB:
            self._strip_drag = 'hi'
        elif x_lo < ev.x < x_hi:
            self._strip_drag = 'band'
            self._drag_start_x  = ev.x
            self._drag_start_lo = self._h_lo
            self._drag_start_hi = self._h_hi
        else:
            # Click outside band â€” start a new band at this point and
            # let the user drag to set the other edge.
            v = self._x_to_val(ev.x)
            self._h_lo = self._h_hi = v
            self._strip_drag = 'hi'
            self._redraw_strip()
            self._fire()

    def _on_motion(self, ev):
        if self._strip_drag is None:
            return
        if self._strip_drag == 'lo':
            v = self._x_to_val(ev.x)
            self._h_lo = min(v, self._h_hi)
        elif self._strip_drag == 'hi':
            v = self._x_to_val(ev.x)
            self._h_hi = max(v, self._h_lo)
        elif self._strip_drag == 'band':
            usable = self._w_px - 2 * self.PAD
            dv = int(round((ev.x - self._drag_start_x) / usable * 179.0))
            width = self._drag_start_hi - self._drag_start_lo
            new_lo = max(0, min(179 - width,
                                self._drag_start_lo + dv))
            self._h_lo = new_lo
            self._h_hi = new_lo + width
        self._redraw_strip()
        self._fire()

    def _on_release(self, _ev):
        self._strip_drag = None

    def _fire(self):
        if self._on_change:
            try:
                self._on_change(self._h_lo, self._h_hi)
            except Exception as e:
                print(f"[HueStripPicker] on_change error: {e}")


class SVFieldPicker(tk.Frame):
    """2D Saturation Ã— Value scatter with a draggable acceptance rectangle.

    The X axis is saturation (0..255, increasing left â†’ right).
    The Y axis is value (0..255, increasing bottom â†’ top).
    Cookie pixels are plotted as small dots; the active rectangle is
    drawn in green; the cookie's (s, v) median is marked.

    Interactions:
      - Click + drag inside the rectangle to slide it.
      - Click + drag on an edge or corner to resize.
      - Click + drag outside the rectangle to draw a new one.

    Fires ``on_change(s_lo, s_hi, v_lo, v_hi)`` while dragging.
    """
    PAD       = 6
    EDGE_GRAB = 8

    def __init__(self, parent, width=380, height=200,
                 on_change=None, **kwargs):
        bg = kwargs.pop("bg", C["bg2"])
        kwargs.pop("width",  None)
        kwargs.pop("height", None)
        super().__init__(parent, bg=bg, **kwargs)
        self._w_px      = int(width)
        self._h_px      = int(height)
        self._on_change = on_change

        self._s_lo = 0;   self._s_hi = 255
        self._v_lo = 0;   self._v_hi = 255
        self._cookie_s = None
        self._cookie_v = None
        self._hsv_sample = None
        # Pre-rasterized scatter cloud; None until set_data is called.
        self._scatter_photo = None

        self._canvas = tk.Canvas(self, width=self._w_px, height=self._h_px,
                                 bg=C["panel"], highlightthickness=0,
                                 cursor="crosshair")
        self._canvas.pack(fill=tk.X)
        self._canvas.bind("<Button-1>",        self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)

        self._drag = None   # one of: 'move', 'l', 'r', 't', 'b',
                            #         'tl', 'tr', 'bl', 'br', 'new'
        self._drag_anchor = (0, 0)
        self._drag_start_rect = (0, 0, 0, 0)
        self._redraw()

    # â”€â”€ public API â”€â”€
    def set_rect(self, s_lo, s_hi, v_lo, v_hi, fire=True):
        self._s_lo = int(max(0, min(255, s_lo)))
        self._s_hi = int(max(self._s_lo, min(255, s_hi)))
        self._v_lo = int(max(0, min(255, v_lo)))
        self._v_hi = int(max(self._v_lo, min(255, v_hi)))
        self._redraw()
        if fire and self._on_change:
            try:
                self._on_change(self._s_lo, self._s_hi,
                                self._v_lo, self._v_hi)
            except Exception as e:
                print(f"[SVFieldPicker] on_change error: {e}")

    def set_data(self, hsv_sample, s_lo, s_hi, v_lo, v_hi,
                 cookie_s=None, cookie_v=None):
        self._hsv_sample = hsv_sample
        self._cookie_s   = cookie_s
        self._cookie_v   = cookie_v
        self._s_lo = int(max(0, min(255, s_lo)))
        self._s_hi = int(max(self._s_lo, min(255, s_hi)))
        self._v_lo = int(max(0, min(255, v_lo)))
        self._v_hi = int(max(self._v_lo, min(255, v_hi)))
        # Rebuild the cached scatter image â€” expensive, done once per
        # cookie. Subsequent drag redraws reuse this and only repaint
        # the lightweight overlay.
        self._scatter_photo = self._build_scatter_photo()
        self._redraw()

    def _build_scatter_photo(self):
        """Rasterize the cookie's HSV pixels into a single PhotoImage.

        Way faster to blit one image on each redraw than to paint
        thousands of Canvas rectangles during drag motions.
        """
        if self._hsv_sample is None or len(self._hsv_sample) == 0:
            return None
        # Start with the canvas's panel background.
        bg_rgb = self._hex_to_rgb(C["panel"])
        img = np.full((self._h_px, self._w_px, 3),
                      bg_rgb, dtype=np.uint8)
        sample = self._hsv_sample
        rgb = cv2.cvtColor(
            sample.reshape(-1, 1, 3),
            cv2.COLOR_HSV2RGB).reshape(-1, 3)
        usable_x = self._w_px - 2 * self.PAD
        usable_y = self._h_px - 2 * self.PAD
        # Vectorised pixel placement.
        s_vals = sample[:, 1].astype(np.float32)
        v_vals = sample[:, 2].astype(np.float32)
        xs = (self.PAD + s_vals / 255.0 * usable_x).astype(int)
        ys = (self._h_px - self.PAD
              - v_vals / 255.0 * usable_y).astype(int)
        xs = np.clip(xs, 0, self._w_px - 1)
        ys = np.clip(ys, 0, self._h_px - 1)
        img[ys, xs] = rgb
        # A second pass with neighbours so the dots look like 2Ã—2 blocks,
        # otherwise they're too tiny to see comfortably.
        for dy in (0, 1):
            for dx in (0, 1):
                yy = np.clip(ys + dy, 0, self._h_px - 1)
                xx = np.clip(xs + dx, 0, self._w_px - 1)
                img[yy, xx] = rgb
        return ImageTk.PhotoImage(Image.fromarray(img))

    @staticmethod
    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    # â”€â”€ coordinate helpers â”€â”€
    def _s_to_x(self, s):
        usable = self._w_px - 2 * self.PAD
        return self.PAD + (s / 255.0) * usable

    def _v_to_y(self, v):
        usable = self._h_px - 2 * self.PAD
        return self._h_px - self.PAD - (v / 255.0) * usable

    def _x_to_s(self, x):
        usable = self._w_px - 2 * self.PAD
        if usable <= 0:
            return 0
        return int(round(max(0, min(255,
                    (x - self.PAD) / usable * 255.0))))

    def _y_to_v(self, y):
        usable = self._h_px - 2 * self.PAD
        if usable <= 0:
            return 0
        return int(round(max(0, min(255,
                    (self._h_px - self.PAD - y) / usable * 255.0))))

    # â”€â”€ drawing â”€â”€
    def _redraw(self):
        c = self._canvas
        c.delete("all")

        # Faint axes labels.
        c.create_text(self._w_px - 4, self._h_px - 10,
                      text="S ->", anchor="e",
                      fill=C["text3"], font=FONT_SMALL)
        c.create_text(8, 8, text="^ V", anchor="nw",
                      fill=C["text3"], font=FONT_SMALL)

        # Pixel cloud â€” pre-rasterized; one blit instead of thousands
        # of canvas items.
        if self._scatter_photo is not None:
            c.create_image(0, 0, image=self._scatter_photo, anchor="nw")

        # Acceptance rectangle.
        x0 = self._s_to_x(self._s_lo); x1 = self._s_to_x(self._s_hi)
        y0 = self._v_to_y(self._v_hi); y1 = self._v_to_y(self._v_lo)
        c.create_rectangle(x0, y0, x1, y1,
                           outline=C["green"], width=2)
        # Faint fill so it reads as a "region", not just a frame.
        c.create_rectangle(x0 + 1, y0 + 1, x1 - 1, y1 - 1,
                           outline="", fill=C["green"], stipple="gray12")

        # Cookie marker.
        if self._cookie_s is not None and self._cookie_v is not None:
            cx = self._s_to_x(float(self._cookie_s))
            cy = self._v_to_y(float(self._cookie_v))
            r = 4
            c.create_oval(cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1,
                          outline="black", width=2)
            in_box = (self._s_lo <= self._cookie_s <= self._s_hi and
                      self._v_lo <= self._cookie_v <= self._v_hi)
            c.create_oval(cx - r, cy - r, cx + r, cy + r,
                          fill=C["green"] if in_box else C["red"],
                          outline=C["text"])

    # â”€â”€ mouse handling â”€â”€
    def _hit_test(self, x, y):
        x0 = self._s_to_x(self._s_lo); x1 = self._s_to_x(self._s_hi)
        y0 = self._v_to_y(self._v_hi); y1 = self._v_to_y(self._v_lo)
        near_l = abs(x - x0) <= self.EDGE_GRAB
        near_r = abs(x - x1) <= self.EDGE_GRAB
        near_t = abs(y - y0) <= self.EDGE_GRAB
        near_b = abs(y - y1) <= self.EDGE_GRAB
        inside = (x0 <= x <= x1) and (y0 <= y <= y1)

        if near_l and near_t: return 'tl'
        if near_r and near_t: return 'tr'
        if near_l and near_b: return 'bl'
        if near_r and near_b: return 'br'
        if near_l and (y0 <= y <= y1): return 'l'
        if near_r and (y0 <= y <= y1): return 'r'
        if near_t and (x0 <= x <= x1): return 't'
        if near_b and (x0 <= x <= x1): return 'b'
        if inside: return 'move'
        return 'new'

    def _on_press(self, ev):
        self._drag = self._hit_test(ev.x, ev.y)
        self._drag_anchor     = (ev.x, ev.y)
        self._drag_start_rect = (self._s_lo, self._s_hi,
                                 self._v_lo, self._v_hi)
        if self._drag == 'new':
            # Start a zero-size rect at the press point and let
            # motion grow the bottom-right corner.
            v_press_s = self._x_to_s(ev.x)
            v_press_v = self._y_to_v(ev.y)
            self._s_lo = self._s_hi = v_press_s
            self._v_lo = self._v_hi = v_press_v
            self._drag = 'br'
            self._redraw()
            self._fire()

    def _on_motion(self, ev):
        if self._drag is None:
            return
        s_lo0, s_hi0, v_lo0, v_hi0 = self._drag_start_rect

        if self._drag == 'move':
            usable_x = self._w_px - 2 * self.PAD
            usable_y = self._h_px - 2 * self.PAD
            ds = int(round((ev.x - self._drag_anchor[0]) / usable_x * 255))
            dv = -int(round((ev.y - self._drag_anchor[1]) / usable_y * 255))
            w_s = s_hi0 - s_lo0
            w_v = v_hi0 - v_lo0
            new_s_lo = max(0, min(255 - w_s, s_lo0 + ds))
            new_v_lo = max(0, min(255 - w_v, v_lo0 + dv))
            self._s_lo = new_s_lo
            self._s_hi = new_s_lo + w_s
            self._v_lo = new_v_lo
            self._v_hi = new_v_lo + w_v
        else:
            new_s = self._x_to_s(ev.x)
            new_v = self._y_to_v(ev.y)
            if 'l' in self._drag:
                self._s_lo = min(new_s, self._s_hi)
            if 'r' in self._drag:
                self._s_hi = max(new_s, self._s_lo)
            if 't' in self._drag:
                self._v_hi = max(new_v, self._v_lo)
            if 'b' in self._drag:
                self._v_lo = min(new_v, self._v_hi)
        self._redraw()
        self._fire()

    def _on_release(self, _ev):
        self._drag = None

    def _fire(self):
        if self._on_change:
            try:
                self._on_change(self._s_lo, self._s_hi,
                                self._v_lo, self._v_hi)
            except Exception as e:
                print(f"[SVFieldPicker] on_change error: {e}")


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
        self._val = tk.Label(self, text="â€”", bg=C["panel"],
                             fg=color, font=FONT_VALUE)
        self._val.pack(anchor=tk.W, padx=10)
        if unit:
            tk.Label(self, text=unit, bg=C["panel"],
                     fg=C["text3"], font=FONT_SMALL).pack(
                anchor=tk.W, padx=10, pady=(0, 8))

    def set(self, value):
        self._val.config(text=str(value))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Main Application
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
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

        # Default frontend: minimal labels-only view. Toggled with 'O'.
        self._overlays_visible = False

        self.v_yolo     = tk.BooleanVar(value=True)
        self.v_mask     = tk.BooleanVar(value=True)
        self.v_grid     = tk.BooleanVar(value=False)
        self.v_diameter = tk.BooleanVar(value=True)
        self.v_feret    = tk.BooleanVar(value=True)
        self.v_min_feret = tk.BooleanVar(value=True)
        self.v_through  = tk.BooleanVar(value=True)
        self.v_conf     = tk.DoubleVar(value=DEFAULT_CONF)

        # â”€â”€ Zoom / pan state â”€â”€
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

        # â”€â”€ Live camera state â”€â”€
        self._camera        = None       # Camera instance (lazy)
        self._live_mode     = False      # True when displaying live feed
        self._live_after_id = None       # Tk after() handle for the live loop
        self._calibrating   = False      # blocks re-entry of calibrate flow

        # â”€â”€ Quick-capture state â”€â”€
        # True while the live â†’ snapshot â†’ inspect chain is in progress;
        # prevents re-entry from multiple Q presses.
        self._quick_capture_active = False
        # Set by the Raspberry Pi hardware button path to ask
        # _inference_worker to skip the expensive texture map and
        # per-cookie metrology (Feret/min-Feret/through-center).
        # Cleared in _inference_worker's finally block.
        self._fast_inference = False
        # Filesystem path of the JPEG written by the most recent quick
        # capture (Q hotkey or Pi GPIO button). Populated in
        # _quick_capture_snapshot, embedded into the walkthrough JSON,
        # and reset on the next capture.
        self._last_snap_path = None

        # â”€â”€ Per-cookie walkthrough popup state â”€â”€
        # _wt_win is the Toplevel; _wt_idx is the cookie currently shown.
        self._wt_win   = None
        self._wt_idx   = 0
        self._wt_photo = None                            # PhotoImage ref
        # Per-cookie data the user enters in the popup. Each value is
        # {"name_override": str | None,
        #  "diameter_mm":   float | None,
        #  "tex_lo":        int,
        #  "tex_hi":        int}
        # Reset per walkthrough run; persists across BACK/NEXT within
        # one run so re-visiting a cookie shows what you typed.
        self._wt_data = {}                               # {cookie_idx: dict}
        # Cookie indices the user explicitly SKIPped â€” excluded from
        # the JSON output.
        self._wt_skipped = set()

        # â”€â”€ Texture analysis state â”€â”€
        self._variance_map     = None    # image-wide LBP variance map
        self._hsv_img          = None    # HSV version of orig_img
        self._mpl_canvases     = []      # tracked figures in detail panel
        self._tex_mpl_canvases = []      # tracked figures in texture panel
        self._texture_panel    = None    # texture analysis frame

        # â”€â”€ Busy overlay state â”€â”€
        # Centered modal-style Toplevel shown while inspection is
        # running (R / Q / Pi button paths). All three handles must
        # be valid Tk objects when the overlay is up, and all three
        # are reset to None when it's hidden.
        self._busy_overlay     = None
        self._busy_msg_var     = None
        self._busy_dots_var    = None
        self._busy_dots_after  = None
        self._busy_dots_step   = 0

        self._build()
        # Hotkey: 'Q' (or 'q') triggers the quick-capture pipeline.
        # Bound on root so it works regardless of which widget has focus,
        # as long as no Entry/Text widget is currently capturing keys.
        self.root.bind_all("<Key-q>", self._on_quick_capture_hotkey)
        self.root.bind_all("<Key-Q>", self._on_quick_capture_hotkey)
        # Hotkey: 'C' (or 'c') runs the calibration flow against the
        # current live frame â€” same behaviour as the (hidden) CALIBRATE
        # button. Suppressed when typing into an Entry/Text widget.
        self.root.bind_all("<Key-c>", self._on_calibrate_hotkey)
        self.root.bind_all("<Key-C>", self._on_calibrate_hotkey)
        # Hotkey: 'O' toggles the full diagnostic overlay layer (YOLO
        # boxes, mask fill, feret lines, etc.) on top of the
        # operator-default labels-only view.
        self.root.bind_all("<Key-o>", self._on_overlays_hotkey)
        self.root.bind_all("<Key-O>", self._on_overlays_hotkey)
        # Hotkey: 'T' opens the texture-analysis panel for the
        # currently-selected cookie. Same code path as the (hidden)
        # TEXTURE ANALYSIS button.
        self.root.bind_all("<Key-t>", self._on_texture_hotkey)
        self.root.bind_all("<Key-T>", self._on_texture_hotkey)
        # Hotkey: 'R' triggers RUN INSPECTION on the current frame
        # (same code path as the hidden RUN INSPECTION button).
        self.root.bind_all("<Key-r>", self._on_run_hotkey)
        self.root.bind_all("<Key-R>", self._on_run_hotkey)
        # Hotkey: 'S' opens the WITHIN-SPEC detail panel for the
        # currently-selected cookie (click a cookie on the canvas
        # first to select it).
        self.root.bind_all("<Key-s>", self._on_spec_hotkey)
        self.root.bind_all("<Key-S>", self._on_spec_hotkey)
        # Hotkey: 'U' opens the file dialog to upload a static image
        # (same code path as the hidden UPLOAD IMAGE button).
        self.root.bind_all("<Key-u>", self._on_upload_hotkey)
        self.root.bind_all("<Key-U>", self._on_upload_hotkey)

        # Hardware trigger: GPIO push-button on pin GPIO_BUTTON_PIN. The
        # callback runs on the Tk thread (dispatched via root.after).
        self._gpio_trigger = GPIOQuickCaptureTrigger(
            self.root, on_press=self._on_pi_button_capture)
        self._gpio_trigger.start()

        self._load_model_bg(self.model_path.get(), silent=True)

        # Operator workflow: the live camera feed should be on from the
        # moment the app opens. Defer a moment so the window is mapped
        # and the canvas has its real size before the first frame
        # arrives. If picamera2 isn't installed (dev box), this is a
        # no-op â€” _start_live() bails out cleanly.
        if PICAMERA_AVAILABLE:
            self.root.after(400, self._start_live)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # UI Construction
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _build(self):
        # Header
        hdr = tk.Frame(self.root, bg=C["bg2"],
                       highlightbackground=C["border"],
                       highlightthickness=1)
        hdr.pack(fill=tk.X)

        logo = tk.Frame(hdr, bg=C["bg2"])
        logo.pack(side=tk.LEFT, padx=18, pady=10)
        tk.Label(logo, text="*", bg=C["bg2"],
                 fg=C["accent"], font=("Courier New", 20)).pack(side=tk.LEFT)
        tk.Label(logo, text=" COOKIE INSPECTOR",
                 bg=C["bg2"], fg=C["text"],
                 font=FONT_TITLE).pack(side=tk.LEFT)
        tk.Label(logo, text="  PRO",
                 bg=C["bg2"], fg=C["accent"],
                 font=FONT_TITLE).pack(side=tk.LEFT)

        btns = tk.Frame(hdr, bg=C["bg2"])
        btns.pack(side=tk.RIGHT, padx=18, pady=10)
        # EXPORT button â€” hidden from the operator UI. Widget is kept so
        # internal code paths that reference btn_save still resolve.
        self.btn_save = FlatButton(btns, "v  EXPORT",
                                   command=self.on_save,
                                   bg=C["bg3"], fg=C["text2"],
                                   font=FONT_SMALL)
        # self.btn_save.pack(...)   # intentionally not packed
        # RUN INSPECTION â€” hidden from operator UI. Triggered via the
        # 'R' hotkey. Widget kept alive because on_run() and the worker
        # callbacks call btn_run.set_state(...).
        self.btn_run = FlatButton(btns, ">  RUN INSPECTION",
                                  command=self.on_run,
                                  bg=C["accent"], fg=C["bg"],
                                  font=("Courier New", 9, "bold"))
        # self.btn_run.pack(...)   # intentionally not packed
        # â”€â”€ UPLOAD IMAGE & LIVE buttons â”€â”€
        # Hidden from the operator UI (camera-only workflow). Kept as
        # real widget objects so internal references (e.g. _stop_live
        # toggling btn_live's colour, _quick_capture relying on btn_live)
        # don't break. No .pack() call â†’ invisible.
        self.btn_upload = FlatButton(btns, "^  UPLOAD IMAGE",
                                     command=self.on_upload,
                                     bg=C["bg3"], fg=C["text"],
                                     font=FONT_SMALL)
        # self.btn_upload.pack(...)   # hidden; triggered by 'U' hotkey
        self.btn_live = FlatButton(btns, "[CAM]  LIVE",
                                   command=self.on_toggle_live,
                                   bg=C["bg3"], fg=C["text"],
                                   font=FONT_SMALL)
        # self.btn_live.pack(...)     # intentionally not packed
        if not PICAMERA_AVAILABLE:
            self.btn_live.set_state(False)

        # â”€â”€ Quick-capture button â”€â”€
        # One-press shortcut: starts live feed, lets exposure settle,
        # grabs a single frame, stops live, then runs inspection â€” all
        # automatically. Also bound to the 'Q' key (see _bind_hotkeys).
        self.btn_quick = FlatButton(btns, ">>  QUICK CAPTURE  [Q]",
                                    command=self.on_quick_capture,
                                    bg=C["yellow"], fg=C["bg"],
                                    font=("Courier New", 9, "bold"))
        self.btn_quick.pack(side=tk.RIGHT, padx=4)
        if not PICAMERA_AVAILABLE:
            self.btn_quick.set_state(False)

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
        self._status_dot = tk.Label(status_bar, text="*",
                                    bg=C["bg2"], fg=C["green"],
                                    font=FONT_SMALL)
        self._status_dot.pack(side=tk.LEFT, padx=(12, 4), pady=4)
        self._status_var = tk.StringVar(
            value="Ready â€” load a model and upload an image.")
        tk.Label(status_bar, textvariable=self._status_var,
                 bg=C["bg2"], fg=C["text2"],
                 font=FONT_SMALL).pack(side=tk.LEFT)
        tk.Label(status_bar, text="YOLO11 Â· Feret Metrology v2",
                 bg=C["bg2"], fg=C["text3"],
                 font=FONT_SMALL).pack(side=tk.RIGHT, padx=12)

    # â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=C["bg2"], width=300,
                      highlightbackground=C["border"],
                      highlightthickness=1)
        sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(1, 0))
        sb.pack_propagate(False)
        # Keep a reference so the walkthrough panel can hide the
        # (empty) sidebar while it's open and put itself flush against
        # the right edge of the window.
        self._sidebar = sb

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

        # â”€â”€ Model section â”€â”€
        # Hidden from operator UI. The path is fixed in code via
        # DEFAULT_MODEL_PATH and the model auto-loads on startup; the
        # status label (_model_lbl) is still updated by _load_model_bg
        # but lives in an unpacked frame so the UI stays clean.
        # To re-enable: change `mf` to a child of `inner` and pack it.
        mf = tk.Frame(self.root, bg=C["bg2"])    # not packed â†’ hidden
        tk.Entry(mf, textvariable=self.model_path).pack()  # keeps var alive
        self._model_lbl = tk.Label(mf, text="* Not loaded",
                                   bg=C["bg2"], fg=C["red"],
                                   font=FONT_SMALL)
        self._model_lbl.pack()

        # â”€â”€ Inference section â”€â”€
        # Hidden from operator UI. The confidence threshold stays at
        # DEFAULT_CONF (v_conf); the slider widget is built into an
        # unpacked frame to keep _conf_lbl alive.
        inf = tk.Frame(self.root, bg=C["bg2"])    # not packed â†’ hidden
        self._conf_lbl = tk.Label(inf, bg=C["bg2"],
                                  fg=C["accent"], font=FONT_MONO,
                                  text=f"{self.v_conf.get():.2f}")
        self._conf_lbl.pack()

        # â”€â”€ Calibration section â”€â”€
        # Hidden from operator UI. Calibration runs via the C hotkey
        # (see _bind_hotkeys at end of __init__) which calls
        # on_calibrate() exactly the same way the button did. Label
        # widgets are kept alive because on_calibrate / on_reset
        # update their text, so they need to be valid Tk widgets.
        cal = tk.Frame(self.root, bg=C["bg2"])   # not packed â†’ hidden
        self._cal_value_lbl = tk.Label(
            cal, bg=C["bg2"], fg=C["accent"], font=FONT_MONO,
            text=f"{SETTINGS.pixel_to_mm:.3f}")
        self._cal_value_lbl.pack()
        self._cal_when_lbl = tk.Label(
            cal, bg=C["bg2"], fg=C["text3"], font=FONT_SMALL,
            text=self._format_calibration_when())
        self._cal_when_lbl.pack()
        self.btn_calibrate = FlatButton(
            cal, "O  CALIBRATE  WITH  CURRENT  FRAME",
            command=self.on_calibrate,
            bg=C["accent2"], fg=C["bg"],
            font=("Courier New", 9, "bold"),
            pad_x=10, pad_y=7)
        self.btn_calibrate.pack()

        # â”€â”€ Overlay toggles â”€â”€
        # Hidden from operator UI. Toggle widgets are built into an
        # unpacked frame so their BooleanVars stay bound (render_overlays
        # still reads them). Toggle the whole overlay layer on/off
        # at runtime with the 'O' hotkey.
        ov = tk.Frame(self.root, bg=C["bg2"])   # not packed â†’ hidden
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
            tk.Label(row, text="|", bg=C["bg2"],
                     fg=color, font=FONT_SMALL).pack(
                side=tk.LEFT, padx=(0, 6))
            tk.Label(row, text=lbl, bg=C["bg2"],
                     fg=C["text2"], font=FONT_SMALL).pack(side=tk.LEFT)
            Toggle(row, variable=var,
                   command=self._refresh_display).pack(side=tk.RIGHT)

        # â”€â”€ Statistics cards â”€â”€
        # Hidden from operator UI but built as real widgets because
        # _on_done / _reset_cards write into them.
        sf = tk.Frame(self.root, bg=C["bg2"])     # not packed â†’ hidden
        self._card_det = StatCard(sf, "Detected", color=C["accent"])
        self._card_det.pack()
        self._card_lop = StatCard(sf, "Lopsided", color=C["red"])
        self._card_lop.pack()
        self._card_fer = StatCard(sf, "Avg MaxFeret", unit="mm",
                                  color=C["cyan"])
        self._card_fer.pack()
        self._card_mfr = StatCard(sf, "Avg MinFeret", unit="mm",
                                  color=C["green"])
        self._card_mfr.pack()
        self._card_ctr = StatCard(sf, "Avg Center", unit="mm",
                                  color=C["magenta"])
        self._card_ctr.pack()

        # â”€â”€ Analysis section â”€â”€
        # Hidden from operator UI. Texture analysis opens via the 'T'
        # hotkey (see _bind_hotkeys at end of __init__). The button and
        # hint widgets are kept alive because other code paths
        # (_show_detail_panel, _hide_detail_panel) update _texture_hint's
        # text, so it needs to be a valid Tk widget.
        an = tk.Frame(self.root, bg=C["bg2"])    # not packed â†’ hidden
        self.btn_texture = FlatButton(
            an, "[T]  TEXTURE  ANALYSIS",
            command=self.on_open_texture,
            bg=C["accent2"], fg=C["bg"],
            font=("Courier New", 9, "bold"),
            pad_x=10, pad_y=7)
        self.btn_texture.pack()
        self._texture_hint = tk.Label(
            an, bg=C["bg2"], fg=C["text3"], font=FONT_SMALL,
            text="", justify=tk.LEFT)
        self._texture_hint.pack()

        # â”€â”€ Results table â”€â”€
        # Hidden from operator UI but kept alive because
        # _populate_table / _clear_table write into self._tbl.
        tf = tk.Frame(self.root, bg=C["bg2"])     # not packed â†’ hidden
        self._tbl = tk.Frame(tf, bg=C["bg2"])
        self._tbl.pack()

    def _section_hdr(self, parent, title):
        row = tk.Frame(parent, bg=C["bg2"], padx=12, pady=0)
        row.pack(fill=tk.X, pady=(10, 2))
        tk.Label(row, text=title, bg=C["bg2"],
                 fg=C["accent"], font=FONT_SMALL).pack(side=tk.LEFT)
        tk.Frame(row, bg=C["border"], height=1).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

    # â”€â”€ Canvas area â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

        # â”€â”€ Zoom controls (right side of toolbar) â”€â”€
        zoom_frame = tk.Frame(ctool, bg=C["bg2"])
        zoom_frame.pack(side=tk.RIGHT, padx=8, pady=4)
        FlatButton(zoom_frame, "R", command=self._zoom_reset,
                   bg=C["bg3"], fg=C["text2"], font=FONT_SMALL,
                   pad_x=7, pad_y=3).pack(side=tk.RIGHT, padx=(2, 0))
        FlatButton(zoom_frame, "+", command=self._zoom_in,
                   bg=C["bg3"], fg=C["text"],
                   font=("Courier New", 10, "bold"),
                   pad_x=8, pad_y=3).pack(side=tk.RIGHT, padx=2)
        FlatButton(zoom_frame, "-", command=self._zoom_out,
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
            cx, cy-20, text="^", fill=C["border2"],
            font=("Courier New", 36), tags="ph")
        self._canvas.create_text(
            cx, cy+20, text="Upload an image to begin",
            fill=C["text3"], font=("Courier New", 12), tags="ph")
        self._canvas.create_text(
            cx, cy+44, text="Supports: JPG Â· PNG Â· BMP Â· TIFF",
            fill=C["text3"], font=FONT_SMALL, tags="ph")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Model loading
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _load_model_bg(self, path, silent=False):
        def worker():
            try:
                self._set_status("Loading modelâ€¦", "yellow")
                self.root.after(0, lambda: self._model_lbl.config(
                    text="* Loading...", fg=C["yellow"]))
                m = YOLO(path)
                # Override the class names baked into the weights with our
                # friendly labels. Ultralytics caches `names` in several
                # places (the wrapper, the inner nn.Module, and the snapshot
                # the Results object reads at plot time) so we patch them
                # all to make sure r.plot() actually shows the cookie name.
                #
                # In newer Ultralytics versions YOLO.names is a read-only
                # property that just reads from m.model.names, so we have
                # to write to the underlying module instead. We try the
                # wrapper first for older versions, but suppress the error
                # if it's read-only.
                try:
                    friendly = dict(CLASS_NAMES)
                    applied_to = []
                    # 1. The underlying nn.Module â€” this is the real source
                    #    of truth in newer Ultralytics.
                    if hasattr(m, "model") and m.model is not None:
                        try:
                            m.model.names = friendly
                            applied_to.append("m.model.names")
                        except Exception as e:
                            print(f"[Model] m.model.names failed: {e}")
                    # 2. The predictor's model (only exists after first predict)
                    if hasattr(m, "predictor") and m.predictor is not None:
                        try:
                            m.predictor.model.names = friendly
                            applied_to.append("m.predictor.model.names")
                        except Exception:
                            pass
                    # 3. The wrapper itself â€” may fail on newer versions
                    #    (read-only property) and that's fine, it just
                    #    reads from m.model.names which we already set.
                    try:
                        m.names = friendly
                        applied_to.append("m.names")
                    except AttributeError:
                        pass  # property is read-only; m.model.names is enough
                    except Exception as e:
                        print(f"[Model] m.names failed: {e}")
                    print(f"[Model] class names override applied to: "
                          f"{applied_to}")
                except Exception as e:
                    print(f"[Model] could not apply class-name override: {e}")
                self.model = m
                self._set_status(
                    f"Model ready â€” {os.path.basename(path)}", "green")
                self.root.after(0, lambda: self._model_lbl.config(
                    text=f"* {os.path.basename(path)}", fg=C["green"]))
            except Exception as e:
                self.model = None
                self._set_status(f"Model error: {e}", "red")
                self.root.after(0, lambda: self._model_lbl.config(
                    text="* Load failed", fg=C["red"]))
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

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Image upload
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        self._dim_lbl.config(text=f"{img.shape[1]} Ã— {img.shape[0]} px")
        self._clear_table()
        self._reset_cards()
        self._zoom  = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._selected_idx = None
        if self._detail is not None:
            self._detail.place_forget()
        if self._texture_panel is not None:
            self._hide_texture_panel()
        self._set_status(
            f"Loaded: {name}  -  click  > RUN INSPECTION  to analyse",
            "green")
        self._refresh_display()

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Inference
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def on_run(self):
        if self.orig_img is None:
            messagebox.showinfo("No image", "Upload an image first.")
            return
        if self.model is None:
            messagebox.showwarning("No model",
                                   "Model not loaded â€” set path and click LOAD.")
            return
        if self._busy:
            return
        self._busy = True
        self.btn_run.set_state(False)
        # Centered busy overlay â€” visible until _on_done (success) or
        # the inference worker's error path (failure) tears it down.
        self._show_busy_overlay("Running YOLO inference...")
        threading.Thread(target=self._inference_worker, daemon=True).start()

    def _inference_worker(self):
        try:
            self._set_status("Running YOLO inferenceâ€¦", "yellow")
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
            # Force our friendly names onto the result object â€” Ultralytics
            # snapshots `names` into Results at predict time, so even if
            # m.names was overridden the result may still hold the old map.
            try:
                r.names = dict(CLASS_NAMES)
            except Exception as e:
                print(f"[infer] could not override r.names: {e}")
            self.yolo_plotted = r.plot(boxes=True, labels=True, conf=True)

            fast = self._fast_inference

            if fast:
                # Pi-button fast path: skip texture map + metrology.
                self._variance_map = None
                self._hsv_img      = cv2.cvtColor(
                    self.orig_img, cv2.COLOR_BGR2HSV)
                variance_map = None
                hsv_img      = self._hsv_img
            else:
                # Compute the image-wide texture map ONCE so each cookie
                # only needs to slice into it. This is the expensive step.
                self._set_status("Analysing textureâ€¦", "yellow")
                variance_map, hsv_img = compute_image_texture(self.orig_img)
                self._variance_map = variance_map
                self._hsv_img      = hsv_img

            cookies = []
            if r.masks is not None:
                n = len(r.masks)
                if fast:
                    self._set_status(
                        f"Building masks for {n} cookie(s)â€¦", "yellow")
                else:
                    self._set_status(
                        f"Computing metrology for {n} mask(s)â€¦", "yellow")
                # Pull per-detection class IDs from the YOLO result.
                # boxes.cls is a tensor aligned with masks (one entry per
                # detection). If boxes is missing for any reason we fall
                # back to None so analyze_cookie still works.
                cls_ids = None
                try:
                    if r.boxes is not None and r.boxes.cls is not None:
                        cls_ids = r.boxes.cls.cpu().numpy().astype(int).tolist()
                except Exception as e:
                    print(f"[infer] could not read class IDs: {e}")
                    cls_ids = None

                for i, mi in enumerate(r.masks):
                    mc = mi.xy[0].astype(np.int32)
                    if len(mc) < 3:
                        continue
                    rm = mi.data[0].cpu().numpy()
                    rm = cv2.resize(rm, (self.orig_img.shape[1],
                                        self.orig_img.shape[0]))
                    cid = (cls_ids[i] if cls_ids is not None
                           and i < len(cls_ids) else None)
                    if fast:
                        # Minimal cookie dict: enough for the walkthrough
                        # popup (mask + label + center) and to keep the
                        # table/cards renderers from crashing on missing
                        # metrology fields.
                        M = cv2.moments(mc)
                        if M['m00'] == 0:
                            continue
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        d = {
                            'mask':           rm,
                            'cls_id':         cid,
                            'label':          class_label(cid),
                            'center':         (cx, cy),
                            'mec_center':     None,
                            'mec_radius':     0.0,
                            'drift_ratio':    0.0,
                            'lopsided':       False,
                            'tc_chord':       None,
                            'tc_mm':          0.0,
                            'feret_pair':     None,
                            'feret_mm':       0.0,
                            'min_feret_pair': None,
                            'min_feret_mm':   0.0,
                            'elongation':     1.0,
                            'grid':           [],
                        }
                    else:
                        d = analyze_cookie(mc, rm, variance_map, hsv_img,
                                           cls_id=cid)
                    if d:
                        cookies.append(d)
            self.cookies = cookies
            self.root.after(0, self._on_done)
        except Exception as e:
            self._set_status(f"Error: {e}", "red")
            self.root.after(
                0, lambda: messagebox.showerror("Inference Error", str(e)))
            # Tear down the busy overlay on the error path so the user
            # isn't left staring at a "running..." dialog forever.
            self.root.after(0, self._hide_busy_overlay)
        finally:
            self._busy = False
            # Always drop fast-mode flag so the next Q press / RUN button
            # gets the full pipeline back.
            self._fast_inference = False
            self.root.after(0, lambda: self.btn_run.set_state(True))

    def _on_done(self):
        # Inspection finished successfully â€” drop the busy overlay
        # before we start populating UI and opening the walkthrough.
        self._hide_busy_overlay()
        n        = len(self.cookies)
        n_lop    = sum(1 for d in self.cookies if d['lopsided'])
        ferets   = [d['feret_mm']     for d in self.cookies if d['feret_mm']     > 0]
        minfers  = [d['min_feret_mm'] for d in self.cookies if d['min_feret_mm'] > 0]
        centers  = [d['tc_mm']        for d in self.cookies if d['tc_mm']        > 0]
        self._card_det.set(str(n))
        self._card_lop.set(str(n_lop))
        self._card_fer.set(
            f"{sum(ferets)/len(ferets):.1f}"  if ferets  else "â€”")
        self._card_mfr.set(
            f"{sum(minfers)/len(minfers):.1f}" if minfers else "â€”")
        self._card_ctr.set(
            f"{sum(centers)/len(centers):.1f}" if centers else "â€”")
        self._populate_table()
        suffix = f"  Â·  {n_lop} lopsided" if n_lop else ""
        self._set_status(
            f"Inspection complete â€” {n} cookie(s) detected{suffix}", "green")
        self._refresh_display()
        # Auto-open the per-cookie walkthrough popup if anything was found.
        if n > 0:
            self.root.after(120, self._open_walkthrough)

    def _reset_cards(self):
        for c in (self._card_det, self._card_lop,
                  self._card_fer, self._card_mfr, self._card_ctr):
            c.set("â€”")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Per-cookie walkthrough popup (minimal: diameter + tolerance slider)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Default texture value applied when a cookie is first viewed.
    # Stored as (lo, hi); slider is now a single value V so lo == hi == V.
    _WT_DEFAULT_TEX = (0, 0)
    # Cropped masked image is fit into this square (px). Sized so the
    # overlay texture slider can span the image edge-to-edge.
    _WT_IMAGE_PX = 380

    def _open_walkthrough(self):
        """Create (or re-open) the per-cookie walkthrough Toplevel.

        The popup is intentionally minimal: for each detected cookie the
        user types its real diameter (mm) into an entry field and uses a
        dual-handle slider (-5..+5) to set a tolerance range. Pressing
        SAVE on the last cookie writes all entered values to JSON.
        """
        if not self.cookies:
            return
        if self._wt_win is not None:
            try:
                self._wt_win.destroy()
            except Exception:
                pass
            self._wt_win = None

        # Adapt the cookie-image size to the available canvas height so
        # the panel's bottom buttons (BACK / SKIP / NEXT), diameter
        # entry, and skip-hint label never fall off the visible area
        # on smaller displays (Raspberry Pi touchscreens, etc.).
        # Reserve ~430 px for everything that isn't the image:
        #   titlebar, header, progress bar, dropdown, separators,
        #   slider, name hint, diameter label + entry, skip hint,
        #   button row, and all the paddings between them.
        canvas_h = self._canvas.winfo_height()
        if canvas_h < 100:
            canvas_h = self.root.winfo_height() - 80
        self._WT_IMAGE_PX = max(140, min(300, canvas_h - 430))

        # Hide any open side panels so the snapshot canvas is fully
        # visible while the user walks through cookies.
        if self._detail is not None:
            self._detail.place_forget()
        if self._texture_panel is not None:
            self._hide_texture_panel()

        self._wt_idx = 0
        # Per-cookie data and skip flags are per-run.
        self._wt_data = {}
        self._wt_skipped = set()

        # Docked side-panel parented to the main body (canvas's
        # grandparent) and packed to the right edge so the canvas
        # reflows beside it rather than being covered. Fixed width;
        # height fills the body area.
        WT_PANEL_W = 440
        parent = self._canvas.master.master   # body frame
        win = tk.Frame(parent, bg=C["bg2"],
                       highlightbackground=C["border"],
                       highlightthickness=1,
                       width=WT_PANEL_W)
        win.pack_propagate(False)
        self._wt_win = win
        # Hide the empty operator-mode sidebar so the walkthrough
        # sits flush against the absolute right edge (directly under
        # the QUICK CAPTURE button).
        if getattr(self, "_sidebar", None) is not None:
            try:
                self._sidebar.pack_forget()
            except Exception:
                pass
        win.pack(side=tk.RIGHT, fill=tk.Y)

        # Inline titlebar â€” we're no longer a Toplevel so we render our
        # own close button in place of the window-manager X.
        titlebar = tk.Frame(win, bg=C["bg3"], height=34)
        titlebar.pack(fill=tk.X)
        titlebar.pack_propagate(False)
        tk.Label(titlebar, text="COOKIE  WALKTHROUGH",
                 bg=C["bg3"], fg=C["accent"],
                 font=("Courier New", 9, "bold")).pack(
            side=tk.LEFT, padx=12, pady=8)
        FlatButton(titlebar, "X",
                   command=self._close_walkthrough,
                   bg=C["bg3"], fg=C["text2"],
                   font=("Courier New", 10, "bold"),
                   pad_x=10, pad_y=4).pack(side=tk.RIGHT, padx=4, pady=4)

        # â”€â”€ Header (cookie name + step counter) â”€â”€
        hdr = tk.Frame(win, bg=C["bg2"])
        hdr.pack(fill=tk.X, padx=20, pady=(14, 0))
        self._wt_title = tk.Label(hdr, text="",
                                  bg=C["bg2"], fg=C["accent"],
                                  font=("Courier New", 14, "bold"))
        self._wt_title.pack(side=tk.LEFT)
        self._wt_counter = tk.Label(hdr, text="",
                                    bg=C["bg2"], fg=C["text3"],
                                    font=("Courier New", 9))
        self._wt_counter.pack(side=tk.RIGHT, pady=(4, 0))

        # Progress bar â€” segmented dots reflecting position in walkthrough.
        prog_wrap = tk.Frame(win, bg=C["bg2"])
        prog_wrap.pack(fill=tk.X, padx=20, pady=(6, 0))
        self._wt_progress = tk.Canvas(prog_wrap, height=3,
                                      bg=C["bg2"],
                                      highlightthickness=0, bd=0)
        self._wt_progress.pack(fill=tk.X)
        self._wt_progress.bind(
            "<Configure>",
            lambda _e: self._wt_draw_progress(
                self._wt_idx + 1, max(1, len(self.cookies))))

        # â”€â”€ Class dropdown directly under the title â”€â”€
        cls_wrap = tk.Frame(win, bg=C["bg2"])
        cls_wrap.pack(fill=tk.X, padx=20, pady=(10, 0))

        # Switch ttk theme to "clam" so our color overrides actually apply
        # (the default "vista" theme on Windows ignores fieldbackground for
        # readonly Combobox state, which makes the field appear bright blue
        # and the text invisible).
        # The dropdown shows full marketing names (Red Velvet Cookie, â€¦)
        # but the model / JSON / canvas overlay all keep using the short
        # internal labels (red_velvet, â€¦). Two parallel lists + a
        # short<->full lookup keep the conversion centralised.
        self._wt_class_options = [CLASS_NAMES[k]
                                  for k in sorted(CLASS_NAMES.keys())]
        self._wt_class_options_full = [
            class_full_name(s) for s in self._wt_class_options]
        self._wt_full_to_short = {
            class_full_name(s): s for s in self._wt_class_options}
        self._wt_short_to_full = {
            s: class_full_name(s) for s in self._wt_class_options}
        style = ttk.Style()
        try:
            if style.theme_use() != "clam":
                style.theme_use("clam")
        except Exception:
            pass
        style.configure("Walkthrough.TCombobox",
                        fieldbackground=C["bg3"],
                        background=C["bg3"],
                        foreground=C["text"],
                        arrowcolor=C["text2"],
                        bordercolor=C["border"],
                        lightcolor=C["bg3"],
                        darkcolor=C["bg3"],
                        insertcolor=C["text"],
                        selectbackground=C["bg3"],
                        selectforeground=C["text"],
                        padding=(10, 6))
        style.map("Walkthrough.TCombobox",
                  fieldbackground=[("readonly", C["bg3"]),
                                   ("focus",    C["bg3"]),
                                   ("active",   C["bg3"])],
                  foreground=[("readonly", C["text"]),
                              ("disabled", C["text3"])],
                  selectbackground=[("readonly", C["bg3"])],
                  selectforeground=[("readonly", C["text"])],
                  bordercolor=[("focus",  C["accent"]),
                               ("active", C["border2"])],
                  arrowcolor=[("active", C["accent"])])
        # Style the popdown listbox (the menu that appears on click) too,
        # otherwise it inherits a light system theme and looks out of place.
        self.root.option_add("*TCombobox*Listbox.background",       C["bg3"])
        self.root.option_add("*TCombobox*Listbox.foreground",       C["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", C["accent"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", C["bg"])
        self.root.option_add("*TCombobox*Listbox.font",             FONT_MONO)
        self.root.option_add("*TCombobox*Listbox.borderWidth",      0)
        self.root.option_add("*TCombobox*Listbox.relief",           "flat")

        self._wt_class_var = tk.StringVar(value="")
        self._wt_class_combo = ttk.Combobox(
            cls_wrap, textvariable=self._wt_class_var,
            values=self._wt_class_options_full,
            state="readonly", font=FONT_MONO,
            style="Walkthrough.TCombobox")
        self._wt_class_combo.pack(fill=tk.X, ipady=3)
        self._wt_class_combo.bind("<<ComboboxSelected>>",
                                  self._wt_on_class_change)

        tk.Frame(win, bg=C["separator"], height=1).pack(
            fill=tk.X, padx=0, pady=(18, 0))

        # â”€â”€ Image card (image on top, slider docked directly below) â”€â”€
        img_outer = tk.Frame(win, bg=C["bg2"])
        img_outer.pack(padx=20, pady=(10, 4))

        # One bordered card holding both the masked image and the slider
        # so they read as a single "cookie image" object.
        img_wrap = tk.Frame(img_outer, bg=C["panel"],
                            highlightbackground=C["border2"],
                            highlightthickness=1)
        img_wrap.pack()

        self._wt_img_lbl = tk.Label(img_wrap, bg=C["panel"])
        self._wt_img_lbl.pack(padx=8, pady=(8, 4))

        # Texture slider â€” single signed value in [-5, +5], integer snap.
        # Packed directly beneath the image, inside the same card.
        self._wt_tex_slider = Slider(
            img_wrap, vmin=-5, vmax=5, value=0,
            width=self._WT_IMAGE_PX,
            on_change=self._wt_on_tex_change,
            on_release=self._wt_on_tex_change,
            value_format=lambda v: (f"+{v}" if v > 0 else str(v)),
            track_color=C["border2"],
            fill_color=C["accent"],
            bg=C["panel"])
        self._wt_tex_slider.pack(padx=8, pady=(0, 6))

        # Correction hint â€” only visible if the user picks a class
        # different from the YOLO prediction.
        self._wt_name_lbl = tk.Label(img_outer, text="",
                                     bg=C["bg2"], fg=C["yellow"],
                                     font=("Courier New", 8))
        self._wt_name_lbl.pack(pady=(4, 0))

        # â”€â”€ Body â”€â”€
        # The frame is created here so its child widgets can be built
        # in the natural reading order, but the actual `body.pack(...)`
        # call is deferred until *after* the button row and footer
        # separator are packed (see below). That way `body` stacks
        # directly above them as a BOTTOM-anchored element, and the
        # diameter entry can never be pushed off-screen on small
        # displays.
        body = tk.Frame(win, bg=C["bg2"])

        # Real-diameter input.
        tk.Label(body, text="REAL  DIAMETER  (MM)",
                 bg=C["bg2"], fg=C["accent2"],
                 font=("Courier New", 8, "bold")).pack(
                    anchor=tk.W, pady=(0, 4))
        self._wt_diam_var = tk.StringVar(value="")
        diam_row = tk.Frame(body, bg=C["bg3"],
                            highlightbackground=C["border"],
                            highlightcolor=C["accent"],
                            highlightthickness=1)
        diam_row.pack(fill=tk.X, pady=(0, 10))
        # Numeric-only validator: accepts the empty string, plain ints
        # like "42", decimals like "3.5", and the mid-typing state
        # "3." (so the user can keep typing after the dot). Anything
        # with letters, symbols, more than one dot, or a leading dot is
        # rejected at the keystroke level â€” the Entry never displays it.
        vcmd = (self.root.register(self._validate_diameter_input), "%P")
        self._wt_diam_entry = tk.Entry(
            diam_row, textvariable=self._wt_diam_var,
            bg=C["bg3"], fg=C["text"],
            insertbackground=C["accent"],
            relief=tk.FLAT, font=FONT_MONO,
            bd=0, highlightthickness=0,
            validate="key", validatecommand=vcmd)
        self._wt_diam_entry.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                 padx=(12, 0), ipady=8)
        tk.Label(diam_row, text="mm", bg=C["bg3"], fg=C["text3"],
                 font=FONT_MONO).pack(side=tk.RIGHT, padx=(0, 12))
        # Persist on every keystroke so a quick NEXT doesn't lose typing.
        self._wt_diam_var.trace_add("write", self._wt_on_diam_change)

        # Skip-state hint (only visible when this cookie is skipped).
        self._wt_skip_lbl = tk.Label(body, text="",
                                     bg=C["bg2"], fg=C["yellow"],
                                     font=FONT_SMALL)
        self._wt_skip_lbl.pack(anchor=tk.W, pady=(4, 0))

        # â”€â”€ Footer separator + button row â”€â”€
        # Packed with side=BOTTOM so they're always anchored to the
        # bottom of the panel. On small displays (Raspberry Pi
        # touchscreens) where the panel can't fit all the content, the
        # image card above gets visually clipped instead of the
        # buttons disappearing off the bottom of the screen.
        btns = tk.Frame(win, bg=C["bg2"])
        btns.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(10, 12))
        tk.Frame(win, bg=C["separator"], height=1).pack(
            side=tk.BOTTOM, fill=tk.X, padx=0, pady=(8, 0))

        # ASCII-only glyphs so any font (incl. the Pi's stock font
        # which lacks U+25C0 / U+25B6 / U+2713) renders them properly.
        # Paddings are tight so all four buttons (BACK / SKIP / NEXT /
        # SAVE) fit on the 440 px-wide panel.
        self._wt_btn_back = FlatButton(btns, "<  BACK",
                                       command=self._wt_back,
                                       bg=C["bg3"], fg=C["text"],
                                       font=FONT_SMALL,
                                       pad_x=10, pad_y=9)
        self._wt_btn_back.pack(side=tk.LEFT)

        self._wt_btn_skip = FlatButton(btns, "SKIP",
                                       command=self._wt_skip,
                                       bg=C["yellow_dim"], fg=C["text"],
                                       font=FONT_SMALL,
                                       pad_x=10, pad_y=9)
        self._wt_btn_skip.pack(side=tk.LEFT, padx=(6, 0))

        # NEXT packed first with side=RIGHT â†’ rightmost; SAVE packed
        # after with side=RIGHT â†’ lands to NEXT's left. Final order
        # left-to-right reads: BACK | SKIP | SAVE | NEXT.
        self._wt_btn_next = FlatButton(btns, "NEXT  >",
                                       command=self._wt_next,
                                       bg=C["accent"], fg=C["bg"],
                                       font=("Courier New", 9, "bold"),
                                       pad_x=12, pad_y=9)
        self._wt_btn_next.pack(side=tk.RIGHT)

        # Always-visible SAVE. Records everything up to and including
        # the current cookie, auto-skipping any cookies after it.
        self._wt_btn_save = FlatButton(btns, "SAVE",
                                       command=self._wt_save_partial,
                                       bg=C["green"], fg=C["bg"],
                                       font=("Courier New", 9, "bold"),
                                       pad_x=12, pad_y=9)
        self._wt_btn_save.pack(side=tk.RIGHT, padx=(0, 6))

        # Pack the diameter body LAST with side=BOTTOM so it stacks
        # directly above the footer separator + button row. This
        # guarantees the REAL DIAMETER label, entry field, and skip
        # hint stay visible on small Pi 5 touchscreens â€” only the
        # cookie image above will get squeezed if anything.
        body.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(6, 4))

        # First render.
        self._wt_render()

        # Panel is positioned via place() above; no Toplevel geometry
        # math required. Just focus the diameter entry so the user can
        # start typing immediately.
        self._wt_diam_entry.focus_set()

    def _wt_render(self):
        """Refresh the popup body for the current self._wt_idx."""
        if self._wt_win is None or not self.cookies:
            return
        n   = len(self.cookies)
        idx = max(0, min(self._wt_idx, n - 1))
        self._wt_idx = idx
        d = self.cookies[idx]

        # Seed per-cookie data on first view of this index.
        data = self._wt_data.get(idx)
        if data is None:
            data = {"name_override": None,
                    "diameter_mm":   None,
                    "tex_lo":        self._WT_DEFAULT_TEX[0],
                    "tex_hi":        self._WT_DEFAULT_TEX[1]}
            self._wt_data[idx] = data

        # â”€â”€ Cookie name (title) â”€â”€
        predicted = d.get('label') or "unknown"
        current_name = data["name_override"] or predicted
        # Popup title shows the short label (e.g. "CLASSIC CHOC") so it
        # matches the on-cookie canvas overlay at a glance. The
        # dropdown below the title still spells out the full marketing
        # name, and the correction hint quotes the short labels too.
        pretty = (current_name.replace("_", " ").upper()
                  if current_name != "unknown" else "UNKNOWN")

        # Title = the cookie's short name; counter + progress on the right.
        self._wt_title.config(text=pretty)
        self._wt_counter.config(text=f"STEP  {idx + 1} / {n}")
        self._wt_draw_progress(idx + 1, n)

        # Correction hint â€” only show when the user has overridden YOLO.
        if data["name_override"] and data["name_override"] != predicted:
            self._wt_name_lbl.config(
                text=f"corrected from "
                     f"{predicted.replace('_', ' ').upper()}")
        else:
            self._wt_name_lbl.config(text="")

        # Populate widgets without firing their callbacks.
        self._wt_suppress_trace = True
        # Dropdown shows the currently-effective class as its full
        # marketing name; we still look up by the short label.
        if current_name in self._wt_class_options:
            self._wt_class_var.set(self._wt_short_to_full[current_name])
        else:
            # Predicted label not in our friendly list (e.g. "unknown") â€”
            # leave the combobox blank to indicate "no selection".
            self._wt_class_var.set("")
        self._wt_diam_var.set("" if data["diameter_mm"] is None
                              else str(data["diameter_mm"]))
        # Slider holds a single signed value. For data that was saved with
        # the old symmetric range, prefer the high end (e.g. (-2, 2) â†’ 2).
        lo_v, hi_v = int(data["tex_lo"]), int(data["tex_hi"])
        tex_val = hi_v if lo_v == -hi_v else (
            lo_v if lo_v == hi_v else (lo_v + hi_v) // 2)
        self._wt_tex_slider.set_value(tex_val)
        self._wt_suppress_trace = False

        # â”€â”€ Masked cookie image â”€â”€
        photo = self._wt_make_cookie_image(d)
        if photo is not None:
            self._wt_photo = photo  # keep ref so Tk doesn't GC it
            self._wt_img_lbl.config(image=photo, width=0, height=0)
        else:
            self._wt_photo = None
            self._wt_img_lbl.config(image="", text="(no image)",
                                    fg=C["text3"], font=FONT_SMALL,
                                    width=30, height=15)

        # Skip-state line.
        if idx in self._wt_skipped:
            self._wt_skip_lbl.config(
                text="This cookie is marked SKIPPED â€” it won't be saved.")
        else:
            self._wt_skip_lbl.config(text="")

        self._wt_btn_back.set_state(idx > 0)
        # NEXT just advances; on the last cookie there's nowhere to
        # advance to, so disable it. SAVE remains live so the user can
        # still finalize from the last step.
        self._wt_btn_next.set_state(idx < n - 1)

        # Highlight the active cookie on the main canvas so the user
        # can match the popup to a specific cookie in the snapshot.
        # Reuses the existing _selected_idx halo+ring rendering.
        self._selected_idx = idx
        self._refresh_display()

    def _wt_draw_progress(self, step, total):
        """Render the walkthrough progress bar as filled/empty segments."""
        c = getattr(self, "_wt_progress", None)
        if c is None or total <= 0:
            return
        c.update_idletasks()
        w = max(1, c.winfo_width())
        h = 3
        gap = 4
        seg_w = max(2, (w - gap * (total - 1)) / total)
        c.delete("all")
        for i in range(total):
            x0 = i * (seg_w + gap)
            x1 = x0 + seg_w
            fill = C["accent"] if i < step else C["border"]
            c.create_rectangle(x0, 0, x1, h, fill=fill, outline="")

    def _wt_make_cookie_image(self, d):
        """Build a PhotoImage of cookie d, masked and centred in the box.

        Pixels outside the cookie's mask are painted in the popup's
        panel colour. The output is a square crop centred on the
        cookie's bbox, sized so the cookie fills most of the display
        box with a small even margin on every side.
        """
        if self.orig_img is None:
            return None
        mask = d.get('mask')
        if mask is None:
            return None
        try:
            mb = mask > 0.5
            if not mb.any():
                return None
            H, W = self.orig_img.shape[:2]
            # Match the mask's resolution to the source frame.
            if mb.shape[:2] != (H, W):
                mb = cv2.resize(
                    mb.astype(np.uint8), (W, H),
                    interpolation=cv2.INTER_NEAREST).astype(bool)

            # Apply the mask on a full-frame copy first so anything we
            # later include in the crop margin is also panel-coloured.
            masked = self.orig_img.copy()
            bg_val = np.array([28, 32, 41], dtype=masked.dtype)
            masked[~mb] = bg_val

            # Square crop centred on the cookie's bbox. Side = longest
            # bbox edge times 1.25 so the cookie fills ~80% of the box.
            ys, xs = np.where(mb)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            side = int(max(y1 - y0, x1 - x0) * 1.25)
            side = max(side, 8)
            half = side // 2

            # Build a padded canvas so the square can extend past the
            # frame edges without clipping or shifting the cookie off-
            # centre near a border.
            canvas = np.full((H + 2 * side, W + 2 * side, 3),
                             bg_val, dtype=masked.dtype)
            canvas[side:side + H, side:side + W] = masked
            cy_c, cx_c = cy + side, cx + side
            sq = canvas[cy_c - half:cy_c + half,
                        cx_c - half:cx_c + half]

            sq_rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(sq_rgb)
            pil = pil.resize((self._WT_IMAGE_PX, self._WT_IMAGE_PX),
                             Image.LANCZOS)
            return ImageTk.PhotoImage(pil)
        except Exception as e:
            print(f"[walkthrough] could not build cookie image: {e}")
            traceback.print_exc()
            return None

    def _wt_on_class_change(self, _event=None):
        """User picked a different class from the dropdown."""
        if getattr(self, "_wt_suppress_trace", False):
            return
        idx = self._wt_idx
        if idx not in self._wt_data:
            return
        # The combobox displays full marketing names, but storage,
        # canvas overlay and JSON keep the short label. Convert here.
        picked_full = self._wt_class_var.get()
        picked = self._wt_full_to_short.get(picked_full, picked_full)
        d = self.cookies[idx]
        predicted = d.get('label') or "unknown"
        # Store as override only if it actually differs from the prediction.
        if picked == predicted:
            self._wt_data[idx]["name_override"] = None
        else:
            self._wt_data[idx]["name_override"] = picked
        # Refresh the name label so the "(corrected)" hint appears/disappears.
        self._wt_render()

    def _wt_on_diam_change(self, *_args):
        if getattr(self, "_wt_suppress_trace", False):
            return
        idx = self._wt_idx
        if idx not in self._wt_data:
            return
        raw = self._wt_diam_var.get().strip()
        if raw == "":
            self._wt_data[idx]["diameter_mm"] = None
        else:
            try:
                self._wt_data[idx]["diameter_mm"] = float(raw)
            except ValueError:
                # User mid-typing ("3" â†’ "3." â†’ "3.5"); leave stored value alone.
                pass
        # Auto-dismiss the "diameter required" warning as soon as the
        # user supplies a positive value â€” they don't need to click
        # NEXT again to see the warning clear.
        if self._wt_current_diameter_valid() and idx not in self._wt_skipped:
            try:
                self._wt_skip_lbl.config(text="", fg=C["yellow"])
            except Exception:
                pass

    @staticmethod
    def _validate_diameter_input(proposed):
        """Tk key-validation callback for the diameter entry.

        Accepts only what could become a valid non-negative decimal
        number:
          - empty string          (clearing the field)
          - digits only           ("42")
          - digits + trailing "." ("3.")
          - digits + "." + digits ("3.5", "10.25")

        Rejects letters, symbols, whitespace, more than one decimal
        point, or a leading dot â€” the Entry simply ignores those
        keystrokes so the user sees nothing typed.
        """
        if proposed == "":
            return True
        if proposed.count(".") > 1:
            return False
        # Strip a single optional dot and check the rest is all digits.
        # This rejects leading "." (e.g. ".5") and any non-digit chars.
        stripped = proposed.replace(".", "", 1)
        if not stripped.isdigit():
            return False
        # Disallow a leading "." like ".5" â€” user must type "0.5".
        if proposed.startswith("."):
            return False
        return True

    def _wt_on_tex_change(self, value):
        """Single signed texture value in [-5, +5]; stored as lo == hi == V."""
        if getattr(self, "_wt_suppress_trace", False):
            return
        idx = self._wt_idx
        if idx not in self._wt_data:
            return
        v = int(value)
        self._wt_data[idx]["tex_lo"] = v
        self._wt_data[idx]["tex_hi"] = v

    def _wt_back(self):
        if self._wt_idx > 0:
            self._wt_idx -= 1
            self._wt_render()

    def _wt_current_diameter_valid(self):
        """True iff the current cookie has a real diameter entered
        AND that value is at least 1 mm. Used to gate NEXT and SAVE
        so partially-filled rows aren't written to JSON. SKIP bypasses
        this check on purpose.
        """
        entry = self._wt_data.get(self._wt_idx, {})
        d = entry.get("diameter_mm")
        try:
            return d is not None and float(d) >= 1
        except (TypeError, ValueError):
            return False

    def _wt_warn_missing_diameter(self, action):
        """Surface the validation failure to the operator and focus
        the diameter entry so they can fix it immediately.
        ``action`` is the verb shown in the status message
        (e.g. 'continue', 'save')."""
        self._set_status(
            f"Enter a real diameter >= 1 mm (or SKIP) to {action}.", "red")
        try:
            self._wt_skip_lbl.config(
                text="Real diameter is required before continuing.",
                fg=C["red"])
        except Exception:
            pass
        try:
            self._wt_diam_entry.focus_set()
        except Exception:
            pass

    def _wt_next(self):
        # Block forward navigation until the current cookie has a
        # real diameter > 0. SKIP is the escape hatch.
        if not self._wt_current_diameter_valid():
            self._wt_warn_missing_diameter("continue")
            return
        if self._wt_idx < len(self.cookies) - 1:
            self._wt_idx += 1
            self._wt_render()

    def _wt_skip(self):
        """Mark the current cookie as skipped and advance.

        Skipped cookies are excluded from the JSON save entirely.
        If we're already on the last cookie, fall through to SAVE so
        the user doesn't get stuck.
        """
        idx = self._wt_idx
        self._wt_skipped.add(idx)
        # Drop the per-cookie entry â€” won't be saved.
        self._wt_data.pop(idx, None)
        self._set_status(f"Skipped cookie #{idx:02d}.", "yellow")
        if idx < len(self.cookies) - 1:
            self._wt_idx += 1
            self._wt_render()
        else:
            # On the last cookie SKIP triggers the save flow directly.
            self._wt_save()

    def _wt_save_partial(self):
        """SAVE button â€” finalize the run from the current cookie.

        Keeps everything the user has already entered for cookies
        0..current (inclusive) and auto-skips any cookies after the
        current one, then writes the JSON via the normal save path.
        Useful when the operator is satisfied partway through a batch
        and wants to stop walking through every remaining cookie.

        The current cookie must have a real diameter > 0 entered, or
        it must already be SKIPped. Otherwise we'd be writing a null
        diameter for whatever row the user happened to be on.
        """
        cur = self._wt_idx
        if cur not in self._wt_skipped and \
                not self._wt_current_diameter_valid():
            self._wt_warn_missing_diameter("save")
            return
        n = len(self.cookies)
        for i in range(cur + 1, n):
            self._wt_skipped.add(i)
            # Drop any data that was seeded but never confirmed for
            # cookies past the current one.
            self._wt_data.pop(i, None)
        self._wt_save()

    def _wt_save(self):
        """Save every cookie's entered diameter and tolerance range to JSON.

        Writes one timestamped file under ``./walkthrough_saves/`` so
        consecutive saves don't overwrite each other.
        """
        if not self.cookies:
            self._set_status("Nothing to save â€” no cookies.", "red")
            self._close_walkthrough()
            return

        out_dir = Path(__file__).resolve().parent / "walkthrough_saves"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror(
                "Save error",
                f"Could not create output folder:\n{out_dir}\n\n{e}")
            return

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"walkthrough_{ts}.json"

        def f(v, nd=3):
            try:
                return round(float(v), nd)
            except (TypeError, ValueError):
                return None

        cookies_out = []
        for i, d in enumerate(self.cookies):
            if i in self._wt_skipped:
                continue
            entry = self._wt_data.get(i, {})
            real_d   = entry.get("diameter_mm")
            tex_val = int(entry.get("tex_hi", self._WT_DEFAULT_TEX[1]))
            override = entry.get("name_override")
            predicted = d.get('label', 'unknown')
            final_label = override if override else predicted
            cls_id = d.get('cls_id')
            # Per-class spec targets pulled from COOKIE_INFO, so
            # downstream consumers can compare the measured diameter
            # and the user-entered texture value against the
            # production-line ideal without re-doing the lookup.
            ideal_d, ideal_min, ideal_max = ideal_info(final_label)
            cookies_out.append({
                "index":             i,
                "cls_id":            (int(cls_id)
                                      if cls_id is not None else None),
                "predicted_label":   predicted,
                "label":             final_label,
                "name_overridden":   bool(override and override != predicted),
                "real_diameter_mm":  (f(real_d)
                                      if real_d is not None else None),
                "texture":           tex_val,
                "ideal_diameter":    f(ideal_d) if ideal_d is not None else None,
                "ideal_min":         f(ideal_min) if ideal_min is not None else None,
                "ideal_max":         f(ideal_max) if ideal_max is not None else None,
                "measured_mm": {
                    "max_feret": f(d.get('feret_mm')),
                    "min_feret": f(d.get('min_feret_mm')),
                },
            })

        # Image dimensions of the frame the user actually walked
        # through. None if no frame is loaded (shouldn't happen at
        # save time, but guard anyway).
        img_w = img_h = None
        if self.orig_img is not None:
            img_h, img_w = self.orig_img.shape[:2]

        # â”€â”€ Two grouped objects: camera (capture / calibration side)
        # and cookie (inspection / per-cookie side). The top-level
        # only holds file metadata.
        camera_obj = {
            "snapshot_path":    getattr(self, "_last_snap_path", None),
            "source_image":     (str(self.image_path)
                                 if getattr(self, 'image_path', None)
                                 else None),
            "image_dimensions": {"width":  img_w,
                                 "height": img_h},
            "pixel_to_mm":      float(SETTINGS.pixel_to_mm),
            "calibrated_at":    SETTINGS.calibrated_at,
        }

        cookie_obj = {
            "model_path":      (self.model_path.get()
                                if hasattr(self, 'model_path') else None),
            "totals": {
                "detected":   len(self.cookies),
                "saved":      len(cookies_out),
                "skipped":    len(self._wt_skipped),
            },
            "skipped_indices": sorted(self._wt_skipped),
            "items":           cookies_out,
        }

        payload = {
            "schema_version": 5,
            "saved_at":       datetime.now().isoformat(timespec="seconds"),
            "camera":         camera_obj,
            "cookie":         cookie_obj,
        }

        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror(
                "Save error",
                f"Failed to write JSON:\n{out_path}\n\n{e}")
            return

        skipped_part = (f"  ({len(self._wt_skipped)} skipped)"
                        if self._wt_skipped else "")
        self._set_status(
            f"Saved {len(cookies_out)} cookie(s){skipped_part} "
            f"to {out_path.name}", "green")
        print(f"[walkthrough] saved â†’ {out_path}")
        self._close_walkthrough()

    def _close_walkthrough(self):
        if self._wt_win is not None:
            try:
                self._wt_win.destroy()
            except Exception:
                pass
            self._wt_win = None
        # Restore the (empty) operator-mode sidebar that was hidden
        # while the walkthrough panel was visible.
        if getattr(self, "_sidebar", None) is not None:
            try:
                self._sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(1, 0))
            except Exception:
                pass
        # Drop the active-cookie highlight on the canvas.
        self._selected_idx = None
        self._refresh_display()

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Table
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Display
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _current_frame(self):
        if self.orig_img is None:
            return None
        # Operator mode: minimal labels-only view. Toggle with 'O' for
        # the full diagnostic overlay (boxes, masks, feret lines, etc.).
        if not self._overlays_visible:
            if not self.cookies:
                return self.orig_img.copy()
            return self._render_minimal(self.orig_img)
        opts = {
            'yolo_overlay':   self.v_yolo.get(),
            'mask':           self.v_mask.get(),
            'grid':           self.v_grid.get(),
            'diameter':       self.v_diameter.get(),
            'feret':          self.v_feret.get(),
            'min_feret':      self.v_min_feret.get(),
            'through_center': self.v_through.get(),
            'selected_idx':   self._selected_idx,
            'orig_img':       self.orig_img,
        }
        base = (self.yolo_plotted
                if opts['yolo_overlay'] and self.yolo_plotted is not None
                else self.orig_img)
        return render_overlays(base, self.cookies, opts) if self.cookies \
            else base.copy()

    def _render_minimal(self, base):
        """Operator-friendly overlay: draws only the cookie's class name
        on each detection. Highlights the currently-selected cookie
        (e.g. the active walkthrough step) with a cyan outline so
        ``selected_idx`` still has visual meaning.
        """
        out = base.copy()
        sel = self._selected_idx
        for i, d in enumerate(self.cookies):
            mask = d.get('mask')
            is_sel = (i == sel)
            # Highlight only the selected cookie â€” kept lightweight so
            # the rest of the cookies stay completely clean.
            if is_sel and mask is not None:
                mask_u8 = (mask > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contours:
                    # Pink (BGR) â€” picks out the active walkthrough
                    # cookie against any background. Thick stroke so
                    # it stays readable on small Pi touchscreens.
                    cv2.drawContours(out, contours, -1, (180, 105, 255), 8)
            # Cookie name centered on its centroid, with a black halo so
            # it stays readable over any background.
            if 'center' in d:
                cx, cy = d['center']
                label = d.get('label') or 'unknown'
                text = (label.replace('_', ' ').upper()
                        if label != 'unknown' else '?')
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                tx = int(cx - tw / 2)
                ty = int(cy + th / 2)
                cv2.putText(out, text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(out, text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 212, 255), 2, cv2.LINE_AA)
        return out

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

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Zoom helpers
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

        # Keep the image point under the cursor stationary
        img = self._current_frame()
        if img is not None:
            h, w = img.shape[:2]
            fit  = min(cw / w, ch / h)
            old_ox = (cw - w * fit * old_zoom) / 2 + self._pan_x
            old_oy = (ch - h * fit * old_zoom) / 2 + self._pan_y
            img_x  = (pivot_x - old_ox) / (fit * old_zoom)
            img_y  = (pivot_y - old_oy) / (fit * old_zoom)
            self._pan_x = pivot_x - (cw - w * fit * new_zoom) / 2 \
                          - img_x * fit * new_zoom
            self._pan_y = pivot_y - (ch - h * fit * new_zoom) / 2 \
                          - img_y * fit * new_zoom

        self._zoom = new_zoom
        self._refresh_display()

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Mouse-wheel zoom
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _on_mousewheel(self, event):
        # Determine scroll direction (cross-platform)
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = 1.15
        else:
            factor = 1 / 1.15
        self._apply_zoom(factor, pivot_x=event.x, pivot_y=event.y)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Pan (drag)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Distance (in pixels) the mouse must move during a press to be
    # treated as a drag rather than a click
    CLICK_THRESHOLD = 5

    def _on_pan_start(self, event):
        self._pan_start    = (event.x, event.y)
        self._pan_origin   = (self._pan_x, self._pan_y)
        self._was_dragged  = False

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

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Click-to-inspect
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _canvas_to_image_coords(self, cx_canvas, cy_canvas):
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
        for i, d in enumerate(self.cookies):
            mask = d.get('mask')
            if mask is None:
                continue
            if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]:
                if mask[iy, ix] > 0.5:
                    return i
        return None

    def _on_canvas_click(self, event):
        """Clicking a cookie just selects it (updates the canvas
        highlight + makes it the target of the T / S hotkeys). The
        WITHIN-SPEC detail panel no longer opens on click â€” press
        'S' to show it."""
        if not self.cookies:
            return
        pt = self._canvas_to_image_coords(event.x, event.y)
        if pt is None:
            return
        ix, iy = pt
        idx = self._hit_test(ix, iy)
        if idx is not None:
            # Close any panel that's already open for an old selection.
            if getattr(self, "_detail", None) is not None and \
                    self._detail.winfo_ismapped():
                self._hide_detail_panel()
            self._selected_idx = idx
            self._refresh_display()
        else:
            # Clicking empty space clears selection and hides panel.
            self._hide_detail_panel()

    def _on_hover(self, event):
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

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Detail panel
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _show_detail_panel(self, idx):
        d = self.cookies[idx]
        if self._texture_panel is not None and \
                self._texture_panel.winfo_ismapped():
            self._hide_texture_panel()
        if getattr(self, '_detail', None) is None:
            self._build_detail_panel()
        self._detail.lift()
        self._detail.place(relx=1.0, rely=0.0, x=-12, y=12, anchor='ne')
        self._detail.update_idletasks()
        self._populate_detail(idx, d)
        self._selected_idx = idx
        try:
            self._texture_hint.config(
                text=f"Selected: cookie #{idx:02d}\nClick the button above.",
                fg=C["text2"])
        except Exception:
            pass
        self._refresh_display()

    def _hide_detail_panel(self):
        if getattr(self, '_detail', None) is not None:
            self._detail.place_forget()
        for c in getattr(self, '_mpl_canvases', []):
            try:
                c.get_tk_widget().destroy()
                if MPL_AVAILABLE:
                    plt.close(c.figure)
            except Exception:
                pass
        self._mpl_canvases = []
        self._selected_idx = None
        try:
            self._texture_hint.config(
                text="Click a cookie on the image first,\n"
                     "then open texture analysis.",
                fg=C["text3"])
        except Exception:
            pass
        self._refresh_display()

    def _build_detail_panel(self):
        self._detail = tk.Frame(
            self._canvas.master,
            bg=C["panel"],
            highlightbackground=C["accent"],
            highlightthickness=1,
            width=380,
            height=520,
        )
        self._detail.pack_propagate(False)

        hdr = tk.Frame(self._detail, bg=C["bg3"], height=36)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        self._detail_title = tk.Label(
            hdr, text="COOKIE #â€”", bg=C["bg3"], fg=C["accent"],
            font=("Courier New", 11, "bold"))
        self._detail_title.pack(side=tk.LEFT, padx=12, pady=8)
        FlatButton(hdr, "X", command=self._hide_detail_panel,
                   bg=C["bg3"], fg=C["text2"],
                   font=("Courier New", 11, "bold"),
                   pad_x=10, pad_y=4).pack(side=tk.RIGHT, padx=4, pady=4)

        outer = tk.Frame(self._detail, bg=C["panel"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._detail_canvas = tk.Canvas(outer, bg=C["panel"],
                                        highlightthickness=0)
        scroll = tk.Scrollbar(outer, orient=tk.VERTICAL,
                              command=self._detail_canvas.yview)
        self._detail_canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._detail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._detail_body_outer = tk.Frame(self._detail_canvas, bg=C["panel"])
        body_id = self._detail_canvas.create_window(
            (0, 0), window=self._detail_body_outer, anchor=tk.NW)
        self._detail_body = tk.Frame(self._detail_body_outer, bg=C["panel"])
        self._detail_body.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        self._detail_body_outer.bind(
            "<Configure>",
            lambda e: self._detail_canvas.configure(
                scrollregion=self._detail_canvas.bbox("all")))

        def _sync_body_width(event):
            if event.width > 50:
                self._detail_canvas.itemconfig(body_id, width=event.width)
        self._detail_canvas.bind("<Configure>", _sync_body_width)

        def _on_panel_wheel(event):
            delta = -1 if (event.num == 5 or
                           (hasattr(event, 'delta') and event.delta < 0)) else 1
            self._detail_canvas.yview_scroll(-delta, "units")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._detail_canvas.bind(seq, _on_panel_wheel)
            self._detail_body.bind(seq,   _on_panel_wheel)
            self._detail_body_outer.bind(seq, _on_panel_wheel)

    def _populate_detail(self, idx, d):
        self._detail.config(height=520)

        for c in getattr(self, '_mpl_canvases', []):
            try:
                c.get_tk_widget().destroy()
                plt.close(c.figure)
            except Exception:
                pass
        self._mpl_canvases = []

        self._detail_title.config(text=f"COOKIE  #{idx:02d}")
        # If we have a real class label, prepend the cookie name so users
        # can see what type of cookie they clicked on at a glance.
        if d.get('label') and d['label'] != "unknown":
            name_display = d['label'].replace("_", " ").upper()
            self._detail_title.config(
                text=f"{name_display}  #{idx:02d}")
        for w in self._detail_body.winfo_children():
            w.destroy()

        is_lop = d['lopsided']
        pill_bg = C["red"] if is_lop else C["green"]
        pill_fg = C["bg"]
        status_txt = "LOPSIDED" if is_lop else "WITHIN SPEC"
        pill = tk.Label(self._detail_body, text=f"  {status_txt}  ",
                        bg=pill_bg, fg=pill_fg,
                        font=("Courier New", 9, "bold"), padx=8, pady=4)
        pill.pack(anchor=tk.W, pady=(0, 12))

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

        # Cookie class / type
        tk.Label(self._detail_body, text="CLASSIFICATION",
                 bg=C["panel"], fg=C["accent"],
                 font=FONT_SMALL).pack(anchor=tk.W, pady=(4, 6))
        kv_row("Type", d.get('label', 'unknown').replace("_", " "),
               C["yellow"])
        kv_row("Class ID",
               str(d['cls_id']) if d.get('cls_id') is not None else "â€”",
               C["text3"])

        tk.Frame(self._detail_body, bg=C["border"], height=1).pack(
            fill=tk.X, pady=10)

        tk.Label(self._detail_body, text="DIMENSIONS",
                 bg=C["panel"], fg=C["accent"],
                 font=FONT_SMALL).pack(anchor=tk.W, pady=(4, 6))
        kv_row("Max Feret",        f"{d['feret_mm']:.2f}",      C["cyan"],    "mm",
               "longest span")
        kv_row("Min Feret",        f"{d['min_feret_mm']:.2f}",  C["green"],   "mm",
               "narrowest span")
        kv_row("Through-center",   f"{d['tc_mm']:.2f}",         C["magenta"], "mm",
               "longest chord through centroid")

        tk.Frame(self._detail_body, bg=C["border"], height=1).pack(
            fill=tk.X, pady=10)

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

        tk.Frame(self._detail_body, bg=C["border"], height=1).pack(
            fill=tk.X, pady=10)

        tk.Label(self._detail_body, text="GEOMETRY",
                 bg=C["panel"], fg=C["accent"],
                 font=FONT_SMALL).pack(anchor=tk.W, pady=(0, 6))

        cx, cy = d['center']
        kv_row("Centroid",     f"({cx}, {cy})",  C["text"], "px")
        kv_row("Encl. radius", f"{d['mec_radius']}", C["text"], "px",
               f"= {d['mec_radius']/SETTINGS.pixel_to_mm:.2f} mm")

        tk.Frame(self._detail_body, bg=C["border"], height=1).pack(
            fill=tk.X, pady=10)
        tk.Label(self._detail_body,
                 text="Click another cookie to inspect it,\n"
                      "or click empty space to close.\n\n"
                      "For texture, colour, and quality analysis,\n"
                      "use the TEXTURE ANALYSIS button in the sidebar.",
                 bg=C["panel"], fg=C["text3"], font=FONT_SMALL,
                 justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Texture analysis panel
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def on_open_texture(self):
        if not self.cookies:
            messagebox.showinfo(
                "No cookies",
                "Run inspection first to detect cookies.")
            return
        if self._selected_idx is None:
            messagebox.showinfo(
                "No cookie selected",
                "Click a cookie on the image first, "
                "then open texture analysis.")
            return
        d = self.cookies[self._selected_idx]
        if 'tex_median' not in d:
            messagebox.showwarning(
                "Texture data unavailable",
                "Texture maps weren't computed.\n"
                "Make sure scikit-image is installed:\n"
                "    pip install scikit-image")
            return
        if getattr(self, '_detail', None) is not None and \
                self._detail.winfo_ismapped():
            self._detail.place_forget()
        self._show_texture_panel(self._selected_idx, d)

    def _show_texture_panel(self, idx, d):
        if self._texture_panel is None:
            self._build_texture_panel()
        self._texture_panel.lift()
        self._texture_panel.place(relx=1.0, rely=0.0,
                                  x=-12, y=12, anchor='ne')
        self._texture_panel.update_idletasks()
        self._populate_texture_panel(idx, d)

    def _hide_texture_panel(self):
        if self._texture_panel is not None:
            self._texture_panel.place_forget()
        for c in self._tex_mpl_canvases:
            try:
                c.get_tk_widget().destroy()
                if MPL_AVAILABLE:
                    plt.close(c.figure)
            except Exception:
                pass
        self._tex_mpl_canvases = []

    def _build_texture_panel(self):
        self._texture_panel = tk.Frame(
            self._canvas.master,
            bg=C["panel"],
            highlightbackground=C["accent2"],
            highlightthickness=1,
            width=420,
            height=780,
        )
        self._texture_panel.pack_propagate(False)

        hdr = tk.Frame(self._texture_panel, bg=C["bg3"], height=36)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        self._tex_title = tk.Label(
            hdr, text="[T]  TEXTURE  ANALYSIS",
            bg=C["bg3"], fg=C["accent2"],
            font=("Courier New", 11, "bold"))
        self._tex_title.pack(side=tk.LEFT, padx=12, pady=8)
        FlatButton(hdr, "X", command=self._hide_texture_panel,
                   bg=C["bg3"], fg=C["text2"],
                   font=("Courier New", 11, "bold"),
                   pad_x=10, pad_y=4).pack(side=tk.RIGHT, padx=4, pady=4)

        outer = tk.Frame(self._texture_panel, bg=C["panel"])
        outer.pack(fill=tk.BOTH, expand=True)
        self._tex_canvas = tk.Canvas(outer, bg=C["panel"],
                                     highlightthickness=0)
        scroll = tk.Scrollbar(outer, orient=tk.VERTICAL,
                              command=self._tex_canvas.yview)
        self._tex_canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._tex_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._tex_body_outer = tk.Frame(self._tex_canvas, bg=C["panel"])
        body_id = self._tex_canvas.create_window(
            (0, 0), window=self._tex_body_outer, anchor=tk.NW)
        self._tex_body = tk.Frame(self._tex_body_outer, bg=C["panel"])
        self._tex_body.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        self._tex_body_outer.bind(
            "<Configure>",
            lambda e: self._tex_canvas.configure(
                scrollregion=self._tex_canvas.bbox("all")))

        def _sync_w(event):
            if event.width > 50:
                self._tex_canvas.itemconfig(body_id, width=event.width)
        self._tex_canvas.bind("<Configure>", _sync_w)

        def _on_wheel(event):
            delta = -1 if (event.num == 5 or
                           (hasattr(event, 'delta') and event.delta < 0)) else 1
            self._tex_canvas.yview_scroll(-delta, "units")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._tex_canvas.bind(seq, _on_wheel)
            self._tex_body.bind(seq,   _on_wheel)
            self._tex_body_outer.bind(seq, _on_wheel)

    def _populate_texture_panel(self, idx, d):
        for c in self._tex_mpl_canvases:
            try:
                c.get_tk_widget().destroy()
                if MPL_AVAILABLE:
                    plt.close(c.figure)
            except Exception:
                pass
        self._tex_mpl_canvases = []
        for w in self._tex_body.winfo_children():
            w.destroy()

        self._tex_title.config(text=f"[T]  COOKIE  #{idx:02d}  TEXTURE")
        if d.get('label') and d['label'] != "unknown":
            name_display = d['label'].replace("_", " ").upper()
            self._tex_title.config(
                text=f"[T]  {name_display}  #{idx:02d}  TEXTURE")

        def kv_row(label, value, color=C["text"], unit="", note=""):
            row = tk.Frame(self._tex_body, bg=C["panel"])
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label, bg=C["panel"], fg=C["text3"],
                     font=FONT_SMALL, anchor=tk.W,
                     width=18).pack(side=tk.LEFT)
            tk.Label(row, text=value, bg=C["panel"], fg=color,
                     font=("Courier New", 11, "bold")).pack(side=tk.LEFT)
            if unit:
                tk.Label(row, text=f" {unit}", bg=C["panel"],
                         fg=C["text3"], font=FONT_SMALL).pack(side=tk.LEFT)
            if note:
                tk.Label(row, text=f"   {note}", bg=C["panel"],
                         fg=C["text3"], font=FONT_SMALL).pack(side=tk.LEFT)

        tk.Label(self._tex_body, text="TEXTURE  STATISTICS",
                 bg=C["panel"], fg=C["accent2"],
                 font=FONT_SMALL).pack(anchor=tk.W, pady=(0, 6))
        kv_row("Tex. median",  f"{d['tex_median']:.2f}",
               C["yellow"], note="LBP local variance")
        kv_row("Stable min",   f"{d['tex_p5']:.2f}",
               C["text"],   note="P5 (robust)")
        kv_row("Stable max",   f"{d['tex_p95']:.2f}",
               C["text"],   note="P95 (robust)")

        tk.Frame(self._tex_body, bg=C["border"], height=1).pack(
            fill=tk.X, pady=10)

        tk.Label(self._tex_body, text="COLOUR  &  QUALITY",
                 bg=C["panel"], fg=C["accent2"],
                 font=FONT_SMALL).pack(anchor=tk.W, pady=(0, 6))
        if 'hue_median' in d:
            kv_row("Median hue",   f"{d['hue_median']:.1f}",
                   C["text"], "deg", "0-179 in OpenCV HSV")
        if 'quality_pct' in d:
            q = d['quality_pct']
            qcolor = (C["green"] if q >= 75 else
                      C["yellow"] if q >= 40 else C["red"])
            kv_row("Bake quality", f"{q:.1f}",
                   qcolor, "%", "% in golden hue range")

        tk.Frame(self._tex_body, bg=C["border"], height=1).pack(
            fill=tk.X, pady=10)

        if not MPL_AVAILABLE:
            tk.Label(self._tex_body,
                     text="matplotlib not installed â€” plots disabled\n"
                          "    pip install matplotlib",
                     bg=C["panel"], fg=C["text3"],
                     font=FONT_SMALL, justify=tk.LEFT).pack(
                anchor=tk.W, pady=(8, 0))
            return

        tk.Label(self._tex_body, text="COLOUR  UNIFORMITY  MAP",
                 bg=C["panel"], fg=C["accent2"],
                 font=FONT_SMALL).pack(anchor=tk.W, pady=(2, 4))
        self._add_texture_signature_plot(d)

        tk.Label(self._tex_body, text="BATCH  TEXTURE  DISTRIBUTION",
                 bg=C["panel"], fg=C["accent2"],
                 font=FONT_SMALL).pack(anchor=tk.W, pady=(10, 4))
        self._add_batch_histogram_plot(d)

        tk.Label(self._tex_body, text="QUALITY  CALIBRATION  SCALE",
                 bg=C["panel"], fg=C["accent2"],
                 font=FONT_SMALL).pack(anchor=tk.W, pady=(10, 4))
        self._add_hue_calibration_plot(d)

        tk.Frame(self._tex_body, bg=C["border"], height=1).pack(
            fill=tk.X, pady=10)
        tk.Label(self._tex_body,
                 text="Pick another cookie and click TEXTURE\n"
                      "ANALYSIS again to compare.",
                 bg=C["panel"], fg=C["text3"], font=FONT_SMALL,
                 justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

    def _add_texture_signature_plot(self, d):
        if d.get('mask') is None or self.orig_img is None:
            return
        mask = d['mask'] > 0.5
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return

        pad = 8
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(self.orig_img.shape[1], int(xs.max()) + pad)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(self.orig_img.shape[0], int(ys.max()) + pad)

        bgr_crop  = self.orig_img[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]

        lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
        L   = lab[:, :, 0]

        smooth = cv2.bilateralFilter(
            L, d=15, sigmaColor=40, sigmaSpace=40)

        masked = np.where(mask_crop, smooth.astype(np.float32), np.nan)

        cookie_vals = smooth[mask_crop]
        if cookie_vals.size == 0:
            return
        vmin = float(np.percentile(cookie_vals, 2))
        vmax = float(np.percentile(cookie_vals, 98))
        if vmax - vmin < 5:
            vmin, vmax = vmin - 5, vmax + 5

        fig = Figure(figsize=(3.8, 2.8), dpi=90, facecolor=C["panel"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(C["panel"])
        im = ax.imshow(masked, cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(C["border"])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("brightness (LAB L)", color=C["text2"], fontsize=7)
        cbar.ax.tick_params(colors=C["text2"], labelsize=7)
        cbar.outline.set_edgecolor(C["border"])
        fig.tight_layout(pad=0.4)
        self._mount_figure_tex(fig)

    def _add_batch_histogram_plot(self, d):
        all_vals = [c['tex_values'] for c in self.cookies if 'tex_values' in c]
        if not all_vals:
            return
        batch = np.concatenate(all_vals)
        if batch.size > 200000:
            idx = np.random.choice(batch.size, 200000, replace=False)
            batch = batch[idx]
        batch_median = float(np.median(batch))
        this_median  = d['tex_median']

        fig = Figure(figsize=(3.8, 2.2), dpi=90, facecolor=C["panel"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(C["panel"])
        ax.hist(batch, bins=60, color="#00d4ff", alpha=0.55, edgecolor="none")
        ax.axvline(batch_median, color=C["text"], lw=1.5,
                   linestyle="--", label=f"Batch  {batch_median:.1f}")
        ax.axvline(this_median,  color=C["red"],  lw=2.0,
                   label=f"This   {this_median:.1f}")
        ax.tick_params(colors=C["text2"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(C["border"])
        ax.set_xlabel("LBP variance", color=C["text2"], fontsize=8)
        ax.set_ylabel("count",        color=C["text2"], fontsize=8)
        leg = ax.legend(facecolor=C["bg2"], edgecolor=C["border"], fontsize=7)
        for txt in leg.get_texts():
            txt.set_color(C["text"])
        fig.tight_layout(pad=0.4)
        self._mount_figure_tex(fig)

    def _add_hue_calibration_plot(self, d):
        if 'hue_median' not in d:
            return

        H_MAX = 180
        grad   = np.linspace(0, H_MAX - 1, 600).astype(np.uint8)
        strip  = np.stack([grad] * 60, axis=0)
        hsv    = np.zeros((60, 600, 3), dtype=np.uint8)
        hsv[:, :, 0] = strip
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = 230
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        hue   = float(d['hue_median'])
        in_gold = (10 <= hue <= 25)

        fig = Figure(figsize=(3.8, 1.6), dpi=90, facecolor=C["panel"])
        ax  = fig.add_subplot(111)
        ax.set_facecolor(C["panel"])

        ax.imshow(rgb, extent=[0, H_MAX, 0, 1], aspect='auto')
        ax.axvspan(10, 25, color=C["green"], alpha=0.18)
        ax.axvline(10, color=C["green"], lw=1.2, alpha=0.9)
        ax.axvline(25, color=C["green"], lw=1.2, alpha=0.9)

        marker_color = C["green"] if in_gold else C["red"]
        ax.axvline(hue, color="black", lw=4)
        ax.axvline(hue, color=marker_color, lw=2)
        label_x = max(6, min(H_MAX - 6, hue))
        ha = ('left'  if hue < H_MAX * 0.15 else
              'right' if hue > H_MAX * 0.85 else 'center')
        ax.annotate(
            f"{hue:.1f}",
            xy=(hue, 1.0), xytext=(label_x, 1.18),
            ha=ha, va='bottom',
            color=marker_color, fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor=C["bg2"], edgecolor=marker_color, lw=1))

        ax.set_xticks([0, 10, 25, 60, 120, 179])
        ax.set_xlim(0, H_MAX)
        ax.set_ylim(0, 1.45)
        ax.set_yticks([])
        ax.tick_params(colors=C["text2"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(C["border"])

        status = "in golden range" if in_gold else "OUT of golden range"
        ax.set_xlabel(
            f"hue (OpenCV HSV) â€” this cookie {status}",
            color=marker_color if not in_gold else C["text2"],
            fontsize=8)
        fig.tight_layout(pad=0.3)
        self._mount_figure_tex(fig)

    def _mount_figure_tex(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self._tex_body)
        widget = canvas.get_tk_widget()
        widget.configure(bg=C["panel"], highlightthickness=0)
        widget.pack(fill=tk.X, pady=(0, 4))
        canvas.draw()
        self._tex_mpl_canvases.append(canvas)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Live camera mode
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                self._camera = Camera()
            self._camera.start()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Camera error",
                                 f"Could not start camera:\n{e}")
            self._camera = None
            return

        self._live_mode = True
        self.btn_live.config(text="#  STOP LIVE", bg=C["red"])
        self.btn_live._bg  = C["red"]
        self.btn_live._dim = FlatButton._darken(C["red"])
        self._file_lbl.config(text="* LIVE FEED", fg=C["red"])
        self._set_status("Live feed active â€” press CALIBRATE or RUN INSPECTION",
                         "green")
        self.cookies      = []
        self.yolo_plotted = None
        self._selected_idx = None
        if self._detail is not None:
            self._detail.place_forget()
        if self._texture_panel is not None:
            self._hide_texture_panel()
        self._zoom  = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
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
        self.btn_live.config(text="[CAM]  LIVE", bg=C["bg3"])
        self.btn_live._bg  = C["bg3"]
        self.btn_live._dim = FlatButton._darken(C["bg3"])
        self._set_status("Live feed stopped.", "green")

    def _tick_live(self):
        """Grab one frame from the camera and refresh the display.

        Hardened so a bad frame can't kill the loop:
        - exceptions in capture/display are caught and printed
        - the next tick is always rescheduled while _live_mode is True
        """
        if not self._live_mode or self._camera is None:
            return
        try:
            frame = self._camera.get_frame()
            if frame is not None:
                self.orig_img = frame
                if not self._busy:
                    self.yolo_plotted = None
                    self.cookies      = []
                h, w = frame.shape[:2]
                self._dim_lbl.config(text=f"{w} Ã— {h} px Â· LIVE")
                self._refresh_display()
            else:
                print("[live] get_frame() returned None")
        except Exception as e:
            print(f"[live] tick error: {e}")
            traceback.print_exc()
            self._set_status(f"Live error: {e}", "red")
        # Always reschedule while live mode is on.
        # ~10 FPS at max-res is plenty for inspection (33ms â‰ˆ 30 FPS but
        # 3280x2464 frames are heavy; 100ms â‰ˆ 10 FPS is realistic).
        if self._live_mode:
            self._live_after_id = self.root.after(100, self._tick_live)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Quick capture â€” one-press live â†’ snapshot â†’ inspect pipeline
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Number of milliseconds to let the live feed run before grabbing
    # the snapshot. This lets the Pi camera's auto-exposure / auto-WB
    # settle so the first usable frame isn't dark or color-shifted.
    QUICK_CAPTURE_WARMUP_MS = 800

    def _on_calibrate_hotkey(self, event):
        """Keyboard handler for the C hotkey â€” runs calibration against
        whatever the live feed is currently showing. Same code path as
        the (hidden) CALIBRATE button, just keyboard-triggered.

        Suppressed when the user is typing into an Entry/Text widget.
        """
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return
        self.on_calibrate()

    def _on_overlays_hotkey(self, event):
        """Keyboard handler for the O hotkey â€” toggles the full
        diagnostic overlay layer (boxes, masks, feret lines, labels
        with confidence, etc.) on top of the labels-only operator view.
        """
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return
        self._overlays_visible = not self._overlays_visible
        state = "ON" if self._overlays_visible else "OFF  (labels only)"
        self._set_status(f"Overlays {state}",
                         "yellow" if self._overlays_visible else "green")
        self._refresh_display()

    def _on_texture_hotkey(self, event):
        """Keyboard handler for the T hotkey â€” opens texture analysis
        for the currently-selected cookie. Same code path as the
        (hidden) TEXTURE ANALYSIS button.
        """
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return
        self.on_open_texture()

    def _on_run_hotkey(self, event):
        """Keyboard handler for the R hotkey â€” runs inspection against
        the current frame. Same code path as the (hidden) RUN
        INSPECTION button.
        """
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return
        self.on_run()

    def _on_upload_hotkey(self, event):
        """Keyboard handler for the U hotkey â€” opens the upload dialog.
        Same code path as the (hidden) UPLOAD IMAGE button.
        """
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return
        self.on_upload()

    def _on_spec_hotkey(self, event):
        """Keyboard handler for the S hotkey â€” opens the WITHIN-SPEC
        detail panel for the currently-selected cookie. Same code path
        as the old click-to-show behaviour, just keyboard-triggered.
        Press 'S' again to close it.
        """
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return
        # Toggle: if already open, close it.
        if getattr(self, "_detail", None) is not None and \
                self._detail.winfo_ismapped():
            self._hide_detail_panel()
            return
        if not self.cookies:
            messagebox.showinfo(
                "No cookies",
                "Run inspection first to detect cookies.")
            return
        if self._selected_idx is None:
            messagebox.showinfo(
                "No cookie selected",
                "Click a cookie on the image first, then press S.")
            return
        self._show_detail_panel(self._selected_idx)

    def _on_quick_capture_hotkey(self, event):
        """Keyboard handler for the Q hotkey.

        Suppressed when the user is typing into an Entry/Text widget so
        the letter 'q' still works in those fields (e.g. typing a path).
        """
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return
        self.on_quick_capture()

    def _on_pi_button_capture(self):
        """Hardware-button entry point. Same flow as Q, but flags the
        inference worker to skip texture analysis + per-cookie metrology
        so the Pi 5 spends its cycles on YOLO + walkthrough only.

        On systemd autostart the model loads in a background thread, so
        a button press during boot can race ahead of self.model being
        set. Instead of surfacing the modal "No model" popup, wait
        quietly (up to ~30 s) for the loader to finish, then proceed.
        """
        if self.model is None:
            if getattr(self, "_pi_wait_ticks", 0) >= 150:  # 150 * 200ms = 30s
                self._pi_wait_ticks = 0
                self._set_status(
                    "Model still loading â€” press button again shortly.",
                    "yellow")
                return
            self._pi_wait_ticks = getattr(self, "_pi_wait_ticks", 0) + 1
            self._set_status("Model loading â€” waitingâ€¦", "yellow")
            self.root.after(200, self._on_pi_button_capture)
            return
        self._pi_wait_ticks = 0
        self._fast_inference = True
        self.on_quick_capture()

    def on_quick_capture(self):
        """Public entry point for the QUICK CAPTURE button / Q hotkey.

        Guards against re-entry, missing camera, missing model, and
        in-flight inspection. On success it kicks off the async state
        machine below.
        """
        if self._quick_capture_active:
            self._set_status("Quick capture already in progressâ€¦", "yellow")
            return
        if not PICAMERA_AVAILABLE:
            messagebox.showwarning(
                "Camera unavailable",
                "picamera2 is not installed â€” quick capture needs a "
                "live camera feed.")
            return
        if self.model is None:
            messagebox.showwarning(
                "No model",
                "Model not loaded â€” set path and click LOAD before "
                "using quick capture.")
            return
        if self._busy:
            self._set_status("Inspection already running â€” please wait.",
                             "yellow")
            return

        self._quick_capture_active = True
        # Audible cue: beep as soon as inspection commits. Wired here
        # (not only in the GPIO ISR) so every trigger path -- Q hotkey,
        # on-screen QUICK CAPTURE button, hardware GPIO button -- beeps
        # the same way. The beep runs on its own thread inside
        # GPIOQuickCaptureTrigger.beep(), so this call returns instantly.
        try:
            self._gpio_trigger.beep()
        except Exception as e:
            print(f"[quick] buzzer trigger failed: {e}")
        # Visual cue on the button so the user knows the chain is running.
        self.btn_quick.config(text="...  CAPTURING...")
        self.btn_quick.set_state(False)
        self._set_status("Quick capture: starting live feedâ€¦", "yellow")
        # Show the centered busy overlay immediately so the user knows
        # the press registered. It will be updated to "running yolo"
        # by on_run() shortly, and torn down by _on_done.
        self._show_busy_overlay("Warming up camera...")

        # Step 1: start live feed (unless one is already running, in
        # which case we just snapshot whatever's on screen).
        if not self._live_mode:
            self._start_live()
            if not self._live_mode:
                # _start_live() bailed out (camera error). Reset and stop.
                self._quick_capture_reset()
                return

        # Step 2: wait for auto-exposure to settle, then snapshot.
        self.root.after(self.QUICK_CAPTURE_WARMUP_MS,
                        self._quick_capture_snapshot)

    def _quick_capture_snapshot(self):
        """Grab one frame from the live feed and stop the camera.

        Runs on the Tk main thread (called via root.after). After
        successfully capturing, it triggers the inference worker.
        """
        if not self._quick_capture_active:
            return  # cancelled somehow

        # Pull the latest frame the live loop has already deposited into
        # self.orig_img. If that's missing (e.g. timing edge case),
        # ask the camera directly.
        frame = None
        try:
            if self.orig_img is not None and self._live_mode:
                frame = self.orig_img.copy()
            elif self._camera is not None:
                frame = self._camera.get_frame()
        except Exception as e:
            print(f"[quick] snapshot capture error: {e}")
            traceback.print_exc()

        if frame is None:
            self._set_status("Quick capture: no frame from camera.", "red")
            self._stop_live()
            self._quick_capture_reset()
            return

        # Step 3: freeze that frame as the working image, stop live feed.
        self.orig_img = frame
        h, w = frame.shape[:2]
        self._file_lbl.config(text="* QUICK SNAPSHOT", fg=C["yellow"])
        self._dim_lbl.config(text=f"{w} Ã— {h} px Â· SNAPSHOT")

        # Persist the captured frame to disk so it can be reviewed
        # later. One file per capture, timestamped to avoid collisions.
        # Path is also embedded in the walkthrough JSON below.
        self._last_snap_path = None
        try:
            out_dir = Path(__file__).resolve().parent / "saved_snapped"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = out_dir / f"snap_{ts}.jpg"
            if cv2.imwrite(str(snap_path), frame):
                self._last_snap_path = str(snap_path)
                print(f"[quick] snapshot saved â†’ {snap_path}")
            else:
                print(f"[quick] cv2.imwrite returned False for {snap_path}")
        except Exception as e:
            print(f"[quick] could not save snapshot: {e}")
            traceback.print_exc()

        self._set_status("Quick capture: snapshot taken, stopping liveâ€¦",
                         "yellow")
        self._stop_live()
        self._refresh_display()

        # Step 4: kick off inspection. on_run() handles its own
        # threading; we hook into its completion via a polling check
        # so we can restore the button when it's done.
        self._set_status("Quick capture: running inspectionâ€¦", "yellow")
        self.on_run()
        # Poll for inspection completion to clear the button state.
        self.root.after(200, self._quick_capture_wait_for_inspection)

    def _quick_capture_wait_for_inspection(self):
        """Poll until on_run()'s worker has finished, then reset UI."""
        if not self._quick_capture_active:
            return
        if self._busy:
            # Still running â€” keep polling.
            self.root.after(200, self._quick_capture_wait_for_inspection)
            return
        # Done.
        self._quick_capture_reset()
        self._set_status("Quick capture complete.", "green")

    def _quick_capture_reset(self):
        """Restore the QUICK CAPTURE button and clear the active flag."""
        self._quick_capture_active = False
        try:
            self.btn_quick.config(text=">>  QUICK CAPTURE  [Q]")
            self.btn_quick.set_state(True)
        except Exception:
            pass
        # Belt-and-suspenders: if the quick-capture chain bailed out
        # before _on_done fired, the busy overlay would still be up.
        # _on_done() already calls _hide_busy_overlay() on the happy
        # path, so this is a no-op there.
        self._hide_busy_overlay()

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Calibration
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def on_calibrate(self):
        if self._calibrating:
            return
        if self.orig_img is None:
            messagebox.showinfo(
                "No frame",
                "Capture a frame first â€” start LIVE feed or upload an image.")
            return
        if self.model is None:
            messagebox.showwarning(
                "No model",
                "Model not loaded â€” set path and click LOAD.")
            return
        self._calibrating = True
        self.btn_calibrate.set_state(False)
        threading.Thread(target=self._calibration_worker,
                         daemon=True).start()

    def _calibration_worker(self):
        try:
            frame = self.orig_img.copy()
            self._set_status("Calibrating â€” segmenting cookieâ€¦", "yellow")
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
                areas = []
                for mi in r.masks:
                    mc = mi.xy[0].astype(np.int32)
                    if len(mc) < 3:
                        areas.append(0.0)
                        continue
                    areas.append(cv2.contourArea(mc))
                best_i = int(np.argmax(areas))
                mask_inst = r.masks[best_i]
                msg_extra = (f" ({n} detected â€” used the largest)")
            else:
                mask_inst = r.masks[0]
                msg_extra = ""

            mc = mask_inst.xy[0].astype(np.int32)
            if len(mc) < 3:
                self.root.after(0, lambda: self._calibration_failed(
                    "Detected mask is degenerate."))
                return

            _, feret_px = max_feret_diameter(mc)
            if feret_px <= 0:
                self.root.after(0, lambda: self._calibration_failed(
                    "Could not compute Feret diameter."))
                return

            try:
                r.names = dict(CLASS_NAMES)
            except Exception as e:
                print(f"[calib] could not override r.names: {e}")
            self.yolo_plotted = r.plot(boxes=True, labels=True, conf=True)
            self.root.after(0,
                lambda: self._prompt_real_diameter(feret_px, msg_extra))
        except Exception as e:
            self.root.after(0, lambda: self._calibration_failed(str(e)))
        finally:
            self._calibrating = False
            self.root.after(0, lambda: self.btn_calibrate.set_state(True))

    def _prompt_real_diameter(self, feret_px, msg_extra=""):
        self._refresh_display()
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
        self._cal_value_lbl.config(text=f"{new_ratio:.3f}")
        self._cal_when_lbl.config(text=self._format_calibration_when())
        self._set_status(
            f"Calibration saved â€” {new_ratio:.3f} px/mm "
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

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Save / Export
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Busy overlay â€” centered "running" indicator shown during
    # any inspection trigger (R hotkey / Q hotkey / Pi GPIO button)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _show_busy_overlay(self, message="Running inspection..."):
        """Display (or update) the centered busy modal.

        Safe to call repeatedly: the second+ call just updates the
        status line without rebuilding the window. Must be called on
        the Tk main thread.
        """
        if self._busy_overlay is not None:
            try:
                if self._busy_msg_var is not None:
                    self._busy_msg_var.set(message)
            except Exception:
                pass
            return

        bw, bh = 380, 150

        win = tk.Toplevel(self.root)
        # Hide while we position so the user never sees it briefly
        # flash at (0, 0) before the geometry call lands.
        win.withdraw()
        win.overrideredirect(True)               # no title bar
        win.configure(bg=C["accent"])            # 2 px accent border
        try:
            win.transient(self.root)
        except Exception:
            pass

        # Centre on the application window. Some window managers
        # ignore the geometry's x/y when overrideredirect is set, so
        # we re-apply it once after the window is mapped (below).
        # Fall back to screen-centre if the root hasn't been mapped
        # yet (e.g. very-early call before mainloop is fully alive).
        self.root.update_idletasks()
        rw = self.root.winfo_width()
        rh = self.root.winfo_height()
        if rw > 100 and rh > 100:
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            x  = rx + (rw - bw) // 2
            y  = ry + (rh - bh) // 2
        else:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            x  = (sw - bw) // 2
            y  = (sh - bh) // 2

        geom = f"{bw}x{bh}+{x}+{y}"
        win.geometry(geom)

        inner = tk.Frame(win, bg=C["bg2"])
        inner.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        tk.Label(inner, text="INSPECTION  RUNNING",
                 bg=C["bg2"], fg=C["accent"],
                 font=("Courier New", 13, "bold")).pack(pady=(22, 6))

        self._busy_msg_var = tk.StringVar(value=message)
        tk.Label(inner, textvariable=self._busy_msg_var,
                 bg=C["bg2"], fg=C["text2"],
                 font=("Courier New", 9)).pack()

        self._busy_dots_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self._busy_dots_var,
                 bg=C["bg2"], fg=C["accent"],
                 font=("Courier New", 18, "bold"),
                 width=6, anchor="center").pack(pady=(8, 0))

        self._busy_overlay   = win
        self._busy_dots_step = 0

        # Map the window, force layout, then re-apply the geometry.
        # Many WMs (incl. some on Pi OS) only honour the x/y portion
        # of geometry() after the window has been mapped at least
        # once, so a single pre-map call can land at (0,0).
        win.deiconify()
        try:
            win.update_idletasks()
            win.geometry(geom)
            win.lift()
            win.attributes("-topmost", True)
        except Exception:
            pass
        self._tick_busy_dots()

    def _tick_busy_dots(self):
        """Animate the dot indicator under the busy overlay."""
        if self._busy_overlay is None or self._busy_dots_var is None:
            return
        self._busy_dots_step = (self._busy_dots_step + 1) % 4
        try:
            self._busy_dots_var.set("." * self._busy_dots_step)
        except Exception:
            return
        self._busy_dots_after = self.root.after(400, self._tick_busy_dots)

    def _hide_busy_overlay(self):
        """Tear down the busy overlay (no-op if it's not showing)."""
        if self._busy_dots_after is not None:
            try:
                self.root.after_cancel(self._busy_dots_after)
            except Exception:
                pass
            self._busy_dots_after = None
        if self._busy_overlay is not None:
            try:
                self._busy_overlay.destroy()
            except Exception:
                pass
        self._busy_overlay  = None
        self._busy_msg_var  = None
        self._busy_dots_var = None

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Status helper
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _set_status(self, msg, level="green"):
        color = {"green": C["green"],
                 "yellow": C["yellow"],
                 "red": C["red"]}.get(level, C["text2"])
        self.root.after(0, lambda: (
            self._status_var.set(msg),
            self._status_dot.config(fg=color)
        ))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Entry point
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def main():
    root = tk.Tk()
    app = CookieInspectorApp(root)

    def _on_close():
        try:
            if app._live_mode:
                app._stop_live()
        except Exception:
            pass
        try:
            app._gpio_trigger.stop()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
