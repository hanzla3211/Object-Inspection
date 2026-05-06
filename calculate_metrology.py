"""
Cookie metrology with two diameter measurements:
  1. Max Feret diameter   - longest distance between any two boundary points
                            (rotating calipers on convex hull). This is the
                            true "longest dimension" of the cookie.
  2. Through-center diameter - longest chord passing through the centroid.
                               Useful for symmetry/roundness checks.
The gap between them is itself a roundness indicator.
Controls:
  m  toggle mask overlay
  g  toggle angle grid
  d  toggle diameter lines
  f  toggle Feret diameter
  c  toggle through-center diameter
  q / Esc  quit
"""
import cv2
import numpy as np
from ultralytics import YOLO
import math

# --- Calibration ---
PIXEL_TO_MM = 6.31956

# --- UI Toggles ---
show_mask = True
show_grid = True
show_diameter = True
show_feret = True
show_through_center = True

# --- Tuning ---
NUM_ANGLES = 180                  # 1-degree resolution for through-center sweep
LOPSIDED_THRESHOLD = 0.15         # flag cookie if centroid drifts >15% of enclosing radius


def ray_contour_intersections(contour, center, angle_rad):
    cx, cy = center
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    intersections = []
    n = len(contour)
    for i in range(n):
        p1 = contour[i]
        p2 = contour[(i + 1) % n]
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        ex = x2 - x1
        ey = y2 - y1
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-9:
            continue
        u = (dx * (y1 - cy) - dy * (x1 - cx)) / denom
        if 0.0 <= u <= 1.0:
            ix = x1 + u * ex
            iy = y1 + u * ey
            intersections.append((ix, iy))
    return intersections


def longest_through_center_chord(contour, center, num_angles):
    best_len = 0.0
    best_chord = None
    chords = []
    for angle in np.linspace(0.0, math.pi, num_angles, endpoint=False):
        pts = ray_contour_intersections(contour, center, angle)
        if len(pts) < 2:
            continue
        dx, dy = math.cos(angle), math.sin(angle)
        projected = [(p[0] * dx + p[1] * dy, p) for p in pts]
        projected.sort(key=lambda x: x[0])
        p_min = projected[0][1]
        p_max = projected[-1][1]
        chords.append((p_min, p_max))
        length = math.hypot(p_max[0] - p_min[0], p_max[1] - p_min[1])
        if length > best_len:
            best_len = length
            best_chord = (p_min, p_max)
    return best_chord, best_len, chords


def max_feret_diameter(contour):
    hull = cv2.convexHull(contour.reshape(-1, 1, 2)).reshape(-1, 2)
    n = len(hull)
    if n < 2:
        return None, 0.0
    best_len = 0.0
    best_pair = None
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(
                float(hull[i][0]) - float(hull[j][0]),
                float(hull[i][1]) - float(hull[j][1]),
            )
            if d > best_len:
                best_len = d
                best_pair = (tuple(hull[i].astype(int)),
                             tuple(hull[j].astype(int)))
    return best_pair, best_len


def lopsidedness_score(centroid, contour):
    (mx, my), r = cv2.minEnclosingCircle(contour.reshape(-1, 1, 2))
    drift = math.hypot(centroid[0] - mx, centroid[1] - my)
    ratio = drift / r if r > 1e-6 else 0.0
    return drift, ratio, (int(mx), int(my)), int(r)


def analyze_cookie(mask_coords, raw_mask):
    M = cv2.moments(mask_coords)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    tc_chord, tc_len_px, all_chords = longest_through_center_chord(
        mask_coords, (cx, cy), NUM_ANGLES
    )
    feret_pair, feret_len_px = max_feret_diameter(mask_coords)
    drift_px, drift_ratio, mec_center, mec_radius = lopsidedness_score(
        (cx, cy), mask_coords
    )

    return {
        'mask': raw_mask,
        'center': (cx, cy),
        'mec_center': mec_center,
        'mec_radius': mec_radius,
        'drift_ratio': drift_ratio,
        'lopsided': drift_ratio > LOPSIDED_THRESHOLD,
        'tc_chord': tc_chord,
        'tc_px': tc_len_px if tc_chord else 0.0,
        'tc_mm': tc_len_px / PIXEL_TO_MM if tc_chord else 0.0,
        'feret_pair': feret_pair,
        'feret_px': feret_len_px if feret_pair else 0.0,
        'feret_mm': feret_len_px / PIXEL_TO_MM if feret_pair else 0.0,
        'grid': all_chords,
    }


def to_int_pt(p):
    return (int(round(p[0])), int(round(p[1])))


def render(img, cookies):
    display = img.copy()
    for d in cookies:
        # Mask overlay
        if show_mask and d['mask'] is not None:
            overlay = display.copy()
            overlay[d['mask'] > 0.5] = (0, 255, 0)
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

        # Angle grid
        if show_grid:
            for c in d['grid']:
                cv2.line(display, to_int_pt(c[0]), to_int_pt(c[1]),
                         (80, 80, 80), 1)

        # Through-center diameter (magenta)
        if show_diameter and show_through_center and d['tc_chord']:
            p1, p2 = d['tc_chord']
            cv2.line(display, to_int_pt(p1), to_int_pt(p2), (255, 0, 255), 2)

        # Max Feret diameter (cyan)
        if show_diameter and show_feret and d['feret_pair']:
            cv2.line(display, d['feret_pair'][0], d['feret_pair'][1],
                     (255, 255, 0), 2)

        # Centroid + min-enclosing-circle center
        cv2.circle(display, d['center'], 5, (0, 0, 255), -1)
        if d['lopsided']:
            cv2.circle(display, d['mec_center'], 5, (0, 165, 255), -1)
            cv2.line(display, d['center'], d['mec_center'],
                     (0, 165, 255), 1)

        # Labels
        cx, cy = d['center']
        y = cy - 10
        if show_feret:
            cv2.putText(display,
                        f"Feret: {d['feret_px']:.1f}px | {d['feret_mm']:.2f}mm",
                        (cx - 80, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)
            y -= 22
        if show_through_center:
            cv2.putText(display,
                        f"Center: {d['tc_px']:.1f}px | {d['tc_mm']:.2f}mm",
                        (cx - 80, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 255), 2)
            y -= 22
        if d['lopsided']:
            cv2.putText(display, f"LOPSIDED ({d['drift_ratio']*100:.1f}%)",
                        (cx - 80, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 165, 255), 2)

    return display


def main():
    model = YOLO('best.pt')
    img_path = ('snapshot5.jpg')
    results = model.predict(img_path, conf=0.5, retina_masks=True)

    global show_mask, show_grid, show_diameter, show_feret, show_through_center

    for r in results:
        img = r.orig_img.copy()
        if r.masks is None:
            continue

        cookies = []
        for mask_instance in r.masks:
            mask_coords = mask_instance.xy[0].astype(np.int32)
            raw_mask = mask_instance.data[0].cpu().numpy()
            raw_mask = cv2.resize(raw_mask, (img.shape[1], img.shape[0]))
            data = analyze_cookie(mask_coords, raw_mask)
            if data:
                cookies.append(data)

        # Console summary
        print(f"\nDetected {len(cookies)} cookie(s):")
        for i, d in enumerate(cookies):
            flag = "  <-- LOPSIDED" if d['lopsided'] else ""
            print(f"  [{i}] Feret: {d['feret_px']:.1f}px ({d['feret_mm']:.2f}mm)   "
                  f"Through-center: {d['tc_px']:.1f}px ({d['tc_mm']:.2f}mm)   "
                  f"Drift: {d['drift_ratio']*100:.1f}%{flag}")

        win = "Cookie Metrology"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1200, 800)

        while True:
            cv2.imshow(win, render(img, cookies))
            key = cv2.waitKey(20) & 0xFF
            if key == ord('m'):   show_mask = not show_mask
            elif key == ord('g'): show_grid = not show_grid
            elif key == ord('d'): show_diameter = not show_diameter
            elif key == ord('f'): show_feret = not show_feret
            elif key == ord('c'): show_through_center = not show_through_center
            elif key in (ord('q'), 27): break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
