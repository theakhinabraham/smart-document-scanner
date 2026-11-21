"""Click tool for recording the TRUE page corners of each photo (ground truth).

Usage:
    python annotate.py                         # annotate photos in data/raw/
    python annotate.py --redo IMG_0042.jpg     # re-annotate one photo
    python annotate.py --conditions-template   # create/update data/conditions.csv

Controls (click inside the image window):
    left click          add a corner (any order - they are sorted automatically)
    u / Backspace       undo the last corner
    r                   reset all corners of this photo
    Enter / Space / n   save the 4 corners and go to the next photo
    s                   skip this photo (it will be offered again next time)
    q / Esc             quit (progress is already saved)

Corners are saved to data/ground_truth.csv as
    filename, x1, y1, x2, y2, x3, y3, x4, y4
in top-left, top-right, bottom-right, bottom-left order and in ORIGINAL image
pixel coordinates (the window may show the photo smaller; clicks are scaled
back). Photos already in the CSV are skipped, so you can stop and resume.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from scanner import load_image, order_points
from scanner.pipeline import list_images

WINDOW = "annotate - click 4 page corners"
GT_HEADER = ["filename", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
CONDITION_HEADER = ["filename", "angle", "lighting", "background", "page"]
MAX_DISPLAY = (1400, 900)  # max window width, height in pixels


def read_annotated(gt_path):
    """Return the set of filenames that already have ground truth."""
    if not gt_path.exists():
        return set()
    with open(gt_path, newline="", encoding="utf-8") as f:
        return {row["filename"] for row in csv.DictReader(f) if row.get("filename")}


def remove_row(gt_path, filename):
    """Delete one photo's row from the CSV (used by --redo)."""
    if not gt_path.exists():
        return
    with open(gt_path, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["filename"] != filename]
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GT_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def append_row(gt_path, filename, corners):
    """Append one photo's ordered corners to the CSV, writing the header if the file is new."""
    is_new = not gt_path.exists() or gt_path.stat().st_size == 0
    with open(gt_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(GT_HEADER)
        writer.writerow([filename] + [f"{v:.1f}" for v in np.asarray(corners).ravel()])


def make_conditions_template(images_dir, conditions_path):
    """Create or update conditions.csv: one row per photo, condition columns left blank.

    Existing rows (and anything you already typed) are kept; only new photos
    are added. Allowed values:
        angle:      straight / tilted / steep
        lighting:   good / dim / shadow / glare
        background: plain / cluttered / low-contrast
        page:       flat / curled / folded
    """
    existing = {}
    if conditions_path.exists():
        with open(conditions_path, newline="", encoding="utf-8") as f:
            existing = {row["filename"]: row for row in csv.DictReader(f) if row.get("filename")}
    names = sorted(set(existing) | {p.name for p in list_images(images_dir)})
    conditions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(conditions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONDITION_HEADER)
        writer.writeheader()
        for name in names:
            row = existing.get(name, {})
            writer.writerow({col: row.get(col, "") if col != "filename" else name
                             for col in CONDITION_HEADER})
    added = len(names) - len(existing)
    print(f"{conditions_path}: {len(names)} photos ({added} new). Fill in the blank columns.")


def render(display, points, scale, info):
    """Draw the clicked corners, the polygon and the help text on a copy of the display image."""
    canvas = display.copy()
    pts = [(int(x * scale), int(y * scale)) for x, y in points]
    if len(pts) == 4:
        ordered = order_points(np.array(pts, np.float32)).astype(np.int32)
        cv2.polylines(canvas, [ordered.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
    elif len(pts) > 1:
        cv2.polylines(canvas, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 1)
    for i, p in enumerate(pts, start=1):
        cv2.circle(canvas, p, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)
    help_text = "click 4 corners | u undo | r reset | Enter save | s skip | q quit"
    for text, y in ((info, 25), (help_text, 52)):
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def annotate_image(image, info):
    """Let the user click 4 corners on one image.

    Returns (action, corners): action is "save", "skip" or "quit"; corners is
    a (4, 2) array in original-image coordinates when action == "save".
    """
    h, w = image.shape[:2]
    scale = min(1.0, MAX_DISPLAY[0] / w, MAX_DISPLAY[1] / h)
    display = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    points = []  # in ORIGINAL image coordinates

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))  # undo the display scaling

    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        cv2.imshow(WINDOW, render(display, points, scale, f"{info}  [{len(points)}/4]"))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("u"), 8, 127) and points:        # 8/127 = Backspace on Win/Linux / macOS
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (13, 10, 32, ord("n")):             # Enter (13 or 10) or Space
            if len(points) == 4:
                return "save", order_points(np.array(points, np.float32))
            print("  need exactly 4 corners before saving")
        elif key == ord("s"):
            return "skip", None
        elif key in (ord("q"), 27):                      # 27 = Esc
            return "quit", None
        # Closing the window with the mouse also quits.
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return "quit", None


def main():
    parser = argparse.ArgumentParser(description="Click the 4 true page corners of each photo.")
    parser.add_argument("--images", default="data/raw", help="folder of photos")
    parser.add_argument("--gt", default="data/ground_truth.csv", help="CSV to write")
    parser.add_argument("--redo", default=None, help="re-annotate this one filename")
    parser.add_argument("--conditions-template", action="store_true",
                        help="create/update the conditions CSV and exit")
    parser.add_argument("--conditions", default="data/conditions.csv", help="conditions CSV path")
    args = parser.parse_args()

    images_dir, gt_path = Path(args.images), Path(args.gt)
    if args.conditions_template:
        make_conditions_template(images_dir, Path(args.conditions))
        return 0

    files = list_images(images_dir)
    if not files:
        print(f"No images in {images_dir}/. Copy your photos there first (JPEG/PNG).")
        return 1
    if args.redo:
        remove_row(gt_path, args.redo)
        files = [p for p in files if p.name == args.redo]
        if not files:
            print(f"{args.redo} not found in {images_dir}/")
            return 1

    done = read_annotated(gt_path)
    todo = [p for p in files if p.name not in done]
    print(f"{len(done)} already annotated, {len(todo)} to go.")
    if not todo:
        return 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    for index, path in enumerate(todo, start=1):
        image = load_image(path)  # same loader as the scanner -> same coordinates
        if image is None:
            print(f"[skip] {path.name}: unreadable (HEIC? export as JPEG)")
            continue
        action, corners = annotate_image(image, f"{index}/{len(todo)}  {path.name}")
        if action == "save":
            append_row(gt_path, path.name, corners)
            print(f"[saved] {path.name}")
        elif action == "skip":
            print(f"[skipped] {path.name}")
        else:
            print("Quit. Run again to continue where you left off.")
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
