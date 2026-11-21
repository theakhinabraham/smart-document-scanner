"""Generate synthetic demo photos in data/samples/ (kept separate from data/raw/).

    python make_samples.py

Creates:
    data/samples/sample_XX.jpg        synthetic "phone photos"
    data/samples/ground_truth.csv     exact page corners (known, because we drew them)
    data/samples/conditions.csv       the condition of each sample
    data/samples/ocr_truth/*.txt      the exact text printed on each page

This lets you run scan.py, evaluate.py, ocr.py and app.py before taking real
photos. The samples deliberately include hard cases (glare, low contrast,
clutter, curl, an inner frame) so the failure analysis has something to show.
Synthetic results are NOT a substitute for the real-photo evaluation.
"""

import csv
from pathlib import Path

import cv2

from scanner.synthetic import make_scene

OUT = Path("data/samples")

#  name         seed  angle       lighting  background      page      inner frame
SAMPLES = [
    ("sample_01", 11, "straight", "good",   "plain",        "flat",   False),
    ("sample_02", 12, "tilted",   "good",   "plain",        "flat",   False),
    ("sample_03", 13, "steep",    "good",   "plain",        "flat",   False),
    ("sample_04", 14, "tilted",   "dim",    "plain",        "flat",   False),
    ("sample_05", 15, "tilted",   "shadow", "plain",        "flat",   False),
    ("sample_06", 16, "tilted",   "glare",  "plain",        "flat",   False),
    ("sample_07", 17, "tilted",   "good",   "cluttered",    "flat",   False),
    ("sample_08", 18, "tilted",   "good",   "low-contrast", "flat",   False),
    ("sample_09", 19, "tilted",   "good",   "plain",        "curled", False),
    ("sample_10", 20, "tilted",   "good",   "plain",        "folded", False),
    ("sample_11", 21, "steep",    "shadow", "cluttered",    "flat",   False),
    ("sample_12", 22, "straight", "good",   "plain",        "flat",   True),
]


def main():
    (OUT / "ocr_truth").mkdir(parents=True, exist_ok=True)
    with open(OUT / "ground_truth.csv", "w", newline="") as gt_file, \
            open(OUT / "conditions.csv", "w", newline="") as cond_file:
        gt = csv.writer(gt_file)
        cond = csv.writer(cond_file)
        gt.writerow(["filename", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])
        cond.writerow(["filename", "angle", "lighting", "background", "page"])
        for name, seed, angle, lighting, background, page, inner in SAMPLES:
            image, corners, text = make_scene(seed, angle, lighting, background, page, inner)
            filename = f"{name}.jpg"
            cv2.imwrite(str(OUT / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            gt.writerow([filename] + [f"{v:.1f}" for v in corners.ravel()])
            cond.writerow([filename, angle, lighting, background, page])
            (OUT / "ocr_truth" / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
            print(f"wrote {OUT / filename}  ({angle}, {lighting}, {background}, {page})")


if __name__ == "__main__":
    main()
