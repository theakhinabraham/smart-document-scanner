"""Optional OCR (stretch goal): extract editable text from a scan with Tesseract.

Usage:
    python ocr.py output/photo_scan.png          # print + save text next to the image
    python ocr.py --evaluate                     # CER: raw photo vs scanned output
    python ocr.py --evaluate --images data/samples --truth data/samples/ocr_truth --out results/samples

OCR is optional: if Tesseract (the OCR engine) is not installed, every
function prints an install hint and returns None instead of crashing.

OCR evaluation: for every data/ocr_truth/<photo name>.txt (the true text,
typed by hand) we OCR (a) the raw photo and (b) our scanned output and compute
the Character Error Rate (CER) of each. A lower CER for (b) shows that the
scanner measurably helps OCR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

INSTALL_HINT = """\
Tesseract OCR is not installed (or not on PATH), so OCR is skipped.
Install it with:
  macOS:   brew install tesseract
  Linux:   sudo apt install tesseract-ocr
  Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
           (then add C:\\Program Files\\Tesseract-OCR to PATH)
and make sure the Python wrapper is installed:  pip install pytesseract"""


def tesseract_available():
    """Return True if both the pytesseract package and the Tesseract program are available."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # ImportError or pytesseract.TesseractNotFoundError
        return False


def print_install_hint():
    """Print how to install Tesseract on each operating system."""
    print(INSTALL_HINT)


def extract_text(image):
    """Run Tesseract on an image (numpy array) and return the text, or None if OCR is unavailable.

    We pass our black-and-white scan: Tesseract itself binarises its input,
    and it works best on dark text on a clean white background, which is
    exactly what adaptive thresholding produces.
    """
    if not tesseract_available():
        print_install_hint()
        return None
    import pytesseract
    import cv2

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Tesseract expects RGB order
    # --psm 3 = fully automatic page segmentation (Tesseract's default for whole pages).
    return pytesseract.image_to_string(image, config="--psm 3")


def save_text(image_path, text):
    """Save `text` as a .txt file next to `image_path` (same name). Returns the path."""
    text_path = Path(image_path).with_suffix(".txt")
    text_path.write_text(text, encoding="utf-8")
    return text_path


# --- Character Error Rate ---------------------------------------------------

def normalise(text):
    """Collapse all runs of whitespace into single spaces so line breaks do not count as errors."""
    return " ".join(text.split())


def edit_distance(a, b):
    """Levenshtein distance: the minimum number of single-character insertions,
    deletions and substitutions needed to turn string `a` into string `b`.

    Dynamic programming: previous[j] holds the distance between the first i-1
    characters of `a` and the first j characters of `b`. We only keep two rows
    of the table, so memory is O(len(b)) instead of O(len(a) * len(b)).
    """
    if len(a) < len(b):
        a, b = b, a  # make `b` the shorter string to keep the rows small
    previous = list(range(len(b) + 1))  # distance from "" to b[:j] is j insertions
    for i, char_a in enumerate(a, start=1):
        current = [i]  # distance from a[:i] to "" is i deletions
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(
                previous[j] + 1,          # delete char_a
                current[j - 1] + 1,       # insert char_b
                previous[j - 1] + cost,   # substitute (or match)
            ))
        previous = current
    return previous[-1]


def character_error_rate(reference, hypothesis):
    """CER = edit_distance(reference, hypothesis) / len(reference), after normalising whitespace.

    0.0 means perfect OCR; 1.0 means as many errors as there are characters
    (it can exceed 1.0 when the OCR output contains lots of extra junk).
    """
    reference, hypothesis = normalise(reference), normalise(hypothesis or "")
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def evaluate_ocr(images_dir="data/raw", truth_dir="data/ocr_truth", out_dir="results"):
    """Compare OCR on raw photos vs scanned outputs for every photo that has a truth file.

    Writes <out_dir>/ocr_cer.csv, <out_dir>/ocr_summary.txt and
    <out_dir>/figures/fig6_ocr_cer.png. Returns the results DataFrame (or None).
    """
    import pandas as pd

    from scanner import load_image, scan_document
    from scanner.pipeline import list_images

    truth_dir, out_dir = Path(truth_dir), Path(out_dir)
    if not truth_dir.is_dir() or not any(truth_dir.glob("*.txt")):
        print(f"No OCR ground truth found in {truth_dir}/ (expected <photo name>.txt files).")
        return None
    if not tesseract_available():
        print_install_hint()
        return None

    rows = []
    for path in list_images(images_dir):
        truth_file = truth_dir / f"{path.stem}.txt"
        if not truth_file.exists():
            continue
        truth = truth_file.read_text(encoding="utf-8")
        image = load_image(path)
        if image is None:
            continue
        raw_text = extract_text(image)
        result = scan_document(image)
        scan_text = extract_text(result.scanned) if result.success else None
        row = {
            "filename": path.name,
            "method": result.method,
            "cer_raw": round(character_error_rate(truth, raw_text), 4),
            "cer_scan": round(character_error_rate(truth, scan_text), 4) if result.success else None,
        }
        rows.append(row)
        print(f"{path.name}: CER raw={row['cer_raw']:.3f}  scanned={row['cer_scan']}")

    if not rows:
        print("No photo had a matching OCR truth file.")
        return None

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ocr_cer.csv", index=False)

    scanned = df.dropna(subset=["cer_scan"])
    summary = [
        f"OCR evaluation on {len(df)} photos with ground-truth text",
        f"Mean CER on raw photos:      {df['cer_raw'].mean():.3f}",
        f"Mean CER on scanned outputs: {scanned['cer_scan'].mean():.3f} "
        f"(over {len(scanned)} photos where a page was detected)",
        f"Photos where scanning lowered the CER: "
        f"{int((scanned['cer_scan'] < scanned['cer_raw']).sum())} / {len(scanned)}",
    ]
    (out_dir / "ocr_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary))

    from evaluate import plot_ocr_cer  # shared figure style lives in evaluate.py
    plot_ocr_cer(df, out_dir / "figures" / "fig6_ocr_cer.png")
    return df


def main():
    parser = argparse.ArgumentParser(description="Optional OCR with Tesseract.")
    parser.add_argument("image", nargs="?", help="image to OCR (e.g. output/photo_scan.png)")
    parser.add_argument("--evaluate", action="store_true", help="run the CER evaluation")
    parser.add_argument("--images", default="data/raw", help="photo folder for --evaluate")
    parser.add_argument("--truth", default="data/ocr_truth", help="folder of <photo>.txt truth files")
    parser.add_argument("--out", default="results", help="results folder for --evaluate")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_ocr(args.images, args.truth, args.out)
        return 0
    if not args.image:
        parser.print_help()
        return 1

    import cv2
    image = cv2.imread(args.image)
    if image is None:
        print(f"Could not read {args.image}")
        return 1
    text = extract_text(image)
    if text is None:
        return 0  # hint already printed; not an error
    print(text)
    print(f"Saved to {save_text(args.image, text)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
