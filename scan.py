"""Command-line scanner.

Examples:
    python scan.py path/to/photo.jpg                 # one photo  -> output/
    python scan.py data/raw/ --batch                 # whole folder
    python scan.py photo.jpg --save-stages --ocr     # also save stages + OCR text
    python scan.py photo.jpg --no-threshold          # colour scan only

For each photo it writes:
    output/<name>_scan.png     black-and-white scan (unless --no-threshold)
    output/<name>_color.png    flattened colour scan
    output/stages/<name>_*.png intermediate stages (with --save-stages)
    output/<name>_scan.txt     OCR text (with --ocr and Tesseract installed)
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

from scanner import load_image, scan_document
from scanner.enhance import DEFAULT_BLOCK_SIZE, DEFAULT_C
from scanner.pipeline import list_images, save_stages


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Document Scanner (classical CV).")
    parser.add_argument("input", help="photo file, or a folder when --batch is given")
    parser.add_argument("--batch", action="store_true", help="process every image in the folder")
    parser.add_argument("--output", default="output", help="output folder (default: output/)")
    parser.add_argument("--save-stages", action="store_true", help="save every intermediate stage")
    parser.add_argument("--ocr", action="store_true", help="run Tesseract OCR on the scan")
    parser.add_argument("--no-threshold", action="store_true",
                        help="skip adaptive thresholding (save colour scan only)")
    parser.add_argument("--canny-low", type=int, default=75, help="Canny lower threshold")
    parser.add_argument("--canny-high", type=int, default=200, help="Canny upper threshold")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                        help="adaptive threshold block size (odd)")
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="adaptive threshold constant C")
    return parser.parse_args()


def scan_one(path, args, output_dir, ocr_ready):
    """Scan a single file and write its outputs. Returns the detection method."""
    image = load_image(path)
    if image is None:
        print(f"[skip] {path.name}: could not read image (HEIC? export as JPEG)")
        return "unreadable"

    result = scan_document(
        image,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        block_size=args.block_size,
        c=args.c,
        apply_threshold=not args.no_threshold,
        keep_stages=args.save_stages,
    )

    if args.save_stages and result.stages:
        save_stages(result.stages, output_dir / "stages", path.stem)

    if not result.success:
        print(f"[FAIL] {path.name}: {result.message} ({result.time_ms:.0f} ms)")
        return result.method

    cv2.imwrite(str(output_dir / f"{path.stem}_color.png"), result.warped)
    final_path = output_dir / f"{path.stem}_color.png"
    if result.scanned is not None:
        final_path = output_dir / f"{path.stem}_scan.png"
        cv2.imwrite(str(final_path), result.scanned)
    print(f"[ok]   {path.name}: method={result.method}, {result.time_ms:.0f} ms -> {final_path}")

    if ocr_ready:
        from ocr import extract_text, save_text  # imported lazily: OCR is optional
        source = result.scanned if result.scanned is not None else result.warped
        text = extract_text(source)
        if text is not None:
            text_path = save_text(final_path, text)
            print(f"       OCR text -> {text_path} ({len(text.split())} words)")
    return result.method


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch or input_path.is_dir():
        files = list_images(input_path)
        if not files:
            print(f"No images found in {input_path}")
            return 1
    elif input_path.is_file():
        files = [input_path]
    else:
        print(f"File not found: {input_path}")
        return 1

    ocr_ready = False
    if args.ocr:
        from ocr import tesseract_available, print_install_hint
        ocr_ready = tesseract_available()
        if not ocr_ready:
            print_install_hint()
            print("Continuing without OCR.\n")

    methods = Counter(scan_one(path, args, output_dir, ocr_ready) for path in files)

    if len(files) > 1:
        detected = sum(n for m, n in methods.items() if m not in ("failed", "unreadable"))
        print(f"\nProcessed {len(files)} images: page detected in {detected}.")
        print("Detection methods used: " + ", ".join(f"{m}={n}" for m, n in sorted(methods.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
