"""scan_document(): runs the whole pipeline on one photo and returns a ScanResult.

    original photo
      -> resize copy to 500 px tall         (preprocess.resize_for_detection)
      -> grayscale -> Gaussian blur          (preprocess)
      -> find 4 page corners                 (detect: canny / otsu / minarearect)
      -> scale corners back to full size     (multiply by ratio)
      -> order corners + perspective warp    (transform) on the ORIGINAL photo
      -> adaptive threshold                  (enhance)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detect import MIN_AREA_RATIO, detect_page
from .enhance import DEFAULT_BLOCK_SIZE, DEFAULT_C, adaptive_threshold
from .preprocess import DETECTION_HEIGHT, blur, resize_for_detection, to_grayscale
from .transform import four_point_transform, order_points

# Names (and order) of the intermediate images saved with --save-stages.
STAGE_NAMES = ["1_gray", "2_blurred", "3_edges", "4_contours", "5_warped", "6_thresholded"]

# File types OpenCV can read. iPhone HEIC photos are NOT supported: export them
# as JPEG first (or set the camera to "Most Compatible").
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder):
    """Return the image files in `folder`, sorted by name (non-recursive)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."))


def save_stages(stages, folder, stem):
    """Write every intermediate image to `folder` as <stem>_<stage>.png. Returns the paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in STAGE_NAMES:
        if name in stages:
            path = folder / f"{stem}_{name}.png"
            cv2.imwrite(str(path), stages[name])
            paths.append(path)
    return paths


@dataclass
class ScanResult:
    """Everything produced by one run of the scanner.

    Attributes:
        success:  True if a page was found and flattened.
        method:   "canny", "otsu", "minarearect" or "failed".
        corners:  (4, 2) float32 page corners in ORIGINAL image coordinates,
                  ordered TL, TR, BR, BL; None if no page was found.
        warped:   flattened colour page (the "colour scan"), or None.
        scanned:  black-and-white thresholded page, or None (also None when
                  thresholding is switched off).
        stages:   intermediate images by name (only if keep_stages=True).
        time_ms:  processing time in milliseconds.
        message:  human-readable status, e.g. "page not detected".
    """

    success: bool
    method: str
    corners: Optional[np.ndarray] = None
    warped: Optional[np.ndarray] = None
    scanned: Optional[np.ndarray] = None
    stages: dict = field(default_factory=dict)
    time_ms: float = 0.0
    message: str = ""


def load_image(path):
    """Read an image from disk as a BGR array, or return None if it cannot be read.

    cv2.imread automatically applies the EXIF orientation tag that phones
    write, so a portrait photo is loaded upright. annotate.py uses the same
    function, so ground-truth clicks and detections share one coordinate system.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image  # None if the file is missing or not an image (e.g. HEIC)


def _as_bgr(image):
    """Make sure the input is a 3-channel BGR uint8 image (accepts gray or BGRA)."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def draw_outline(image, corners, color=(0, 255, 0), thickness=None):
    """Return a copy of `image` with the quadrilateral `corners` drawn on it.

    The line thickness scales with the image size so it is visible both on a
    500 px preview and on a 4000 px photo.
    """
    output = _as_bgr(image).copy()
    if corners is None:
        return output
    if thickness is None:
        thickness = max(2, int(round(max(output.shape[:2]) / 250)))
    pts = np.round(np.asarray(corners)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(output, [pts], True, color, thickness, cv2.LINE_AA)
    for x, y in pts.reshape(-1, 2):
        cv2.circle(output, (int(x), int(y)), thickness * 2, color, -1, cv2.LINE_AA)
    return output


def scan_document(image, canny_low=75, canny_high=200, block_size=DEFAULT_BLOCK_SIZE,
                  c=DEFAULT_C, apply_threshold=True, keep_stages=False,
                  min_area_ratio=MIN_AREA_RATIO):
    """Scan one photo: detect the page, flatten it and make it black-and-white.

    This function never raises for bad input; if anything goes wrong it
    returns a ScanResult with success=False, method="failed" and a message.

    Args:
        image:           BGR photo (numpy array) as loaded by load_image().
        canny_low:       Canny lower threshold (default 75).
        canny_high:      Canny upper threshold (default 200).
        block_size:      adaptive-threshold neighbourhood size (odd, default 25).
        c:               adaptive-threshold constant C (default 10).
        apply_threshold: False skips the black-and-white step (colour scan only).
        keep_stages:     True stores every intermediate image in result.stages.
        min_area_ratio:  minimum fraction of the image the page must cover.

    Returns:
        ScanResult.
    """
    start = time.perf_counter()
    stages = {}

    def finish(result):
        result.time_ms = (time.perf_counter() - start) * 1000.0
        result.stages = stages if keep_stages else {}
        return result

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return finish(ScanResult(False, "failed", message="no image / could not read image"))

    try:
        original = _as_bgr(image)

        # Step 1: small copy for detection; `ratio` maps small -> original coordinates.
        small, ratio = resize_for_detection(original, DETECTION_HEIGHT)

        # Step 2: grayscale + 5x5 Gaussian blur.
        gray = to_grayscale(small)
        blurred = blur(gray, 5)

        # Steps 3-5: edges, contours and the fallback chain.
        detection = detect_page(blurred, canny_low, canny_high, min_area_ratio)

        if keep_stages:
            stages["1_gray"] = gray
            stages["2_blurred"] = blurred
            # Show the binary image that the successful method actually used.
            stages["3_edges"] = detection.mask if detection.method == "otsu" else detection.edges
            overlay = small.copy()
            cv2.drawContours(overlay, detection.candidates, -1, (0, 200, 255), 1)
            stages["4_contours"] = draw_outline(overlay, detection.corners, (0, 255, 0), 2)

        if detection.corners is None:
            return finish(ScanResult(False, "failed",
                                     message="page not detected (no 4-corner outline found)"))

        # Step 6: order corners and scale them to the original resolution.
        corners = order_points(detection.corners * ratio)

        # Step 7: perspective warp on the ORIGINAL full-resolution photo.
        warped = four_point_transform(original, corners)

        # Step 8: adaptive threshold for the black-and-white scan.
        scanned = adaptive_threshold(warped, block_size, c) if apply_threshold else None

        if keep_stages:
            stages["5_warped"] = warped
            if scanned is not None:
                stages["6_thresholded"] = scanned

        return finish(ScanResult(True, detection.method, corners, warped, scanned,
                                 message=f"page detected with {detection.method}"))
    except (cv2.error, ValueError) as error:  # e.g. degenerate corners -> warp fails
        return finish(ScanResult(False, "failed", message=f"processing error: {error}"))
