"""Stage 4-5 of the pipeline: find the page's four corners.

Three methods are tried in order (the "fallback chain"). The first one that
succeeds wins, and its name is recorded so the evaluation can report how
often each method was needed:

    1. "canny"       Canny edges -> contours -> 4-point polygon
    2. "otsu"        Otsu threshold mask -> contours -> 4-point polygon
    3. "minarearect" rotated bounding box of the largest shape
    4. "failed"      nothing usable found (no corners returned, never crashes)

All coordinates in this module are in the SMALL detection image (500 px tall).
The pipeline scales them back up to the original photo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .preprocess import close_edges, detect_edges, has_enough_contrast, otsu_mask

# --- Detection settings -------------------------------------------------------

# Only the 5 largest contours are checked. The page is normally the biggest
# object in the photo, so it is almost always in the top few; checking many
# more mostly adds chances to accept a wrong shape (a table, a photo on the page).
TOP_N_CONTOURS = 5

# approxPolyDP tolerance as a fraction of the contour perimeter. 2% means a
# point may move up to 2% of the perimeter while simplifying: big enough to
# turn a slightly wobbly page edge (paper curl, JPEG noise) into one straight
# line, small enough that a real corner is not cut off. 1% often leaves 5-8
# points; 5% can collapse a page into a triangle.
EPSILON_RATIO = 0.02

# The page must cover at least 20% of the image. Phone photos of a document
# are framed around the page, so anything smaller is most likely a text block,
# a sticky note or a shape on the page rather than the page itself.
MIN_AREA_RATIO = 0.20

# Shapes covering more than 98% of the image are the image border itself
# (e.g. Otsu marking the whole photo white), not a page.
MAX_AREA_RATIO = 0.98

# The contour's perimeter may be at most 20% longer than the perimeter of its
# 4-point simplification. Why: a ragged blob (page edge merged with wood
# grain or clutter) has a very long, zig-zag perimeter, so 2% of it is a huge
# tolerance and approxPolyDP can squash the blob into 4 points. A real page
# outline - even a curled one - is only ~2-5% longer than its quadrilateral,
# while merged blobs measured 3-5x longer in our tests.
MAX_PERIMETER_RATIO = 1.2


@dataclass
class Detection:
    """Result of the page search on the small detection image.

    Attributes:
        corners:    (4, 2) float32 corners in detection-image coordinates, or None.
        method:     "canny", "otsu", "minarearect" or "failed".
        edges:      the closed Canny edge map (always computed; used for figures).
        mask:       the Otsu mask, if the Otsu fallback was run (else None).
        candidates: contours that were examined (drawn in the overlay figure).
    """

    corners: Optional[np.ndarray]
    method: str
    edges: np.ndarray
    mask: Optional[np.ndarray] = None
    candidates: list = field(default_factory=list)


def find_quadrilateral(binary, min_area_ratio=MIN_AREA_RATIO, top_n=TOP_N_CONTOURS,
                       epsilon_ratio=EPSILON_RATIO):
    """Search a binary image for a large, convex, 4-cornered shape (the page).

    Steps:
      1. findContours with RETR_LIST returns EVERY outline, including inner
         ones. We need inner outlines because a page edge in an edge map is a
         thin closed line: its inner side is the cleanest page outline when
         background clutter touches the outer side.
      2. Sort by area, largest first, and keep the top `top_n`.
      3. Simplify each with approxPolyDP (Douglas-Peucker algorithm) using
         epsilon = epsilon_ratio * perimeter.
      4. Accept the first polygon that has exactly 4 points, is convex (a
         flat page seen in perspective is always convex), covers between
         min_area_ratio and MAX_AREA_RATIO of the image, and whose original
         contour is not much longer than the 4-sided polygon (so it really
         is quadrilateral-shaped, not a ragged blob - see MAX_PERIMETER_RATIO).

    Args:
        binary:         binary image (edge map or threshold mask).
        min_area_ratio: minimum page area as a fraction of the image area.
        top_n:          how many of the largest contours to check.
        epsilon_ratio:  approxPolyDP tolerance as a fraction of the perimeter.

    Returns:
        (corners, candidates): corners is a (4, 2) float32 array or None;
        candidates is the list of contours that were checked.
    """
    image_area = binary.shape[0] * binary.shape[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n]

    for contour in candidates:
        perimeter = cv2.arcLength(contour, True)  # True = closed shape
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        area_ratio = cv2.contourArea(approx) / image_area
        if not (min_area_ratio <= area_ratio <= MAX_AREA_RATIO):
            continue
        if perimeter > MAX_PERIMETER_RATIO * cv2.arcLength(approx, True):
            continue
        return approx.reshape(4, 2).astype(np.float32), candidates
    return None, candidates


def min_area_rect_corners(binaries, min_area_ratio=MIN_AREA_RATIO):
    """Last-resort detection: rotated bounding box of the largest shape.

    Why: when the page outline is not a clean 4-sided polygon (a curled or
    folded page, a corner covered by a hand, a bitten-off corner),
    approxPolyDP gives 5+ points and the earlier methods reject it. The
    minimum-area rotated rectangle around the largest contour still gives a
    rough page location. It cannot model perspective (it is always a true
    rectangle), so expect a lower IoU - but it is better than nothing.

    Args:
        binaries:       list of binary images to take contours from
                        (the Otsu mask and/or the Canny edge map).
        min_area_ratio: the box must cover at least this fraction of the image.

    Returns:
        (4, 2) float32 corners, or None if no box of acceptable size exists.
    """
    height, width = binaries[0].shape[:2]
    image_area = height * width

    largest = None
    for binary in binaries:
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if largest is None or cv2.contourArea(contour) > cv2.contourArea(largest):
                largest = contour
    if largest is None:
        return None

    rect = cv2.minAreaRect(largest)          # ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)                # its 4 corner points
    # The rotated box can stick out of the image slightly; clip it back inside.
    box[:, 0] = np.clip(box[:, 0], 0, width - 1)
    box[:, 1] = np.clip(box[:, 1], 0, height - 1)

    area_ratio = cv2.contourArea(box.reshape(-1, 1, 2)) / image_area
    if min_area_ratio <= area_ratio <= MAX_AREA_RATIO:
        return box.astype(np.float32)
    return None


def detect_page(blurred, canny_low=75, canny_high=200, min_area_ratio=MIN_AREA_RATIO):
    """Run the full fallback chain on a blurred grayscale detection image.

    Args:
        blurred:        blurred grayscale image (500 px tall).
        canny_low:      Canny lower threshold (see preprocess.detect_edges).
        canny_high:     Canny upper threshold.
        min_area_ratio: minimum fraction of the image the page must cover.

    Returns:
        A Detection with the corners (or None) and the method that worked.
    """
    # Method 1: Canny edges.
    edges = close_edges(detect_edges(blurred, canny_low, canny_high))
    corners, candidates = find_quadrilateral(edges, min_area_ratio)
    if corners is not None:
        return Detection(corners, "canny", edges, None, candidates)

    # If the image is (almost) uniform there is nothing to find: Otsu would
    # just split noise into random blobs, so stop here.
    if not has_enough_contrast(blurred):
        return Detection(None, "failed", edges, None, candidates)

    # Method 2: Otsu threshold + closing.
    mask = otsu_mask(blurred)
    corners, otsu_candidates = find_quadrilateral(mask, min_area_ratio)
    if corners is not None:
        return Detection(corners, "otsu", edges, mask, otsu_candidates)

    # Method 3: minAreaRect on the largest contour (Otsu mask first, then edges).
    corners = min_area_rect_corners([mask, edges], min_area_ratio)
    if corners is not None:
        return Detection(corners, "minarearect", edges, mask, otsu_candidates)

    return Detection(None, "failed", edges, mask, candidates + otsu_candidates)
