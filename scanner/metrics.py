"""Evaluation metric: Intersection over Union (IoU) of two quadrilaterals.

IoU = area(A and B) / area(A or B). It is 1.0 when the detected page outline
matches the ground truth exactly and 0.0 when they do not overlap at all.
We rasterise both polygons into binary masks with cv2.fillPoly and count
pixels; this avoids an extra geometry library (such as shapely).
"""

import cv2
import numpy as np


def polygon_area(polygon):
    """Area in pixels of a polygon given as an (N, 2) array (shoelace formula via OpenCV)."""
    if polygon is None:
        return 0.0
    return float(cv2.contourArea(np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)))


def polygon_iou(poly_a, poly_b):
    """Intersection over Union of two polygons, computed with pixel masks.

    How: both polygons are drawn as filled white shapes on two black masks
    covering just their combined bounding box (so a 12 MP photo does not need
    two 12 MP masks). Intersection = pixels white in both masks, union =
    pixels white in either.

    Args:
        poly_a, poly_b: (N, 2) arrays of (x, y) points, or None.

    Returns:
        IoU in [0, 1]. Returns 0.0 if either polygon is missing.
    """
    if poly_a is None or poly_b is None:
        return 0.0
    a = np.asarray(poly_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(poly_b, dtype=np.float64).reshape(-1, 2)

    both = np.vstack([a, b])
    x0, y0 = np.floor(both.min(axis=0)).astype(int)
    x1, y1 = np.ceil(both.max(axis=0)).astype(int)
    width, height = x1 - x0 + 1, y1 - y0 + 1

    mask_a = np.zeros((height, width), dtype=np.uint8)
    mask_b = np.zeros((height, width), dtype=np.uint8)
    offset = np.array([x0, y0])
    cv2.fillPoly(mask_a, [np.round(a - offset).astype(np.int32)], 1)
    cv2.fillPoly(mask_b, [np.round(b - offset).astype(np.int32)], 1)

    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection) / union if union > 0 else 0.0
