"""Smart Document Scanner - classical computer-vision document scanner.

The package is split into one module per pipeline stage so each step can be
explained (and tested) on its own:

    preprocess.py  -> resize, grayscale, blur, edges, Otsu mask
    detect.py      -> find the page's 4 corners (with a fallback chain)
    transform.py   -> order the corners and flatten the page (perspective warp)
    enhance.py     -> adaptive thresholding for the black-and-white "scan" look
    pipeline.py    -> scan_document() runs all of the above in order
    metrics.py     -> polygon IoU used by the evaluation and the tests
    synthetic.py   -> generates fake "phone photos" for tests and demos
"""

from .pipeline import ScanResult, load_image, scan_document, draw_outline
from .transform import order_points, four_point_transform
from .metrics import polygon_iou

__all__ = [
    "ScanResult",
    "load_image",
    "scan_document",
    "draw_outline",
    "order_points",
    "four_point_transform",
    "polygon_iou",
]
