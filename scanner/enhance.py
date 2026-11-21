"""Stage 8 of the pipeline: turn the flattened colour page into a clean black-and-white scan."""

import cv2

from .preprocess import to_grayscale

# Default adaptive-threshold settings (tuned for full-resolution phone photos).
DEFAULT_BLOCK_SIZE = 25
DEFAULT_C = 10


def make_odd(value, minimum=3):
    """Return `value` as an odd integer >= `minimum`.

    adaptiveThreshold requires an odd blockSize (so the window has a centre
    pixel). Sliders and command-line flags can produce even numbers, so we
    round them up instead of crashing.
    """
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


def adaptive_threshold(warped, block_size=DEFAULT_BLOCK_SIZE, c=DEFAULT_C):
    """Convert the flattened page to pure black text on a white background.

    Why ADAPTIVE thresholding: a single global threshold fails on phone photos
    because lighting is uneven - one side of the page may be in shadow and
    the other bright. Adaptive thresholding computes a separate threshold for
    every pixel from its neighbourhood, so a shadowed area gets a lower
    threshold than a bright area.

    For each pixel:
        threshold = (Gaussian-weighted mean of the block_size x block_size
                     neighbourhood) - C
        pixel becomes white if it is brighter than threshold, otherwise black.

    Args:
        warped:     flattened page (colour or grayscale).
        block_size: neighbourhood size in pixels (odd). It must be larger than
                    the thickness of a pen/text stroke, otherwise the inside
                    of thick letters is compared only to other dark pixels and
                    turns white (hollow letters). 25 px suits text in 12 MP
                    photos; increase it for very high-resolution images.
        c:          constant subtracted from the local mean. Paper is never
                    perfectly uniform; subtracting C=10 means a pixel must be
                    clearly darker than its surroundings to count as ink, which
                    removes paper texture and faint noise. Larger C = cleaner
                    background but thinner/fainter text.

    Returns:
        Single-channel image with values 0 (ink) and 255 (paper).
    """
    gray = to_grayscale(warped)
    block_size = make_odd(block_size)
    return cv2.adaptiveThreshold(
        gray,
        255,                              # value given to "white" pixels
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # Gaussian-weighted mean (smoother than plain mean)
        cv2.THRESH_BINARY,                # bright -> 255, dark -> 0
        block_size,
        c,
    )
