"""Stage 1-3 of the pipeline: resize, grayscale, blur and edge detection.

All of these steps work on a SMALL copy of the photo (500 px tall). Phone
photos are 3000-4000 px tall; detecting the page on a small copy is much
faster and also less sensitive to fine texture (paper grain, wood grain),
because that texture mostly disappears when the image is shrunk.
"""

import cv2
import numpy as np

# Height (in pixels) of the copy used for page detection. 500 px keeps enough
# detail to locate the page outline but is ~40x fewer pixels than a 12 MP photo.
DETECTION_HEIGHT = 500


def resize_for_detection(image, height=DETECTION_HEIGHT):
    """Shrink the image to a fixed height, keeping its aspect ratio.

    Why: detection is faster and more robust on a small image. We return the
    scale `ratio` so that corners found on the small image can be multiplied
    back up to the original full-resolution image later (step 7 warps the
    ORIGINAL photo, so the final scan keeps full quality).

    Args:
        image:  BGR image (numpy array, H x W x 3).
        height: target height in pixels for the detection copy.

    Returns:
        (small_image, ratio) where ratio = original_height / height.
        Multiply small-image coordinates by `ratio` to get original coordinates.
    """
    original_height = image.shape[0]
    ratio = original_height / float(height)
    new_width = max(1, int(round(image.shape[1] / ratio)))
    # INTER_AREA is the recommended interpolation for shrinking: it averages
    # the pixels that fall into each output pixel, which avoids aliasing.
    small = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_AREA)
    return small, ratio


def to_grayscale(image):
    """Convert a BGR colour image to a single-channel grayscale image.

    Why: edges and thresholds only need brightness, not colour. Working on one
    channel instead of three is simpler and three times less data.
    """
    if image.ndim == 2:  # already grayscale
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur(gray, ksize=5):
    """Smooth the image with a Gaussian blur.

    Why: camera noise and paper texture create thousands of tiny "edges".
    Blurring removes that high-frequency noise so that Canny only responds to
    strong, large-scale edges such as the border between page and table.

    Args:
        gray:  grayscale image.
        ksize: kernel size (must be odd). 5x5 removes pixel-level noise on a
               500 px image without smearing the page border too much; 3x3 is
               too weak for noisy phone photos and 7x7+ starts to round corners.
    """
    # sigmaX=0 lets OpenCV compute sigma from the kernel size (about 1.1 for 5x5).
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def detect_edges(blurred, low=75, high=200):
    """Find edges with the Canny edge detector.

    How Canny works (short version): it computes the brightness gradient,
    thins edges to 1 px, and then uses two thresholds (hysteresis):
      * pixels with gradient > `high` are definitely edges ("strong"),
      * pixels with gradient < `low` are definitely not edges,
      * pixels in between are kept only if they touch a strong edge.

    Why 75 / 200: a white page on a darker surface gives a gradient well above
    200 along its border, so the border is always "strong". Weak texture
    (< 75) is ignored. The 1:2.7 ratio is close to the 1:2-1:3 ratio that
    Canny recommended in the original paper. These are the values used in the
    PyImageSearch document-scanner tutorial [see REFERENCES.md].

    Args:
        blurred: blurred grayscale image.
        low:     lower hysteresis threshold.
        high:    upper hysteresis threshold.
    """
    return cv2.Canny(blurred, low, high)


def close_edges(edges, close_ksize=5):
    """Join broken edge segments with a small morphological closing + dilation.

    Why: the page border is often broken into pieces (a shadow, a bit of glare
    or a low-contrast patch interrupts it). findContours can only produce a
    closed page outline if the border is one connected line.
      * Closing (dilate then erode) with a 5x5 kernel bridges gaps up to
        ~5 px without making the line thicker overall.
      * One extra 3x3 dilation thickens 1-px edges so that diagonal steps in
        the line stay connected for the contour tracer.

    Args:
        edges:       binary Canny edge map.
        close_ksize: size of the square closing kernel (gap size to bridge).
    """
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(closed, dilate_kernel, iterations=1)


def otsu_mask(blurred, close_ksize=9):
    """Separate the bright page from the background with Otsu thresholding.

    Used as the FIRST FALLBACK when Canny cannot find a clean outline (for
    example when the border is soft or partly in shadow).

    How it works: Otsu's method looks at the brightness histogram and picks
    the single threshold that best splits the pixels into two groups (it
    minimises the variance inside each group). For a document photo the two
    groups are usually "page" and "background". Unlike Canny it looks at
    whole regions, not at sharp brightness changes.

    A morphological closing (9x9) then fills the dark text holes and small
    notches along the page border so the page becomes one solid white blob.
    9x9 is larger than the edge-closing kernel because text lines on a 500 px
    image are ~5-8 px tall and must be filled completely.

    Args:
        blurred:     blurred grayscale image.
        close_ksize: closing kernel size.

    Returns:
        Binary mask (0 or 255), page usually white.
    """
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def has_enough_contrast(gray, min_std=8.0):
    """Return True if the image has enough brightness variation to contain a page.

    Why: Otsu ALWAYS returns a threshold, even for a completely blank image,
    and then splits pure noise into random blobs. If the standard deviation
    of brightness is tiny (< 8 grey levels out of 255), there is no visible
    object to find, so we skip the threshold-based fallbacks.
    """
    return float(np.std(gray)) >= min_std
