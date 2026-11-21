"""Stage 6-7 of the pipeline: order the corners and flatten the page.

The four-point perspective transform follows the well-known pattern from the
PyImageSearch "4 Point OpenCV getPerspective Transform" tutorial and the
OpenCV geometric-transformations documentation (see REFERENCES.md [1], [3]).
The corner-ordering function is a variation: it sorts by angle around the
centre instead of by x+y / y-x sums, which avoids ties when the page is
rotated by about 45 degrees.
"""

import cv2
import numpy as np


def order_points(points):
    """Return the 4 corners in a fixed order: top-left, top-right, bottom-right, bottom-left.

    Why: contour detection returns the corners in an arbitrary order (it
    depends on where the contour tracer started). The perspective transform
    needs source and destination corners to correspond one-to-one, so we
    must always put them in the same order.

    Method:
      1. Compute the centre (mean) of the 4 points.
      2. Sort the points by their angle around the centre. In image
         coordinates (y grows downwards) increasing angle = clockwise, so the
         order becomes TL -> TR -> BR -> BL, starting somewhere in that cycle.
      3. Rotate the list so that it starts at the point with the smallest
         x + y, i.e. the one nearest the image's top-left corner.

    Args:
        points: array-like of shape (4, 2) with (x, y) coordinates.

    Returns:
        float32 array of shape (4, 2) in TL, TR, BR, BL order.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    clockwise = pts[np.argsort(angles)]
    start = int(np.argmin(clockwise.sum(axis=1)))  # smallest x + y = top-left
    return np.roll(clockwise, -start, axis=0)


def four_point_transform(image, corners):
    """Warp the quadrilateral `corners` of `image` into a flat, upright rectangle.

    Why: a photo taken at an angle shows the page as a trapezoid (perspective
    distortion). A perspective transform (homography) is a 3x3 matrix that
    maps any 4 points onto any other 4 points; mapping the page's corners onto
    the corners of a rectangle "un-tilts" the page as if scanned from directly
    above.

    Output size: the width is the LONGER of the top and bottom edges, the
    height is the LONGER of the left and right edges (measured in pixels with
    the Euclidean distance). Using the longer edge keeps the most detail.

    Args:
        image:   the image to warp (we pass the ORIGINAL full-resolution photo).
        corners: (4, 2) corners in that image's coordinates, any order.

    Returns:
        The flattened page image.
    """
    tl, tr, br, bl = order_points(corners)

    width_bottom = np.linalg.norm(br - bl)
    width_top = np.linalg.norm(tr - tl)
    max_width = max(int(round(width_bottom)), int(round(width_top)), 1)

    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    max_height = max(int(round(height_right)), int(round(height_left)), 1)

    source = np.array([tl, tr, br, bl], dtype=np.float32)
    destination = np.array(
        [
            [0, 0],                          # top-left
            [max_width - 1, 0],              # top-right
            [max_width - 1, max_height - 1], # bottom-right
            [0, max_height - 1],             # bottom-left
        ],
        dtype=np.float32,
    )

    # getPerspectiveTransform solves for the 3x3 homography from 4 point pairs.
    matrix = cv2.getPerspectiveTransform(source, destination)
    # warpPerspective applies it to every pixel (bilinear interpolation by default).
    return cv2.warpPerspective(image, matrix, (max_width, max_height))
