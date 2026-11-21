"""Generate synthetic "phone photos" of a document with known corner positions.

Used by the unit tests and by make_samples.py so the whole workflow can be
demonstrated before real photos are taken. Because we create the image
ourselves, the true page corners (ground truth) are known exactly.

A scene is built in four steps:
    1. draw a white page with lines of black text,
    2. draw a background (plain, textured, cluttered or low-contrast),
    3. paste the page onto the background with a random perspective warp,
    4. add lighting effects (dim / shadow / glare) and camera noise.
"""

from __future__ import annotations

import cv2
import numpy as np

WORDS = (
    "the scanner finds page edges using contours and then applies a perspective "
    "transform so that the document looks flat adaptive threshold makes text dark "
    "and paper white computer vision course project report results method image "
    "corner detection quality light shadow table photo camera angle evaluation"
).split()

PAGE_WIDTH, PAGE_HEIGHT = 850, 1100   # roughly A4 proportions (1 : 1.29)
CANVAS_WIDTH, CANVAS_HEIGHT = 1200, 1600


def make_page(rng, inner_box=False):
    """Draw a white page with black text lines. Returns (page_bgr, text)."""
    page = np.full((PAGE_HEIGHT, PAGE_WIDTH, 3), 245, dtype=np.uint8)
    lines = []
    y = 120
    cv2.putText(page, "SCAN TEST", (70, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 4)
    lines.append("SCAN TEST")
    y += 80
    while y < PAGE_HEIGHT - 90:
        words = list(rng.choice(WORDS, size=int(rng.integers(4, 7))))
        line = " ".join(words)
        cv2.putText(page, line, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2)
        lines.append(line)
        y += 55
    if inner_box:
        # A thick printed frame inside the page: a typical "inner contour" trap.
        cv2.rectangle(page, (40, 40), (PAGE_WIDTH - 40, PAGE_HEIGHT - 40), (20, 20, 20), 8)
    return page, "\n".join(lines)


def make_background(rng, kind="plain", width=CANVAS_WIDTH, height=CANVAS_HEIGHT):
    """Create a background image.

    kind: "plain" (dark table), "textured" (noisy wood-like texture),
          "cluttered" (table with random objects), "low-contrast" (light grey,
          like a white page on a white table).
    """
    if kind == "low-contrast":
        base = rng.integers(200, 216)
        bg = np.full((height, width, 3), base, dtype=np.float32)
        bg += rng.normal(0, 3, (height, width, 1))
        return np.clip(bg, 0, 255).astype(np.uint8)

    colour = np.array([rng.integers(40, 90), rng.integers(60, 110), rng.integers(80, 140)])
    bg = np.ones((height, width, 3), dtype=np.float32) * colour
    # Gentle left-to-right brightness gradient, like light from a window.
    bg *= np.linspace(0.85, 1.1, width)[None, :, None]

    if kind == "textured":
        noise = rng.normal(0, 40, (height // 4, width // 4)).astype(np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        bg += noise[:, :, None]
        for _ in range(25):  # wood-grain-like streaks
            y = int(rng.integers(0, height))
            cv2.line(bg, (0, y), (width, y + int(rng.integers(-80, 80))),
                     tuple(float(v) for v in colour * 0.6), int(rng.integers(1, 4)))
    elif kind == "cluttered":
        for _ in range(14):  # pens, phones, coasters, books ...
            c = tuple(int(v) for v in rng.integers(0, 256, 3))
            if rng.random() < 0.5:
                x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
                w, h = int(rng.integers(40, 260)), int(rng.integers(20, 200))
                cv2.rectangle(bg, (x, y), (x + w, y + h), c, -1)
            else:
                centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
                cv2.circle(bg, centre, int(rng.integers(20, 120)), c, -1)
    return np.clip(bg, 0, 255).astype(np.uint8)


def random_corners(rng, angle="tilted", width=CANVAS_WIDTH, height=CANVAS_HEIGHT):
    """Random page corners (TL, TR, BR, BL) on the canvas for a given camera angle.

    angle: "straight" (camera almost above the page), "tilted" (moderate
           perspective and rotation) or "steep" (strong keystone effect).
    """
    settings = {  # (keystone, max rotation in degrees, corner jitter)
        "straight": (0.00, 3, 0.01),
        "tilted": (0.10, 12, 0.05),
        "steep": (0.32, 10, 0.04),
    }
    keystone, max_rot, jitter = settings[angle]
    for scale in np.linspace(0.62, 0.40, 12):  # shrink until the page fits
        h = height * scale
        w = h * PAGE_WIDTH / PAGE_HEIGHT
        k = keystone * w / 2  # the far (top) edge looks shorter
        pts = np.array([[-w / 2 + k, -h / 2], [w / 2 - k, -h / 2],
                        [w / 2, h / 2], [-w / 2, h / 2]])
        theta = np.radians(rng.uniform(-max_rot, max_rot))
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pts = pts @ rotation.T
        pts += rng.uniform(-jitter, jitter, pts.shape) * np.array([w, h])
        pts += np.array([width / 2, height / 2]) + rng.uniform(-0.04, 0.04, 2) * np.array([width, height])
        margin = 30
        if (pts[:, 0].min() > margin and pts[:, 0].max() < width - margin
                and pts[:, 1].min() > margin and pts[:, 1].max() < height - margin):
            return pts.astype(np.float32)
    return pts.astype(np.float32)


def apply_lighting(image, rng, kind="good"):
    """Apply a lighting condition: "good", "dim", "shadow" or "glare"."""
    img = image.astype(np.float32)
    h, w = img.shape[:2]
    if kind == "dim":
        img *= 0.42
        img += rng.normal(0, 5, img.shape)
    elif kind == "shadow":
        # A soft-edged shadow over one side of the image (e.g. the photographer's arm).
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = rng.uniform(-1, 1)
        shadow = ((xx - w * rng.uniform(0.35, 0.6)) + a * (yy - h / 2)) > 0
        shadow = cv2.GaussianBlur(shadow.astype(np.float32), (0, 0), 25)
        img *= (1.0 - 0.5 * shadow)[:, :, None]
    elif kind == "glare":
        # A bright, saturated hotspot (reflection of a lamp on glossy paper).
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w * rng.uniform(0.25, 0.75), h * rng.uniform(0.2, 0.45)
        sigma = w * 0.15
        img += 200 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def _curl(page, amount):
    """Bend the page so its top and bottom edges bow (simulates a curled sheet).

    The page is padded first so the bent edges have room to move. Returns the
    padded page, its alpha mask and the padding size. The 4 corners do not
    move, so the ground-truth corners stay valid.
    """
    pad = int(abs(amount)) + 10
    padded = cv2.copyMakeBorder(page, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    alpha = np.zeros(padded.shape[:2], np.float32)
    alpha[pad:-pad, pad:-pad] = 1.0
    hh, ww = padded.shape[:2]
    xs = np.arange(ww, dtype=np.float32)
    bend = amount * np.sin(np.pi * np.clip((xs - pad) / PAGE_WIDTH, 0, 1))
    map_x = np.tile(xs, (hh, 1))
    map_y = np.arange(hh, dtype=np.float32)[:, None] + bend[None, :]
    padded = cv2.remap(padded, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    alpha = cv2.remap(alpha, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    return padded, alpha, pad


def make_scene(seed=0, angle="tilted", lighting="good", background="plain",
               page_condition="flat", inner_box=False):
    """Create one synthetic photo.

    Args:
        seed:           random seed (same seed -> same image).
        angle:          "straight", "tilted" or "steep".
        lighting:       "good", "dim", "shadow" or "glare".
        background:     "plain", "textured", "cluttered" or "low-contrast".
        page_condition: "flat", "curled" or "folded".
        inner_box:      draw a thick frame inside the page (inner-contour trap).

    Returns:
        (image_bgr, corners, text): corners is a (4, 2) float32 array of the
        true page corners in TL, TR, BR, BL order; text is the printed text.
    """
    rng = np.random.default_rng(seed)
    page, text = make_page(rng, inner_box)
    bg = make_background(rng, background)
    corners = random_corners(rng, angle)

    if page_condition == "folded":
        # A crease across the middle: a dark line and slightly darker lower half.
        mid = PAGE_HEIGHT // 2 + int(rng.integers(-60, 60))
        page[mid:] = (page[mid:].astype(np.float32) * 0.9).astype(np.uint8)
        cv2.line(page, (0, mid), (PAGE_WIDTH, mid), (150, 150, 150), 3)

    if page_condition == "curled":
        src_page, alpha, pad = _curl(page, amount=float(rng.uniform(35, 55)))
    else:
        src_page, alpha, pad = page, np.ones(page.shape[:2], np.float32), 0

    source = np.array([[pad, pad], [pad + PAGE_WIDTH, pad],
                       [pad + PAGE_WIDTH, pad + PAGE_HEIGHT], [pad, pad + PAGE_HEIGHT]],
                      dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, corners)
    size = (bg.shape[1], bg.shape[0])
    warped_page = cv2.warpPerspective(src_page, matrix, size)
    warped_alpha = cv2.warpPerspective(alpha, matrix, size)[:, :, None]

    scene = bg.astype(np.float32) * (1 - warped_alpha) + warped_page.astype(np.float32) * warped_alpha
    scene = apply_lighting(scene.astype(np.uint8), rng, lighting)

    # Camera effects: slight optical blur and sensor noise.
    scene = cv2.GaussianBlur(scene, (3, 3), 0.8).astype(np.float32)
    scene += rng.normal(0, 3, scene.shape)
    scene = np.clip(scene, 0, 255).astype(np.uint8)
    return scene, corners, text
