"""Unit tests for the scanner, using synthetic images generated in code.

Run with:  python -m pytest -v
"""

import itertools
import os
import sys

import numpy as np
import pytest

# Make the project root importable when pytest is run from any folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scanner import order_points, polygon_iou, scan_document  # noqa: E402
from scanner.enhance import adaptive_threshold, make_odd  # noqa: E402
from scanner.synthetic import make_scene  # noqa: E402

SEEDS = range(6)


@pytest.mark.parametrize("seed", SEEDS)
def test_detects_page_on_plain_background(seed):
    """A tilted white page on a darker plain background must be found accurately."""
    image, true_corners, _ = make_scene(seed=seed, angle="tilted", background="plain")
    result = scan_document(image)
    assert result.success, result.message
    assert result.method in ("canny", "otsu")
    assert polygon_iou(result.corners, true_corners) >= 0.95


@pytest.mark.parametrize("seed", SEEDS)
def test_detects_page_on_textured_background(seed):
    """Same, but on a noisy, textured (wood-grain-like) background."""
    image, true_corners, _ = make_scene(seed=100 + seed, angle="tilted", background="textured")
    result = scan_document(image)
    assert result.success, result.message
    assert polygon_iou(result.corners, true_corners) >= 0.95


@pytest.mark.parametrize("value", [0, 128, 255])
def test_blank_image_fails_cleanly(value):
    """An image with no page must return method 'failed' without raising."""
    image = np.full((900, 700, 3), value, dtype=np.uint8)
    result = scan_document(image)
    assert not result.success
    assert result.method == "failed"
    assert result.corners is None
    assert "not detected" in result.message


def test_noisy_blank_image_fails_cleanly():
    """Mild camera noise on an empty grey surface is still 'no page'."""
    rng = np.random.default_rng(0)
    image = np.clip(rng.normal(128, 3, (900, 700, 3)), 0, 255).astype(np.uint8)
    result = scan_document(image)
    assert result.method == "failed"


def test_bad_inputs_do_not_crash():
    """None, an empty array and a tiny image are handled without exceptions."""
    assert scan_document(None).method == "failed"
    assert scan_document(np.zeros((0, 0, 3), np.uint8)).method == "failed"
    assert scan_document(np.zeros((5, 5, 3), np.uint8)).method == "failed"


def test_grayscale_input_is_accepted():
    image, true_corners, _ = make_scene(seed=7)
    gray = image.mean(axis=2).astype(np.uint8)
    result = scan_document(gray)
    assert result.success
    assert polygon_iou(result.corners, true_corners) >= 0.95


def test_corner_order_is_consistent_for_shuffled_points():
    """order_points returns TL, TR, BR, BL whatever order the input is in."""
    expected = np.array([[100, 80], [520, 120], [560, 700], [60, 650]], dtype=np.float32)
    for permutation in itertools.permutations(range(4)):
        ordered = order_points(expected[list(permutation)])
        np.testing.assert_allclose(ordered, expected)


def test_corner_order_for_rotated_square():
    """A square rotated by ~45 degrees (where x+y sums tie) still gives 4 distinct corners."""
    diamond = np.array([[300, 100], [500, 300], [300, 500], [100, 300]], dtype=np.float32)
    rng = np.random.default_rng(1)
    first = order_points(diamond)
    assert len({tuple(p) for p in first}) == 4
    for _ in range(10):
        np.testing.assert_allclose(order_points(rng.permutation(diamond)), first)


def test_polygon_iou_basic_cases():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    shifted = square + np.array([50, 0], dtype=np.float32)
    far_away = square + 1000
    assert polygon_iou(square, square) == pytest.approx(1.0)
    assert polygon_iou(square, shifted) == pytest.approx(1 / 3, abs=0.02)
    assert polygon_iou(square, far_away) == 0.0
    assert polygon_iou(square, None) == 0.0


def test_scan_outputs_have_expected_types():
    image, _, _ = make_scene(seed=3)
    result = scan_document(image, keep_stages=True)
    assert result.warped.ndim == 3
    assert set(np.unique(result.scanned)) <= {0, 255}
    for name in ["1_gray", "2_blurred", "3_edges", "4_contours", "5_warped", "6_thresholded"]:
        assert name in result.stages
    no_threshold = scan_document(image, apply_threshold=False)
    assert no_threshold.scanned is None and no_threshold.warped is not None


def test_even_block_size_is_made_odd():
    assert make_odd(24) == 25
    assert make_odd(1) == 3
    page = np.full((60, 60, 3), 255, np.uint8)
    assert adaptive_threshold(page, block_size=24).shape == (60, 60)


def test_edit_distance_and_cer():
    """The hand-written Levenshtein distance and CER used for the OCR evaluation."""
    from ocr import character_error_rate, edit_distance

    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("same", "same") == 0
    assert character_error_rate("hello world", "hello world") == 0.0
    assert character_error_rate("hello  world\n", "hello world") == 0.0  # whitespace normalised
    assert character_error_rate("abcd", "abxd") == pytest.approx(0.25)
    assert character_error_rate("abcd", None) == 1.0
