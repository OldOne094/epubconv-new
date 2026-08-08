import numpy as np

from epubconv.preprocessing import deskew, estimate_skew_angle, preprocess, to_grayscale


def test_to_grayscale_passthrough_for_2d():
    gray = np.zeros((10, 10), dtype=np.uint8)
    assert to_grayscale(gray) is gray


def test_to_grayscale_converts_rgb():
    rgb = np.full((10, 10, 3), 128, dtype=np.uint8)
    gray = to_grayscale(rgb)
    assert gray.shape == (10, 10)


def test_estimate_skew_angle_on_blank_page_is_zero():
    blank = np.full((100, 100), 255, dtype=np.uint8)
    assert estimate_skew_angle(blank) == 0.0


def test_deskew_is_noop_for_tiny_angle():
    image = np.zeros((20, 20), dtype=np.uint8)
    result = deskew(image, 0.01)
    assert result is image


def test_deskew_is_noop_for_noise_level_angle_from_a_straight_page():
    # Regression test: angle estimation on a real, perfectly straight PDF-rendered
    # page still reports small nonzero angles (measured up to ~0.26 degrees) that
    # are pure noise, not real skew. Actually rotating for one of these measurably
    # hurt OCR accuracy, so anything under a full degree must stay a no-op.
    image = np.zeros((20, 20), dtype=np.uint8)
    result = deskew(image, 0.26)
    assert result is image


def test_deskew_still_corrects_a_genuinely_skewed_page():
    image = np.zeros((20, 20), dtype=np.uint8)
    result = deskew(image, 5.0)
    assert result is not image


def test_preprocess_returns_three_channel_same_size():
    rgb = np.full((50, 60, 3), 200, dtype=np.uint8)
    result = preprocess(rgb)
    assert result.shape == (50, 60, 3)
