"""Deskew and denoise a page image before OCR.

Uses opencv-python only. The prior version depended on opencv-contrib-python at
runtime without declaring it in pyproject.toml; nothing here needs contrib-only
modules, so that dependency is avoided entirely.
"""

from __future__ import annotations

import cv2
import numpy as np

MAX_PLAUSIBLE_SKEW_DEGREES = 20.0
# Angle estimation is noisy on a page that isn't actually skewed at all — measured
# on a real, cleanly-rendered PDF book: every page came back with a "detected"
# angle between 0.0 and 0.26 degrees despite being perfectly straight. Correcting
# that non-existent skew still rotates+interpolates the image, which measurably
# hurt OCR accuracy across a 10-page sample (higher avg confidence and fewer
# low-confidence words with correction disabled). 1 degree is comfortably above
# that noise floor while still catching genuinely skewed/photographed scans.
MIN_CORRECTION_DEGREES = 1.0


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate the rotation (in degrees) needed to deskew a scanned page."""
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > MAX_PLAUSIBLE_SKEW_DEGREES:
        return 0.0
    return angle


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < MIN_CORRECTION_DEGREES:
        return image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    border_value = int(np.median(image))
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=7)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Deskew and denoise a raw page image, returning a 3-channel image for OCR."""
    gray = to_grayscale(image)
    angle = estimate_skew_angle(gray)
    gray = deskew(gray, angle)
    gray = denoise(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
