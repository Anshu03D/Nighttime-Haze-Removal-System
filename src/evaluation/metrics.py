"""
Image Quality Evaluation Module

This module calculates objective image quality metrics for the
Nighttime Haze Removal System.
"""

from typing import Dict

import cv2
import numpy as np


def calculate_entropy(gray_image: np.ndarray) -> float:
    """Calculate image entropy."""

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    hist = hist.ravel()
    hist = hist / hist.sum()
    hist = hist[hist > 0]

    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def calculate_brightness(gray_image: np.ndarray) -> float:
    """Calculate average brightness."""

    return float(np.mean(gray_image))


def calculate_contrast(gray_image: np.ndarray) -> float:
    """Calculate RMS contrast."""

    return float(np.std(gray_image))


def calculate_sharpness(gray_image: np.ndarray) -> float:
    """Calculate image sharpness using Variance of Laplacian."""

    return float(
        cv2.Laplacian(
            gray_image,
            cv2.CV_64F
        ).var()
    )


def calculate_colorfulness(image: np.ndarray) -> float:
    """
    Calculate colorfulness using
    Hasler & Susstrunk algorithm.
    """

    image = image.astype("float")

    (b, g, r) = cv2.split(image)

    rg = np.abs(r - g)

    yb = np.abs(
        0.5 * (r + g) - b
    )

    std_rg = np.std(rg)
    std_yb = np.std(yb)

    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)

    return float(
        np.sqrt(
            std_rg ** 2 +
            std_yb ** 2
        )
        +
        0.3 * np.sqrt(
            mean_rg ** 2 +
            mean_yb ** 2
        )
    )


def evaluate_image(
    original_image: np.ndarray,
    enhanced_image: np.ndarray,
    processing_time: float
) -> Dict:
    """
    Evaluate image quality.
    """

    original_gray = cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2GRAY
    )

    enhanced_gray = cv2.cvtColor(
        enhanced_image,
        cv2.COLOR_RGB2GRAY
    )

    original_brightness = calculate_brightness(
        original_gray
    )

    enhanced_brightness = calculate_brightness(
        enhanced_gray
    )

    brightness_gain = (
        (
            enhanced_brightness -
            original_brightness
        )
        /
        max(original_brightness, 1)
    ) * 100

    original_contrast = calculate_contrast(
        original_gray
    )

    enhanced_contrast = calculate_contrast(
        enhanced_gray
    )

    contrast_gain = (
        (
            enhanced_contrast -
            original_contrast
        )
        /
        max(original_contrast, 1)
    ) * 100

    entropy = calculate_entropy(
        enhanced_gray
    )

    sharpness = calculate_sharpness(
        enhanced_gray
    )

    colorfulness = calculate_colorfulness(
        enhanced_image
    )

    return {

        "original_brightness":
            round(original_brightness, 2),

        "enhanced_brightness":
            round(enhanced_brightness, 2),

        "brightness_gain":
            round(brightness_gain, 2),

        "contrast_gain":
            round(contrast_gain, 2),

        "entropy":
            round(entropy, 2),

        "sharpness":
            round(sharpness, 2),

        "colorfulness":
            round(colorfulness, 2),

        "processing_time":
            round(processing_time, 2),

        "resolution":
            f"{enhanced_image.shape[1]} × {enhanced_image.shape[0]}"
    }