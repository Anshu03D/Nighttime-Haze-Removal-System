"""
Histogram Visualization Module
"""

import os
import cv2
import matplotlib.pyplot as plt


def generate_histogram(original_image, enhanced_image):
    """
    Generate grayscale histogram comparison and save it.
    """

    os.makedirs("graphs", exist_ok=True)

    original_gray = cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2GRAY
    )

    enhanced_gray = cv2.cvtColor(
        enhanced_image,
        cv2.COLOR_RGB2GRAY
    )

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(
        original_gray.ravel(),
        bins=256,
        range=(0, 256),
        color="blue"
    )
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.hist(
        enhanced_gray.ravel(),
        bins=256,
        range=(0, 256),
        color="green"
    )
    plt.title("Enhanced")

    plt.tight_layout()

    output_path = "graphs/histogram.png"

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    return output_path