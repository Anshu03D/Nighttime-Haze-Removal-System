"""
Benchmark Evaluation Module

Runs quantitative evaluation on a dataset of nighttime hazy images.
"""

import os
import time
import pandas as pd
import numpy as np
from PIL import Image

from model_runner import run_dehaze
from src.evaluation.metrics import evaluate_image


def benchmark_dataset(
    dataset_folder="dataset_test",
    output_csv="benchmark_results/benchmark.csv"
):
    """
    Benchmark the dehazing model on all images in a dataset.
    """

    os.makedirs("benchmark_results", exist_ok=True)

    supported_formats = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif"
    )

    results = []

    for filename in sorted(os.listdir(dataset_folder)):

        if not filename.lower().endswith(supported_formats):
            continue

        input_path = os.path.join(dataset_folder, filename)

        print(f"Processing: {filename}")

        start = time.perf_counter()

        output_path = run_dehaze(input_path)

        end = time.perf_counter()

        processing_time = end - start

        if not os.path.exists(output_path):
            print(f"Skipping {filename}: output image was not generated.")
            continue

        original = np.array(
            Image.open(input_path).convert("RGB")
        )

        enhanced = np.array(
            Image.open(output_path).convert("RGB")
        )

        metrics = evaluate_image(
            original,
            enhanced,
            processing_time
        )

        metrics["image"] = filename

        results.append(metrics)

    df = pd.DataFrame(results)

    columns = [
        "image",
        "original_brightness",
        "enhanced_brightness",
        "brightness_gain",
        "contrast_gain",
        "entropy",
        "sharpness",
        "colorfulness",
        "processing_time",
        "resolution"
    ]

    df = df[columns]

    df.to_csv(output_csv, index=False)

    return df