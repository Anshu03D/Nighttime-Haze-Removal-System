"""
Performance Dashboard Module

Generates benchmark graphs for the Nighttime Haze Removal System.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def create_dashboard(
    benchmark_csv="benchmark_results/benchmark.csv",
    output_folder="graphs"
):
    """
    Generate performance graphs from benchmark results.
    """

    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(benchmark_csv)

    metrics = [
        "brightness_gain",
        "contrast_gain",
        "entropy",
        "sharpness",
        "processing_time"
    ]

    for metric in metrics:

        plt.figure(figsize=(8,4))

        plt.bar(
            df["image"],
            df[metric]
        )

        plt.title(metric.replace("_", " ").title())

        plt.xlabel("Images")

        plt.ylabel(metric.replace("_", " ").title())

        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_folder,
                f"{metric}.png"
            ),
            dpi=300
        )

        plt.close()

    print("Dashboard graphs generated successfully.")