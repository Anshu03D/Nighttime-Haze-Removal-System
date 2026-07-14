from PIL import Image
import numpy as np

from src.visualization.histogram import generate_histogram

original = Image.open(
    "report_results/original/038.jpg"
)

enhanced = Image.open(
    "report_results/dehazed/038.jpg"
)

path = generate_histogram(
    np.array(original),
    np.array(enhanced)
)

print(path)