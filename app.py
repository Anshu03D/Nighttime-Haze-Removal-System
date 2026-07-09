import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import time
import pandas as pd

from model_runner import run_dehaze

st.set_page_config(
    page_title="Nighttime Haze Removal",
    page_icon="🌙",
    layout="wide"
)

st.sidebar.title("Project Information")
st.sidebar.markdown("""
### Developer
Snehanshu Daripa

### Institution
Techno India College Of Technology

### Technology Stack
- Python
- PyTorch
- Streamlit
- OpenCV
- GAPSF
""")

st.sidebar.markdown("""
### Nighttime Haze Removal System

Research-based image enhancement system using a deep learning dehazing network.

### Features

✅ Nighttime Dehazing

✅ Visibility Enhancement

✅ Brightness Analysis

✅ Contrast Analysis

✅ Entropy Measurement

✅ Image Download

### Developer

Final Year Project
""")

st.title("🌙 Nighttime Haze Removal System")

with st.expander("Project Abstract"):

    st.write("""
    This project enhances visibility in nighttime hazy images using
    a deep learning based dehazing network. The system removes haze,
    improves scene visibility, and evaluates image quality through
    brightness, contrast, entropy, and processing time metrics.
    """)

uploaded_file = st.file_uploader(
    "Upload a Nighttime Hazy Image",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:

    os.makedirs("temp/input", exist_ok=True)

    input_path = os.path.join(
        "temp/input",
        uploaded_file.name
    )

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    original = Image.open(input_path)

    start_time = time.time()

    with st.spinner("Removing haze..."):
        output_path = run_dehaze(input_path)

    end_time = time.time()
    processing_time = round(end_time - start_time, 2)

    dehazed = Image.open(output_path)

    dehazed_np = np.array(dehazed)

    gamma = 0.7

    table = np.array([
    ((i / 255.0) ** gamma) * 255
    for i in np.arange(256)
    ]).astype("uint8")

    dehazed_np = cv2.LUT(
    dehazed_np,
    table
)

    dehazed = Image.fromarray(
        dehazed_np
    )
    original.save(
    f"report_results/original/{uploaded_file.name}"
)

    dehazed.save(
    f"report_results/dehazed/{uploaded_file.name}"
)

    # -------------------
    # Image Comparison
    # -------------------

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            original,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            dehazed,
            caption="Dehazed Image",
            use_container_width=True
        )

    def calculate_entropy(gray):
        hist = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256]
        )
        hist = hist.ravel()
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    # -------------------
    # Metrics
    # -------------------

    original_np = np.array(original)
    dehazed_np = np.array(dehazed)

    original_gray = cv2.cvtColor(
        original_np,
        cv2.COLOR_RGB2GRAY
    )

    dehazed_gray = cv2.cvtColor(
        dehazed_np,
        cv2.COLOR_RGB2GRAY
    )

    original_brightness = np.mean(original_gray)
    dehazed_brightness = np.mean(dehazed_gray)

    brightness_gain = (
        (
            dehazed_brightness
            - original_brightness
        )
        / max(original_brightness, 1)
    ) * 100

    original_contrast = np.std(original_gray)

    dehazed_contrast = np.std(dehazed_gray)

    contrast_gain = (
    (
        dehazed_contrast
        - original_contrast
    )
    / max(original_contrast, 1)
) * 100

    entropy = calculate_entropy(
    dehazed_gray
)

    st.subheader("Performance Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Original Brightness",
        f"{original_brightness:.2f}"
    )

    m2.metric(
        "Enhanced Brightness",
        f"{dehazed_brightness:.2f}"
    )

    m3.metric(
        "Brightness Gain",
        f"{brightness_gain:.2f}%"
    )
    
    m4.metric(
        "Contrast Gain",
        f"{contrast_gain:.2f}%"
    )

    m5.metric(
        "Entropy",
        f"{entropy:.2f}"
    )

    # -------------------
    # Processing Time
    # -------------------

    st.info(
        f"Processing Time: {processing_time} seconds"
    )

    # -------------------
    # Download Button
    # -------------------

    import io

    buffer = io.BytesIO()

    dehazed.save(
    buffer,
    format="JPEG"
)

    st.download_button(
    label="📥 Download Dehazed Image",
    data=buffer.getvalue(),
    file_name="dehazed_image.jpg",
    mime="image/jpeg"
)
    metrics_data = {
        "Image Name": uploaded_file.name,
        "Original Brightness": round(original_brightness, 2),
        "Enhanced Brightness": round(dehazed_brightness, 2),
        "Brightness Gain (%)": round(brightness_gain, 2),
        "Contrast Gain (%)": round(contrast_gain, 2),
        "Entropy": round(entropy, 2),
        "Processing Time (s)": processing_time
    }

    csv_file = "metrics/results.csv"

    os.makedirs("metrics", exist_ok=True)

    df = pd.DataFrame([metrics_data])

    if os.path.exists(csv_file):
        df.to_csv(
            csv_file,
            mode="a",
            header=False,
            index=False
        )
    else:
        df.to_csv(
            csv_file,
            index=False
        )

    st.success("Processing Complete!")