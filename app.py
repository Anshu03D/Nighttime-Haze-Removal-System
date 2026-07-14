import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import time
import pandas as pd

from model_runner import run_dehaze
from src.visualization.histogram import generate_histogram

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
        st.subheader("📈 Histogram Comparison")

st.image(
    histogram_path,
    use_container_width=True
)
    # -------------------
    # Image Evaluation
    # -------------------

    original_np = np.array(original)
    dehazed_np = np.array(dehazed)
    histogram_path = generate_histogram(
    original_np,
    dehazed_np
)

    metrics = evaluate_image(
        original_image=original_np,
        enhanced_image=dehazed_np,
        processing_time=processing_time
    )
    st.subheader("Performance Metrics")

    st.subheader("📊 Image Quality Evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
        "Original Brightness",
        metrics["original_brightness"]
    )

        st.metric(
        "Enhanced Brightness",
        metrics["enhanced_brightness"]
    )

    with col2:
        st.metric(
        "Brightness Gain",
        f"{metrics['brightness_gain']}%"
    )

        st.metric(
        "Contrast Gain",
        f"{metrics['contrast_gain']}%"
    )

    with col3:
        st.metric(
        "Entropy",
        metrics["entropy"]
    )

    st.metric(
        "Sharpness",
        metrics["sharpness"]
    )

    st.metric(
    "Colorfulness",
    metrics["colorfulness"]
)

    st.metric(
    "Resolution",
    metrics["resolution"]
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
        "Original Brightness": metrics["original_brightness"],
        "Enhanced Brightness": metrics["enhanced_brightness"],
        "Brightness Gain (%)": metrics["brightness_gain"],
        "Contrast Gain (%)": metrics["contrast_gain"],
        "Entropy": metrics["entropy"],
        "Sharpness": metrics["sharpness"],
        "Colorfulness": metrics["colorfulness"],
        "Resolution": metrics["resolution"],
        "Processing Time (s)": metrics["processing_time"]
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