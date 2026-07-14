import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import time
import pandas as pd
import io

from model_runner import run_dehaze
from src.evaluation.metrics import evaluate_image
from src.visualization.histogram import generate_histogram

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Nighttime Haze Removal System",
    page_icon="🌙",
    layout="wide"
)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("🌙 Project Information")

st.sidebar.markdown("""
### Developer
**Snehanshu Daripa**

### Institution
Techno India College Of Technology

### Technology Stack

- Python
- PyTorch
- Streamlit
- OpenCV
- NumPy
- Pandas
- Matplotlib

---

### Features

✅ Nighttime Image Dehazing

✅ Deep Learning Enhancement

✅ Histogram Visualization

✅ Image Quality Metrics

✅ CSV Report Generation

✅ Download Enhanced Image

---

### Final Year Project
Computer Science & Engineering
""")

# -------------------------------------------------------
# Main Title
# -------------------------------------------------------

st.title("🌙 Nighttime Haze Removal System")

st.markdown(
"""
Deep Learning based image enhancement system for removing
nighttime haze and improving scene visibility.
"""
)

# -------------------------------------------------------
# Abstract
# -------------------------------------------------------

with st.expander("📄 Project Abstract"):

    st.write("""
This project enhances nighttime hazy images using a
deep learning based dehazing network.

The system performs:

- Nighttime haze removal
- Visibility enhancement
- Histogram comparison
- Image quality evaluation
- Performance measurement
- CSV logging
""")

# -------------------------------------------------------
# Upload Image
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Nighttime Hazy Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------

if uploaded_file is not None:

    os.makedirs("temp/input", exist_ok=True)
    os.makedirs("report_results/original", exist_ok=True)
    os.makedirs("report_results/dehazed", exist_ok=True)

    input_path = os.path.join(
        "temp/input",
        uploaded_file.name
    )

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    original = Image.open(input_path).convert("RGB")

    # -----------------------------------
    # Run Model
    # -----------------------------------

    start_time = time.perf_counter()

    with st.spinner("Removing haze..."):

        output_path = run_dehaze(input_path)

    end_time = time.perf_counter()

    processing_time = round(
        end_time - start_time,
        2
    )

    dehazed = Image.open(output_path).convert("RGB")

    # -----------------------------------
    # Gamma Correction
    # -----------------------------------

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

    # -----------------------------------
    # Save Images
    # -----------------------------------

    original.save(
        f"report_results/original/{uploaded_file.name}"
    )

    dehazed.save(
        f"report_results/dehazed/{uploaded_file.name}"
    )

    # -----------------------------------
    # Image Comparison
    # -----------------------------------

    st.subheader("🖼 Image Comparison")

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
            caption="Enhanced Image",
            use_container_width=True
        )

    # -----------------------------------
    # Image Evaluation
    # -----------------------------------

    original_np = np.array(original)

    dehazed_np = np.array(dehazed)

    metrics = evaluate_image(
        original_image=original_np,
        enhanced_image=dehazed_np,
        processing_time=processing_time
    )

    histogram_path = generate_histogram(
        original_np,
        dehazed_np
    )
        # -----------------------------------
    # Histogram
    # -----------------------------------

    st.subheader("📈 Histogram Comparison")

    st.image(
        histogram_path,
        use_container_width=True
    )

    # -----------------------------------
    # Image Quality Metrics
    # -----------------------------------

    st.subheader("📊 Image Quality Evaluation")

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:

        st.metric(
            "Original Brightness",
            metrics["original_brightness"]
        )

        st.metric(
            "Enhanced Brightness",
            metrics["enhanced_brightness"]
        )

    with row1_col2:

        st.metric(
            "Brightness Gain",
            f"{metrics['brightness_gain']}%"
        )

        st.metric(
            "Contrast Gain",
            f"{metrics['contrast_gain']}%"
        )

    with row1_col3:

        st.metric(
            "Entropy",
            metrics["entropy"]
        )

        st.metric(
            "Sharpness",
            metrics["sharpness"]
        )

    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:

        st.metric(
            "Colorfulness",
            metrics["colorfulness"]
        )

    with row2_col2:

        st.metric(
            "Resolution",
            metrics["resolution"]
        )

    with row2_col3:

        st.metric(
            "Processing Time",
            f"{metrics['processing_time']} sec"
        )

    # -----------------------------------
    # Processing Information
    # -----------------------------------

    st.info(
        f"⏱ Total Processing Time : {processing_time} seconds"
    )

    # -----------------------------------
    # Save Metrics
    # -----------------------------------

    metrics_data = {

        "Image Name": uploaded_file.name,

        "Original Brightness":
            metrics["original_brightness"],

        "Enhanced Brightness":
            metrics["enhanced_brightness"],

        "Brightness Gain (%)":
            metrics["brightness_gain"],

        "Contrast Gain (%)":
            metrics["contrast_gain"],

        "Entropy":
            metrics["entropy"],

        "Sharpness":
            metrics["sharpness"],

        "Colorfulness":
            metrics["colorfulness"],

        "Resolution":
            metrics["resolution"],

        "Processing Time (s)":
            metrics["processing_time"]

    }

    os.makedirs(
        "metrics",
        exist_ok=True
    )

    csv_file = "metrics/results.csv"

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

    # -----------------------------------
    # Download Button
    # -----------------------------------

    buffer = io.BytesIO()

    dehazed.save(
        buffer,
        format="JPEG"
    )

    st.download_button(
        label="📥 Download Enhanced Image",
        data=buffer.getvalue(),
        file_name=f"dehazed_{uploaded_file.name}",
        mime="image/jpeg"
    )

    # -----------------------------------
    # Completion Message
    # -----------------------------------

    st.success("✅ Processing Completed Successfully!")

    # -----------------------------------
    # Footer
    # -----------------------------------

    st.markdown("---")

    st.caption(
        "Nighttime Haze Removal System | Final Year Project | "
        "Techno India College Of Technology"
    )