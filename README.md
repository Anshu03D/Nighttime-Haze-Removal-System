# 🌙 Nighttime Haze Removal System

A Deep Learning Based Web Application for Enhancing Visibility in Nighttime Hazy Images using the GAPSF Network (ACM Multimedia 2023).

---

## 📖 Project Overview

Nighttime haze significantly reduces the visibility of roads, vehicles, pedestrians, and urban scenes, making navigation and surveillance difficult. Traditional image enhancement methods often fail because nighttime images suffer from both low illumination and atmospheric scattering.

This project presents a **web-based Nighttime Haze Removal System** that utilizes a **deep learning-based GAPSF (Guided APSF and Gradient Adaptive Convolution) network** to remove haze from nighttime images while preserving scene details.

The application is developed using **Python**, **PyTorch**, **OpenCV**, and **Streamlit**, providing an intuitive interface for uploading images, processing them through the dehazing model, and evaluating the results using multiple image quality metrics.

---

# ✨ Features

* 🌙 Nighttime image dehazing
* 📤 Upload hazy images
* 🤖 Deep Learning based GAPSF model
* 🖼️ Side-by-side image comparison
* 📊 Brightness Analysis
* 📈 Contrast Gain Analysis
* 📉 Entropy Measurement
* ⏱️ Processing Time Calculation
* 📥 Download enhanced images
* 📋 Automatic CSV logging of experimental results

---

# 🏗️ System Architecture

```text
              Input Image
                   │
                   ▼
          Image Upload Module
                   │
                   ▼
         GAPSF Deep Learning Model
                   │
                   ▼
         Gamma-based Enhancement
                   │
                   ▼
      Image Quality Evaluation
      (Brightness, Contrast,
       Entropy, Time)
                   │
                   ▼
         Display & Download
```

---

# 🛠️ Technologies Used

| Technology | Purpose                 |
| ---------- | ----------------------- |
| Python     | Programming Language    |
| PyTorch    | Deep Learning Framework |
| OpenCV     | Image Processing        |
| Streamlit  | Web Application         |
| Pillow     | Image Handling          |
| NumPy      | Numerical Computation   |
| Pandas     | Metrics Logging         |

---

# 📂 Project Structure

```
NighttimeHazeRemoval_Final/

├── app.py
├── model_runner.py
├── model/
│   ├── DEGLOW_test.py
│   ├── dataset.py
│   ├── networks.py
│   └── utils.py
│
├── weights/
│   └── dehaze.pt
│
├── metrics/
│   └── results.csv
│
├── report_results/
│   ├── original/
│   └── dehazed/
│
├── temp/
│   ├── input/
│   └── output/
│
└── assets/
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Anshu03D/Nighttime-Haze-Removal-System.git
```

Move into the project directory

```bash
cd Nighttime-Haze-Removal-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📥 Model Weights

The trained model file **dehaze.pt** is **not included** in this repository because of GitHub's file size limitations.

Download the pretrained model from the original research implementation and place it inside:

```
weights/dehaze.pt
```

---

# 📊 Performance Evaluation

The application evaluates every processed image using:

* Original Brightness
* Enhanced Brightness
* Brightness Gain (%)
* Contrast Gain (%)
* Entropy
* Processing Time (seconds)

All experimental results are automatically stored in:

```
metrics/results.csv
```

---

# 📸 Application Preview

> Add screenshots inside the **assets/** folder and replace the placeholders below.

### Home Page

```
assets/home.png
```

### Image Upload

```
assets/upload.png
```

### Before vs After

```
assets/comparison.png
```

### Metrics Dashboard

```
assets/metrics.png
```

---

# 🎯 Applications

* Intelligent Transportation Systems
* Autonomous Driving
* CCTV Surveillance
* Smart City Monitoring
* Night Vision Enhancement
* Road Safety Systems

---

# 📚 Research Reference

This project is based on the research paper:

**Enhancing Visibility in Nighttime Haze Images Using Guided APSF and Gradient Adaptive Convolution**

**ACM Multimedia (ACM MM), 2023**

Authors:

* Yeying Jin
* Beibei Lin
* Wending Yan
* Yuan Yuan
* Wei Ye
* Robby T. Tan

Original Repository:

https://github.com/jinyeying/nighttime_dehaze

This project extends the research implementation by developing a complete deployment pipeline, web interface, automated evaluation metrics, CSV logging, visualization, and user interaction components.

---

# 🔮 Future Scope

* Real-time video dehazing
* Live webcam enhancement
* Cloud deployment
* Mobile application
* Batch image processing
* Performance optimization for edge devices

---

# 👨‍💻 Developer

**Sweta Mondal**

B.Tech – Computer Science & Engineering

Techno India College Of Technology

---

# ⭐ Acknowledgement

The pretrained deep learning model and network architecture are based on the published ACM Multimedia 2023 research. The web application, deployment workflow, testing pipeline, evaluation metrics, and user interface were developed as part of this undergraduate final-year project.
