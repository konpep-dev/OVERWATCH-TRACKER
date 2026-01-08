<div align="center">
  
<img src="https://img.shields.io/badge/🎯-OVERWATCH-black?style=for-the-badge&labelColor=red" alt="Logo" height="60">

# OVERWATCH TRACKER

### Real-time Person Tracking System with AI-Powered Pose Estimation

<br>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-blue?style=flat-square)
![Version](https://img.shields.io/badge/Version-1.0-orange?style=flat-square)

<br>

<p align="center">
  <b>Advanced computer vision system featuring skeleton tracking,<br>threat level analysis, and real-time object detection</b>
</p>

<br>

```
coded by konpep
```

<br>

[Features](#-features) •
[Installation](#-installation) •
[Usage](#-camera-configuration) •
[Requirements](#%EF%B8%8F-system-requirements)

</div>

<br>

---

<br>

## ✨ Features

<table>
<tr>
<td width="50%">

### 🦴 Skeleton Tracking
Real-time pose estimation with 17-point body keypoints visualization

### 🎯 Head Targeting  
Precise head position tracking with visual crosshair overlay

### 📊 Threat Analysis
Dynamic threat level system based on behavior patterns

</td>
<td width="50%">

### 🔴 Movement Trail
Visual path showing head movement history over time

### 📦 Object Detection
Identifies objects being held (phone, bottle, knife, etc.)

### ⚡ GPU Acceleration
Support for NVIDIA CUDA & AMD DirectML

</td>
</tr>
</table>

<br>

---

<br>

## 📊 Threat Level System

<div align="center">

| Level | Status | Color | Triggers |
|:-----:|:------:|:-----:|:---------|
| **0-25%** | `RELAXED` | 🟢 | Standing still, no objects detected |
| **25-50%** | `ALERT` | 🟡 | Movement detected, holding harmless objects |
| **50-70%** | `DANGER` | 🟠 | Fast movement, close proximity |
| **70-100%** | `CRITICAL` | 🔴 | Holding dangerous objects, aggressive movement |

</div>

<br>

---

<br>

## 🚀 Installation

### Prerequisites

> - Python 3.8 or higher
> - Webcam or IP Webcam app on smartphone

<br>

### Quick Start

```bash
# Clone the repository
git clone https://github.com/konpep/OVERWATCH-Tracker.git

# Navigate to directory
cd OVERWATCH-Tracker

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

<br>

> 💡 **Tip:** Dependencies are auto-installed on first run if missing

<br>

---

<br>

## 📱 Camera Configuration

<table>
<tr>
<td>

### Setup Steps

1. Install **IP Webcam** from Play Store
2. Open app → Tap **Start Server**
3. Note the IP address shown
4. Edit `config.py`:

```python
CAMERA_IP = "192.168.1.XXX"
CAMERA_PORT = 8080
```

</td>
<td>

### Supported Sources

| Source | Status |
|--------|:------:|
| IP Webcam (Android) | ✅ |
| USB Webcam | ✅ |
| DroidCam | ✅ |
| Built-in Camera | ✅ |

</td>
</tr>
</table>

<br>

---

<br>

## ⚙️ Performance Presets

<div align="center">

| Preset | Model | Resolution | Best For |
|:------:|:-----:|:----------:|:---------|
| 🟢 **LOW** | YOLOv8n | 256px | Low-end CPUs, best FPS |
| 🟡 **MEDIUM** | YOLOv8n | 320px | Balanced performance |
| 🟠 **HIGH** | YOLOv8s | 416px | Gaming PCs / Entry GPUs |
| 🔴 **ULTRA** | YOLOv8m | 640px | High-end GPUs only |

</div>

<br>

---

<br>

## 🖥️ System Requirements

<table>
<tr>
<th></th>
<th>Minimum</th>
<th>Recommended</th>
</tr>
<tr>
<td><b>CPU</b></td>
<td>Any x64 processor</td>
<td>Intel i5 / AMD Ryzen 5</td>
</tr>
<tr>
<td><b>RAM</b></td>
<td>4 GB</td>
<td>8 GB+</td>
</tr>
<tr>
<td><b>GPU</b></td>
<td>Not required</td>
<td>NVIDIA GTX 1060+ / AMD RX 580+</td>
</tr>
<tr>
<td><b>Python</b></td>
<td>3.8</td>
<td>3.10+</td>
</tr>
</table>

<br>

---

<br>

## 🛠️ Tech Stack

<div align="center">

| Technology | Purpose |
|:----------:|:--------|
| **YOLOv8** | Object Detection & Pose Estimation |
| **OpenCV** | Video Processing & Visualization |
| **PyTorch** | Deep Learning Backend |
| **ONNX Runtime** | DirectML Support for AMD GPUs |

</div>

<br>

---

<br>

## 📁 Project Structure

```
OVERWATCH-Tracker/
│
├── 📄 main.py              # Main application
├── ⚙️ config.py            # Configuration
├── 📋 requirements.txt     # Dependencies
├── 🪟 install.bat          # Windows installer
├── 🐧 install.sh           # Linux/Mac installer
├── 📜 LICENSE              # MIT License
└── 📖 README.md            # This file
```

<br>

---

<br>

## 🎮 Controls

<div align="center">

| Key | Action |
|:---:|:-------|
| `Q` | Quit application |

</div>

<br>

---

<br>

## 📄 License

<div align="center">

This project is licensed under the **MIT License**

See [LICENSE](LICENSE) for details

</div>

<br>

---

<br>

<div align="center">

### Built with 🐍 Python and ❤️

<br>

**© 2026 konpep**

<br>

[![GitHub](https://img.shields.io/badge/GitHub-konpep-181717?style=for-the-badge&logo=github)](https://github.com/konpep)

</div>
