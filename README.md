<div align="center">

<img src="https://img.shields.io/badge/YOLO-v8%20%7C%20v11-FFD700?style=for-the-badge&logo=pytorch&logoColor=black" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />

# 🐝 Bee Detection System

**Real-time Queen & Worker Bee detection using dual YOLO models with a FastAPI backend and Streamlit frontend.**

[Live Demo](#-demo) · [Model Comparison](#-model-comparison) · [Quick Start](#-quick-start) · [Architecture](#-system-architecture)

</div>

---

## 📌 Overview

This project implements an end-to-end computer vision pipeline for detecting and classifying **Queen Bees** and **Worker Bees** in beehive images. Two YOLO models were trained and compared — a lightweight **YOLOv8n** and a high-capacity **YOLO11m** — and deployed through a **FastAPI** inference server with a **Streamlit** dashboard for real-time detection.

| | |
|---|---|
| **Task** | Object Detection — Queen Bee vs Worker Bee |
| **Models** | YOLOv8n · YOLO11m |
| **Backend** | FastAPI (REST inference endpoint) |
| **Frontend** | Streamlit (interactive web app) |
| **Input** | JPG / PNG beehive images |
| **Output** | Annotated image + detection metadata (class, bbox, count) |

---

## 🎯 Demo

> Upload a beehive image → FastAPI runs YOLO inference → Streamlit renders annotated results

**Bounding box color guide:**

| Color | Class |
|---|---|
| 🟥 Red | Queen Bee 👑 |
| 🟨 Yellow | Worker Bee 🐝 |

---

## 📊 Model Comparison

Two models were trained independently and evaluated on their respective validation sets.

| Metric | YOLOv8n (`best.pt`) | YOLO11m (`best11m.pt`) | Winner |
|---|---|---|---|
| Architecture | YOLOv8 Nano | YOLO11 Medium | — |
| Parameters | ~3.05M | ~20.1M | YOLOv8n (×6.6 smaller) |
| File Size | **5.96 MB** | 38.64 MB | YOLOv8n |
| mAP@0.5 | 80.77% | **80.78%** | Tie |
| mAP@0.5:0.95 | 52.91% | **55.79%** | YOLO11m (+2.88 pp) |
| Precision | **89.52%** | 86.87% | YOLOv8n (+2.65 pp) |
| Recall | 74.62% | **78.55%** | YOLO11m (+3.93 pp) |
| Fitness Score | 0.529 | **0.558** | YOLO11m |
| val/box\_loss | 1.137 | **0.941** | YOLO11m (−17.2%) |
| Epochs Trained | 50 | 91 | — |
| Est. CPU FPS | ~50–80 | ~15–25 | YOLOv8n (~3× faster) |
| Edge Deployment | ✅ Excellent | ⚠️ Limited | YOLOv8n |



### Key Takeaways

- **YOLOv8n** is the better choice for real-time or edge deployment — 6.6× smaller with nearly identical coarse accuracy.
- **YOLO11m** achieves superior localization (mAP@0.5:0.95 +2.88 pp), recall (+3.93 pp), and lower validation loss across all three loss heads — preferred when accuracy is the priority.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User (Browser)                          │
│                   Streamlit Frontend                         │
│         Upload Image ──► Run Detection Button               │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP POST /predict (multipart/form-data)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend                             │
│              http://127.0.0.1:8000                           │
│    ┌──────────────────────────────────────────────────┐     │
│    │  YOLO Inference Engine                           │     │
│    │  ├── best.pt   (YOLOv8n  · 3.05M params)        │     │
│    │  └── best11m.pt (YOLO11m · 20.1M params)        │     │
│    └──────────────────────────────────────────────────┘     │
│    Returns: { detections: [{class, box, confidence}],       │
│               total_detected: N }                           │
└────────────────────┬────────────────────────────────────────┘
                     │ JSON response
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Streamlit renders annotated image               │
│         Red boxes = Queen Bee · Yellow = Worker Bee          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Bee_-Detection-_system/
│
├── app.py                  # Streamlit frontend
├── main.py                 # FastAPI backend (inference server)
├── best.pt                 # Trained YOLOv8n weights (5.96 MB)
├── best11m.pt              # Trained YOLO11m weights (38.6 MB)
├── yolov8n.pt              # Base pretrained YOLOv8n (Ultralytics)
├── data.yaml               # Dataset config (classes, paths)
├── requirments.txt         # Python dependencies
│
├── yolo11mModel.ipynb      # Training notebook (YOLO11m on Colab)
├── yolo_model_comparison.html  # Interactive model comparison report
│
├── train/                  # Training images + labels
├── valid/                  # Validation images + labels
├── test/                   # Test images + labels
└── runs/detect/            # YOLO training output (weights, plots)
```

---

## ⚡ Quick Start

### Prerequisites

```bash
Python 3.10+
```

### 1. Clone the repo

```bash
git clone https://github.com/nouralaa311/Bee_-Detection-_system.git
cd Bee_-Detection-_system
```

### 2. Install dependencies

```bash
pip install -r requirments.txt
```

### 3. Start the FastAPI backend

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 4. Launch the Streamlit frontend

```bash
streamlit run app.py
```

### 5. Open your browser

Navigate to `http://localhost:8501`, upload a beehive image, and click **Run AI Detection**.

---

## 🔬 Training Details

### YOLOv8n (`best.pt`)

| Config | Value |
|---|---|
| Base model | `yolov8n.pt` (pretrained COCO) |
| Dataset | `bee_dataset` |
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Patience (early stop) | 100 |
| Precision | fp16 (HalfStorage) |

### YOLO11m (`best11m.pt`)

| Config | Value |
|---|---|
| Base model | `yolo11m.pt` (pretrained) |
| Dataset | `Queen-Bee-4` |
| Epochs | 120 (converged at ~91) |
| Batch size | 8 |
| Image size | 640 × 640 |
| Patience (early stop) | 25 |
| Precision | fp16 (HalfStorage) |

Training was conducted on **Google Colab** (GPU). See `yolo11mModel.ipynb` for the full YOLO11m training pipeline.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) |
| Frontend | [Streamlit](https://streamlit.io/) |
| Image Processing | Pillow (PIL), NumPy |
| Training Platform | Google Colab |
| Dataset Source | [Roboflow](https://roboflow.com/) |

---

## 📦 API Reference

**`POST /predict`**

Accepts a multipart image upload and returns detection results.

**Request:**
```
Content-Type: multipart/form-data
file: <image file>
```

**Response:**
```json
{
  "total_detected": 7,
  "detections": [
    {
      "class_name": "Queen-Bee",
      "confidence": 0.91,
      "box": [120, 85, 310, 275]
    },
    {
      "class_name": "Worker-Bee",
      "confidence": 0.87,
      "box": [400, 200, 530, 320]
    }
  ]
}
```

---

## 🔮 Future Work

- [ ] Add confidence threshold slider to Streamlit UI
- [ ] Support video / webcam stream detection
- [ ] Retrain both models on the same dataset split for a fair comparison
- [ ] Export models to ONNX / TFLite for true edge deployment
- [ ] Deploy FastAPI backend to cloud (Render / Railway / HuggingFace Spaces)
- [ ] Add per-class detection count dashboard

---

## 👤 Author

**Nour Alaa**

[![GitHub](https://img.shields.io/badge/GitHub-nouralaa311-181717?style=flat-square&logo=github)](https://github.com/nouralaa311)

---

## 📄 Dataset License

Dataset sourced via Roboflow. See `README.dataset.txt` and `README.roboflow.txt` for full dataset terms and attribution.

---

<div align="center">
<sub>Built with ❤️ and 🐝 | YOLOv8 · YOLO11 · FastAPI · Streamlit</sub>
</div>
