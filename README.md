# 🦺 Construction Site Safety Monitor

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-construction--safety.streamlit.app-ff4b4b?style=for-the-badge&logo=streamlit)](https://construction-safety.streamlit.app/)

A real-time **PPE (Personal Protective Equipment) violation detection** system for construction sites, built with YOLOv11 and Streamlit. The system detects workers missing hardhats, safety vests, and masks — and generates downloadable violation reports.

> 🌐 **Try it live:** [https://construction-safety.streamlit.app](https://construction-safety.streamlit.app/)

---

## 📸 Features

- 🎥 **Video Upload & Analysis** — Upload any site footage and run automated safety checks
- 🧠 **YOLOv11-powered Detection** — Trained on 10 construction safety classes
- 📊 **Live Dashboard** — Real-time frame preview with violation counters
- 📥 **Downloadable Reports** — Export violation summary as CSV and annotated video
- 🌑 **Dark-themed UI** — Clean, professional Streamlit interface

---

## 🏷️ Detection Classes

| Class | Type |
|-------|------|
| Hardhat | ✅ Compliant |
| NO-Hardhat | ❌ Violation |
| Mask | ✅ Compliant |
| NO-Mask | ❌ Violation |
| Safety Vest | ✅ Compliant |
| NO-Safety Vest | ❌ Violation |
| Person | 👷 Person |
| Safety Cone | 🔶 Object |
| Machinery | 🏗️ Object |
| Vehicle | 🚛 Object |

---

## 🗂️ Project Structure

```
CONSTRUCTION/
│
├── dashboard.py          # Streamlit web app (main UI)
├── train.py              # YOLOv11 model training script
├── video_test.py         # Standalone video inference + CSV report
├── model_test.py         # Quick model inference test
├── data.yaml             # Dataset configuration (classes & paths)
│
├── DATA/                 # Dataset (not included — see below)
│   ├── train/
│   ├── valid/
│   └── test/
│
└── runs/                 # Training outputs (not included — auto-generated)
    └── detect/
        └── train-4/
            └── weights/
                └── best.pt   ← your trained model weights
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/dannygeorge2026-coder/construction-safety-detection.git
cd construction-safety-detection
```

### 2. Install Dependencies

```bash
pip install ultralytics streamlit opencv-python
```

### 3. Get the Dataset

Download the dataset from [Roboflow](https://roboflow.com) or [Kaggle](https://kaggle.com) and place it in the `DATA/` folder matching the structure in `data.yaml`.

### 4. Train the Model

```bash
python train.py
```

This will train for **50 epochs** on your dataset and save weights to `runs/detect/train/weights/best.pt`.

### 5. Run the Dashboard

```bash
streamlit run dashboard.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎬 Usage

1. Open the Streamlit dashboard
2. Set the model weights path in the **sidebar** (default: `models/best.pt`)
3. Adjust **confidence threshold** as needed
4. Upload a construction site video (`.mp4`, `.avi`, `.mov`, `.mkv`)
5. Click **▶ Run Analysis**
6. Download the **CSV report** or **annotated video**

---

## 📦 Standalone Video Inference (No UI)

```bash
python video_test.py
```

This processes `worker_safety.mp4`, shows a live window, saves `output.mp4`, and writes `violation_report.csv`.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) | Object detection |
| [Streamlit](https://streamlit.io) | Web dashboard |
| [OpenCV](https://opencv.org) | Video processing |
| Python 3.10+ | Core language |

---

## 👤 Author

**Danny George**  
GitHub: [@dannygeorge2026-coder](https://github.com/dannygeorge2026-coder)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
