# Road Damage Detection System (UAV/Drone Imagery)

## 📌 Project Overview
This project is an advanced **AI-powered Road Inspection System** designed to detect road surface damage (cracks, potholes, deterioration) from UAV/Drone imagery or standard camera feeds. 

It utilizes a **Hybrid Detection Engine** combining Deep Learning (YOLOv8) with **Adaptive Computer Vision** algorithms to ensure high accuracy in diverse environments, specifically filtering out complex backgrounds like forests and bridges.

---

## 🚀 How to Run

### 1. Prerequisites
- Python 3.8+
- CUDA (Optional, for GPU acceleration)

### 2. Installation
1.  **Clone/Download** this repository.
2.  Navigate to the project folder:
    ```bash
    cd Roaddamgedetection
    ```
3.  **Install Dependencies**:
    Double-click `install_dependencies.bat` OR run:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Start the Application
Run the following command in your terminal:
```bash
uvicorn main:app --reload
```
*Note: If you get a "Forbidden" or "Socket" error, make sure no other application is using port 8000.*

### 4. Use the System
1.  Open your browser and go to **[http://127.0.0.1:8000].
2.  **Upload** an image of a road.
3.  The system will automatically analyze and display the results.

---

## 🏗️ Model Architecture

The system uses a **Hybrid Architecture** to maximize robustness.

```mermaid
graph TD
    A[Input Image] --> B{Pre-Processing}
    B --> C[YOLOv8 Object Detection]
    B --> D[Adaptive Road Segmentation]
    
    C -- "Detects" --> E[Vehicles / People]
    D -- "Generates" --> F[Road ROI Mask]
    
    E --> G{Exclusion Logic}
    F --> G
    
    G -- "Filtered Region" --> H[Texture Analysis Engine]
    H --> I[Edge Density Scan]
    I --> J{Damage Threshold}
    
    J -- "High Density" --> K[Damage Detected (Red Grid)]
    J -- "Low Density" --> L[Safe Surface (Green Grid)]
    
    K --> M[Final Result Overlay]
    L --> M
```

### Key Components:
1.  **YOLOv8 (Ultralytics)**:
    *   Used as a "Negative Filter". It detects non-road objects (Cars, Trucks, Pedestrians) so the system knows *what to ignore*.
2.  **Adaptive Road Segmentation (Auto-Calibrated)**:
    *   Rather than hardcoding colors, the system **samples** the road pixels from the bottom-center of the image (statistically almost always the road).
    *   It builds a dynamic color profile (Means & Variance) for *that specific image*.
    *   It expands a mask to cover the connected asphalt surface, automatically stopping at grass, trees, or sky.
3.  **Texture Analysis**:
    *   The masked road area is scanned for "Texture Energy" (using Canny Edge Density).
    *   Smooth roads have low energy. Cracks and Potholes have high energy.

---

## 🔄 Data Flow Pipeline

1.  **Client Layer (Frontend)**:
    *   User uploads image via `index.html`.
    *   Image is sent to `/predict` endpoint via `FormData`.

2.  **Server Layer (FastAPI)**:
    *   `main.py` receives the file and saves it to `static/uploads`.

3.  **Processing Layer (Logic)**:
    *   **Step 1: Calibration**: `get_adaptive_road_mask()` samples the ROI.
    *   **Step 2: Masking**: HSV Thresholding applied to isolate road from foliage.
    *   **Step 3: Object Avoidance**: YOLO boxes are checked; if an area is a "Car", it is skipped.
    *   **Step 4: Grid Scan**: The image is divided into a 5x5 grid. Each cell is analyzed for damage texture.

4.  **Presentation Layer**:
    *   Processed image is saved with a Green/Red grid overlay.
    *   `result.html` renders the original vs. processed image with a status banner (Safe/Damage).

---

## 📂 Project Structure
```
Roaddamgedetection/
│
├── main.py              # Core Backend (FastAPI + Logic)
├── requirements.txt     # Dependencies
├── install_dependencies # Setup Scripts
│
├── static/              # Static assets
│   └── uploads/         # Temp storage for analyzed images
│
├── templates/           # Frontend HTML
│   ├── index.html       # Upload Interface
│   └── result.html      # Analysis Dashboard
│
└── dataset/             # (Optional) Training data
```
