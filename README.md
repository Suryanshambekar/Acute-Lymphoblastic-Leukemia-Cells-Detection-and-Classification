# Acute Lymphoblastic Leukemia Cells Detection and Classification

This project implements a deep learning-based system for detecting and classifying Acute Lymphoblastic Leukemia (ALL) cells in microscopic images using the YOLOv8 object detection model. It provides a Streamlit web interface for uploading images and visualizing detected leukemia cells with their classification and confidence scores.

---

## Features

- Detects leukemia cells in input images using a trained YOLOv8 model.
- Classifies detected cells into four classes: Benign, Early, Pre, and Pro.
- Displays bounding boxes and confidence scores on detected cells.
- Provides average confidence and final classification based on the most common detected class.
- Training pipeline included for model training, evaluation, and visualization of metrics.
- Saves inference results to CSV for further analysis.

---
## Deployment

The project is deployed on:
- **Streamlit Cloud**: [https://acute-lymphoblastic-leukemia-cells-detection-and-classification.streamlit.app](https://acute-lymphoblastic-leukemia-cells-detection-and-classification.streamlit.app)
  
## Installation

### Prerequisites

- Python 3.7 or higher
- GPU recommended for training and inference

### Install dependencies

```bash
pip install -r requirements.txt
```

The main dependencies include:

- ultralytics (YOLOv8)
- roboflow (dataset management)
- opencv-python
- matplotlib
- pandas
- seaborn
- scikit-learn
- kagglehub
- Pillow
- streamlit
- torch

Additionally, install system package:

```bash
sudo apt-get install libgl1
```

---

## Usage

### Run the Streamlit app for inference

```bash
streamlit run app.py
```

Upload an image through the web interface to detect and classify leukemia cells.

### Train the model

The training pipeline is provided in `dsip_project.py` (originally a Jupyter notebook). It includes dataset download, model training, evaluation, and visualization.

Run the script or notebook to train the YOLOv8 model on the leukemia dataset.

---

## Model Details

- Model: YOLOv8 (Ultralytics)
- Classes: 4 (Benign, Early, Pre, Pro)
- Trained weights saved in `runs/detect/train/weights/best.pt`

---

## Dataset

The dataset is managed via Roboflow and Kagglehub, containing labeled images of leukemia cells categorized into the four classes.

---

## Results and Evaluation

- Metrics such as Precision, Recall, mAP@50, and mAP@50-95 are calculated and visualized.
- Confusion matrix heatmap is generated to analyze classification performance.
- Sample inference images with bounding boxes are saved and displayed.

---

## Folder Structure

```
Acute-Lymphoblastic-Leukemia-Cells-Detection-and-Classification/
├── app.py                      # Streamlit app for inference
├── dsip_project.py             # Training and evaluation pipeline
├── packages.txt                # System package dependencies
├── requirements.txt            # Python dependencies
├── runs/                       # Training and inference outputs
│   └── detect/
│       └── train/
│           └── weights/        # Trained model weights
└── README.md                   # This file
```


## Contact

For questions or support, please contact Suryansh Ambekar suryanshambekar@gmail.com
