# Military and Civilian Vehicles Classification

## Overview

This repository contains code and data organization for detecting and classifying military and civilian vehicles using YOLOv5. The project includes dataset split utilities and a copy of YOLOv5 for training and inference.

## Features

- Prepare dataset in YOLO format (split train / val)
- YOLOv5 training and inference (uses the `yolov5/` folder)
- Image / video detection using YOLOv5 scripts

## Tech Stack

- Python
- PyTorch (via YOLOv5)
- YOLOv5
- pandas (dataset utilities)

## Project Structure

Below are the main folders after cleanup:

```
.
├── Military and Civilian Vehicles Classification/   # dataset and split script
│   ├── Images/
│   ├── Labels/
│   └── split_dataset.py
├── yolov5/                                         # YOLOv5 code (vendor copy)
├── custom_data.yaml                                # YOLOv5 dataset config
├── README.md
├── requirements.txt                                # minimal requirements for dataset script
└── .gitignore
```

## Dataset

The dataset used to develop this project is not included in this repository. Source: https://data.mendeley.com/datasets/njdjkbxdpn/1

Folder layout expected under `Military and Civilian Vehicles Classification/`:

- `Images/` - original images (JPEG/PNG)
- `Labels/` - labels in multiple formats (CSV, TXT, XML). TXT labels should be in YOLO format for training.

Do not commit full datasets to GitHub. Place your dataset outside the repo or follow the instructions below to prepare a small sample.

## Dataset preparation

From the project root run:

```
cd "Military and Civilian Vehicles Classification"
python split_dataset.py
```

This copies images and TXT label files into `train_data/images/(train|val)` and `train_data/labels/(train|val)` using the CSV files in `Labels/CSV Format/`.

## Training

Training uses the YOLOv5 code in the `yolov5/` folder. Install YOLOv5 dependencies and run the training script from the repository root or inside `yolov5/`:

```
pip install -r yolov5/requirements.txt
python yolov5/train.py --data custom_data.yaml --img 640 --batch 16 --epochs 50
```

Notes:

- Do not change model training logic. Adjust `--img`, `--batch`, `--epochs`, and `--weights` according to your setup and GPU memory.
- If you prefer, run training from inside the `yolov5/` directory.

## Inference

Use the yolov5 detect script for image or video inference:

```
python yolov5/detect.py --weights path/to/best.pt --source path/to/image_or_video
```

## Results

Training artifacts (runs, checkpoints) are excluded from the repository (`.gitignore`). Keep a small `assets/` folder with representative, compressed example images if you want to display results in this README.

## Installation (quick)

```
git clone <repository-url>
cd <repository-name>
pip install -r yolov5/requirements.txt
pip install -r requirements.txt
```

## Future improvements

- Add evaluation scripts and a small `assets/` folder with demo images
- Improve label quality and augmentations
- Export a lightweight ONNX/TensorRT model for deployment

---

Project prepared for sharing. See the repository root for `.gitignore` and `yolov5/` (vendor code).
