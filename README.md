# 🅿️ Smart Parking & Surveillance AI Model Challenge

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Computer Vision](https://img.shields.io/badge/Task-Computer%20Vision-lightblue)



## 🚗 A Computer Vision-Based Solution for Vehicle Detection, Edge Deployment & Human Action Recognition

This repository contains the solution for the **Smart Parking & Surveillance AI Model Challenge**, which involves developing a deep learning-based system for:

- **Vehicle detection in parking spaces along with the available and occupied parking place counts**
- **Real-time deployment on edge devices(Mobile)**
- **(Bonus) Human action detection using YOLO(Fall Detection)**

The project is organized into three tasks, each in its own folder, and follows the challenge requirements closely to ensure a clear, functional, and reproducible solution. The goal is to showcase AI's capability in real-time surveillance and smart city infrastructure, particularly in improving parking management and safety monitoring.

---
---

## 📁 Project Structure

```bash

Smart-Parking-Surveillance-AI-Model/
├── Vehicle_Detection_in_Parking_Spaces/
│   ├── best.pt
│   ├── bounding_boxes.json
│   ├── model_training.ipynb
│   ├── ParkingArea_Mapping.py
│   ├── ParkingManagement_usingCV.py
│   ├── ParkingManagement_usingUltralytics.py
│   └── VIDEOS.md
│
├── Task_2_Deploymnet/
│   ├── converting_into_tflite.ipynb
│   ├── android_app/
│   ├── best.onnx
│   ├── best_float32.tflite
│   └── best.pt
│
├── FallDetection_Folder/
│   ├── best.pt
│   ├── fall_detection.ipynb
│   └── VIDEOS.md
│
├── requirements.txt
├── .gitignore
├── README.md
├── Smart_Parking_AI_Model_Report.docx
└── LICENSE


```

---

---

## 🧠 Task 1: Vehicle Detection in Parking Spaces

### 📁 Folder: `Vehicle_Detection_in_Parking_Spaces`

This folder contains all the necessary files and scripts related to **Task 1**, which aims to detect the number of vehicles parked and the available parking slots using a custom-trained YOLOv8 model.

---

### 📄 Contents:

- `best.pt`  
  Trained YOLOv8s model file. The model was trained using a combination of the **CARPK dataset** and a custom-created dataset featuring **top-down (eagle view) vehicle images** to improve detection accuracy in parking lots.

  📌 **Note**: The vehicle detection model was trained using a combination of the publicly available [CARPK dataset](https://paperswithcode.com/dataset/carpk) and a custom dataset created with top-down images of vehicles. Please ensure appropriate attribution when reusing or redistributing the data.


- `bounding_boxes.json`  
  Contains the predefined parking slot coordinates. These were manually marked using the **Ultralytics Parking Solution UI**, which allows you to define parking spaces and export them in JSON format.

- `model_training.ipynb`  
  Jupyter notebook that includes the complete code for training the YOLOv8s model on the combined dataset.

- `ParkingArea_Mapping.py`  
  A script to launch the Ultralytics-provided UI for mapping and marking parking areas. The JSON file generated from this UI (`bounding_boxes.json`) is used during detection.

- `ParkingManagement_usingCV.py`  
  Script to detect parked vehicles and available slots on both images and videos using OpenCV method *pointpolygontest* and the trained YOLO model (`best.pt`).

- `ParkingManagement_usingUltralytics.py`  
  Alternative implementation using **Ultralytics' official Parking Management System** instead of OpenCV, for streamlined deployment and visualization.

- `VIDEOS.md`  
  Contains Google Drive links to the demo videos for:
  - Input video used for testing
  - Output from the OpenCV-based solution
  - Output from the Ultralytics-based system

---

---

## 📱 Task 2: Deployment Demonstration on Edge Devices

### 📁 Folder: `Task_2_Deploymnet`

This folder contains the code and relevant files for converting and deploying the trained YOLOv8 model (`best.pt`) on edge devices such as mobile phones.

> **Note:** We used an open-source [GitHub repository](https://github.com/surendramaran/YOLOv8-TfLite-Object-Detector) that provided an `android_app` structure for developing and running the TFLite model inside an Android app using Android Studio.

---

### 📄 Contents:

- `converting_into_tflite.ipynb`  
  Jupyter notebook demonstrating the process of converting the `best.pt` model to a **TensorFlow Lite (.tflite)** format, which is optimized for mobile and other lightweight edge devices.

- `android_app/`  
  Android Studio project folder used to build the mobile app that integrates and runs the YOLOv8 model.

- `best.pt`, `best_float32.tflite`, `best.onnx`  
  Model files generated and used during deployment.

---

### 🚀 Deployment Process & Tools Used

Below are the steps and technologies used to deploy the model on an edge device (e.g., Android mobile):

#### 1. **Model Conversion**  
- The YOLOv8 model (`best.pt`) was first converted to ONNX format (`best.onnx`) and then to TensorFlow Lite format (`best_float32.tflite`) for use in mobile deployment.

#### 2. **Integration into Mobile App** 
- We utilized the provided Android app template and integrated the converted model into the app.
- The model was placed in the `assets` directory, and paths to the model and labels were updated in the `Constants.kt` file.

#### 3. **Tools & Frameworks**
- Ultralytics (for model training and conversion)
- ONNX (intermediate format)
- TensorFlow Lite (for edge optimization)
- Android Studio (for app development)
- Kotlin (for app codebase)

---

---

## 🧍‍♂️ Task 3 (Bonus): YOLO Extension for Action Detection

### 📁 Folder: `FallDetection_Folder`

This folder contains the implementation for the optional bonus task — using YOLOv8 to detect specific human actions, specifically **fall detection**, which can be useful for safety monitoring in surveillance systems.

---

### 📄 Contents:

- `fall_detection.ipynb`  
  Jupyter notebook showing the process of training the **YOLOv8s** model on a **fall detection dataset**.  
  - The dataset was obtained from **Roboflow** using a direct link and downloaded in runtime in **Google Colab**.  
  - The model was trained and tested on both **images** and **videos** to validate performance.

- `best.pt`  
  The trained YOLOv8s model (`best.pt`) that was trained on the fall detection dataset.

  📌 **Note**: The fall detection model was trained using a publicly available dataset from Roboflow. You can access the dataset here: [Fall Detection Dataset (YOLOv8 format)](https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8/dataset/4/download/yolov8). Please ensure appropriate attribution when reusing or modifying this dataset.


- `VIDEOS.md`  
  Contains Google Drive links to demo videos showing:
  - Input test videos
  - Output videos of the trained fall detection model in action

---
---

---
## 👤 Author

**Babaji Byri**  
📧 [byribabajimudhiraj@gmail.com]  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/babajimudhiraj)  
🎥 [Project Videos](https://drive.google.com/drive/folders/1oBzPA0mDvXiUHEdRKVwh5OfDmyo0hqAW)

---