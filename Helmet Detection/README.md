Helmet Violation Detection System using YOLOv8 

This project detects **two-wheeler riders without helmets** from a video stream using **YOLOv8 object detection**, plays an **alert sound**, and **saves violation images automatically** for reporting or legal evidence.

---

## 📌 Project Overview

Road safety violations, especially riding without helmets, contribute heavily to road accident deaths.  
This AI-based system helps traffic authorities **monitor CCTV/video footage automatically** and flag **helmet violators in real time**.

✅ Detects riders with or without helmets  
✅ Draws **green** box for "With Helmet" & **red** box for "Without Helmet"  
✅ Plays alert sound when violation occurs  
✅ Saves violator frame as image in `/violations` folder  
✅ Works on recorded video or live webcam feed  

---

## 🔥 Features

| Feature | Description |
|---------|-------------|
| 🎯 YOLOv8 Object Detection | Trained helmet classification model |
| 🔊 Real-time Alert System | Plays `alert.wav` when violation detected |
| 🖼️ Auto Screenshot Capture | Saves non-helmet frames with timestamp |
| 🎥 Video/Camera Support | Works with `.mp4` file or webcam (`0`) |
| 🧠 Fast Inference | Optimized for real-time performance |
| 🛠️ Easy to Run | Only Python + OpenCV + Ultralytics |

---

## 🧰 Tech Stack

| Component | Library / Framework |
|-----------|--------------------|
| Programming Language | Python |
| Model | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV |
| ML Backend | PyTorch |
| Audio Alert | playsound |

---

## 📂 Folder Structure
Helmet-Detection/
│── main.py
│── best.pt # YOLOv8 trained model
│── alert.wav # Alert sound for violations
│── Traffic1.mp4 # Sample input video
│── requirements.txt
│── README.md
│── /violations/ # Auto-saved violation images
│── /samples/ # (optional) demo screenshots