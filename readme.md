# Pneumonia Detection using Deep Learning

An end-to-end deep learning system for detecting pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs) and Transfer Learning. This project focuses on **model comparison, safety-critical evaluation, and real-world deployment**.

---

## Project Highlights

* Built and compared **5 different models**:

  * Custom CNN
  * Transfer Learning:

    * VGG16
    * ResNet50
    * MobileNetV2
    * EfficientNetB0

* Designed with a **medical-first mindset**:

  * Prioritized **Recall (no missed pneumonia cases)**
  * Analyzed **False Negatives (FN) and False Positives (FP)**

* Deployed using **Streamlit** with:

  * Real-time predictions
  * Confidence scores
  * Human-like AI responses (LLM-style UX)

---

## Problem Statement

Pneumonia is a potentially life-threatening condition that requires early detection.
This project aims to build a model that:

* Accurately detects pneumonia from X-rays
* Minimizes **false negatives (critical in healthcare)**
* Is efficient enough for real-time use

---

## 🏗️ Models Explored

| Model          | Type              | Notes                               |
| -------------- | ----------------- | ----------------------------------- |
| Custom CNN     | Scratch           | Baseline model                      |
| VGG16          | Transfer Learning | High accuracy, heavy                |
| ResNet50       | Transfer Learning | Deep & stable                       |
| MobileNetV2    | Transfer Learning | Lightweight, fast                   |
| EfficientNetB0 | Transfer Learning | Best performance-efficiency balance |

---

## 📊 Model Comparison

![alt text](model_accuracy.png.png "Title")

---

## 🏆 Final Model Selection

**EfficientNetB0** was selected as the final model.

### ✅ Why EfficientNet?

* **0 False Negatives** → No missed pneumonia cases
* **Highest Accuracy (96%)**
* **Low False Positives (1)**
* **Efficient inference time (80 ms)**

> The model achieves the best trade-off between **safety, performance, and efficiency**, making it suitable for real-world deployment.

---

## 📈 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall (**primary metric**)
* F1-score
* Confusion Matrix (TP, FP, FN, TN)
* Prediction Time (Latency)

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib
* Streamlit

---

## 🖥️ Application (Streamlit UI)

Features:

* Upload chest X-ray image
* Real-time prediction
* Confidence score display
* AI-generated interpretation text
* Clean and minimal UI

---

---

## ▶️ How to Run

### 1. Clone repo

```bash
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run app

```bash
streamlit run app.py
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It is **not a substitute for professional medical diagnosis**.

---

## 🔮 Future Improvements

* Grad-CAM visualization for interpretability
* Larger and more diverse dataset
* API deployment (FastAPI / Docker)
* Model quantization for edge devices

---

## 💡 Key Takeaway

> This project goes beyond model building by focusing on **real-world constraints, safety-critical evaluation, and deployment readiness**.

---

## 📬 Connect

If you found this interesting or have suggestions, feel free to connect!

---
