# 🚧 AI BASED POTHOLE DETECTOR

This project is a **Streamlit web application** that detects potholes in road images using a **Convolutional Neural Network (CNN)**.
It also highlights potholes on the uploaded image and estimates the **CO₂ savings** if those potholes were fixed, promoting sustainability.

---

## ✨ Features

* ✅ **Train a CNN model** on road images (Plain vs. Pothole).
* ✅ **Handles imbalanced datasets** using computed class weights.
* ✅ **Evaluates model performance** with Classification Report & Confusion Matrix.
* ✅ **Upload and predict** on new road images.
* ✅ **Pothole localization** with bounding boxes drawn on the image.
* ✅ **Sustainability impact estimation** (CO₂ savings if potholes are repaired).
* ✅ **Export sustainability report** as CSV.

---

## 📂 Dataset Structure

The dataset must be organized in the following directory format:
```
My Dataset/
│── train/
│   ├── Plain/
│   ├── Pothole/
│
│── test/
│   ├── Plain/
│   ├── Pothole/

```

Images are available in jpeg/png/jpg format.
---

## ⚙️ Installation

### 1️⃣ Clone Repository

git clone https://github.com/your-username/pothole-detector.git
cd pothole-detector


### 2️⃣ Create Virtual Environment

python -m venv venv
source venv/bin/activate
venv\Scripts\activate

### 3️⃣ Install Dependencies

pip install -r requirements.txt


### 4️⃣ Setup Environment Variables

Create a `.env` file in the project root:

DATASET_PATH=path/to/dataset

---

## ▶️ Run the App

streamlit run app.py

---

## 📊 Model Training

* Go to the sidebar and click **"Train / Re-train Model"**.
* The model will:

  * Load dataset
  * Apply data augmentation
  * Compute class weights for imbalance handling
  * Train CNN with early stopping & checkpointing
* The best model is saved at:

models/pothole_cnn_model_v3.keras

---

## 🔍 Prediction

* Upload a new road image (`.jpg`, `.jpeg`, `.png`).
* The app will:

  * Classify as **Plain** or **Pothole**
  * Draw bounding boxes around detected potholes
  * Estimate **CO₂ savings** if potholes are repaired
  * Optionally export a sustainability report

---

## 📈 Evaluation

* **Classification Report** (Precision, Recall, F1-Score, Accuracy)
* **Confusion Matrix**
* **Confidence Scores** for predictions

---

## 📦 Dependencies

* Python 3.8+
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas
* scikit-learn
* Streamlit
* python-dotenv

Install via:

pip install tensorflow opencv-python-headless numpy pandas scikit-learn streamlit python-dotenv

---

## 🌍 Sustainability Impact

* Assumption: Fixing **1 pothole saves \~5 kg CO₂/year per 1000 vehicles**.
* Helps cities & municipalities prioritize road repairs for **eco-friendly transport infrastructure**.

---

## 📌 Future Improvements

* Multi-class classification (severity levels of potholes).
* Integration with **real-time video streams**.
* GPS tagging of potholes for **smart city applications**.
* Dashboard with **trend analysis & heatmaps**.

---

## 🤝 Contributing

1. Fork the repo
2. Create a new branch (`feature-xyz`)
3. Commit changes
4. Push to branch
5. Open Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.

---
