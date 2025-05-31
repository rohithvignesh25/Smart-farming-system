# 🌾 Smart Agriculture System using Machine Learning

This project aims to build a smart agriculture system that leverages machine learning models to provide:
- 📋 Crop recommendations
- 🧪 Fertilizer suggestions
- 🌿 Plant disease detection from images

## 📁 Project Structure

```
├── Crop.py                         # SVM model for crop recommendation
├── Fertilizer.py                  # Random Forest model for fertilizer recommendation
├── disease.py                     # CNN model for disease classification
├── Combined_Crop_Recommendation.csv
├── f2.csv
├── crop_recommendation_model.pkl
├── crop_encoder.pkl
├── fertilizer_recommendation_model.pkl
├── fertilizer_encoder.pkl
├── plant_disease_cnn_model.h5
```

---

## 🌱 1. Crop Recommendation System

- **Script**: `Crop.py`
- **Model**: Support Vector Machine (SVM)
- **Dataset**: `Combined_Crop_Recommendation.csv`
- **Input Features**:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - pH
  - Rainfall
- **Output**: Most suitable crop to grow
- **Preprocessing**:
  - StandardScaler for normalization
  - LabelEncoder for encoding crop labels
- **Model Evaluation**:
  - F1 Score, Accuracy
  - Confusion Matrix
- **Saved Artifacts**:
  - `crop_recommendation_model.pkl`
  - `crop_encoder.pkl` (Label encoder)
  - `scaler.pkl`

---

## 💊 2. Fertilizer Recommendation System

- **Script**: `Fertilizer.py`
- **Model**: Random Forest Classifier
- **Dataset**: `f2.csv`
- **Input Features**:
  - Soil Type
  - Crop Type
  - Moisture
  - Nitrogen
  - Phosphorus
  - Potassium
  - Temperature
  - Humidity
- **Output**: Best fertilizer to use
- **Preprocessing**:
  - Label Encoding for categorical variables
  - MinMaxScaler for feature scaling
- **Model Evaluation**:
  - Accuracy, F1 Score
  - Confusion Matrix, ROC Curve (multi-class)
- **Saved Artifacts**:
  - `fertilizer_recommendation_model.pkl`
  - `fertilizer_encoder.pkl`, `soil_encoder.pkl`, `crop_encoder.pkl`
  - `fertilizer_scaler.pkl`

---

## 🌿 3. Plant Disease Detection

- **Script**: `disease.py`
- **Model**: Convolutional Neural Network (CNN)
- **Input**: Leaf images
- **Dataset Directory**: `new/` (contains subfolders for each disease class)
- **Model Details**:
  - 3 convolutional + max-pooling layers
  - Dense + Dropout layers
  - Output layer with `softmax` activation
- **Preprocessing**:
  - ImageDataGenerator with augmentation (rotation, flip, zoom)
- **Training**:
  - Trained for 10 epochs with validation
- **Output**: Disease class prediction
- **Saved Model**:
  - `plant_disease_cnn_model.h5`

---

## 🛠️ Requirements

Install the required libraries using:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:
```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
tensorflow
```

---

## 📌 Future Improvements

- Integrate all models into a unified web dashboard.
- Add real-time inference using webcam or sensors.
- Optimize model performance on edge devices.

---
