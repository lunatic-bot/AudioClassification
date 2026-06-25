# Audio Classification using UrbanSound8K

## Overview

This project focuses on **Environmental Sound Classification** using the **UrbanSound8K dataset**. The goal is to classify audio recordings into predefined sound categories by extracting audio features and training a deep learning model.

The project was developed using **Jupyter Notebooks** and covers the complete machine learning workflow, including:

* Exploratory Data Analysis (EDA)
* Audio Data Preprocessing
* Feature Extraction
* Model Training
* Model Evaluation
* Sound Prediction on New Audio Samples

---

## Dataset

This project uses the **UrbanSound8K Dataset**:

Dataset Link: https://urbansounddataset.weebly.com/urbansound8k.html

### About the Dataset

UrbanSound8K is a dataset of urban environmental sounds containing:

* **8,732 labeled audio clips**
* Audio clips of duration **up to 4 seconds**
* Organized into **10 folds** for cross-validation
* **10 sound classes**

### Classes

| Class ID | Sound Category   |
| -------- | ---------------- |
| 0        | Air Conditioner  |
| 1        | Car Horn         |
| 2        | Children Playing |
| 3        | Dog Bark         |
| 4        | Drilling         |
| 5        | Engine Idling    |
| 6        | Gun Shot         |
| 7        | Jackhammer       |
| 8        | Siren            |
| 9        | Street Music     |

### Dataset Structure

```text
UrbanSound8K/
│
├── audio/
│   ├── fold1/
│   ├── fold2/
│   ├── ...
│   └── fold10/
│
└── metadata/
    └── UrbanSound8K.csv
```

The metadata file contains information such as:

* File name
* Fold number
* Class ID
* Class name
* Audio slice information

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand:

* Class distribution
* Dataset balance
* Audio duration statistics
* Sample audio visualization
* Waveform analysis

---

### 2. Audio Preprocessing

The audio files were processed using the **Librosa** library.

Preprocessing steps included:

* Loading audio files
* Resampling audio
* Noise handling
* Feature extraction preparation

---

### 3. Feature Extraction

Mel-Frequency Cepstral Coefficients (**MFCCs**) were extracted from each audio file.

MFCCs are widely used in audio and speech processing because they effectively capture the perceptually important characteristics of sound.

Example:

```python
mfccs = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)
```

The extracted MFCC features were converted into fixed-length feature vectors for model training.

---

### 4. Model Training

A Deep Neural Network (DNN) was built using TensorFlow/Keras.

Typical architecture:

* Dense Layers
* ReLU Activation
* Dropout Layers
* Softmax Output Layer

The model was trained on extracted MFCC features and corresponding class labels.

---

### 5. Model Evaluation

The trained model was evaluated using:

* Training Accuracy
* Validation Accuracy
* Loss Curves
* Test Predictions

Performance metrics were used to assess the model's ability to classify environmental sounds accurately.

---

### 6. Audio Prediction

The trained model can classify unseen audio clips into one of the ten UrbanSound8K classes.

Prediction pipeline:

```text
Audio File
    ↓
Feature Extraction (MFCC)
    ↓
Model Prediction
    ↓
Sound Class
```

---

## Technologies Used

* Python
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Librosa
* TensorFlow / Keras
* Scikit-Learn

---

## Results

The model successfully learns audio patterns from urban environmental sounds and can classify unseen audio clips into their respective categories.

Model performance may vary depending on:

* Feature extraction strategy
* Network architecture
* Hyperparameter tuning
* Train-test split

---

## Future Improvements

* CNN-based Audio Classification
* Spectrogram-based Classification
* Data Augmentation
* Transfer Learning
* Real-time Audio Classification
* Model Deployment using FastAPI or Streamlit

---

## Acknowledgements

* UrbanSound8K Dataset
* Librosa Audio Processing Library
* TensorFlow/Keras
* Scikit-Learn
* Jupyter Notebook

---
