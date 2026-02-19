# Radar & AI for Hand Gesture Classification (HCI)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omarbenomar/Radar_AI_For_Hand_Gesture_Classification/blob/main/Final%20Project%20-%20Omar%20Ben%20Omar%20(6564062)%20-%20Pauline%20De%20Baets%20(6544819).ipynb)

## Project Overview
This project implements a complete machine learning and signal processing pipeline to classify human hand gestures using **micro-Doppler radar signatures**. Developed as part of the Machine Learning for Electrical Engineering course at **TU Delft**, the system utilizes the **DopNet dataset** to recognize four distinct gestures: **Wave, Pinch, Swipe, and Click**.

The project explores the transition from raw radar data to high-dimensional feature vectors, comparing classical statistical methods with modern deep learning architectures.

---

## Research Foundations & Methodology
The technical approach is grounded in two primary research papers providing the framework for feature extraction and signature analysis:

### 1. Discrete Chebyshev Moments
*Reference: "Classification of micro-Doppler radar hand-gesture signatures by means of Chebyshev moments" (Pallotta et al.)*
We implement **Discrete Chebyshev Moments** extracted from the **Cadence Velocity Diagram (CVD)**. Unlike continuous moments (like Zernike), Chebyshev moments are defined on a discrete set, eliminating discretization errors and providing a highly efficient, symmetrical feature set. This allows us to reduce a complex spectrogram into a compact 1D feature vector that captures the repetitive motion cycles of a gesture.

### 2. Micro-Doppler Envelopes
*Reference: "Hand Gesture Recognition based on Radar Micro-Doppler Signature Envelopes" (Amin et al.)*
This methodology focuses on the **positive and negative frequency envelopes** of the micro-Doppler signature. These envelopes represent the kinetic boundaries of the hand's motion toward and away from the radar. By analyzing these boundaries, the models can effectively distinguish between gestures with similar frequency content but different temporal "shapes," such as a Click versus a Pinch.

---

## Technical Pipeline

### 1. Signal Processing & Feature Extraction
* **Time-Frequency Analysis:** Transformation of raw radar returns into spectrograms via Short-Time Fourier Transform (STFT).
* **CVD Computation:** Moduli of the spectrograms are used to generate the Cadence Velocity Diagram to isolate periodic components.
* **Mathematical Moments:** Implementation of discrete Chebyshev polynomials to extract high-order moments (up to order 20) as optimized feature vectors.

### 2. Classification Models
* **Classical Machine Learning:** Implementation of **Logistic Regression** with **L1 (Lasso)** and **L2 (Ridge)** regularization to evaluate feature importance and baseline accuracy.
* **Deep Learning (1D CNN):** Architected a **Convolutional Neural Network** using **TensorFlow/Keras**. The model utilizes 1D convolutions, MaxPooling, and Dropout layers to learn spatial hierarchies in the moment-based features while ensuring robust generalization across different users in the dataset.

---

## Project Structure
* `Final Project - Omar Ben Omar (6564062) - Pauline De Baets (6544819).ipynb` — The primary notebook containing data preprocessing, mathematical function definitions, model training, and performance evaluation.
* `Project_Description_EE4C12_WICOS_2025.pdf` — The formal project requirements and context provided by TU Delft.
* `Reference Papers/` — Detailed research papers by Pallotta et al. and Amin et al. regarding Chebyshev moments and micro-Doppler envelopes.

---

## Getting Started
To view or run the notebooks, ensure you have Jupyter and the required Python libraries installed:
```bash
pip install jupyter numpy pandas matplotlib scikit-learn
jupyter notebook
```

## Contributors
* Omar Ben Omar 
* Pauline De Baets

#
