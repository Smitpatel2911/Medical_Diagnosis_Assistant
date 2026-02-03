# 🩺 Medical Diagnosis Assistant: Heart Disease Prediction

This repository contains a machine learning pipeline designed to predict the presence of heart disease using clinical patient data. The project spans the full ML lifecycle: from **Exploratory Data Analysis (EDA)** and **Feature Engineering** to **Model Evaluation** and **Serialization**.

## 📊 Project Highlights

* **Targeted Analysis:** Focuses on key indicators like `cp` (Chest Pain Type), `thalach` (Max Heart Rate), and `chol` (Cholesterol).
* **Preprocessing Pipeline:** Includes handling of categorical variables via One-Hot Encoding and feature scaling using `StandardScaler`.
* **Model Zoo:** Experimentation with Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest Classifiers.
* **Production Ready:** Trained models are serialized into the `models/` directory for immediate inference.

---

## 🏗️ Repository Architecture

```text
Medical_Diagnosis_Assistant/
├── data/               # Raw and processed CSV files
├── models/             # Saved .pkl or .joblib model files
├── notebooks/          # Step-by-step EDA and training logic
│   └── Heart_Disease_Analysis.ipynb
├── main.py             # CLI entry point for predictions
├── requirements.txt    # Environment dependencies
└── README.md           # Project documentation

```

## 🛠️ Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

### 2. Setup

```bash
# Clone the repo
git clone https://github.com/Smitpatel2911/Medical_Diagnosis_Assistant.git
cd Medical_Diagnosis_Assistant

# Install dependencies
pip install pandas scikit-learn matplotlib seaborn

```

### 3. Running the Assistant

To run the prediction script:

```bash
python main.py

```

---

## 🧪 Evaluation Metrics

The models are evaluated primarily on **Recall** and **F1-Score**, as minimizing False Negatives is critical in a medical diagnostic context.

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.85 | 0.82 | 0.88 | 0.85 |
| Random Forest | 0.83 | 0.84 | 0.81 | 0.82 |
| KNN | 0.68 | 0.69 | 0.65 | 0.67 |

---

## 📝 Roadmap & Future Add-ons

* [ ] **Streamlit Integration:** Build a web-based UI for non-technical users.
* [ ] **Hyperparameter Tuning:** Implement `GridSearchCV` for Random Forest optimization.
* [ ] **SHAP Explainability:** Add visualizations to show which features most influenced a specific prediction.
* [ ] **Deep Learning:** Experiment with a simple Neural Network using TensorFlow/Keras.

---

**Disclaimer:** *This assistant is a proof-of-concept for educational purposes and should not be used for real-world medical diagnosis.*

---

**Would you like me to help you write the `requirements.txt` file or perhaps a `main.py` template that loads your saved model?**
