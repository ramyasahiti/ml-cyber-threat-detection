# 🔐 ML-Based Cyber Threat Detection & Risk Management Framework

## 📌 Overview

In modern IT project environments, cybersecurity has become a critical challenge due to the increasing use of distributed teams, shared repositories, and cloud-based tools. Traditional rule-based security systems often fail to detect sophisticated and evolving threats such as Denial-of-Service (DoS) attacks, exploits, and insider misuse.

This project presents a **Machine Learning-Based Cyber Threat Detection and Risk Management Framework** that leverages advanced supervised learning techniques to detect, classify, and assess cyber threats in real time. The system integrates **automated detection, explainability, and risk-based decision support** into a unified, scalable framework.

---

## 🎯 Objectives

* Detect cyber threats using machine learning
* Classify attacks using binary and multi-class models
* Perform feature selection for improved performance
* Provide real-time monitoring through a dashboard
* Enable risk-based decision-making with severity scoring

---

## 🧠 Technologies Used

* **Programming Language:** Python
* **Libraries:** Scikit-learn, XGBoost, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Dashboard:** Streamlit
* **Explainability:** SHAP

---

## ⚙️ Models Used

* Random Forest (initial experimentation)
* **XGBoost (final selected model)**

### 📌 Why XGBoost?

XGBoost was chosen as the final model because it:

* Achieved higher accuracy than Random Forest
* Handled imbalanced data more effectively
* Provided better generalization
* Delivered faster and more efficient performance

---

## System Architecture

The framework uses a **dual-model architecture**:

### 🔹 Binary Classification Model

* Classifies traffic as **Normal or Malicious**
* Ensures high-confidence threat detection

### 🔹 Multi-Class Classification Model

* Identifies specific attack types such as:

  * DoS
  * Exploits
  * Reconnaissance
  * Fuzzers
  * Worms
* Provides deeper insight into threat categories

---

## 📊 Dataset

This project uses the **UNSW-NB15**, a modern benchmark dataset for network intrusion detection.

### 📌 Features

* Realistic network traffic with synthetic attacks
* Multiple attack categories
* Suitable for binary and multi-class classification

---

### ⚠️ Dataset Availability

Due to GitHub file size limitations, the dataset is not included in this repository.

### 📥 How to Use

1. Download dataset from:
   https://research.unsw.edu.au/projects/unsw-nb15-dataset

2. Place files inside:

```
project/
├── Data/
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
```

3. Ensure folder name is exactly **`Data`**

---

## 📈 Results

### 🔹 Binary Classification

* **Accuracy:** 99.24%
* **Precision:** 0.99
* **Recall:** 0.99
* **F1-Score:** 0.99

### 🔹 Multi-Class Classification

* **Accuracy:** 87%
* **Weighted F1-Score:** 0.86

---

## 🔄 Workflow

1. Data Collection (UNSW-NB15)
2. Data Preprocessing
3. Feature Engineering & Selection
4. Model Training (Random Forest & XGBoost)
5. Model Evaluation
6. Real-Time Threat Detection
7. Risk Scoring & Mitigation Mapping
8. Visualization via Dashboard

---

## 🚨 Risk Management Framework

The system converts predictions into **actionable risk insights**:

| Severity | Description               |
| -------- | ------------------------- |
| Low      | Minimal threat            |
| Medium   | Moderate risk             |
| High     | Serious threat            |
| Critical | Immediate action required |

Each detected threat is mapped to **recommended mitigation strategies**, enabling effective decision-making.

---

## 📊 Explainability

* Integrated **SHAP (SHapley Additive exPlanations)**
* Provides feature-level importance for predictions
* Enhances transparency and trust in model decisions

---

## 🖥️ Dashboard Features

Built using Streamlit, the dashboard provides:

* 📡 Real-time threat monitoring
* 📊 Attack distribution visualization
* 🚨 Severity-based alerts
* 🛡️ Mitigation recommendations
* 🔍 Explainability insights using SHAP
* 📥 Downloadable logs

---

## 📂 Project Structure

```
project/
│
├── Data/                         # Dataset (not included)
├── models/                       # Trained models (.pkl)
│
├── cyber_dashboard.py            # Streamlit dashboard
├── train_models.py               # Model training
├── preprocess_data.py            # Data preprocessing
├── load_data.py                  # Data loading
├── realtime_detection.py         # Real-time detection
├── feature_selection_retrain.py  # Feature selection
├── xgb_binary_final.py           # Binary model
├── xgb_multiclass_final.py       # Multi-class model
├── xgb_leaky_version.py          # Experimental version
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Dashboard

```bash
streamlit run cyber_dashboard.py
```

---

## ⚠️ Notes

* Dataset is not included due to size constraints
* Ensure dataset is placed correctly before execution
* Trained models are required for prediction

---

## 🚀 Key Contributions

* **High-Accuracy Threat Detection**
  Achieved 99.24% accuracy in binary classification and 87% in multi-class classification

* **Dual-Model Architecture**
  Enables both detection and detailed attack classification

* **Explainable AI Integration**
  SHAP-based insights improve transparency

* **Risk-Based Severity Mapping**
  Converts predictions into actionable insights

* **Real-Time Monitoring Dashboard**
  Interactive and user-friendly interface

* **Scalable Design**
  Extendable for real-world deployment

---

## 🔮 Future Scope

* Integration with real-time network monitoring systems
* Cloud deployment
* Deep learning-based detection

---
