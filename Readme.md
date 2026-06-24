# 🚦 Cyclical Metro Traffic Prediction Engine

An end-to-end machine learning pipeline for forecasting metro traffic volumes using advanced time-series feature engineering, cyclical temporal encoding, and automated hyperparameter optimization.

## 📌 Overview

Urban transportation systems generate large volumes of temporal data that exhibit strong cyclical patterns across hours, days, and weeks. Traditional linear models often fail to capture these periodic trends effectively.

This project develops a robust forecasting engine that leverages cyclical temporal encoding and gradient boosting techniques to accurately predict metro traffic volumes while maintaining low inference latency for real-world deployment.

## ✨ Key Features

* 📊 Processed and analyzed **48,000+ metro traffic records**
* 🔄 Applied **cyclical temporal feature engineering** using sine-cosine transformations
* 🤖 Benchmarked **6 machine learning architectures**
* 🚀 Achieved **12% higher accuracy** than traditional linear baselines
* ⚡ Automated hyperparameter optimization using **Optuna**
* 📈 Obtained **R² Score: 0.9698**
* 📉 Achieved **RMSE: 346**
* 💾 Serialized production-ready model using **Joblib**
* 🔥 Designed for low-latency inference and deployment

---

## 🏗️ Project Architecture

```text
Raw Traffic Data
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Cyclical Feature Engineering
(Hour, Day, Week Encodings)
        │
        ▼
Train/Test Split
        │
        ▼
Model Benchmarking
(6 ML Algorithms)
        │
        ▼
Optuna Hyperparameter Tuning
        │
        ▼
Best Model Selection (XGBoost)
        │
        ▼
Model Serialization (Joblib)
        │
        ▼
Inference Pipeline
```

## 🛠️ Tech Stack

| Category              | Technology          |
| --------------------- | ------------------- |
| Language              | Python              |
| ML Framework          | XGBoost             |
| Hyperparameter Tuning | Optuna              |
| Data Processing       | Pandas, NumPy       |
| Visualization         | Matplotlib, Seaborn |
| Model Persistence     | Joblib              |
| Evaluation            | Scikit-Learn        |

---

## 📂 Dataset

The dataset contains metro traffic observations with temporal information such as:

* Timestamp
* Hour of Day
* Day of Week
* Traffic Volume
* Historical Traffic Patterns

Total Records: **48,000+**

---

## 🔍 Feature Engineering

To capture periodic traffic behavior, cyclical encoding was applied:

```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)

day_sin = sin(2π × day / 7)
day_cos = cos(2π × day / 7)
```

This enables the model to learn natural temporal continuity and recurring traffic cycles.

---

## 🤖 Models Evaluated

The following algorithms were benchmarked:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. LightGBM
6. XGBoost

### 🏆 Best Performing Model

**XGBoost Regressor**

Performance improvements:

* 12% accuracy gain over linear baselines
* Superior generalization capability
* Lower prediction error

---

## 📈 Results

| Metric               | Score  |
| -------------------- | ------ |
| R² Score             | 0.9698 |
| RMSE                 | 346    |
| Accuracy Improvement | 12%    |

---

## 🚀 Running the Project

### Clone Repository

```bash
git clone https://github.com/yourusername/Cyclical-traffic-prediction-engine.git
cd Cyclical-traffic-prediction-engine
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Generate Predictions

```bash
python predict.py
```

---

## 💡 Future Enhancements

* Real-time traffic forecasting API
* Weather-aware traffic prediction
* Deep Learning (LSTM/Transformer) experimentation
* Interactive dashboard for visualization
* Cloud deployment with automated retraining

---

## 👨‍💻 Author

**Vanshraj Tandon**

* GitHub: https://github.com/Vanshraj30
* LinkedIn: https://linkedin.com/in/vanshraj-tandon-a404b2271

If you found this project useful, consider giving it a ⭐ on GitHub.
