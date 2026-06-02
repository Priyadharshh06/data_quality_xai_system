# 🔍 Explainable AI-Based Automated Data Quality Management System
### For Noise Detection in Numerical Data Pipelines

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

## 📌 Overview

This project presents an automated data quality management system that detects noise and anomalies in numerical data pipelines using an ensemble of unsupervised machine learning models. The system integrates **Explainable AI (XAI)** via SHAP values to provide transparent, human-interpretable explanations for every anomaly detected — making it suitable for real-world data engineering pipelines.

---

## 🎯 Key Features

- ✅ Ensemble anomaly detection using **Isolation Forest**, **Local Outlier Factor**, and **One-Class SVM**
- ✅ **Consensus voting mechanism** (2-of-3 majority) for reliable anomaly flagging
- ✅ **SHAP-based explainability** — identifies which features contribute most to each anomaly
- ✅ Interactive **Streamlit dashboard** for real-time data quality monitoring
- ✅ **SQLite storage** for logging detected anomalies and quality reports
- ✅ Overall **Data Quality Score** computed automatically
- ✅ Cleaned dataset export after noise removal

---

## 🗂️ Project Structure

```
data_quality_xai_system/
│
├── app.py                        # Streamlit dashboard (main entry point)
├── anomaly_detection.py          # Ensemble model logic (IF + LOF + OCSVM)
├── shap_explainability.py        # SHAP explanation generation
├── data_quality_pipeline.py      # Full pipeline orchestration
├── requirements.txt              # Python dependencies
│
├── data/
│   └── credit_worthiness.csv     # Dataset used for testing (614 rows, 13 columns)
│
├── outputs/
│   ├── shap_summary_plot.png     # SHAP summary visualization
│   ├── shap_waterfall_plot.png   # SHAP waterfall chart
│   └── shap_force_plot.html      # Interactive SHAP force plot
│
└── data_quality.db               # SQLite database for anomaly logs
```

---

## 📊 Dataset

- **Name:** Credit Worthiness Dataset
- **Rows:** 614 | **Columns:** 13
- **Type:** Numerical and categorical financial data
- **Source:** Standard loan eligibility benchmark dataset

---

## ⚙️ System Architecture

The pipeline follows a 4-stage process:

```
Raw Data Input
      ↓
Preprocessing & Feature Extraction
      ↓
Ensemble Anomaly Detection (IF + LOF + OCSVM)
      ↓
Consensus Voting (2-of-3 Majority)
      ↓
SHAP Explainability Layer
      ↓
Streamlit Dashboard + SQLite Storage
      ↓
Cleaned Dataset Output + Quality Score
```

---

## 📈 Results

| Model | Anomalies Detected | F1 Score |
|---|---|---|
| Isolation Forest | 123 | 0.865 |
| Local Outlier Factor | 123 | 0.646 |
| One-Class SVM | 123 | 0.891 |
| **Consensus (2-of-3)** | **106** | — |

- 🏆 **Overall Data Quality Score: 90.6%**
- 🧹 **Cleaned Dataset Size: 508 rows**
- 🔑 **Top SHAP Features: Credit_History, Loan_Amount_Term**

---

## 🖥️ Dashboard Screenshots

## 🖥️ Dashboard Screenshots

![Dashboard Overview](Screenshots/Dashboard%20Overview.png)
![Data Overview](Screenshots/Data%20Overview.png)
![Noise and Data Quality Report](Screenshots/Noise%20and%20Data%20Quality%20Report.png)
![Model Comparison](Screenshots/Model%20Comparison.png)
![Evaluation Metrics](Screenshots/Evaluation%20Metrics.png)
![Anomaly Distribution per Feature](Screenshots/Anomaly%20Dis%20per%20Feature.png)

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Priyadharshh06/data_quality_xai_system.git
cd data_quality_xai_system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

### 4. Open in Browser
```
http://localhost:8501
```

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML Models | Scikit-learn (IsolationForest, LOF, OneClassSVM) |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Storage | SQLite |
| Visualization | Matplotlib, Seaborn |
| Version Control | Git, GitHub |

---

## 👩‍💻 Author

**Priya Dharshini**
M.Sc. Information Technology - VISTAS, Chennai
Supervised by: Dr. A. Akila

[![GitHub](https://img.shields.io/badge/GitHub-Priyadharshh06-black?logo=github)](https://github.com/Priyadharshh06)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
