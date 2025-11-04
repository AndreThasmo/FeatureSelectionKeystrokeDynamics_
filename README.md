# 🧠 Keystroke Dynamics Evaluation

This project implements a robust and modular framework for evaluating biometric authentication based on keystroke dynamics.  
It supports multiple feature selection techniques, classifier models, advanced evaluation metrics, and includes a polished interactive dashboard for result exploration.

---

## 🧪 Overview

Keystroke dynamics capture how individuals type a password or phrase, enabling behavioral biometrics for user authentication. This framework simulates real-world intrusion scenarios, trains multiple models per user, and reports both traditional and security-focused metrics.

---

## 📦 Main Features

- **Feature Selection Algorithms**
  - T-Score
  - Fisher Score
  - Low Variance Filter (configurable threshold)

- **Classification Models**
  - Random Forest
  - Logistic Regression
  - Decision Tree
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)

- **Evaluation Metrics**
  - Accuracy, Balanced Accuracy, Precision, Recall, F1-Score
  - Specificity, Confusion Matrix
  - False Positive Rate (FPR), False Negative Rate (FNR)
  - Matthews Correlation Coefficient (MCC)

- **Result Export**
  - Outputs timestamped `.csv` and `.xlsx` result files
  - Generates `.png` plots for each experiment
  - Automatically zips all figures for download

- **Streamlit Dashboard**
  - Filter by user, feature selector, classifier, and number of features
  - Compare classifiers visually across metrics
  - Summary indicators with dynamic alerts for high FPR/FNR or low MCC
  - Correlation heatmap, precision-recall analysis, and more

---

## 📁 Project Structure

```
FeatureSelectionKeystrokeDynamics/
├── app/
│   ├── data_loader.py
│   ├── feature_selector.py
│   ├── keystroke_evaluator.py
│   ├── model_trainer.py
│   ├── user_dataset_builder.py
│   └── visualization.py
├── data/
│   ├── DSL-StrongPasswordData.csv
│   └── output/
├── figures/
├── generate_results.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Environment Setup

```bash
python -m venv env
source env/bin/activate        # macOS/Linux
.\env\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Run Experiment Suite

```bash
python generate_results.py
```

### 3. Launch the Dashboard

```bash
streamlit run streamlit_app.py
```

---

## 📊 Example Visualizations

*Insert screenshots of the dashboard or figures here.*

---

## 📜 License

Distributed under the MIT License.

---

## 👨‍🔬 Author

Developed and evaluated by **@andrethasmo**  
Technical implementation supported by ChatGPT 🤖
