# 💳 Credit Card Fraud Detection: Predictive Analytics Pipeline

**Seoul Christian University | AI & Big Data Capstone Design - Project 1**

This repository contains the completed end-to-end predictive analytics pipeline for our Project 1 Capstone Design. The objective is to predict whether a credit card transaction is legitimate or fraudulent based on anonymized numerical features. 

Because fraudulent transactions account for only **0.172%** of the dataset, this project focuses heavily on strict data leakage prevention, imbalanced classification techniques (SMOTE, Cost-Sensitive Learning), and Precision-Recall Area Under Curve (PR-AUC) optimization.

## 🏆 Final Model Performance (Hold-Out Test Set)

After evaluating candidate models via 5-fold stratified cross-validation, the final tuned models were evaluated once on the 20% hold-out test set. **XGBoost** was selected as the final production recommendation for its superior balance of catching fraud while minimizing operational false alarms.

| Model | Tuned Threshold | Precision | Recall | F1-Score | PR-AUC | False Alarms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Benchmark** | - | 0.000 | 0.000 | 0.000 | 0.002 | 0 |
| **Logistic Regression + SMOTE** | 0.97 | 0.830 | 0.768 | 0.798 | 0.687 | 15 |
| **Random Forest (Balanced)** | 0.54 | 0.932 | 0.726 | 0.817 | 0.799 | 5 |
| **XGBoost (scale_pos_weight)** | **0.90** | **0.960** | **0.758** | **0.847** | **0.817** | **3** |

## 👥 Team Members & Task Allocation

| Name | Role | Responsibilities |
| :--- | :--- | :--- |
| **Chaulagai Shishir** | Project Manager & Repo Lead | Repository management, final report coordination, reproducibility assurance. |
| **Deula Ritika** | Data Engineer (Acquisition) | Dataset acquisition, source/license documentation, initial quality evaluation. |
| **Budhathoki Sandesh**| Data Engineer (Features) | Preprocessing pipeline, train/test separation, scaling implementation. |
| **Malla Abiraj** | Exploratory Data Analyst | Target distribution analysis, class imbalance visualization. |
| **KC hikalpit** | Validation & Eval Lead | Stratified validation design, metric justification (Precision, Recall, PR-AUC). |
| **Rawat Nabraj** | Modeling Specialist 1 | Naive benchmark establishment, interpretable baseline (Logistic Regression). |
| **Thada Magar Mansi** | Modeling Specialist 2 | Advanced candidate models (Random Forest, GBM), cross-validation. |
| **Aayusha Bista** | Interpretation Analyst | Error analysis, feature importance extraction, responsible-use documentation. |

## 📊 Dataset Requirements

*   **Source:** [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
*   **Context:** The dataset contains transactions made by European cardholders in September 2013 over two days. 
*   **Data Handling Note:** The `creditcard.csv` dataset is approximately 150MB and is deliberately excluded from this repository via `.gitignore` to comply with GitHub file size limits and data versioning best practices. 

## 🚀 How to Run the Code

This project is consolidated into a single, highly reproducible Google Colab script.

1.  Clone this repository to your local machine.
2.  Download the `creditcard.csv` file from the Kaggle link above.
3.  Open the `01_EDA_and_Preprocessing.ipynb` notebook located in the `/notebooks` folder via Google Colab.
4.  Upload `creditcard.csv` directly into the Colab session storage.
5.  Run the notebook from top to bottom. The global `RANDOM_STATE = 42` ensures results are exactly reproducible.


## 🔍 Key Findings

*   **Leakage Prevention:** Standard scaling and resampling techniques cause data leakage if applied before splitting. This pipeline strictly applies `train_test_split` prior to `RobustScaler`, and encapsulates `SMOTE` within an `imblearn` pipeline to isolate validation folds.
*   **The Cost of SMOTE:** While SMOTE helped the linear Logistic Regression model identify fraud (high recall), it generated an unacceptable number of False Positives.
*   **Algorithmic Weighting:** XGBoost natively handled the extreme imbalance using `scale_pos_weight`, drastically reducing False Positives (only 3 in the test set) without relying on synthetic data generation.
*   **Interpretation Limitations:** Features `V14`, `V4`, and `V12` dominate predictive importance across all models. However, because these are PCA-transformed variables, their real-world meaning remains obscured, limiting model explainability in a regulatory context.

## 👨🏻‍🏫 Instructor 
*   Prof. Dinesh Paudel PHD
