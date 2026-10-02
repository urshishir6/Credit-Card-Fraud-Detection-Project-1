# 💳 Credit Card Fraud Detection: Predictive Analytics Pipeline

**Seoul Christian University | AI & Big Data Capstone Design - Project 1**

This repository contains the end-to-end predictive analytics pipeline for our Project 1 Capstone Design. The objective is to predict whether a credit card transaction is legitimate or fraudulent based on anonymized numerical features. 

Because fraudulent transactions account for less than 0.2% of the dataset, this project focuses heavily on **imbalanced classification techniques**, rigorous preprocessing to prevent data leakage, and strict validation strategies.

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

## 📊 Dataset Description

*   **Source:** [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
*   **Context:** The dataset contains transactions made by European cardholders in September 2013 over two days. 
*   **Features:** 
    *   `V1` to `V28`: Principal Component Analysis (PCA) transformed features to protect user identities.
    *   `Time`: Seconds elapsed between the transaction and the first transaction.
    *   `Amount`: The transaction monetary value.
    *   `Class`: Target variable (1 = Fraudulent, 0 = Legitimate).
*   **Challenge:** The dataset is highly unbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.

## 🛠️ Project Structure & Workflow

1.  **Exploratory Data Analysis (EDA):** Visualizing the extreme class imbalance and identifying threshold markers in `Amount` and `Time`.
2.  **Data Preprocessing:** Robust scaling of un-transformed features and strict train/test separation before applying sampling techniques (e.g., SMOTE) to prevent data leakage.
3.  **Baseline Modeling:** Establishing a naive benchmark and an interpretable Logistic Regression model.
4.  **Advanced Modeling:** Training and comparing Random Forest and Gradient Boosting Machines (XGBoost/LightGBM).
5.  **Evaluation:** Measuring model performance using metrics suited for imbalanced datasets, primarily Precision, Recall, F1-Score, and Precision-Recall Area Under Curve (PR-AUC).

## 📁 Repository Navigation

*   `/data`: Contains instructions for downloading the dataset (raw data is excluded from version control due to file size limits).
*   `/notebooks`: Jupyter notebooks containing EDA, preprocessing, and modeling experiments.
*   `/reports`: Technical reports, Pre-Report and the Final Project 1 Submission.
## 👨🏻‍🏫 Instructor 
*   Prof. Dinesh Paudel PHD
