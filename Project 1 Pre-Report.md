**Project 1 Pre-Report: Credit Card Fraud Detection**

**Team Members and Task Allocation**

To ensure a clear division of responsibilities for this end-to-end analytical workflow, our eight-member team has assigned specific roles.

| **Team Member**        | **Assigned Role**            | **Primary Responsibilities for Project 1**                                                                                                      |
| ---------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Chaulagai Shishir**  | Project Manager & Repo Lead  | Coordinates the 7-10 page technical report, manages the version-controlled repository, and ensures reproducibility of all analysis code.        |
| **Deula Ritika**       | Data Engineer (Acquisition)  | Acquires the dataset, documents the data source and license conditions, and evaluates initial data quality.                                     |
| **Budhathoki Sandesh** | Data Engineer (Features)     | Builds the reproducible preprocessing pipeline and ensures train/test data separation before any learned preprocessing to prevent data leakage. |
| **Malla Abiraj**       | Exploratory Data Analyst     | Conducts focused EDA to inform modeling choices by investigating the extreme class imbalance and feature distributions.                         |
| **KC hikalpit**        | Validation & Evaluation Lead | Defines the target variable, establishes the stratified validation strategy, and justifies evaluation metrics for imbalanced classification.    |
| **Rawat Nabraj**       | Modeling Specialist 1        | Establishes a defensible naive benchmark and trains the interpretable baseline model (Logistic Regression).                                     |
| **Thada Magar Mansi**  | Modeling Specialist 2        | Trains two stronger alternative models (Random Forest, GBM) and implements the cross-validation strategy.                                       |
| **Aayusha Bista**      | Interpretation Analyst       | Conducts model error analysis, interprets feature importance, and communicates practical limitations and responsible-use concerns.              |

1. **Problem Definition and Stakeholder Context**

The objective of this project is to predict whether a credit card transaction is legitimate or fraudulent based on anonymized numerical features, structured as an imbalanced classification problem. The primary stakeholders for this predictive model are financial institutions, credit card issuers, and fraud analysts who require accurate detection of rare fraud cases while minimizing false alarms that disrupt customer experience. The unit of observation is a single credit card transaction.

1. **Dataset and Data Dictionary**

The project utilizes the "Credit Card Fraud Detection" dataset sourced from Kaggle. This public dataset satisfies the requirement for an ethically appropriate data source and is rich enough to require meaningful preprocessing.

- Time: Seconds elapsed between the transaction and the first transaction in the dataset (Numeric).
- V1 to V28: Principal Component Analysis (PCA) transformed features to protect user identities (Numeric).
- Amount: The transaction monetary value (Numeric).
- Class: The target variable where 1 represents a fraudulent transaction and 0 represents a legitimate transaction (Categorical/Boolean).

1. **Target and Features**

The primary target variable is Class. The predictive features include all 28 PCA-transformed variables (V1 through V28), the transaction Amount, and the Time feature, which may require temporal engineering.

1. **Exploratory Data Analysis (EDA) Plan**

The exploratory data analysis will focus directly on informing the modeling strategy for rare-event detection. We will analyze the severe class imbalance (where frauds represent less than 0.2% of the dataset) to justify our validation and preprocessing strategies. Additionally, we will visualize the distribution of transaction Amount and Time across both classes to identify potential threshold markers for fraud.

1. **Preprocessing and Feature Preparation Plan**

Our reproducible preprocessing pipeline will begin by separating the data into training and testing sets to absolutely prevent data leakage. Because the PCA features (V1-V28) are already scaled, our feature engineering will focus on applying a Robust Scaler to the Amount and Time variables, which are prone to extreme outliers. To handle the severe class imbalance, we will apply techniques such as SMOTE (Synthetic Minority Over-sampling Technique) or class weighting strictly on the training set during model development.

1. **Baseline and Candidate Models**

To establish a naive benchmark, we will use a model that simply predicts the majority class (legitimate) for all test instances, which will yield high accuracy but zero fraud detection. For our interpretable model, we will train a Logistic Regression classifier to provide clear coefficients for the PCA features. We will compare this against two stronger alternatives suited for imbalanced data: a Random Forest Classifier and a Gradient Boosting Machine (e.g., XGBoost).

1. **Validation Strategy and Evaluation Metrics**

Due to the extreme class imbalance, a standard random split is inappropriate. The team will employ Stratified k-fold cross-validation on the training data to ensure the rare fraud cases are proportionally represented in every fold, keeping a strictly held-out test set for final evaluation. Because standard accuracy is misleading for this dataset, we will evaluate the models using Precision, Recall, the F1-score, and the Precision-Recall Area Under Curve (PR-AUC), explicitly justifying these choices based on the high cost of false negatives in fraud detection.

1. **Repository and Code Draft**

A version-controlled project folder has been initialized on GitHub to fulfill the reproducibility requirements. This repository contains our draft Python notebooks and separates raw data from processed outputs.

- **Repository Link:** \[Insert Team GitHub URL Here\]

1. **References and AI-Use Statement**

- **Data Source:** Credit Card Fraud Detection Dataset, accessed via Kaggle.
- **AI-Use Disclosure:** Generative AI was utilized to help structure this pre-report outline and brainstorm the imbalanced classification approach, strictly adhering to the transparent AI use guidelines. All final data preparation, modeling, and evaluation code will be understood, tested, and independently verified by the team members.