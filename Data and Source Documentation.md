**Data and Source Documentation: Credit Card Fraud Detection**

**Team Project Manager:** Chaulagai Shishir

**Data Engineer (Acquisition):** Deula Ritika

For the Capstone Design requirement here, we elaborate the document data provenance, lawful access conditions, variables, and initial data quality before beginning the preprocessing pipeline.

#### **Data Provenance and Source Access**

- **Dataset Name:** Credit Card Fraud Detection
- **Original Source:** Worldline and the Machine Learning Group (MLG) of Université Libre de Bruxelles (ULB).
- **Platform/Access:** Hosted publicly on Kaggle (\[kaggle.com/datasets/mlg-ulb/creditcardfraud\](<https://kaggle.com/datasets/mlg-ulb/creditcardfraud>)).
- **Lawful Access & License:** The dataset is distributed under the Open Data Commons for Public Domain Dedication and License (PDDL). It is fully authorized for academic research and open-source machine learning projects, fulfilling the course requirement for an ethically appropriate public dataset.

#### **Dataset Context and Scope**

- **Timeframe:** Transactions made by European cardholders occurring over a two-day period in September 2013.
- **Size:** 284,807 total rows (transactions).
- **Scope:** Contains exclusively numerical input variables resulting from a Principal Component Analysis (PCA) transformation to maintain strict user confidentiality.

#### **Data Dictionary (Variables and Data Types)**

The dataset contains 31 continuous numerical features and 1 categorical target variable.

- Time **(Numeric / Float):** The seconds elapsed between each transaction and the very first transaction recorded in the dataset.
- V1 **through** V28 **(Numeric / Float):** 28 independent features that have been mathematically transformed via PCA. Due to privacy and security constraints, the original features (e.g., location, merchant, item type) and background context cannot be provided.
- Amount **(Numeric / Float):** The exact monetary value of the transaction.
- Class **(Categorical / Integer):** The boolean target variable where 1 represents a fraudulent transaction and 0 represents a valid, legitimate transaction.

#### **Class Balance and Target Distribution**

The dataset is explicitly designed to simulate real-world financial environments, resulting in an extreme class imbalance:

- **Legitimate Transactions (Class 0):** 284,315 cases (99.827%)
- **Fraudulent Transactions (Class 1):** 492 cases (0.172%)
- **Modeling Implications:** Because of this distribution, standard accuracy metrics are invalid. The team's evaluation strategy will strictly utilize Precision, Recall, F1-Score, and the Precision-Recall Area Under Curve (PR-AUC).

#### **Data Quality, Missingness, and Constraints**

- **Missing Values:** There are exactly zero NULL or missing values across all 284,807 rows and 31 columns. No imputation strategy is required for this specific dataset.
- **Feature Scaling Discrepancy:** While features V1 to V28 are strictly bounded as a result of the PCA transformation, the Time and Amount features retain their original wide variances.
- **Potential Quality Issues:** The massive range in the Amount variable introduces extreme outliers that can severely skew linear models or distance-based algorithms. A RobustScaler must be applied to these two specific columns, strictly after separating the train and test sets, to prevent data leakage.