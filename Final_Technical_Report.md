Capstone Design Project 1 Final Technical Report 

# **End-to-End Predictive Analytics: Credit Card Fraud Detection** 

**Seoul Christian University** 

Bachelor’s Program - AI & Big Data Big Data Analytics and Modeling 

## **Team and Task Allocation** 

|**Role**|**Member**|**Primary Responsibility**|
|---|---|---|
|**Project Manager**|Chaulagai Shishir|Coordinates report, reproducibility, alternative<br>validation, and final recommendations.|
|**Data Engineer (Acq.)**|Deula Ritika|Acquires dataset, documents provenance, au-<br>dits initial data quality and missingness.|
|**Data Engineer (Feat.)**|Budhathoki Sandesh|Builds leakage-safe preprocessing pipeline<br>(Robust Scaling, SMOTE integration).|
|**Exploratory Data Analyst**|Malla Abiraj|Conducts focused EDA, investigates extreme<br>class imbalance and feature distributions.|
|**Validation & Eval**|KC Hikalpit|Establishes stratified split, cross-validation,<br>and PR-AUC metric strategy.|
|**Modeling Specialist 1**|Rawat Nabraj|Establishes defensible naive baseline and Inter-<br>pretable Logistic Regression model.|
|**Modeling Specialist 2**|Thada Magar Mansi|Trains and evaluates advanced non-linear mod-<br>els (Random Forest and XGBoost).|
|**Interpretation Analyst**|Aayusha Bista|Conducts error analysis, evaluates feature im-<br>portance, and outlines responsible-use con-<br>straints.|



1 

## **Contents** 

|**1 Problem Definition and Stakeholder Context**|**3**|
|---|---|
|**2 Data Acquisition, Provenance, and Quality**|**3**|
|2.1 Data Source and Lawful Access ..............................................................|................................. 3|
|2.2 Data Quality Audit ..................................................................................|................................. 3|
|**3 Exploratory Data Analysis (EDA)**|**4**|
|3.1 Extreme Target Imbalance .......................................................................|................................. 4|
|3.2 Feature Skewness and Outliers ................................................................|................................. 4|
|**4 Train/Test Separation (Leakage Prevention)**|**5**|
|**5 Preprocessing and Feature Engineering**|**5**|
|5.1 Robust Scaling .........................................................................................|................................. 5|
|5.2 Pipeline Integration .................................................................................|................................. 5|
|**6 Baseline and Candidate Models**|**5**|
|**7 Validation Strategy and Model Selection**|**6**|
|7.1 Metric Selection: Precision-Recall AUC (PR-AUC) ...............................|................................. 6|
|7.2 Cross-Validation and Selection ................................................................|................................. 6|
|**8 Final Evaluation and Error Analysis**|**6**|
|8.1 Analysis of Errors ....................................................................................|................................. 7|
|**9 Model Interpretation and Feature Importance**|**7**|
|**10 Final Recommendation and Responsible Use**|**8**|
|10.1 Decision Recommendation ......................................................................|................................. 8|
|10.2 Limitations, Ethics, and Responsible AI Use ..........................................|................................. 9|



2 

## **1 Problem Definition and Stakeholder Context** 

The objective of this project is to construct an end-to-end analytical workflow to predict whether a credit card transaction is legitimate or fraudulent based on anonymized numerical features. The dataset presents an extreme imbalanced classification problem, mirroring real-world financial environments where frauds represent a microscopic fraction of overall traffic. 

The primary stakeholders for this predictive model are financial institutions, credit card issuers, and fraud analysts. The business objective is strictly defined: accurately detect rare fraud cases to prevent financial loss, while aggressively minimizing false alarms (False Positives). High false positive rates disrupt the customer experience by declining legitimate purchases and overwhelm human fraud analysts with unmanageable alert queues. The unit of observation is a single credit card transaction. 

### **Analytical Design Rules Enforced:** 

1. **Strict Leakage Prevention:** The hold-out test set is touched exactly once, and only after all modeling, scaling, and hyperparameter decisions are finalized using the training data. 

2. **Encapsulated Pipelines:** Every algorithm that learns from data (RobustScaler, SMOTE, Model) is encapsulated inside an imblearn.pipeline.Pipeline. This guarantees that during crossvalidation, the algorithms are re-fitted inside each specific fold, completely isolating the validation fold from data leakage. 

## **2 Data Acquisition, Provenance, and Quality** 

### **2.1 Data Source and Lawful Access** 

The dataset utilized is the “Credit Card Fraud Detection” dataset, originally sourced from the Worldline and the Machine Learning Group (MLG) of Université Libre de Bruxelles (ULB). It is publicly hosted on Kaggle. The dataset contains transactions made by European cardholders occurring over a two-day period in September 2013. The dataset is distributed under the Open Data Commons for Public Domain Dedication and License (PDDL), making it fully authorized and ethically appropriate for this academic capstone project. 

### **2.2 Data Quality Audit** 

An programmatic initial audit confirmed the dataset matches the official documentation perfectly, requiring no imputation: 

- **Shape:** 284,807 rows and 31 columns. 

- **Missing Values:** Exactly 0 missing (NULL) values across all 284,807 rows. 

- **Variables:** 28 independent features (V1-V28) that have been mathematically transformed via Principal Component Analysis (PCA) to protect user confidentiality. The dataset also includes unscaled Time (seconds elapsed since the first transaction) and Amount (exact monetary value). 

- **Target Variable (** Class **):** A Boolean integer where 1 represents a Fraudulent transaction and 0 represents a Legitimate transaction. 

3 

## **3 Exploratory Data Analysis (EDA)** 

Exploratory Data Analysis was conducted strictly to inform modeling and preprocessing decisions, focusing on target class imbalance and the statistical distribution of the unscaled features. 

### **3.1 Extreme Target Imbalance** 

The dataset contains 284,315 legitimate transactions and only 492 frauds. As visualized in Figure 1, frauds represent just **0.172%** of the total dataset. Because of this extreme skew, standard accuracy is mathematically invalid as an evaluation metric. A “naive” model that simply predicts 0 for every transaction will achieve 99.83% accuracy while failing entirely to detect fraud. This visual evidence justifies our shift to Precision-Recall Area Under Curve (PR-AUC) and necessitates algorithmic imbalance handling (e.g., SMOTE, Cost-Sensitive Learning). 

### **3.2 Feature Skewness and Outliers** 

Because features V1 through V28 are outputs of PCA, they are naturally centered and scaled. However, Time and Amount retain their original variances. Figure 1 visualizes these distributions. The Amount feature is heavily right-skewed with extreme monetary outliers (a maximum transaction over $25,000 against a median of $22). Standard scaling would be severely skewed by these outliers, justifying the use of a RobustScaler. The Time feature displays a bimodal distribution corresponding to day and night cycles in consumer behavior. 



<!-- Start of picture text -->
226,602 25<br>10° 2.0<br>a<br>ao! 2s<br>Lo<br>w 378<br>ail<br>es oo<br>ul<br>0.08 (5 Fraud ie<br>Lo<br>0.06 &<br>= 208<br>S04 Sos<br>os<br>sua ee | me ee|<br><!-- End of picture text -->

Figure 1: Exploratory data analysis showing extreme target imbalance and right-skewed Amount distributions necessitating scaling. 

4 

## **4 Train/Test Separation (Leakage Prevention)** 

To guarantee an unbiased, scientifically defensible evaluation, the data was separated **before** any scaling, resampling, or model fitting occurred. We utilized a Stratified Split using train_test_split with an 80/20 ratio. Stratification ensures that the microscopic 0.167% fraud rate is maintained perfectly proportionally across both sets, preventing folds with zero fraud cases. 

- **Training Set:** 226,980 transactions (378 frauds) 

- **Hold-Out Test Set:** 56,746 transactions (95 frauds) 

From this point forward, the test set was completely sealed. It played no role in Cross-Validation, SMOTE generation, or scaler learning parameters. 

## **5 Preprocessing and Feature Engineering** 

A reproducible preprocessing pipeline was constructed to prepare the data for machine learning algorithms without inducing data leakage. 

### **5.1 Robust Scaling** 

Because distance-based algorithms and linear models are sensitive to extreme variances, the Amount and Time features required scaling. We applied a RobustScaler, which centers the data using the median and scales it according to the Interquartile Range (IQR). This ensures that the massive monetary outliers identified in EDA do not distort the scaler. Crucially, the scaler was fitted exclusively on the X_train dataset. 

### **5.2 Pipeline Integration** 

To address the class imbalance, we opted to use the Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic fraud examples for our linear models. Standard scikit-learn pipelines do not support resampling. Therefore, we utilized imblearn.pipeline.Pipeline. This guarantees that SMOTE is applied _only_ to the training folds during Cross-Validation, preventing synthetic data from bleeding into the validation fold, which would artificially inflate model performance. 

## **6 Baseline and Candidate Models** 

Four models were established to provide a rigorous comparison between simplistic, interpretable, and advanced ensemble approaches. 

1. **Naive Benchmark:** A baseline array that explicitly predicts 0 (Legitimate) for all transactions. This establishes a floor metric, proving that baseline accuracy is 99.8% but baseline fraud detection (Recall/PR-AUC) is 0.00. 

5 

2. **Logistic Regression (Interpretable):** Encapsulated in a pipeline with SMOTE, this provides a linear boundary. Because it calculates coefficients for each feature, it serves as our interpretable model to explain the drivers of fraud. 

3. **Random Forest Classifier:** A strong, non-linear bagging ensemble. Rather than using SMOTE (which causes memory issues on massive datasets with complex trees), we utilized algorithm-level cost-sensitive learning via class_weight=’balanced’. 

4. **XGBoost (Gradient Boosting):** An advanced boosting ensemble. We implemented scale_pos_weight calculated precisely by the ratio of legitimate to fraudulent transactions in the training set to penalize the misclassification of the minority class heavily. 

## **7 Validation Strategy and Model Selection** 

### **7.1 Metric Selection: Precision-Recall AUC (PR-AUC)** 

Receiver Operating Characteristic (ROC-AUC) can be highly optimistic in severely imbalanced datasets because the massive number of True Negatives drags the False Positive Rate down, making the curve look artificially strong. We explicitly chose **Precision-Recall AUC (PR-AUC)** as our primary decision metric. PR-AUC focuses exclusively on the positive (Fraud) class, penalizing models that throw massive amounts of false alarms while attempting to increase recall. Precision, Recall, and F1-Score were calculated as secondary context metrics. 

### **7.2 Cross-Validation and Selection** 

Models were evaluated using Stratified 5-Fold Cross-Validation strictly on the training data. 

|**Candidate Model**|**PR-AUC (Mean)**|**F1-Score**|**Precision**|**Recall**|
|---|---|---|---|---|
|Naive Benchmark (All Zeros)|0.002|0.000|0.000|0.000|
|Logistic Regression + SMOTE|0.735|0.110|0.058|0.918|
|Random Forest (Balanced)|0.835|0.856|0.941|0.786|
|**XGBoost (scale_pos_weight)**|**0.847**|**0.879**|**0.952**|**0.816**|



Table 1: Cross-Validation results comparing baseline and candidate models. 

**Selection:** XGBoost achieved the highest PR-AUC (0.847) and was formally selected as the optimal model for final evaluation. Thresholds were dynamically tuned on the training data to maximize the F1-Score. 

## **8 Final Evaluation and Error Analysis** 

Having selected XGBoost, we executed a one-time final prediction on the previously sealed hold-out test set using the tuned thresholds. We directly compared the confusion matrices of our chosen advanced model (XGBoost) against the interpretable model (Logistic Regression). 

6 



<!-- Start of picture text -->
10 Se aa fraudLegitimate ii<br>\ be === tuned threshold H<br>os 1 a | i<br>‘aise i i<br>508 “Ss 1 —} H<br>§ —Ss HH<br>Eo St 1 w | ;<br>fecal Predicted fraud probabiity<br><!-- End of picture text -->

Figure 2: Precision-Recall curves across all models and the bimodal score distribution for the selected XGBoost model. 



<!-- Start of picture text -->
Pos RIT vecos Heer AO as wage Rca tsa<br>3 2 B 3 26 6 z Fa 2<br>predicted Predicted Predicted<br><!-- End of picture text -->

Figure 3: Confusion Matrices on the untouched Test Set. Logistic Regression generates excessive False Positives (15) compared to XGBoost (3). 

### **8.1 Analysis of Errors** 

- **Logistic Regression (High Recall, Poor Precision):** Due to SMOTE, the logistic regression model learned to catch almost every fraud. However, it threw 15 False Positives. In a real bank, blocking thousands of legitimate transactions across the total volume would cause massive customer churn. 

- **XGBoost (Optimal Balance):** XGBoost dropped the False Positive count to 3. It caught 72 out of 95 frauds while only mistakenly flagging a handful of normal transactions, proving its viability for deployment. 

- **Analysis of Missed Frauds (False Negatives):** An investigation into the raw values of the frauds that XGBoost missed revealed that they mathematically masqueraded perfectly as legitimate transactions (their V14 and V12 values sat directly on the median of legitimate data). No mathematical model can catch these without secondary contextual data. 

## **9 Model Interpretation and Feature Importance** 

To ensure transparency, we interpreted the models by extracting coefficients from Logistic Regression and Feature Importances from XGBoost. 

7 



<!-- Start of picture text -->
Logistic Regression: standardised coefficients (top 10)<br>red = raises fraud log-odds, blue = lowers<br>V4<br>v4<br>v10<br>Vi12<br>22<br>V8<br>Vil<br>Amount<br>V6<br>v5<br>-0.75 -0.50 -0.25 0.00 0.25 0.50 0.75 1.00<br><!-- End of picture text -->

Figure 4: Logistic Regression Coefficients. Red bars represent features that increase the probability of Fraud, while Blue bars push the prediction toward Legitimate. 



<!-- Start of picture text -->
Permutation importance (drop in PR-AUC)<br>XGBoost (scale_pos_weight) Built-in importance (model-specific)<br>SS— 6<br>o. = ‘a<br>vs ve<br>= oZ<br>s = vo i<br>= ve il<br>we =e vu<br>0.00 0.02 0.04 0.06 0.08 010 000 0.05 010 03s 020025 030<br><!-- End of picture text -->

Figure 5: Permutation Importance and Built-in Feature Importances extracted from the XGBoost Model. Features V14, V4, and V12 heavily dominate the decision trees. 

**Interpretation Synthesis:** Both the linear and tree-based models overwhelmingly agreed that features V14, V4, and V12 were the most critical factors in distinguishing fraud. The engineered Amount and Time variables provided very little predictive power compared to the PCA-transformed variables. 

## **10 Final Recommendation and Responsible Use** 

### **10.1 Decision Recommendation** 

We formally recommend the deployment of the **XGBoost** model utilizing scale_pos_weight for class imbalance. It overwhelmingly beat the baseline and interpretable models, achieving a Test PR-AUC of over 0.81 and an F1-Score of 0.847. It struck the optimal business balance: catching the vast majority of financial fraud (72 out of 95 cases) while minimizing the operational burden of false alarms (only 3). 

8 

### **10.2 Limitations, Ethics, and Responsible AI Use** 

Before this model can be safely deployed into a live financial ecosystem, stakeholders must understand the following limitations: 

1. **Lack of Explainability due to PCA:** Because the most important features (V14, V4) are mathematical outputs of a PCA transformation, they are anonymized. We cannot map these to real-world attributes (e.g., "The model flagged this because the purchase was made in a foreign country"). This lack of explainability may conflict with financial regulations requiring banks to explain why a transaction was declined. 

2. **Customer Friction and False Positives:** Even with a highly precise model like XGBoost, false positives still occur. Blocking a legitimate transaction causes severe customer dissatisfaction. Therefore, this model must be deployed strictly as a **Decision-Support System** . Rather than automatically freezing customer bank accounts, flagged transactions should trigger a 2-Factor Authentication (OTP) prompt to the user’s phone, or enter a prioritization queue for human fraud analysts to review. 

3. **Temporal Concept Drift:** Fraudsters are adversarial and actively adapt to bypass machine learning models. The data in this dataset only spans a two-day timeframe. Fraud patterns from September 2013 will not entirely generalize to today’s threat landscape. Real-world deployment requires MLOps infrastructure for continuous monitoring, automated performance evaluation, and frequent model re-training to prevent concept drift. 

### **AI-Use and Academic Integrity Statement:** 

Generative AI was utilized according to instructor guidelines to assist in formatting structure of this technical report and to suggest the implementation of the imblearn.pipeline to ensure absolute adherence to strict data-leakage prevention rules during Cross-Validation. All programmatic code was independently executed, verified, and analyzed by our team on the actual dataset. No metrics, values, or figures were manually fabricated or edited. 

9 

