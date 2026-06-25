# ⚽ FIFA Player Scouting System — ML Classification & Regression

A complete machine learning pipeline that analyzes FIFA player data to **predict market value** (regression) and **classify player performance tiers** (classification), culminating in a unified scouting system powered by ensemble learning.

---

## 👥 Team Members

| Name            | GitHub                                                     |
| --------------- | ---------------------------------------------------------- |
| [Member 1 Name] | [@FarahHossam126](https://github.com/FarahHossam126)       |
| [Member 2 Name] | [@menna0602](https://github.com/menna0602)                 |
| [Member 3 Name] | [@maii818](https://github.com/maii818)                     |
| [Member 4 Name] | [@Farah-Ibrahim1405](https://github.com/Farah-Ibrahim1405) |
| [Member 5 Name] | [@nourhanessam4](https://github.com/nourhanessam4)         |

---

## 📌 Project Overview

This project spans two assignments built on the **FIFA Player Dataset**, covering the full ML workflow from raw data to a deployed prediction system:

- **Assignment 2** — EDA, preprocessing, polynomial regression with regularization, logistic regression, Naïve Bayes, and cross-validation
- **Assignment 3** — KNN, Random Forest, SVM/SVR, and ensemble learning (Voting + Stacking), all wrapped in a `unified_scouting_system()` that predicts both a player's market value and performance tier from raw input

---

## 📂 Repository Structure

```
ML_Project/
│
├── ML_project.ipynb        # Main Jupyter Notebook (fully executed)
├── Fifa.csv                    # FIFA player dataset
└── README.md                   # This file
```

---

## 📊 Dataset

| Property     | Detail                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------- |
| **Source**   | FIFA Player Dataset                                                                         |
| **Domain**   | Sports Analytics                                                                            |
| **Features** | Age, Future Potential, Total Stats Score, Position, Country, Overall Rating, Value Per M$   |
| **Targets**  | `Value Per M$` (regression) · `Performance Tier` (classification: Low / Mid / High / Elite) |
| **Type**     | Mixed — numeric and categorical                                                             |

---

## 🔬 Pipeline Overview

```
Raw FIFA Data
     ↓
EDA & Visualization
     ↓
Preprocessing (train/test split → deduplication → outlier clipping → OHE → scaling)
     ↓
┌──────────────────────┬──────────────────────────┐
│    Regression        │      Classification       │
│  (Value Per M$)      │    (Performance Tier)     │
│                      │                           │
│  Polynomial Reg      │  Logistic Regression      │
│  Ridge / Lasso       │  Naïve Bayes              │
│  KNN Regressor       │  KNN Classifier           │
│  Random Forest Reg   │  Random Forest Clf        │
│  SVR                 │  SVM (SVC)                │
│  Voting Regressor    │  Voting + Stacking Clf    │
└──────────────────────┴──────────────────────────┘
     ↓
unified_scouting_system(player_data)
→ Predicted Value + Predicted Tier
```

---

## 🗂️ Assignment 2

### Preprocessing

| Step                  | Method                                       | Justification                                      |
| --------------------- | -------------------------------------------- | -------------------------------------------------- |
| Train/Test Split      | 80/20 · `random_state=1`                     | Split before any transformation to prevent leakage |
| Deduplication         | Drop exact duplicates                        | Applied on training set only                       |
| Outlier Treatment     | IQR clipping (fit on train, apply to test)   | Preserves data size while capping extremes         |
| Categorical Encoding  | One-Hot Encoding on `Position` & `Country`   | Fit on train, transformed on both sets             |
| Feature Scaling       | StandardScaler on numeric columns            | Fit on train only                                  |
| Target Classification | Quartile-based binning (Q1=58, Q2=63, Q3=68) | Data-driven, approximately balanced 4-class split  |

### Task 4 — Polynomial Regression

| Degree | Train R²  | Test R²   | Test RMSE |
| ------ | --------- | --------- | --------- |
| 1      | 0.753     | 0.744     | 0.568     |
| 2      | 0.826     | 0.820     | 0.476     |
| 3      | 0.864     | 0.866     | 0.411     |
| **4**  | **0.888** | **0.889** | **0.375** |

- Best degree: **4** (smallest train-test gap, best generalization)
- Regularization: Ridge (`α=0.0001`, RMSE=0.375) outperformed Lasso (`α=0.0001`, RMSE=0.378)
- Lasso zeroed out ~98 polynomial features, confirming most interaction terms were redundant

### Task 5 — Logistic Regression

| Configuration           | Test Accuracy |
| ----------------------- | ------------- |
| Baseline (default C)    | 80.33%        |
| Best C = 1 · L1 penalty | **80.60%**    |
| Best C = 1 · L2 penalty | 80.33%        |

- L1 slightly outperformed L2 by zeroing out redundant features
- Weakest class: `High` (F1 = 0.68) — most confusion with adjacent tiers

### Task 6 — Naïve Bayes

| Variant      | Features Used                                     | Test Accuracy |
| ------------ | ------------------------------------------------- | ------------- |
| GaussianNB   | Numeric only (Age, Future Potential, Total Stats) | 71.89%        |
| BernoulliNB  | Binary OHE columns only                           | —             |
| ComplementNB | All features (shifted)                            | —             |

- GaussianNB was most appropriate — continuous features follow approximately normal distributions
- Scaling had no effect on GaussianNB accuracy (as expected — Gaussian NB is scale-invariant)

### Task 7 — Cross-Validation

| Model                   | CV Strategy       | Mean Score        | Std    |
| ----------------------- | ----------------- | ----------------- | ------ |
| Ridge Regression (best) | 5-Fold KFold      | RMSE = 0.3855     | 0.009  |
| Logistic Regression     | 5-Fold Stratified | Accuracy = 80.47% | 0.0053 |
| GaussianNB              | 5-Fold Stratified | Accuracy = 71.46% | —      |

---

## 🗂️ Assignment 3

### Architecture — sklearn Pipelines

All models in Assignment 3 use `sklearn.Pipeline` with a shared `preprocessing_pipeline` containing:

- `SimpleImputer` for missing values
- Custom `OutlierClipper` (IQR-based, fit on train only)
- `StandardScaler` for numeric features
- `OneHotEncoder` for categorical features

This architecture **prevents data leakage** and makes each model fully self-contained.

### Model 1 — KNN

| Task           | Best Params      | Train Score  | Test Score   | Diagnosis               |
| -------------- | ---------------- | ------------ | ------------ | ----------------------- |
| Regression     | GridSearch tuned | R² = 1.000   | R² = 0.803   | Overfitting (gap=0.197) |
| Classification | GridSearch tuned | Acc = 99.98% | Acc = 79.46% | Overfitting (gap=0.205) |

> KNN memorizes training points by design — small k values cause severe overfitting on this dataset.

### Model 2 — Random Forest

| Task           | CV Score          | Test Score   | Gap   | Diagnosis           |
| -------------- | ----------------- | ------------ | ----- | ------------------- |
| Regression     | Mean R² = 0.830   | R² = 0.856   | 0.088 | Good generalization |
| Classification | Mean Acc = 84.64% | Acc = 85.31% | 0.088 | Good generalization |

- Best RF Classifier: `min_samples_split=10` (regularization to prevent overfitting)
- Per-class F1 (Classification): Elite=0.94 · High=0.79 · Low=0.89 · Mid=0.78

### Model 3 — SVM / SVR

| Task                 | Best Kernel | Test Score   | Gap   | Diagnosis           |
| -------------------- | ----------- | ------------ | ----- | ------------------- |
| Classification (SVC) | RBF         | Acc = 85.89% | 0.032 | Good generalization |
| Regression (SVR)     | RBF         | R² = 0.869   | 0.016 | Good generalization |

- SVM achieved the best generalization gap among individual models
- Linear kernel severely underfitted the regression task

### Ensemble Learning

#### Classification Ensembles

| Model         | Test Accuracy | CV Mean    | Std        | Gap       | Diagnosis        |
| ------------- | ------------- | ---------- | ---------- | --------- | ---------------- |
| KNN           | 79.46%        | 80.26%     | 0.0086     | 0.205     | Overfitting      |
| Random Forest | 85.31%        | 84.64%     | 0.0055     | 0.088     | Good             |
| SVM           | 85.89%        | 85.86%     | 0.0034     | 0.032     | Good             |
| Voting (Soft) | 86.38%        | 85.89%     | 0.0038     | 0.127     | Slight overfit   |
| **Stacking**  | **86.78%**    | **86.14%** | **0.0016** | **0.042** | **Best overall** |

#### Regression Ensembles

| Model            | Test R²   | Diagnosis   |
| ---------------- | --------- | ----------- |
| KNN Regressor    | 0.803     | Overfitting |
| Random Forest    | 0.856     | Good        |
| SVR              | **0.869** | Good        |
| Voting Regressor | 0.868     | Good        |

#### Unified Scouting System

```python
def unified_scouting_system(player_data):
    # Predicts market value (regression) + performance tier (classification)
    predicted_value = voting_reg_pipeline.predict(player_df)[0]
    predicted_tier  = stacking_pipeline.predict(player_df)[0]
    return predicted_value, predicted_tier

# Example
unified_scouting_system({
    "Age": 26,
    "Future Potential": 82,
    "Total_Stats Score": 78,
    "Country": "Egypt",
    "Position": "ST"
})
# → Value: X.XX M$ | Tier: High
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.x
- [Anaconda](https://www.anaconda.com/) or pip

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```bash
jupyter notebook ML_project_2_3.ipynb
```

> Run all cells in order. Assignment 3 depends on variables defined in Assignment 2.

---

## 🔑 Key Findings

- **Polynomial Regression at degree 4 with Ridge** was the best regression baseline (R²=0.889), outperforming even SVR on pure accuracy
- **Stacking Ensemble** was the overall best classifier — 86.78% accuracy with the smallest generalization gap (0.042) and lowest variance across CV folds (std=0.0016)
- **SVM had the best generalization** among individual models (gap=0.032), making it the most reliable single-model choice
- **KNN consistently overfits** on this dataset — both regression and classification showed ~20% train/test gaps despite tuning
- **The `High` player tier is hardest to classify** across all models — it lies between `Mid` and `Elite` with no sharp boundary
- **Ridge outperforms Lasso** when many OHE features are present — Lasso aggressively zeros country/position columns that carry small but meaningful signal

---

## 📝 Notes

- All preprocessing is fit on training data only and applied to test — no data leakage
- `OutlierClipper` is a custom `BaseEstimator / TransformerMixin` compatible with sklearn pipelines
- SVR used 3-fold CV (vs 5-fold elsewhere) for computational efficiency — results are directionally consistent
- `results.json` is auto-generated by the final cell and stores all model metrics
