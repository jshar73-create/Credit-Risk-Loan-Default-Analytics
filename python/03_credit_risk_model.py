import pandas as pd
from pathlib import Path 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from sklearn.model_selection import (TimeSeriesSplit,cross_val_score,GridSearchCV)

# 1. Load Cleaned Dataset
base_path = Path(__file__).resolve().parent.parent

file_path = "../data/cleaned_loans.csv"

df = pd.read_csv(file_path)

print("Cleaned dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# 2. Prepare Features and Target

y = df["default"]

X = df[["revenue","dti","loan_amount","fico_score","emp_length","purpose","home_ownership","state","loan_to_income","issue_year"]]

# 3. Time-Based Train/Test Split

train_data = df[df["issue_year"] <= 2016]

test_data = df[df["issue_year"] >= 2017]

X_train = train_data[["revenue","dti","loan_amount","fico_score","emp_length","purpose","home_ownership","state","loan_to_income","issue_year"]]

y_train = train_data["default"]

X_test = test_data[["revenue","dti","loan_amount","fico_score","emp_length","purpose","home_ownership","state","loan_to_income","issue_year"]]

y_test = test_data["default"]

print("\nTraining data:", X_train.shape)
print("Test data:", X_test.shape)

# 4. Preprocessing
numeric_features = ["revenue","dti","loan_amount","fico_score","loan_to_income","issue_year"]

categorical_features = ["emp_length","purpose","home_ownership","state"]

preprocessor = ColumnTransformer([("numeric",StandardScaler(),numeric_features),
                                  ("categorical", OneHotEncoder(handle_unknown="ignore"),categorical_features)])

# 5. Logistic Regression Model

model = Pipeline([("preprocessing",preprocessor),
                  ("logistic_regression",LogisticRegression(max_iter=1000))])

model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")

# 6. Baseline Model Evaluation

y_pred = model.predict(X_test)

print("\nPredictions completed successfully.")

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

print("\nBaseline Logistic Regression Performance")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. ROC-AUC and PR-AUC

y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)

pr_auc = average_precision_score(y_test, y_prob)

print("\nProbability-Based Metrics")

print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)

# 8. Threshold Analysis
thresholds = [0.50, 0.40, 0.30, 0.20, 0.10]

print("\nThreshold Analysis")

for threshold in thresholds:

    y_pred_threshold = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test,y_pred_threshold)

    recall = recall_score(y_test,y_pred_threshold)

    print(
        "Threshold:", threshold,
        "| Precision:", round(precision, 3),
        "| Recall:", round(recall, 3))

# 9. Number of Borrowers Classified as High Risk

print("\nDefault Risk Borrowers by Threshold")

for threshold in thresholds:

    y_pred_threshold = (y_prob >= threshold).astype(int)

    default_risk_count = (y_pred_threshold == 1).sum()

    default_risk_percentage = (default_risk_count / len(y_test)) * 100

    print(
        "Threshold:", threshold,
        "| Default Risk Borrowers:", default_risk_count,
        "| Percentage:", round(default_risk_percentage, 2), "%")

# 10. Time Series Cross-Validation

time_cv = TimeSeriesSplit(n_splits=3)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=time_cv,
    scoring="roc_auc",
    n_jobs=1
)

print("\nCross-validation ROC-AUC scores:", cv_scores)

print("Mean ROC-AUC:", cv_scores.mean())

# 11. Tune Logistic Regression

param_grid = {"logistic_regression__C": [0.01, 0.1, 1, 10]}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=time_cv,
    scoring="roc_auc",
    n_jobs=1
)

grid_search.fit(X_train, y_train)

print("\nBest C:", grid_search.best_params_)

print("Best CV ROC-AUC:", grid_search.best_score_)

best_model = grid_search.best_estimator_


# 12. Final Tuned Logistic Regression Predictions
# ============================================================

y_pred_best = best_model.predict(X_test)

print("\nFinal predictions completed!")

# 13. Final Tuned Logistic Regression Evaluation
# ============================================================

accuracy_best = accuracy_score(y_test,y_pred_best)

precision_best = precision_score(y_test, y_pred_best)

recall_best = recall_score(y_test,y_pred_best)

y_prob_best = best_model.predict_proba(X_test)[:, 1]

roc_auc_best = roc_auc_score(y_test, y_prob_best)

pr_auc_best = average_precision_score(y_test, y_prob_best)

print("\nFinal Model Performance")

print("Accuracy:", accuracy_best)

print("Precision:", precision_best)

print("Recall:", recall_best)

print("ROC-AUC:", roc_auc_best)

print("PR-AUC:", pr_auc_best)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test,y_pred_best))

# 14. Logistic Regression — Threshold 0.30
# ============================================================

y_prob_best = best_model.predict_proba(X_test)[:, 1]

threshold = 0.30

y_pred_30 = (y_prob_best >= threshold).astype(int)

precision_30 = precision_score(y_test,y_pred_30)

recall_30 = recall_score(y_test,y_pred_30)

print("\nLogistic Regression — Threshold 0.30")

print("Threshold:", threshold)

print("Precision:", precision_30)

print("Recall:", recall_30)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test,y_pred_30))

# 15. Random Forest Model
# ============================================================

rf_preprocessor = ColumnTransformer([
    ("numeric","passthrough",numeric_features),
    ("categorical",OneHotEncoder(handle_unknown="ignore"),categorical_features)])

rf_model = Pipeline([
    ("preprocessing",rf_preprocessor),
    ("random_forest",RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,n_jobs=1))])


# 16. Train Random Forest
# ============================================================

rf_model.fit(X_train,y_train)

print("\nRandom Forest training completed!")

y_pred_rf = rf_model.predict(X_test)

print("Random Forest predictions completed!")

# 17. Random Forest Evaluation
# ============================================================

accuracy_rf = accuracy_score(y_test,y_pred_rf)

precision_rf = precision_score( y_test,y_pred_rf,zero_division=0)

recall_rf = recall_score(y_test, y_pred_rf)

y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

roc_auc_rf = roc_auc_score(y_test,y_prob_rf)

pr_auc_rf = average_precision_score( y_test,y_prob_rf)

print("\nRandom Forest Performance")

print("Accuracy:", accuracy_rf)

print("Precision:", precision_rf)

print("Recall:", recall_rf)

print("ROC-AUC:", roc_auc_rf)

print("PR-AUC:", pr_auc_rf)

print("\nConfusion Matrix:")

print(confusion_matrix( y_test,y_pred_rf))

# 18. Random Forest — Threshold 0.30
# ============================================================

threshold = 0.30

y_pred_rf_30 = (y_prob_rf >= threshold).astype(int)

precision_rf_30 = precision_score(y_test,y_pred_rf_30)

recall_rf_30 = recall_score(y_test,y_pred_rf_30)

print("\nRandom Forest — Threshold 0.30")

print("Precision:", precision_rf_30)

print("Recall:", recall_rf_30)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test,y_pred_rf_30))

# 19. Final Model Selection
# ============================================================

print("\nFinal Model Selection")

print(
    "For this dataset and the specific credit-risk objective, "
    "Logistic Regression was selected as the primary model.")

print(
    "At the 0.30 threshold, Logistic Regression provided "
    "substantially higher recall than Random Forest.")

print(
    "Logistic Regression was also preferred because its "
    "relationships are easier to interpret for credit-risk analysis.")

print("Random Forest was used as a benchmark model for comparison.")

print("\nMachine Learning analysis completed successfully!")
