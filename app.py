!pip install numpy==1.26.4
!pip install pandas==2.2.2
!pip install scipy==1.14.1
!pip install matplotlib==3.9.2
!pip install seaborn==0.13.2
!pip install scikit-learn==1.4.1.post1
!pip install joblib==1.4.2





import os
import gc
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

DATA_PATH = "train_delay.csv"
TARGET_COL = "delay_flag"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

if TARGET_COL not in data.columns:
    raise ValueError("Target column missing")

! pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class 

AV = AutoViz_Class()

import matplotlib.pyplot as plt
%matplotlib INLINE
filename = 'train_delay.csv'
sep =","
dft = AV.AutoViz(
    filename  
)
!

label_encoders = {}

for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

joblib.dump(label_encoders, os.path.join(MODEL_DIR, "all_label_encoders.joblib"))

X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

pipeline_preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=300, n_jobs=1),

    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        n_jobs=1,
        random_state=RANDOM_STATE
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=100,
        max_depth=12,
        n_jobs=1,
        random_state=RANDOM_STATE
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=1,
        random_state=RANDOM_STATE
    )
}

for name, model in models.items():
    print("\n" + "="*80)
    print("MODEL:", name)

    try:
        pipeline = Pipeline([
            ("preprocess", pipeline_preprocess),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
        print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
        print("Train F1:", f1_score(y_train, y_train_pred, average="macro"))
        print("Test F1:", f1_score(y_test, y_test_pred, average="macro"))

        print("\nTEST REPORT")
        print(classification_report(y_test, y_test_pred))

        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.joblib"))

    except Exception as e:
        print("Error:", e)

    finally:
        gc.collect()

USE_SHAP = True

if USE_SHAP:
    print("\nRunning SHAP Analysis...")

    import shap
    import matplotlib.pyplot as plt

    best_model_name = "XGBoost"
    pipeline = joblib.load(os.path.join(MODEL_DIR, f"{best_model_name}.joblib"))

    X_sample = X_test.iloc[:300]

    X_transformed = pipeline.named_steps["preprocess"].transform(X_sample)

    model = pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    feature_names = X.columns.tolist()

    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)
    plt.show()


print("\nSimple Feature Insights:\n")

shap_array = shap_values if not isinstance(shap_values, list) else shap_values[0]

importance = np.abs(shap_array).mean(axis=0)

for i in np.argsort(importance)[-5:][::-1]:
    feature = feature_names[i]
    impact = shap_array[:, i].mean()

    if impact > 0:
        print(f"• {feature}: increases prediction (positive impact)")
    else:
        print(f"• {feature}: decreases prediction (negative impact)")

print("\nSimple Feature Insights:\n")

TOP_N = 10   # 🔥 change this to 15, 20, etc.

shap_array = shap_values if not isinstance(shap_values, list) else shap_values[0]

importance = np.abs(shap_array).mean(axis=0)

# Get top N features
top_indices = np.argsort(importance)[-TOP_N:][::-1]

for rank, i in enumerate(top_indices, start=1):
    feature = feature_names[i]
    impact = shap_array[:, i].mean()

    if impact > 0:
        print(f"{rank}. {feature} → increases prediction (positive impact 📈)")
    else:
        print(f"{rank}. {feature} → decreases prediction (negative impact 📉)")




















