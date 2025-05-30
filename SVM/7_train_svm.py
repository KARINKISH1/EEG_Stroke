# 7_train_svm.py
# -------------------------------------------------------------------
# Step 7: Train and evaluate calibrated SVM on selected features
# Метод: StandardScaler + SVM(RBF) + Platt scaling (CalibratedClassifierCV)
# Вход: results/06_feature_selection_mi/features_selected_mi.csv
# Результаты: модель и метрики в results/07_train_svm/

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json

# Пути
INPUT_FILE = '../results/06_feature_selection_mi/features_elescted_mi.csv'
OUTPUT_DIR = '../results/07_train_svm'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'svm_model.pkl')
METRICS_FILE = os.path.join(OUTPUT_DIR, 'metrics.json')
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading selected features from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, index_col=0)

    # Разделяем на X и y
    X = df.drop(columns=['label'])
    y = df['label']

    # Сплит с стратификацией
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # Сборка пайплайна: шкалирование + калиброванный SVM
    scaler = StandardScaler()
    svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    calibrated = CalibratedClassifierCV(estimator=svm, cv=5, method='sigmoid')
    pipeline = Pipeline([
        ('scaler', scaler),
        ('svm', calibrated)
    ])

    # Обучение
    print("Training calibrated SVM...")
    pipeline.fit(X_train, y_train)

    # Предсказания
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Вычисление метрик
    acc = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        'accuracy': acc,
        'roc_auc': auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

    # Сохраняем модель
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    # Сохраняем метрики
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {METRICS_FILE}")
    print(f"Accuracy: {acc:.4f}, ROC AUC: {auc:.4f}")


if __name__ == '__main__':
    main()
