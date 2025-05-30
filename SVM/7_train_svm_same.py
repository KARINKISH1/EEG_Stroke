# 7_train_svm.py
# -------------------------------------------------------------------
# Step 7: Train and evaluate calibrated SVM on selected features
# Метод: StandardScaler + SVM(RBF) + Platt scaling (CalibratedClassifierCV)
# Вход: results/06_feature_selection_mi/features_selected_mi.csv
# Результаты: модель и метрики в results/07_train_svm/
# Изменение: Разбиение на train/test по именам файлов в колонке 'filename'

import os
import pandas as pd
# train_test_split больше не нужен для этого типа разбиения
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json

# Пути
INPUT_FILE = '../results/06_feature_selection_mi/features_selected_mi.csv'
OUTPUT_DIR = '../results/07_train_svm'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'svm_model.pkl')
METRICS_FILE = os.path.join(OUTPUT_DIR, 'metrics.json')
# TEST_SIZE больше не используется для разбиения, но можно оставить как метаинформацию
# TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading selected features from {INPUT_FILE}...")
    # При загрузке не указываем index_col=0, чтобы колонка 'filename' осталась доступной
    # или убедимся, что 'filename' не используется как индекс на предыдущих шагах
    df = pd.read_csv(INPUT_FILE)  # Убираем index_col=0, если filename в колонке

    # Убедимся, что колонка 'filename' существует
    if 'filename' not in df.columns:
        print("Error: 'filename' column not found in the input CSV. Cannot split by filename.")
        return  # Или поднимите ошибку

    # Определяем список файлов для тестовой выборки
    # s_files = [f's{i:02d}.csv' for i in range(1, 11)]
    # sub_files = [f'sub-{i:02d}_task-motor-imagery_eeg.csv' for i in range(1, 11)]

    # Файлы для первого тестового набора (первые 8 из 15 каждого типа)
    # test_files = s_files[:5] + sub_files[:5]
    # print("Files for Test Set 1:", test_files)

    test_files = [f's{i:02d}.csv' for i in range(1, 11)] + \
                 [f'sub-{i:02d}_task-motor-imagery_eeg.csv' for i in range(1, 11)]

    print(f"Splitting data based on filenames. Test files: {test_files}")

    # Создаем маски для тестовой и обучающей выборок
    test_mask = df['filename'].isin(test_files)
    train_mask = ~test_mask  # Инвертируем маску для обучающей выборки

    # Разделяем DataFrame на train и test по маскам
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    # Разделяем на X и y для train и test
    # Удаляем колонки 'label' и 'filename' из признаков (X)
    X_train = df_train.drop(columns=['label', 'filename'])
    y_train = df_train['label']

    X_test = df_test.drop(columns=['label', 'filename'])
    y_test = df_test['label']

    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # Проверка, что тестовая выборка не пуста
    if len(y_test) == 0:
        print("Error: Test dataset is empty based on the specified filenames.")
        return

    # Сборка пайплайна: шкалирование + калиброванный SVM
    scaler = StandardScaler()
    # probability=True обязательно для CalibratedClassifierCV
    svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    # cv=5 используется для внутренней кросс-валидации калибратора на обучающих данных
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
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # Вероятность положительного класса

    # Вычисление метрик
    # accuracy = (y_pred == y_test).mean() # Более точный расчет через classification_report
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        'roc_auc': auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

    # Добавляем точность из отчета
    if 'accuracy' in report:
        metrics['accuracy'] = report['accuracy']
    # Или просто
    # metrics['accuracy'] = (y_pred == y_test).mean()

    # Сохраняем модель
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    # Сохраняем метрики
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {METRICS_FILE}")
    print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}, ROC AUC: {auc:.4f}")  # Выводим accuracy из словаря


if __name__ == '__main__':
    main()
