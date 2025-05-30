# 5_feature_selection_corr.py
# -------------------------------------------------------------------
# Step 5: Feature selection via correlation (filter) for SVM pipeline
# Метод: удаление признаков с высокой корреляцией (|corr| > threshold)
# Вход: results/04_extract_features/features_basic.csv
# Выход: results/05_feature_selection_corr/features_selected_corr.csv

import os
import pandas as pd
import numpy as np

# Директории
INPUT_FILE = '../results/04_extract_features/features_basic.csv'
OUTPUT_DIR = '../results/05_feature_selection_corr'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'features_selected_corr.csv')

# Порог корреляции для отбора
CORR_THRESHOLD = 0.90


def select_by_correlation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading features from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, index_col=0)

    # Отделяем метку 'label' от признаков
    labels = df['label']
    features = df.drop(columns=['label'])

    # Вычисляем абсолютную корреляционную матрицу
    print("Computing correlation matrix...")
    corr_matrix = features.corr().abs()

    # Формируем маску для верхнего треугольника, исключая диагональ
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    upper_tri = corr_matrix.where(mask)

    # Выбираем признаки для удаления
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > CORR_THRESHOLD)]
    print(f"Dropping {len(to_drop)} features with |corr| > {CORR_THRESHOLD}")

    # Оставляем только не скоррелированные признаки
    selected = features.drop(columns=to_drop)

    # Собираем итоговую матрицу с меткой
    result = selected.copy()
    result['label'] = labels

    # Сохраняем
    result.to_csv(OUTPUT_FILE)
    print(f"Selected features saved to {OUTPUT_FILE}")
    print(f"Original feature count: {features.shape[1]}, Selected: {selected.shape[1]}")


if __name__ == '__main__':
    select_by_correlation()
