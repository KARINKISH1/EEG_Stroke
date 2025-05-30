# 6_feature_selection_mi.py
# -------------------------------------------------------------------
# Step 6: Feature selection via mutual information (filter) for SVM pipeline
# Метод: mutual_info_classif + SelectKBest для выбора наиболее информативных признаков
# Вход: results/05_feature_selection_corr/features_selected_corr.csv
# Выход: results/06_feature_selection_mi/features_selected_mi.csv

import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest

# Директории и параметры
INPUT_FILE = '../results/05_feature_selection_corr/features_selected_corr.csv'
OUTPUT_DIR = '../results/06_feature_selection_mi'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'features_selected_mi.csv')

# Число признаков для выбора (если доступно меньше, будет выбрано все)
K = 10


def select_by_mutual_info():
    # Создание папки для результатов
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading features from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, index_col=0)

    # Разделяем X и y
    X = df.drop(columns=['label'])
    y = df['label']

    # Вычисляем взаимную информацию
    print("Computing mutual information for each feature...")
    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    print("Mutual information scores (descending):")
    print(mi_series)

    # Отбор топ-K признаков
    k = min(K, X.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    selected_columns = X.columns[selector.get_support()]
    print(f"Selected top {k} features:")
    print(list(selected_columns))

    # Сбор итогового DataFrame
    selected_df = X[selected_columns].copy()
    selected_df['label'] = y

    # Сохранение
    selected_df.to_csv(OUTPUT_FILE, index=True)
    print(f"Selected features saved to {OUTPUT_FILE}")
    print(f"Resulting feature matrix shape: {selected_df.shape}")


if __name__ == '__main__':
    select_by_mutual_info()
