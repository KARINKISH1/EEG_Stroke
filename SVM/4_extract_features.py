# 4_extract_features.py
# -------------------------------------------------------------------
# Step 4: Extract basic features (mean, std, min, max) from ICA-cleaned EEG data for SVM
# Данные: results/03_ica_clean/all_eeg_data_ica.parquet
# Результаты: results/04_extract_features/features_basic.csv

import os
import pandas as pd

# Директории
INPUT_DIR = '../results/03_ica_clean'
OUTPUT_DIR = '../results/04_extract_features'
INPUT_FILE = os.path.join(INPUT_DIR, 'all_eeg_data_ica.parquet')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'features_basic.csv')


def extract_basic_features():
    """
    Загружает ICA-очищенные данные, группирует по файлам и вычисляет базовые
    статистические признаки (mean, std, min, max) для каждого канала.
    Сохраняет результирующую матрицу в CSV.
    """
    # Создаём папку для результатов
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading cleaned data from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    # Определяем каналы (все столбцы, кроме label и filename)
    channels = [c for c in df.columns if c not in ('label', 'filename')]

    # Формируем словарь агрегаций: mean, std, min, max для каждого канала
    agg_funcs = {ch: ['mean', 'std', 'min', 'max'] for ch in channels}
    print('Aggregating features per file...')
    grouped = df.groupby('filename').agg(agg_funcs)

    # Выпрямляем MultiIndex колонок
    grouped.columns = [f"{ch}_{stat}" for ch, stat in grouped.columns]

    # Добавляем метку (одна на файл)
    labels = df.groupby('filename')['label'].first()
    features = grouped.copy()
    features['label'] = labels

    # Сохраняем в CSV
    features.to_csv(OUTPUT_FILE, index=True)
    print(f"Features saved to {OUTPUT_FILE}")
    print('Feature matrix shape:', features.shape)
    print('\nFirst 5 rows:')
    print(features.head())


if __name__ == '__main__':
    extract_basic_features()
