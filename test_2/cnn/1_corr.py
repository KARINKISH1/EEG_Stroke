import os
import pandas as pd
import numpy as np

# Директории
INPUT_FILE = '../results/03_ica_clean/all_eeg_data_ica.parquet'
OUTPUT_DIR = '../results/02_cnn_lstm_corr'  # Возможно, стоит назвать 04 или 05, в зависимости от порядка в пайплайне
OUTPUT_FILE = os.path.join(OUTPUT_DIR,
                           'data_selected_corr.parquet')  # Изменено название файла, т.к. это не просто features, а данные с filename и label

# Порог корреляции для отбора (абсолютное значение)
CORR_THRESHOLD = 0.90


def select_channels_by_correlation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading data from {INPUT_FILE}...")

    # Читаем Parquet файл. Убедитесь, что filename и label являются обычными колонками.
    try:
        df = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    except Exception as e:
        print(f"Error loading data from {INPUT_FILE}: {e}")
        return

    # Убедимся, что нужные колонки существуют
    required_cols = ['label', 'filename']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Required columns {required_cols} not found in the dataframe.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Отделяем метку 'label' и имя файла 'filename' от признаков (каналов)
    labels = df['label']
    filenames = df['filename']  # Сохраняем колонку filename
    # Выбираем колонки, которые НЕ являются label или filename - это наши признаки (каналы)
    features = df.drop(columns=required_cols)

    print(f"Original data shape: {df.shape}")
    print(f"Feature shape before correlation check: {features.shape}")
    print(f"Number of channels: {features.shape[1]}")

    # Вычисляем абсолютную корреляционную матрицу для признаков
    print("Computing absolute correlation matrix...")
    corr_matrix = features.corr().abs()

    # Создаем структуру для хранения признаков, которые нужно удалить
    # Используем set, чтобы избежать дубликатов
    features_to_drop = set()

    # Итерируемся по верхнему треугольнику матрицы корреляции (без диагонали)
    print(f"Identifying channels to drop with pairwise |corr| > {CORR_THRESHOLD}...")
    # Используем stack() для преобразования верхнего треугольника в Series
    # Сбрасываем индекс для доступа к именам пар каналов
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)).stack()

    # Итерируемся по парам и их значениям корреляции
    # item() получает значение и имена индекса (multi-index из имен каналов)
    for (feat1, feat2), corr_value in upper_tri.items():
        if corr_value > CORR_THRESHOLD:
            # Если корреляция выше порога, добавляем ОДИН ИЗ ПРИЗНАКОВ пары в set для удаления
            # Здесь мы просто выбираем feat2. Можно добавить более сложную логику,
            # например, сохранить тот канал, у которого меньше пропусков или больше дисперсия,
            # но для простоты удалим feat2.
            features_to_drop.add(feat2)
            # print(f"  Marking '{feat2}' for drop (correlated with '{feat1}', corr={corr_value:.4f})")

    print(f"Identified {len(features_to_drop)} channels to drop.")
    print(f"Channels to drop: {list(features_to_drop)}")  # Преобразуем set в list для вывода

    # Определяем признаки для сохранения
    all_features = features.columns.tolist()
    selected_feature_names = [feat for feat in all_features if feat not in features_to_drop]

    print(f"Original channel count: {len(all_features)}")
    print(f"Selected channel count: {len(selected_feature_names)}")

    # Оставляем только выбранные признаки в DataFrame features
    selected_features_df = features[selected_feature_names].copy()

    # Собираем итоговую матрицу с сохраненными признаками, label и filename
    # Важно сохранить filename для последующего разделения на train/test
    result_df = pd.concat([selected_features_df, labels, filenames], axis=1)

    # Сохраняем результат в Parquet файл. Включаем index=False, т.к. index не несёт полезной инф.
    print(f"Saving selected data (including label and filename) to {OUTPUT_FILE}...")
    try:
        result_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"Selected data saved successfully. Final shape: {result_df.shape}")
    except Exception as e:
        print(f"Error saving data to {OUTPUT_FILE}: {e}")


if __name__ == '__main__':
    select_channels_by_correlation()
