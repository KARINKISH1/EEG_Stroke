# 1_load_data.py
# -------------------------------------------------------------------
# Step 1: Load and concatenate EEG CSV files into a single DataFrame
# Теперь с организацией директорий: входные данные в data/raw, результаты в results/01_load_data

import os
import glob
import pandas as pd

# Путь к сырым данным и результатам
RAW_DIR = '../common_rows_csv'  # Ожидает подпапки 'non_stroke' и 'stroke'
RESULTS_DIR = 'results/01_load_data'  # Здесь будут сохраняться результаты этапа


def load_and_concatenate() -> pd.DataFrame:
    """
    Читает все CSV-файлы из data/raw/non_stroke и data/raw/stroke,
    присваивает метки и объединяет в один DataFrame.
    """
    # Создаём папку для результатов, если её нет
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data_frames = []
    for class_name, class_value in [('non_stroke', 0), ('stroke', 1)]:
        folder = os.path.join(RAW_DIR, class_name)
        csv_files = glob.glob(os.path.join(folder, '*.csv'))
        print(f"[{class_name}] найдено файлов: {len(csv_files)}")

        for file_path in csv_files:
            df = pd.read_csv(file_path)
            df['label'] = class_value
            df['filename'] = os.path.basename(file_path)
            data_frames.append(df)

    full_df = pd.concat(data_frames, ignore_index=True)
    return full_df


def main():
    # Загрузка и объединение данных
    df = load_and_concatenate()

    # Основная статистика
    total_rows = len(df)
    unique_files = df['filename'].nunique()
    lengths = df['filename'].value_counts()

    print(f"Всего строк: {total_rows}")
    print(f"Уникальных файлов: {unique_files}")
    print(f"Строк на файл: min={lengths.min()}, max={lengths.max()}, mean≈{lengths.mean():.1f}")
    print("Баланс меток по строкам:")
    print(df['label'].value_counts())
    print("\nПример строк:")
    print(df.head())

    # Сохранение объединённого DataFrame в Parquet для эффективности
    output_path = os.path.join(RESULTS_DIR, 'all_eeg_data.parquet')
    df.to_parquet(output_path, index=False)
    print(f"\nСохранено all_eeg_data.parquet в {RESULTS_DIR}")


if __name__ == '__main__':
    main()
