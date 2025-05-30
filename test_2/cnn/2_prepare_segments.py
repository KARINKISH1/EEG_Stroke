# # # cnn/2_prepare_segments.py
# # # -------------------------------------------------------------------
# # # Step 1: Segment ICA-cleaned EEG data for CNN/CNN+LSTM models
# # # Метод: разбивка каждого файла на перекрывающиеся окна фиксированной длины
# # # Вход: results/03_ica_clean/all_eeg_data_ica.parquet
# # # Выход: results/08_cnn_data/eeg_segments.npz
# #
# # import os
# # import pandas as pd
# # import numpy as np
# #
# # # Пути
# # INPUT_FILE = '../results/03_ica_clean/all_eeg_data_ica.parquet'
# # OUTPUT_DIR = '../results/08_cnn_data'
# # OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'eeg_segments.npz')
# #
# # # Параметры сегментации
# # WINDOW_SIZE = 500  # длина окна в отсчетах (~1 с при 500 Hz)
# # OVERLAP = 0.5  # доля перекрытия окон (50%)
# # STRIDE = int(WINDOW_SIZE * (1 - OVERLAP))
# #
# #
# # def main():
# #     # Создаем папку для данных, если не существует
# #     os.makedirs(OUTPUT_DIR, exist_ok=True)
# #
# #     # Загружаем очищенные ICA данные
# #     print(f"Loading ICA-cleaned data from {INPUT_FILE}...")
# #     df = pd.read_parquet(INPUT_FILE)
# #
# #     # Определяем список каналов (все, кроме label и filename)
# #     channels = [c for c in df.columns if c not in ('label', 'filename')]
# #
# #     X_segments = []  # список для сегментов
# #     y_labels = []  # список меток
# #
# #     # Разбиваем данные по файлам
# #     for fname, group in df.groupby('filename'):
# #         print(f"Segmenting file {fname}...")
# #         signal = group[channels].values  # форма: (n_samples_file, n_channels)
# #         label = group['label'].iloc[0]
# #         n_samples = signal.shape[0]
# #
# #         # Скользим окном по сигналу
# #         for start in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
# #             end = start + WINDOW_SIZE
# #             segment = signal[start:end, :]
# #             X_segments.append(segment)
# #             y_labels.append(label)
# #
# #     # Преобразуем в numpy-массивы
# #     X = np.stack(X_segments)  # форма: (n_segments, WINDOW_SIZE, n_channels)
# #     y = np.array(y_labels)  # форма: (n_segments,)
# #
# #     # Сохраняем сжато в NPZ
# #     print(f"Saving {X.shape[0]} segments to {OUTPUT_FILE}...")
# #     np.savez_compressed(OUTPUT_FILE, X=X, y=y, channels=channels)
# #     print("Segmentation complete.")
# #
# #
# # if __name__ == '__main__':
# #     main()
#
# # cnn/1_prepare_train_test_segments.py
# # -------------------------------------------------------------------
# # Step 1: Segment ICA-cleaned EEG data for TRAIN and TEST sets
# # Метод: разбивка файлов на перекрывающиеся окна, сохранение в отдельные файлы
# # Вход: results/03_ica_clean/all_eeg_data_ica.parquet
# # Выход: results/08_cnn_data/eeg_train_segments.npz
# #         results/08_cnn_data/eeg_test_segments.npz
#
# import os
# import pandas as pd
# import numpy as np
#
# # Пути
# INPUT_FILE = '../results/02_cnn_lstm_corr/data_selected_corr.parquet'
# OUTPUT_DIR = '../results/08_cnn_data'
# OUTPUT_FILE_TRAIN = os.path.join(OUTPUT_DIR, 'eeg_train_segments.npz')  # Файл для обучающих сегментов
# OUTPUT_FILE_TEST = os.path.join(OUTPUT_DIR, 'eeg_test_segments.npz')  # Файл для тестовых сегментов
#
# # Параметры сегментации
# WINDOW_SIZE = 500  # длина окна в отсчетах (~1 с при 500 Hz)
# OVERLAP = 0.5  # доля перекрытия окон (50%)
# STRIDE = int(WINDOW_SIZE * (1 - OVERLAP))
#
# # Список файлов, которые ДОЛЖНЫ пойти в ТЕСТОВУЮ выборку
# TEST_FILES = [f's{i:02d}.csv' for i in range(1, 11)] + \
#              [f'sub-{i:02d}_task-motor-imagery_eeg.csv' for i in range(1, 11)]
#
#
# def segment_dataframe(df_to_segment, channels, window_size, stride):
#     """Сегментирует данные в DataFrame по именам файлов."""
#     X_segments = []
#     y_labels = []
#
#     unique_files = df_to_segment['filename'].unique()
#     print(f"Segmenting data from {len(unique_files)} files...")
#
#     for fname, group in df_to_segment.groupby('filename'):
#         # print(f"  Processing file {fname}...") # Можно включить для более детального вывода
#         signal = group[channels].values  # форма: (n_samples_file, n_channels)
#         label = group['label'].iloc[0]  # Предполагаем, что метка постоянна в пределах файла
#         n_samples = signal.shape[0]
#
#         # Проверяем, достаточно ли в файле данных для хотя бы одного сегмента
#         if n_samples < window_size:
#             print(
#                 f"  Warning: File {fname} is too short ({n_samples} samples) for window size {window_size}. Skipping.")
#             continue
#
#         # Скользим окном по сигналу
#         for start in range(0, n_samples - window_size + 1, stride):
#             end = start + window_size
#             segment = signal[start:end, :]
#             X_segments.append(segment)
#             y_labels.append(label)
#
#     return X_segments, y_labels
#
#
# def main():
#     # Создаем папку для данных, если не существует
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#     # Загружаем очищенные ICA данные
#     print(f"Loading ICA-cleaned data from {INPUT_FILE}...")
#     df = pd.read_parquet(INPUT_FILE)
#
#     # Убедимся, что колонка 'filename' существует
#     if 'filename' not in df.columns:
#         print("Error: 'filename' column not found in the input Parquet. Cannot split by filename.")
#         return
#
#     # Определяем список каналов (все, кроме label и filename)
#     channels = [c for c in df.columns if c not in ('label', 'filename')]
#
#     # --- Разделение DataFrame на обучающую и тестовую части по именам файлов ---
#
#     # Фильтруем DataFrame: оставляем ТОЛЬКО строки, ЧЕЙ filename В СПИСКЕ TEST_FILES (для теста)
#     df_test = df[df['filename'].isin(TEST_FILES)].copy()
#
#     # Фильтруем DataFrame: оставляем ТОЛЬКО строки, ЧЕЙ filename НЕ В СПИСКЕ TEST_FILES (для обучения)
#     df_train = df[~df['filename'].isin(TEST_FILES)].copy()
#
#     print("-" * 30)
#     print(f"Total files loaded: {df['filename'].nunique()}")
#     print(f"Files for training: {df_train['filename'].nunique()}")
#     print(f"Files for testing:  {df_test['filename'].nunique()}")
#     print("-" * 30)
#
#     # --- Сегментация обучающих данных ---
#     print("Segmenting training data...")
#     if df_train.empty:
#         print("Warning: Training DataFrame is empty. No training segments will be generated.")
#         X_segments_train, y_labels_train = [], []
#     else:
#         X_segments_train, y_labels_train = segment_dataframe(
#             df_train, channels, WINDOW_SIZE, STRIDE
#         )
#
#     # Проверяем, были ли сгенерированы какие-либо обучающие сегменты и сохраняем их
#     if X_segments_train:
#         X_train = np.stack(X_segments_train)  # форма: (n_segments_train, WINDOW_SIZE, n_channels)
#         y_train = np.array(y_labels_train)  # форма: (n_segments_train,)
#         print(f"Saving {X_train.shape[0]} training segments to {OUTPUT_FILE_TRAIN}...")
#         np.savez_compressed(OUTPUT_FILE_TRAIN, X=X_train, y=y_train, channels=channels)
#         print("Training segmentation data saved.")
#     else:
#         print("No training segments generated.")
#
#     print("-" * 30)
#
#     # --- Сегментация тестовых данных ---
#     print("Segmenting testing data...")
#     if df_test.empty:
#         print("Warning: Testing DataFrame is empty. No testing segments will be generated.")
#         X_segments_test, y_labels_test = [], []
#     else:
#         X_segments_test, y_labels_test = segment_dataframe(
#             df_test, channels, WINDOW_SIZE, STRIDE
#         )
#
#     # Проверяем, были ли сгенерированы какие-либо тестовые сегменты и сохраняем их
#     if X_segments_test:
#         X_test = np.stack(X_segments_test)  # форма: (n_segments_test, WINDOW_SIZE, n_channels)
#         y_test = np.array(y_labels_test)  # форма: (n_segments_test,)
#         print(f"Saving {X_test.shape[0]} test segments to {OUTPUT_FILE_TEST}...")
#         np.savez_compressed(OUTPUT_FILE_TEST, X=X_test, y=y_test, channels=channels)
#         print("Test segmentation data saved.")
#     else:
#         print("No test segments generated.")
#
#     print("-" * 30)
#     print("Overall segmentation process complete.")
#
#
# if __name__ == '__main__':
#     main()

import os
import pandas as pd
import numpy as np

# Пути
INPUT_FILE = '../results/02_cnn_lstm_corr/data_selected_corr.parquet'
OUTPUT_DIR = '../results/08_cnn_data'
OUTPUT_FILE_TRAIN = os.path.join(OUTPUT_DIR, 'eeg_train_segments.npz')  # Файл для обучающих сегментов
OUTPUT_FILE_TEST_1 = os.path.join(OUTPUT_DIR, 'eeg_test_segments_1.npz')  # Файл для первого тестового набора
OUTPUT_FILE_TEST_2 = os.path.join(OUTPUT_DIR, 'eeg_test_segments_2.npz')  # Файл для второго (экстра) тестового набора

# Параметры сегментации
WINDOW_SIZE = 500  # длина окна в отсчетах (~1 с при 500 Hz)
OVERLAP = 0.5  # доля перекрытия окон (50%)
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP))

# --- Определение файлов для разных наборов данных ---

# # Исходные списки файлов
# s_files = [f's{i:02d}.csv' for i in range(1, 11)]
# sub_files = [f'sub-{i:02d}_task-motor-imagery_eeg.csv' for i in range(1, 11)]

# # Файлы для первого тестового набора (первые 5 из каждого типа)
# TEST_FILES_1 = s_files[:5] + sub_files[:5]
# print("Files for Test Set 1:", TEST_FILES_1)

# Определяем список файлов для тестовой выборки
s_files = [f's{i:02d}.csv' for i in range(1, 16)]
sub_files = [f'sub-{i:02d}_task-motor-imagery_eeg.csv' for i in range(1, 16)]

# Файлы для первого тестового набора (первые 8 из 15 каждого типа)
TEST_FILES_1 = s_files[:7] + sub_files[:8]
print("Files for Test Set 1:", TEST_FILES_1)

# Файлы для второго (экстра) тестового набора (с 6 по 10 из каждого типа)
TEST_FILES_2 = s_files[7:15] + sub_files[8:15]
print("Files for Test Set 2 (Extra):", TEST_FILES_2)

# Список всех файлов, которые пойдут в любой из тестовых наборов
ALL_TEST_FILES = TEST_FILES_1 + TEST_FILES_2


def segment_dataframe(df_to_segment, channels, window_size, stride, set_name="Unknown"):
    """Сегментирует данные в DataFrame по именам файлов."""
    X_segments = []
    y_labels = []

    unique_files = df_to_segment['filename'].unique()
    print(f"Segmenting data for {set_name} set from {len(unique_files)} files...")

    for fname, group in df_to_segment.groupby('filename'):
        # print(f"  Processing file {fname} for {set_name}...") # Можно включить для более детального вывода
        signal = group[channels].values  # форма: (n_samples_file, n_channels)
        label = group['label'].iloc[0]  # Предполагаем, что метка постоянна в пределах файла
        n_samples = signal.shape[0]

        # Проверяем, достаточно ли в файле данных для хотя бы одного сегмента
        if n_samples < window_size:
            print(
                f"  Warning: File {fname} ({set_name} set) is too short ({n_samples} samples) for window size {window_size}. Skipping.")
            continue

        # Скользим окном по сигналу
        for start in range(0, n_samples - window_size + 1, stride):
            end = start + window_size
            segment = signal[start:end, :]
            X_segments.append(segment)
            y_labels.append(label)

    return X_segments, y_labels


def main():
    # Создаем папку для данных, если не существует
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Загружаем очищенные ICA данные
    print(f"Loading ICA-cleaned data from {INPUT_FILE}...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input Parquet file not found at {INPUT_FILE}")
        return
    except Exception as e:
        print(f"Error loading input Parquet file: {e}")
        return

    # Убедимся, что колонка 'filename' существует
    if 'filename' not in df.columns:
        print("Error: 'filename' column not found in the input Parquet. Cannot split by filename.")
        return

    # Определяем список каналов (все, кроме label и filename)
    channels = [c for c in df.columns if c not in ('label', 'filename')]

    # --- Разделение DataFrame на обучающую и тестовые части по именам файлов ---

    # DataFrame для первого тестового набора (файлы только из TEST_FILES_1)
    df_test_1 = df[df['filename'].isin(TEST_FILES_1)].copy()

    # DataFrame для второго тестового набора (файлы только из TEST_FILES_2)
    df_test_2 = df[df['filename'].isin(TEST_FILES_2)].copy()

    # DataFrame для обучающего набора (файлы, которые НЕ входят ни в TEST_FILES_1, ни в TEST_FILES_2)
    df_train = df[~df['filename'].isin(ALL_TEST_FILES)].copy()

    print("-" * 30)
    print(f"Total unique files in loaded data: {df['filename'].nunique()}")
    print(f"Unique files for training: {df_train['filename'].nunique()}")
    print(f"Unique files for Test Set 1: {df_test_1['filename'].nunique()}")
    print(f"Unique files for Test Set 2 (Extra): {df_test_2['filename'].nunique()}")
    print("-" * 30)

    # --- Сегментация и сохранение обучающих данных ---
    if df_train.empty:
        print("Warning: Training DataFrame is empty. No training segments will be generated.")
    else:
        X_segments_train, y_labels_train = segment_dataframe(
            df_train, channels, WINDOW_SIZE, STRIDE, set_name="Training"
        )
        if X_segments_train:
            X_train = np.stack(X_segments_train)  # форма: (n_segments_train, WINDOW_SIZE, n_channels)
            y_train = np.array(y_labels_train)  # форма: (n_segments_train,)
            print(f"Saving {X_train.shape[0]} training segments to {OUTPUT_FILE_TRAIN}...")
            np.savez_compressed(OUTPUT_FILE_TRAIN, X=X_train, y=y_train, channels=channels)
            print("Training segmentation data saved.")
        else:
            print("No training segments generated.")

    print("-" * 30)

    # --- Сегментация и сохранение первого тестового набора ---
    if df_test_1.empty:
        print("Warning: Test Set 1 DataFrame is empty. No segments will be generated.")
    else:
        X_segments_test_1, y_labels_test_1 = segment_dataframe(
            df_test_1, channels, WINDOW_SIZE, STRIDE, set_name="Test Set 1"
        )
        if X_segments_test_1:
            X_test_1 = np.stack(X_segments_test_1)  # форма: (n_segments_test_1, WINDOW_SIZE, n_channels)
            y_test_1 = np.array(y_labels_test_1)  # форма: (n_segments_test_1,)
            print(f"Saving {X_test_1.shape[0]} test segments for Set 1 to {OUTPUT_FILE_TEST_1}...")
            np.savez_compressed(OUTPUT_FILE_TEST_1, X=X_test_1, y=y_test_1, channels=channels)
            print("Test Set 1 segmentation data saved.")
        else:
            print("No segments generated for Test Set 1.")

    print("-" * 30)

    # --- Сегментация и сохранение второго (экстра) тестового набора ---
    if df_test_2.empty:
        print("Warning: Test Set 2 (Extra) DataFrame is empty. No segments will be generated.")
    else:
        X_segments_test_2, y_labels_test_2 = segment_dataframe(
            df_test_2, channels, WINDOW_SIZE, STRIDE, set_name="Test Set 2 (Extra)"
        )
        if X_segments_test_2:
            X_test_2 = np.stack(X_segments_test_2)  # форма: (n_segments_test_2, WINDOW_SIZE, n_channels)
            y_test_2 = np.array(y_labels_test_2)  # форма: (n_segments_test_2,)
            print(f"Saving {X_test_2.shape[0]} test segments for Set 2 to {OUTPUT_FILE_TEST_2}...")
            np.savez_compressed(OUTPUT_FILE_TEST_2, X=X_test_2, y=y_test_2, channels=channels)
            print("Test Set 2 (Extra) segmentation data saved.")
        else:
            print("No segments generated for Test Set 2 (Extra).")

    print("-" * 30)
    print("Overall segmentation process complete.")


if __name__ == '__main__':
    main()
