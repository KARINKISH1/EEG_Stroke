# 3_ica_clean.py
# -------------------------------------------------------------------
# Step 3: ICA artifact removal (FastICA) on filtered EEG data
# Метод: sklearn.decomposition.FastICA для разложения на независимые компоненты,
#        обнаружение артефактных компонентов по куртозису и удаление их
# Результат сохраняется в results/03_ica_clean/all_eeg_data_ica.parquet

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis

# Директории
INPUT_DIR = 'results/02_filter_data'
OUTPUT_DIR = 'results/03_ica_clean'
INPUT_FILE = os.path.join(INPUT_DIR, 'all_eeg_data_filtered.parquet')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'all_eeg_data_ica.parquet')

# Порог куртозиса, выше которого IC считается артефактным
KURTOSIS_THRESHOLD = 5.0


def remove_artifacts_via_ica(signal: np.ndarray, n_components: int):
    """
    Применяет FastICA к сигналу (n_samples, n_channels),
    вычисляет куртозис каждого IC и удаляет компоненты с |kurtosis| > threshold.
    Возвращает очищенный сигнал той же формы.
    """
    ica = FastICA(n_components=n_components, random_state=0)
    # Преобразование в независимые компоненты
    S = ica.fit_transform(signal)  # shape (n_samples, n_components)
    A = ica.mixing_  # shape (n_components, n_channels)

    # Оценка куртозиса по компонентам (вдоль временной оси)
    kurts = kurtosis(S, axis=0, fisher=False)
    artifact_idx = np.where(np.abs(kurts) > KURTOSIS_THRESHOLD)[0]
    print(f"  Detected {len(artifact_idx)} artifact IC(s): indices {artifact_idx}")

    # Обнуляем артефактные компоненты
    S_clean = S.copy()
    S_clean[:, artifact_idx] = 0

    # Реконструкция сигнала без артефактных IC
    cleaned = S_clean.dot(A.T)
    return cleaned


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading filtered data from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    channels = [c for c in df.columns if c not in ('label', 'filename')]
    cleaned_data = []

    # Применяем ICA к каждому файлу отдельно
    for fname, group in df.groupby('filename'):
        print(f"Processing file {fname} for ICA artifact removal...")
        signal = group[channels].values
        n_channels = signal.shape[1]

        # Удаляем артефакты
        cleaned_signal = remove_artifacts_via_ica(signal, n_components=n_channels)

        # Собираем DataFrame
        clean_df = pd.DataFrame(cleaned_signal, columns=channels)
        clean_df['label'] = group['label'].iloc[0]
        clean_df['filename'] = fname
        cleaned_data.append(clean_df)

    # Конкатенация и сохранение
    result_df = pd.concat(cleaned_data, ignore_index=True)
    result_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"ICA-cleaned data saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
