# 2_filter_data.py
# -------------------------------------------------------------------
# Step 2: Filter EEG signals with Butterworth bandpass (0.5–40 Hz) and notch (50 Hz), then resample to 500 Hz
# Метод: scipy.signal.butter + filtfilt для фильтрации и линейная интерполяция для ресемплинга
# Результаты сохраняются в results/02_filter_data/all_eeg_data_filtered.parquet

import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# Директории
INPUT_DIR = 'results/01_load_data'
OUTPUT_DIR = 'results/02_filter_data'
RAW_PARQUET = os.path.join(INPUT_DIR, 'all_eeg_data.parquet')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'all_eeg_data_filtered.parquet')

# Параметры фильтрации и ресемплинга
LOWCUT = 0.5  # Гц для bandpass
HIGHCUT = 40.0  # Гц для bandpass
NOTCH_FREQ = 50.0  # Гц для notch
FS_ORIG_NON = 512.0  # исходная частота non_stroke
FS_ORIG_ST = 500.0  # исходная частота stroke
FS_TARGET = 500.0  # целевая частота после ресемплинга


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_notch(freq, fs, quality=30):
    nyq = 0.5 * fs
    w0 = freq / nyq
    # интервал вокруг w0 для notch
    bw = w0 / quality
    b, a = butter(2, [w0 - bw, w0 + bw], btype='bandstop')
    return b, a


def resample_signal(signal, fs_old, fs_new):
    """
    Ресемплинг линейной интерполяцией от fs_old к fs_new
    """
    old_len = len(signal)
    new_len = int(np.round(old_len * fs_new / fs_old))
    old_idx = np.linspace(0, 1, old_len)
    new_idx = np.linspace(0, 1, new_len)
    return np.interp(new_idx, old_idx, signal)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading raw data from {RAW_PARQUET}...")
    df = pd.read_parquet(RAW_PARQUET)

    # Перечень каналов
    channels = [c for c in df.columns if c not in ('label', 'filename')]

    # Предварительно вычисляем фильтры для каждой Fs
    band_b_non, band_a_non = butter_bandpass(LOWCUT, HIGHCUT, FS_ORIG_NON)
    notch_b_non, notch_a_non = butter_notch(NOTCH_FREQ, FS_ORIG_NON)
    band_b_st, band_a_st = butter_bandpass(LOWCUT, HIGHCUT, FS_ORIG_ST)
    notch_b_st, notch_a_st = butter_notch(NOTCH_FREQ, FS_ORIG_ST)

    filtered_data = []
    for fname, group in df.groupby('filename'):
        print(f"Filtering and resampling file {fname}...")
        # Выбор набора фильтров по исходному Fs
        if fname.startswith('s'):
            fs = FS_ORIG_NON
            band_b, band_a = band_b_non, band_a_non
            notch_b, notch_a = notch_b_non, notch_a_non
        else:
            fs = FS_ORIG_ST
            band_b, band_a = band_b_st, band_a_st
            notch_b, notch_a = notch_b_st, notch_a_st

        signal = group[channels].values
        n_samples = signal.shape[0]

        # Фильтрация и ресемплинг по каждому каналу
        filtered = np.zeros((int(np.round(n_samples * FS_TARGET / fs)), len(channels)))
        for i in range(len(channels)):
            ch_data = signal[:, i]
            # bandpass
            ch_f = filtfilt(band_b, band_a, ch_data)
            # notch
            ch_f = filtfilt(notch_b, notch_a, ch_f)
            # ресемплинг
            ch_r = resample_signal(ch_f, fs_old=fs, fs_new=FS_TARGET)
            filtered[:, i] = ch_r

        # Сборка нового DataFrame
        new_len = filtered.shape[0]
        new_df = pd.DataFrame(filtered, columns=channels)
        new_df['label'] = group['label'].iloc[0]
        new_df['filename'] = fname
        filtered_data.append(new_df)

    result_df = pd.concat(filtered_data, ignore_index=True)
    result_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Filtered and resampled data saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
