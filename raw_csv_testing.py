# preprocess_single_csv.py
# -------------------------------------------------------------------
# EEG preprocessing pipeline for a single raw CSV file (channels only):
# 1) Select required channel columns
# 2) Apply Butterworth bandpass (0.5–40 Hz) + notch (50 Hz) + resample to 500 Hz
# 3) Remove ICA artifacts (FastICA + kurtosis)
# Input: one CSV file with raw channel data (no labels, no filename)
# Output: printed intermediate DataFrame shapes and samples
import joblib
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis

# --- Settings ---
# RAW_CSV = 'csv/stroke/sub-19_task-motor-imagery_eeg.csv'
RAW_CSV = 'csv/non_stroke/s01.csv'

# Path to the trained SVM model
SVM_MODEL_PATH = 'SVM_model/svm_model.pkl'  # <-- update if needed

required_channels = [
    'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Fcz', 'Fc4', 'Ft7', 'Ft8',
    'Cz', 'C3', 'C4', 'Cp3', 'Cp4', 'Tp7', 'Tp8', 'Pz', 'P3', 'P4', 'O1', 'O2', 'Oz'
]

# Filter parameters
LOWCUT, HIGHCUT = 0.5, 40.0
NOTCH_FREQ = 50.0
FS_ORIG = 512.0  # assume raw data at 512 Hz; update if different
FS_TARGET = 500.0

# ICA parameter
KURTOSIS_THRESHOLD = 5.0


# --- Helper functions ---

def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return b, a


def butter_notch(freq, fs, q=30):
    nyq = 0.5 * fs
    w0 = freq / nyq
    bw = w0 / q
    b, a = butter(2, [w0 - bw, w0 + bw], btype='bandstop')
    return b, a


def resample_signal(sig, fs_old, fs_new):
    old_n = len(sig)
    new_n = int(round(old_n * fs_new / fs_old))
    old_idx = np.linspace(0, 1, old_n)
    new_idx = np.linspace(0, 1, new_n)
    return np.interp(new_idx, old_idx, sig)


def remove_ica_artifacts(signal, thresh):
    n_ch = signal.shape[1]
    ica = FastICA(n_components=n_ch, random_state=0)
    S = ica.fit_transform(signal)
    A = ica.mixing_
    kurts = kurtosis(S, axis=0, fisher=False)
    artifacts = np.where(np.abs(kurts) > thresh)[0]
    print(f"ICA detected {len(artifacts)} artifact ICs: {artifacts}")
    S[:, artifacts] = 0
    return S.dot(A.T)


# --- 1) Load raw CSV and select channels ---
print("Loading raw CSV...")
df = pd.read_csv(RAW_CSV)
print(f"Original shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}\n")

# Keep only required channels
df_sel = df[required_channels].copy()
print(f"After channel selection shape: {df_sel.shape}")
print(df_sel.head(), "\n")

# --- 2) Filter + notch + resample ---
print("Applying bandpass, notch filters and resampling...")
# Compute filter coeffs
bb, ba = butter_bandpass(LOWCUT, HIGHCUT, FS_ORIG)
nn, na = butter_notch(NOTCH_FREQ, FS_ORIG)

filtered = np.zeros((int(round(len(df_sel) * FS_TARGET / FS_ORIG)), len(required_channels)))
for i, ch in enumerate(required_channels):
    data = df_sel[ch].values
    # bandpass
    tmp = filtfilt(bb, ba, data)
    # notch
    tmp = filtfilt(nn, na, tmp)
    # resample
    filtered[:, i] = resample_signal(tmp, FS_ORIG, FS_TARGET)

df_filt = pd.DataFrame(filtered, columns=required_channels)
print(f"After filtering & resampling shape: {df_filt.shape}")
print(df_filt.head(), "\n")

# --- 3) ICA artifact removal ---
print("Performing ICA artifact removal...")
sig = df_filt.values
cleaned = remove_ica_artifacts(sig, KURTOSIS_THRESHOLD)

df_clean = pd.DataFrame(cleaned, columns=required_channels)
print(f"After ICA-clean shape: {df_clean.shape}")
print(df_clean.head(), "\n")

print("Preprocessing complete. Data ready for feature extraction or classification.")


# --- 4) Extract Basic Features (single file) ---
def extract_basic_features_single(data_df):
    """
    Compute basic statistics (mean, std, min, max) for each channel over the entire DataFrame.
    Prints the resulting feature vector and first few values.
    """
    print("Extracting basic features for the single file...")
    stats = {}
    for ch in required_channels:
        vals = data_df[ch].values
        stats[f"{ch}_mean"] = np.mean(vals)
        stats[f"{ch}_std"] = np.std(vals)
        stats[f"{ch}_min"] = np.min(vals)
        stats[f"{ch}_max"] = np.max(vals)
    # Convert to Series for pretty printing
    feat_series = pd.Series(stats)
    print(f"Features vector length: {len(feat_series)}")
    print(feat_series.to_frame(name="Value").T)
    return feat_series


# Call feature extraction on cleaned data
feat_series = extract_basic_features_single(df_clean)

# --- 5) Select features for SVM ---
selected = [
    'Fp1_std', 'F7_mean', 'Cp3_mean', 'Cp4_mean', 'Tp7_mean',
    'Tp7_std', 'Pz_std', 'P3_std', 'P4_mean', 'Oz_std'
]
print("Selecting features for classification:", selected)
# Filter the feature series to only those keys
selected_feats = feat_series[selected]
print("Filtered feature vector:")
print(selected_feats.to_frame(name='Value').T)

# # --- 6) Load pre-trained SVM model and predict --- Stroke/Non-stroke
# import joblib
#
# # Path to the trained SVM model
# SVM_MODEL_PATH = 'test_2/results/07_train_svm/svm_model.pkl'  # <-- update if needed
# print(f"Loading pre-trained SVM model from {SVM_MODEL_PATH}...")
# try:
#     svm_pipe = joblib.load(SVM_MODEL_PATH)
#     # Convert feature vector to 2D array for prediction
#     X_input = selected_feats.values.reshape(1, -1)
#     y_pred = svm_pipe.predict(X_input)[0]
#     # Map numeric labels to text
#     label_map = {0: 'non-stroke', 1: 'stroke'}
#     predicted_text = label_map.get(y_pred, str(y_pred))
#     print(f"Predicted class: {predicted_text} (label {y_pred})")
# except FileNotFoundError:
#     print(f"Error: SVM model file not found at {SVM_MODEL_PATH}.")
# except Exception as e:
#     print(f"Error during SVM prediction: {e}")

# --- 6) Load pre-trained SVM model and predict ---
import joblib

print(f"Loading pre-trained SVM model from {SVM_MODEL_PATH}...")
try:
    svm_pipe = joblib.load(SVM_MODEL_PATH)
    # Convert feature vector to 2D array for prediction
    X_input = selected_feats.values.reshape(1, -1)
    # Predict class and probability
    y_pred = svm_pipe.predict(X_input)[0]
    y_prob = svm_pipe.predict_proba(X_input)[0, 1]  # probability of stroke
    percent = y_prob * 100
    # Map numeric labels to text
    label_map = {0: 'non-stroke', 1: 'stroke'}
    predicted_text = label_map.get(y_pred, str(y_pred))

    # Interpret probability ranges
    if percent < 30:
        interpretation = 'Низкий риск инсульта.'
    elif percent < 60:
        interpretation = 'Умеренный риск инсульта.'
    else:
        interpretation = 'Высокая вероятность инсульта.'

    # Print results
    print(f"Предсказанный класс: {predicted_text} (label {y_pred})")
    print(f"Вероятность инсульта: {percent:.1f}%")
    print(interpretation)

except FileNotFoundError:
    print(f"Error: SVM model file not found at {SVM_MODEL_PATH}.")
except Exception as e:
    print(f"Error during SVM prediction: {e}")
