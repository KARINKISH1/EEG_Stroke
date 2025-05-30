# app.py
# app.py
from flask import Flask, render_template, request, redirect, flash
import os
import io
import pandas as pd
import numpy as np
import joblib
import matplotlib
import mne

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Безопасный ключ
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB
app.config['UPLOAD_FOLDER'] = 'static'

# Модель
SVM_MODEL_PATH = '../SVM_model/svm_model.pkl'

# Параметры обработки
LOWCUT, HIGHCUT = 0.5, 40.0
NOTCH_FREQ = 50.0
FS_ORIG = 512.0
FS_TARGET = 500.0

required_channels = [
    'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Fcz', 'Fc4', 'Ft7', 'Ft8',
    'Cz', 'C3', 'C4', 'Cp3', 'Cp4', 'Tp7', 'Tp8', 'Pz', 'P3', 'P4', 'O1', 'O2', 'Oz'
]


def allowed(filename):
    return filename.lower().endswith('.csv')


def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [low / nyq, high / nyq], btype='band')


def butter_notch(freq, fs, q=30):
    nyq = 0.5 * fs
    w0 = freq / nyq
    bw = w0 / q
    return butter(2, [w0 - bw, w0 + bw], btype='bandstop')


def resample_signal(sig, fs_old, fs_new):
    old_n = len(sig)
    new_n = int(round(old_n * fs_new / fs_old))
    old_idx = np.linspace(0, 1, old_n)
    new_idx = np.linspace(0, 1, new_n)
    return np.interp(new_idx, old_idx, sig)


def remove_ica_artifacts(sig, thresh=5.0):
    n_ch = sig.shape[1]
    ica = FastICA(n_components=n_ch, random_state=0)
    S = ica.fit_transform(sig)
    A = ica.mixing_
    kurts = kurtosis(S, axis=0, fisher=False)
    bad = np.where(np.abs(kurts) > thresh)[0]
    S[:, bad] = 0
    return S.dot(A.T)


def extract_features(df_clean):
    stats = {}
    for ch in required_channels:
        vals = df_clean[ch].values
        stats[f'{ch}_mean'] = np.mean(vals)
        stats[f'{ch}_std'] = np.std(vals)
        stats[f'{ch}_min'] = np.min(vals)
        stats[f'{ch}_max'] = np.max(vals)
    return pd.Series(stats)


def process_signal(df):
    try:
        # Фильтрация
        bb, ba = butter_bandpass(LOWCUT, HIGHCUT, FS_ORIG)
        nn, na = butter_notch(NOTCH_FREQ, FS_ORIG)

        filtered = np.zeros((int(round(len(df) * FS_TARGET / FS_ORIG)), len(required_channels)))
        for i, ch in enumerate(required_channels):
            sig = df[ch].values
            tmp = filtfilt(bb, ba, sig)
            tmp = filtfilt(nn, na, tmp)
            filtered[:, i] = resample_signal(tmp, FS_ORIG, FS_TARGET)

        # Удаление артефактов
        cleaned = remove_ica_artifacts(filtered)
        return pd.DataFrame(cleaned, columns=required_channels)
    except Exception as e:
        raise RuntimeError(f"Ошибка обработки сигнала: {str(e)}")


def validate_csv_structure(df):
    missing = set(required_channels) - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные каналы: {', '.join(missing)}")
    return True


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in {'csv', 'edf'}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Валидация файла
            file = request.files.get('eegfile')
            if not file or not allowed_file(file.filename):
                raise ValueError("Пожалуйста, загрузите CSV или EDF файл")

            if file.filename.endswith('.edf'):
                # Сохраняем временный файл
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.edf')
                file.save(temp_path)

                # Чтение EDF
                raw = mne.io.read_raw_edf(temp_path, preload=True)
                data = raw.get_data()
                channels = [ch.capitalize() for ch in raw.ch_names]  # первая буква заглавная, остальные маленькие
                df = pd.DataFrame(data.T, columns=channels)

                # Удаляем временный файл
                os.remove(temp_path)

                # Обработка CSV
            else:
                df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
                df.columns = [col.capitalize() for col in df.columns]

            validate_csv_structure(df)

            # Обработка сигнала
            df_clean = process_signal(df[required_channels])

            # Прогнозирование
            model = joblib.load(SVM_MODEL_PATH)
            features = extract_features(df_clean)

            selected = [
                'Fp1_std', 'F7_mean', 'Cp3_mean', 'Cp4_mean', 'Tp7_mean',
                'Tp7_std', 'Pz_std', 'P3_std', 'P4_mean', 'Oz_std'
            ]

            selected_feats = features[selected]

            proba = model.predict_proba(selected_feats.values.reshape(1, -1))[0][1] * 100

            # Формирование результата
            if proba < 30:
                msg = 'Низкий риск инсульта'
            elif proba < 60:
                msg = 'Умеренный риск'
            else:
                msg = 'Высокий риск'

            return render_template('index.html',
                                   result=True,
                                   probability=f"{proba:.1f}",
                                   message=msg,
                                   channels=required_channels)

        except Exception as e:
            flash(str(e), 'danger')
            return redirect(request.url)

    return render_template('index.html', result=False)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
