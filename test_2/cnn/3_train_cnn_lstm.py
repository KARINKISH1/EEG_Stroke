# # cnn/3_train_cnn_lstm.py
# # -------------------------------------------------------------------
# # Step 3: Train hybrid CNN+LSTM on segmented EEG data
# # Вход: results/08_cnn_data/eeg_segments.npz
# # Выход: results/10_cnn_lstm_models/best_cnn_lstm_model.h5 + metrics
#
# import os
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import joblib
# import json
#
# # Пути
# INPUT_FILE = '../results/08_cnn_data/eeg_segments.npz'
# OUTPUT_DIR = '../results/10_cnn_lstm_models'
# MODEL_FILE = os.path.join(OUTPUT_DIR, 'best_cnn_lstm_model.h5')
# SCALER_FILE = os.path.join(OUTPUT_DIR, 'scaler_cnn_lstm.pkl')
# METRICS_FILE = os.path.join(OUTPUT_DIR, 'cnn_lstm_metrics.json')
#
# # Параметры
# TEST_SIZE = 0.2
# RANDOM_STATE = 42
# EPOCHS = 100
# BATCH_SIZE = 32
#
#
# def build_cnn_lstm(input_shape):
#     """
#     Сборка гибридной модели: Conv1D-блок → LSTM → Dense
#     """
#     model = Sequential([
#         # Сверточный блок для локальных признаков
#         Conv1D(32, 3, activation='relu', padding='same', input_shape=input_shape),
#         BatchNormalization(),
#         MaxPooling1D(2),
#         Dropout(0.3),
#
#         # LSTM-слой для временной динамики
#         LSTM(64, return_sequences=False),
#         Dropout(0.4),
#
#         # Полносвязный выход
#         Dense(32, activation='relu'),
#         Dropout(0.4),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
#     )
#     return model
#
#
# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     # Загрузка сегментов
#     data = np.load(INPUT_FILE)
#     X, y = data['X'], data['y']
#     print(f"Loaded segments: {X.shape}, labels: {y.shape}")
#
#     # Масштабирование каналов
#     n_samples, window, n_channels = X.shape
#     X_flat = X.reshape(-1, n_channels)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_flat)
#     X = X_scaled.reshape(n_samples, window, n_channels)
#     joblib.dump(scaler, SCALER_FILE)
#     print(f"Scaler saved to {SCALER_FILE}")
#
#     # Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
#     )
#     print(f"Train segments: {X_train.shape}, Test segments: {X_test.shape}")
#
#     # Построение модели
#     model = build_cnn_lstm(input_shape=(window, n_channels))
#     model.summary()
#
#     # Callbacks
#     es = EarlyStopping(monitor='val_auc', mode='max', patience=10, verbose=1)
#     mc = ModelCheckpoint(MODEL_FILE, monitor='val_auc', mode='max', save_best_only=True, verbose=1)
#
#     # Обучение
#     history = model.fit(
#         X_train, y_train,
#         validation_split=0.2,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         callbacks=[es, mc]
#     )
#
#     # Оценка
#     best_model = tf.keras.models.load_model(MODEL_FILE)
#     loss, acc, auc = best_model.evaluate(X_test, y_test, verbose=1)
#     metrics = {'loss': float(loss), 'accuracy': float(acc), 'auc': float(auc)}
#     with open(METRICS_FILE, 'w') as f:
#         json.dump(metrics, f, indent=4)
#     print(f"Test metrics saved to {METRICS_FILE}")
#     print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")
#
#
# if __name__ == '__main__':
#     main()
# cnn/3_train_cnn_lstm.py
# -------------------------------------------------------------------
# Step 3: Train hybrid CNN+LSTM on segmented EEG data (TRAIN set) and evaluate on TEST set
# Вход: results/08_cnn_data/eeg_train_segments.npz
#       results/08_cnn_data/eeg_test_segments.npz
# Выход: results/10_cnn_lstm_models/best_cnn_lstm_model.h5 + metrics + scaler

import os
import numpy as np
# train_test_split больше не нужен
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import json

# Пути
INPUT_FILE_TRAIN = '../results/08_cnn_data/eeg_train_segments.npz'  # Файл для обучения
INPUT_FILE_TEST = '../results/08_cnn_data/eeg_test_segments_1.npz'  # Файл для тестирования
OUTPUT_DIR = '../results/10_cnn_lstm_models'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'best_cnn_lstm_model.h5')
SCALER_FILE = os.path.join(OUTPUT_DIR, 'scaler_cnn_lstm.pkl')
METRICS_FILE = os.path.join(OUTPUT_DIR, 'cnn_lstm_metrics.json')

# Параметры
# TEST_SIZE больше не используется для разбиения
# TEST_SIZE = 0.2
RANDOM_STATE = 42  # Можно использовать для воспроизводимости TF/Keras
EPOCHS = 100
BATCH_SIZE = 32


# Установка сида для воспроизводимости (опционально)
# tf.random.set_seed(RANDOM_STATE)
# np.random.seed(RANDOM_STATE)
# import random
# random.seed(RANDOM_STATE)


def build_cnn_lstm(input_shape):
    """
    Сборка гибридной модели: Conv1D-блок → LSTM → Dense
    """
    model = Sequential([
        # Сверточный блок для локальных признаков
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # LSTM-слой для временной динамики
        # return_sequences=False, т.к. это последний RNN слой перед Dense
        LSTM(units=64, return_sequences=False),
        Dropout(0.4),

        # Полносвязный выход
        Dense(units=32, activation='relu'),
        Dropout(0.4),
        Dense(units=1, activation='sigmoid')  # Для бинарной классификации
    ])
    # Указываем имя метрики auc правильно, как она определена
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Загрузка обучающих данных ---
    print(f"Loading training segments from {INPUT_FILE_TRAIN}...")
    try:
        train_data = np.load(INPUT_FILE_TRAIN)
        X_train, y_train = train_data['X'], train_data['y']
        channels = train_data['channels']  # Получаем список каналов из файла
        print(f"Loaded training segments: {X_train.shape}, labels: {y_train.shape}")
    except FileNotFoundError:
        print(f"Error: Training data file not found at {INPUT_FILE_TRAIN}")
        return
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # --- Загрузка тестовых данных ---
    print(f"Loading test segments from {INPUT_FILE_TEST}...")
    try:
        test_data = np.load(INPUT_FILE_TEST)
        X_test, y_test = test_data['X'], test_data['y']
        # Проверяем, что каналы совпадают (опционально, но хорошая практика)
        test_channels = test_data['channels']
        if not np.array_equal(channels, test_channels):
            print("Warning: Channel order mismatch between train and test data!")
        print(f"Loaded test segments: {X_test.shape}, labels: {y_test.shape}")
    except FileNotFoundError:
        print(f"Error: Test data file not found at {INPUT_FILE_TEST}")
        # Можно продолжить обучение, но оценка будет невозможна
        X_test, y_test = None, None  # Устанавливаем None, чтобы пропустить оценку
    except Exception as e:
        print(f"Error loading test data: {e}")
        X_test, y_test = None, None  # Устанавливаем None

    # --- Масштабирование каналов ---
    # Обучаем скейлер ТОЛЬКО на обучающих данных
    n_train_samples, window, n_channels = X_train.shape
    X_train_flat = X_train.reshape(-1, n_channels)  # Сворачиваем до (все_отсчеты, каналы)

    scaler = StandardScaler()
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)  # Обучение и трансформация на обучении

    # Применяем обученный скейлер к обучающим данным
    X_train_scaled = X_train_scaled_flat.reshape(n_train_samples, window, n_channels)

    # Применяем обученный скейлер к тестовым данным (если они загружены)
    X_test_scaled = None  # Инициализируем None
    if X_test is not None:
        n_test_samples = X_test.shape[0]
        X_test_flat = X_test.reshape(-1, n_channels)
        X_test_scaled_flat = scaler.transform(X_test_flat)  # Только трансформация на тесте
        X_test_scaled = X_test_scaled_flat.reshape(n_test_samples, window, n_channels)

    # Сохраняем скейлер (обученный на обучающих данных)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler saved to {SCALER_FILE}")

    # --- Построение модели ---
    # Входная форма берется из обучающих данных
    input_shape = (window, n_channels)
    model = build_cnn_lstm(input_shape)  # Используем функцию сборки CNN+LSTM
    model.summary()

    # --- Callbacks ---
    # Мониторим 'val_auc' т.к. она добавлена в метрики компиляции
    es = EarlyStopping(monitor='val_auc', mode='max', patience=10, verbose=1)
    mc = ModelCheckpoint(MODEL_FILE, monitor='val_auc', mode='max', save_best_only=True, verbose=1)

    # --- Обучение ---
    # validation_split=0.2 разбивает обучающие данные на train/validation для отслеживания прогресса
    print("Starting model training...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, mc]
    )
    print("Model training finished.")

    # --- Оценка ---
    if X_test_scaled is not None:
        print("Evaluating model on the separate test set...")
        try:
            # Загружаем лучшую модель (сохраненную ModelCheckpoint)
            best_model = tf.keras.models.load_model(MODEL_FILE)
            # Оцениваем на scaled тестовых данных
            loss, acc, auc = best_model.evaluate(X_test_scaled, y_test, verbose=1)
            metrics = {'loss': float(loss), 'accuracy': float(acc), 'auc': float(auc)}

            with open(METRICS_FILE, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"Test metrics saved to {METRICS_FILE}")
            print(f"Test Accuracy: {acc:.4f}, Test AUC: {auc:.4f}")

        except Exception as e:
            print(f"Error during model evaluation: {e}")
            # Сохраняем только метрики обучения/валидации, если оценка на тесте не удалась
            metrics_hist = {k: [float(v) for v in history.history[k]] for k in history.history}
            metrics_hist['evaluation_error'] = str(e)
            with open(METRICS_FILE, 'w') as f:
                json.dump(metrics_hist, f, indent=4)
            print(f"Training history saved to {METRICS_FILE} due to evaluation error.")

    else:
        print("Test data was not loaded, skipping final evaluation on test set.")
        # Опционально: сохранить только историю обучения, если тест недоступен
        metrics_hist = {k: [float(v) for v in history.history[k]] for k in history.history}
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics_hist, f, indent=4)
        print(f"Training history saved to {METRICS_FILE}.")


if __name__ == '__main__':
    main()
