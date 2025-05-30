"""
inference_all_models.py

Load pre-trained SVM, CNN, and CNN+LSTM models and evaluate them on a fixed test set.
- SVM: uses features from results/06_feature_selection_mi/features_selected_mi.csv
- CNN/CNN+LSTM: uses test segments from results/08_cnn_data/eeg_test_segments.npz

Outputs a summary table of key metrics (Accuracy, Precision, Recall, F1-score, ROC AUC)
and generates comparison plots (metrics bar chart, confusion matrix heatmaps)
and plots comparing True vs Predicted labels for each model with simplified numerical X-axis.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# --- Settings ---

RESULTS_DIR = 'results/FINAL_TESTS'  # Directory for saving results (tables and plots)

RANDOM_STATE = 42
BASE_DIR = 'results'
SVM_MODEL_PATH = os.path.join(BASE_DIR, '07_train_svm', 'svm_model.pkl')
FEATURE_CSV = os.path.join(BASE_DIR, '06_feature_selection_mi', 'features_selected_mi.csv')
CNN_MODEL_PATH = os.path.join(BASE_DIR, '09_cnn_models', 'best_cnn_model.h5')
CNN_SCALER_PATH = os.path.join(BASE_DIR, '09_cnn_models', 'scaler_cnn.pkl')
LSTM_MODEL_PATH = os.path.join(BASE_DIR, '10_cnn_lstm_models', 'best_cnn_lstm_model.h5')
LSTM_SCALER_PATH = os.path.join(BASE_DIR, '10_cnn_lstm_models', 'scaler_cnn_lstm.pkl')
SEGMENTS_TEST_PATH = os.path.join(BASE_DIR, '08_cnn_data', 'eeg_test_segments_2.npz')

# # Define fixed test files - used for SVM data loading
test_files = [f's{i:02d}.csv' for i in range(1, 8)] + \
             [f'sub-{i:02d}_task-motor-imagery_eeg.csv' for i in range(1, 9)]
# Исходные списки файлов
# s_files = [f's{i:02d}.csv' for i in range(1, 11)]
# sub_files = [f'sub-{i:02d}_task-motor-imagery_eeg.csv' for i in range(1, 11)]

# Файлы для второго (экстра) тестового набора (с 6 по 10 из каждого типа)
# test_files = s_files[5:10] + sub_files[5:10]
print("Files for Test Set 2 (Extra):", test_files)

print("Test files used:", test_files)

# Define class names
CLASS_NAMES = ['non-stroke', 'stroke']  # 0: non-stroke, 1: stroke

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set seaborn plot style
sns.set_theme(style="whitegrid")


# --- Helper Functions ---

def compute_metrics(name, y_true, y_pred, y_prob=None):
    """Computes and returns a set of key classification metrics."""
    m = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred)
    }
    # ROC AUC requires probabilities
    if y_prob is not None:
        m['ROC_AUC'] = roc_auc_score(y_true, y_prob)
    # Removed MSE, MAE, R2 as they are less standard for binary classification evaluation

    return m


def plot_confusion_matrix(cm, model_name, class_names=CLASS_NAMES, save_path=None, normalize=False):
    """Plots and saves a confusion matrix heatmap. Can display percentages."""
    plt.figure(figsize=(7, 6))
    if normalize:
        # Compute percentages relative to true labels (sum across rows)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Replace NaN with 0 if a true label row is empty for a class
        cm_normalized = np.nan_to_num(cm_normalized)
        annot_fmt = '.1%'
        data_to_plot = cm_normalized
        title = f'Confusion Matrix - {model_name} (Percent %)'
    else:
        annot_fmt = 'd'
        data_to_plot = cm
        title = f'Confusion Matrix - {model_name} (Counts)'

    sns.heatmap(data_to_plot, annot=True, fmt=annot_fmt, cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})

    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Confusion Matrix for {model_name} to {save_path}")
    plt.close()


def plot_metrics_bar_chart(summary_df, metrics_to_plot, save_path=None):
    """Plots and saves a bar chart comparing metrics."""
    # Ensure only metrics present in the DataFrame are plotted
    metrics_to_plot_existing = [m for m in metrics_to_plot if m in summary_df.columns]
    if not metrics_to_plot_existing:
        print("Warning: No metrics available in summary DataFrame to plot bar chart.")
        return

    summary_plot = summary_df[metrics_to_plot_existing]
    # Увеличиваем ширину столбцов (width=0.95, например)
    ax = summary_plot.plot(kind='bar', figsize=(12, 7), colormap='viridis', edgecolor='black', width=0.95)

    plt.title('Model Comparison - Key Metrics', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=0, ha='center', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Форматируем метки значений над столбцами индивидуально и делаем их жирными
    for container in ax.containers:
        labels = []
        for value in container.get_children():
            # Получаем высоту столбца (значение метрики)
            height = value.get_height()
            # Проверяем, является ли число близким к целому
            if abs(height - round(height)) < 1e-6:  # Используем небольшой допуск для сравнения с целым
                labels.append(f'{int(round(height))}')  # Форматируем как целое число
            else:
                labels.append(f'{height:.2f}')  # Форматируем с 2 знаками после запятой

        # Применяем отформатированные метки к контейнеру, делая шрифт жирным
        ax.bar_label(container, labels=labels, label_type='edge', padding=8, fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Metrics Bar Chart to {save_path}")
    plt.close()


def plot_true_vs_predicted(y_true, y_pred, model_name, index_labels=None, save_path=None, max_points=None):
    """
    Plots True vs Predicted labels, highlighting correct and incorrect predictions.
    Plots the predicted label on the y-axis and colors by correctness.
    Uses a numerical index (1, 2, 3...) on the x-axis for plotting and ticks.
    Optionally limits the number of points plotted if max_points is specified.
    """
    # Create a DataFrame for easier plotting
    plot_df = pd.DataFrame({'True_Label': y_true, 'Predicted_Label': y_pred})

    # Use a numerical index for plotting positions on the x-axis (0, 1, 2...)
    x_positions = np.arange(len(plot_df))

    # Add a column indicating correctness
    plot_df['Correct'] = (plot_df['True_Label'] == plot_df['Predicted_Label'])

    # Define colors and labels based on correctness
    colors = {
        True: 'green',  # Correct Prediction
        False: 'red'  # Incorrect Prediction
    }
    labels = {
        True: 'Correct Prediction',
        False: 'Incorrect Prediction'
    }

    # If max_points is specified, sample or slice the data
    if max_points is not None and len(plot_df) > max_points:
        print(f"Plotting only the first {max_points} instances for {model_name} prediction plot.")
        plot_df_subset = plot_df.head(max_points)
        x_positions_subset = x_positions[:max_points]
        title_suffix = f' (First {max_points} Instances)'
    else:
        plot_df_subset = plot_df
        x_positions_subset = x_positions
        title_suffix = ''

    # Check if there are any points to plot after subsetting
    if plot_df_subset.empty:
        print(f"Warning: No instances to plot for {model_name} prediction plot after applying max_points limit.")
        return  # Exit the function if no data to plot

    plt.figure(figsize=(15, 6))  # Wider figure to show more points

    # Plot points, using Predicted_Label on y-axis and coloring by Correctness
    for correctness, color in colors.items():
        # Filter the subset DataFrame based on correctness
        subset = plot_df_subset[plot_df_subset['Correct'] == correctness]

        if not subset.empty:
            # Use the numerical positions (0, 1, 2...) for the rows in the 'subset' DataFrame
            # relative to the start of the plot_df_subset DataFrame.
            # This is the corrected part: use get_indexer to find positions in the subset's index
            numerical_positions_in_subset = plot_df_subset.index.get_indexer(subset.index)

            # Add a small vertical jitter to points to help visualize density
            jittered_predicted_label = subset['Predicted_Label'] + np.random.uniform(-0.05, 0.05, len(subset))

            # Plot using the numerical positions
            plt.scatter(numerical_positions_in_subset, jittered_predicted_label, color=color, label=labels[correctness],
                        alpha=0.6, s=20, marker='o')

    plt.yticks([0, 1], CLASS_NAMES)  # Use class names for y-axis ticks
    # Обновляем подпись оси X, чтобы отразить, что это индекс
    plt.xlabel('Instance Index', fontsize=12)
    plt.ylabel('Predicted Label', fontsize=12)  # Y-axis now represents the predicted label
    plt.title(f'True vs Predicted Labels - {model_name}' + title_suffix, fontsize=14)
    plt.legend(title='Prediction Outcome', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend

    # Устанавливаем числовые метки на оси X (1, 2, 3...)
    num_points = len(plot_df_subset)
    if num_points > 0:
        # Позиции для меток (0, 1, 2...)
        x_tick_positions_raw = np.arange(num_points)
        # Текст меток (1, 2, 3...)
        x_tick_labels_raw = [str(i + 1) for i in x_tick_positions_raw]

        # Выбираем подмножество меток, если их слишком много
        if num_points > 20:  # Показываем все метки, если их <= 20
            step = max(1, num_points // 10)  # Шаг для меток (примерно 10 меток)
            x_tick_positions = x_tick_positions_raw[::step]
            x_tick_labels = x_tick_labels_raw[::step]
        else:
            x_tick_positions = x_tick_positions_raw
            x_tick_labels = x_tick_labels_raw

        plt.xticks(x_tick_positions, x_tick_labels, rotation=0, ha='center', fontsize=10)  # Убираем поворот, центрируем

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved True vs Predicted plot for {model_name} to {save_path}")
    plt.close()


# --- Main Inference and Plotting ---

# 1) SVM Inference
print("\n--- SVM Inference ---")
metrics_svm, cm_svm, y_prob_svm, y_test_svm_actual, y_pred_svm = None, None, None, None, None

try:
    df_feat = pd.read_csv(FEATURE_CSV, index_col=0)
    df_test_svm = df_feat.loc[df_feat.index.isin(test_files)].copy()
    if df_test_svm.empty:
        print(f"Warning: No SVM test data found for files in {test_files}. Skipping SVM inference.")
    else:
        X_test_svm = df_test_svm.drop(columns=['label'])
        y_test_svm_actual = df_test_svm['label']
        svm_pipe = joblib.load(SVM_MODEL_PATH)
        y_pred_svm = svm_pipe.predict(X_test_svm)
        # ROC AUC requires probabilities, so we still compute y_prob_svm
        y_prob_svm = svm_pipe.predict_proba(X_test_svm)[:, 1]
        metrics_svm = compute_metrics('SVM', y_test_svm_actual, y_pred_svm, y_prob_svm)
        cm_svm = confusion_matrix(y_test_svm_actual, y_pred_svm)
        print("Confusion matrix (SVM):\n", cm_svm)
        print(pd.Series(metrics_svm))

        # Plot True vs Predicted for SVM (using filenames as index labels, but numerical x-axis)
        # SVM has only 20 test files, so plot all
        plot_true_vs_predicted(
            y_test_svm_actual, y_pred_svm, 'SVM',
            index_labels=df_test_svm.index,  # Pass filenames as index labels (для информации, не для оси X)
            save_path=os.path.join(RESULTS_DIR, 'svm_true_vs_predicted.png')
            # max_points=None # Plot all for SVM
        )


except FileNotFoundError:
    print(f"Error: SVM feature file not found at {FEATURE_CSV}. Skipping SVM inference.")
except Exception as e:
    print(f"Error during SVM inference: {e}. Skipping SVM inference.")

# 2) CNN & CNN+LSTM Inference
print("\n--- CNN/CNN+LSTM Inference ---")
X_test_seg, y_test_seg_actual = None, None
metrics_cnn, cm_cnn, y_prob_cnn, y_pred_cnn = None, None, None, None
metrics_cl, cm_cl, y_prob_cl, y_pred_cl = None, None, None, None

try:
    data = np.load(SEGMENTS_TEST_PATH)
    X_test_seg = data['X']
    y_test_seg_actual = data['y']
    print(f"Loaded test segments: {X_test_seg.shape}, labels: {y_test_seg_actual.shape}")

    # CNN Inference
    print("\n--- CNN Inference ---")
    try:
        scaler_cnn = joblib.load(CNN_SCALER_PATH)
        cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
        n_seg, w, c = X_test_seg.shape
        X_flat = X_test_seg.reshape(-1, c)
        X_scaled_cnn = scaler_cnn.transform(X_flat).reshape(n_seg, w, c)
        # ROC AUC requires probabilities, so we still compute y_prob_cnn
        y_prob_cnn = cnn.predict(X_scaled_cnn).ravel()
        y_pred_cnn = (y_prob_cnn > 0.5).astype(int)
        metrics_cnn = compute_metrics('CNN', y_test_seg_actual, y_pred_cnn, y_prob_cnn)
        cm_cnn = confusion_matrix(y_test_seg_actual, y_pred_cnn)
        print("Confusion matrix (CNN):\n", cm_cnn)
        print(pd.Series(metrics_cnn))
        del X_scaled_cnn

        # Plot True vs Predicted for CNN (using segment index)
        # Plot ALL segments by removing max_points
        plot_true_vs_predicted(
            y_test_seg_actual, y_pred_cnn, 'CNN',
            save_path=os.path.join(RESULTS_DIR, 'cnn_true_vs_predicted.png')
            # max_points=None # Plot all segments
        )


    except FileNotFoundError as e:
        print(f"Error loading CNN model or scaler: {e}. Skipping CNN inference.")
    except Exception as e:
        print(f"Error during CNN inference: {e}. Skipping CNN inference.")

    # CNN+LSTM Inference
    print("\n--- CNN+LSTM Inference ---")
    try:
        scaler_lstm = joblib.load(LSTM_SCALER_PATH)
        cl = tf.keras.models.load_model(LSTM_MODEL_PATH)
        n_seg, w, c = X_test_seg.shape
        X_flat = X_test_seg.reshape(-1, c)
        X_scaled_lstm = scaler_lstm.transform(X_flat).reshape(n_seg, w, c)
        # ROC AUC requires probabilities, so we still compute y_prob_cl
        y_prob_cl = cl.predict(X_scaled_lstm).ravel()
        y_pred_cl = (y_prob_cl > 0.5).astype(int)
        metrics_cl = compute_metrics('CNN+LSTM', y_test_seg_actual, y_pred_cl, y_prob_cl)
        cm_cl = confusion_matrix(y_test_seg_actual, y_pred_cl)
        print("Confusion matrix (CNN+LSTM):\n", cm_cl)
        print(pd.Series(metrics_cl))
        del X_scaled_lstm

        # Plot True vs Predicted for CNN+LSTM (using segment index)
        # Plot ALL segments by removing max_points
        plot_true_vs_predicted(
            y_test_seg_actual, y_pred_cl, 'CNN+LSTM',
            save_path=os.path.join(RESULTS_DIR, 'cnn_lstm_true_vs_predicted.png')
            # max_points=None # Plot all segments
        )


    except FileNotFoundError as e:
        print(f"Error loading CNN+LSTM model or scaler: {e}. Skipping CNN+LSTM inference.")
    except Exception as e:
        print(f"Error during CNN+LSTM inference: {e}. Skipping CNN+LSTM inference.")


except FileNotFoundError:
    print(f"Error: Test segments file not found at {SEGMENTS_TEST_PATH}. Skipping CNN/CNN+LSTM inference.")
except Exception as e:
    print(f"Error loading test segments: {e}. Skipping CNN/CNN+LSTM inference.")

# --- Summary Table ---
print("\n--- Summary Table ---")
all_metrics = {}
if metrics_svm:
    all_metrics['SVM'] = metrics_svm
if metrics_cnn:
    all_metrics['CNN'] = metrics_cnn
if metrics_cl:
    all_metrics['CNN+LSTM'] = metrics_cl

if all_metrics:
    summary = pd.DataFrame(all_metrics).T
    print(summary)
    summary_path = os.path.join(RESULTS_DIR, 'inference_metrics.csv')
    summary.to_csv(summary_path)
    print("Saved summary table to", summary_path)

    # --- Plotting ---
    print("\n--- Generating Plots ---")

    # Plot Confusion Matrices (Counts and Percentages)
    if cm_svm is not None:
        plot_confusion_matrix(cm_svm, 'SVM', class_names=CLASS_NAMES,
                              save_path=os.path.join(RESULTS_DIR, 'cm_svm_counts.png'), normalize=False)
        plot_confusion_matrix(cm_svm, 'SVM', class_names=CLASS_NAMES,
                              save_path=os.path.join(RESULTS_DIR, 'cm_svm_percent.png'), normalize=True)
    if cm_cnn is not None:
        plot_confusion_matrix(cm_cnn, 'CNN', class_names=CLASS_NAMES,
                              save_path=os.path.join(RESULTS_DIR, 'cm_cnn_counts.png'), normalize=False)
        plot_confusion_matrix(cm_cnn, 'CNN', class_names=CLASS_NAMES,
                              save_path=os.path.join(RESULTS_DIR, 'cm_cnn_percent.png'), normalize=True)
    if cm_cl is not None:
        plot_confusion_matrix(cm_cl, 'CNN+LSTM', class_names=CLASS_NAMES,
                              save_path=os.path.join(RESULTS_DIR, 'cm_cnn_lstm_counts.png'), normalize=False)
        plot_confusion_matrix(cm_cl, 'CNN+LSTM', class_names=CLASS_NAMES,
                              save_path=os.path.join(RESULTS_DIR, 'cm_cnn_lstm_percent.png'), normalize=True)

    # Plot Metrics Bar Chart
    metrics_for_bar_chart = ['Accuracy', 'Precision', 'Recall']
    metrics_to_plot_existing = [m for m in metrics_for_bar_chart if m in summary.columns]

    if metrics_to_plot_existing:
        plot_metrics_bar_chart(summary, metrics_to_plot_existing,
                               save_path=os.path.join(RESULTS_DIR, 'metrics_bar_chart.png'))
    else:
        print("Not enough metrics available to plot bar chart.")

    # --- Prediction Comparison Plots ---
    print("\n--- Generating Prediction Comparison Plots ---")
    # Plots are generated within the inference blocks now.


else:
    print("\nNo model metrics were successfully computed. Cannot generate summary table or plots.")

print("\nInference and comparison complete.")
