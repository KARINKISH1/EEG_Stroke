import pandas as pd
import os

# Указываем путь к папке с CSV файлами
folder_path = 'csv/non_stroke'

# Список колонок, которые нужно оставить
required_columns = [
    'Fp1', 'Fp2',
    'Fz', 'F3', 'F4', 'F7', 'F8',
    'Fcz', 'Fc4', 'Ft7', 'Ft8',
    'Cz', 'C3', 'C4',
    'Cp3', 'Cp4',
    'Tp7', 'Tp8',
    'Pz', 'P3', 'P4',
    'O1', 'O2', 'Oz',
    'label'
]

# Получаем список всех CSV файлов в папке
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

print(f"Найдено {len(csv_files)} файлов для обработки.")

# Обрабатываем каждый файл
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)

    # Читаем CSV файл
    df = pd.read_csv(file_path)

    # Проверяем, есть ли все необходимые колонки в файле
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"В файле {csv_file} отсутствуют следующие колонки: {', '.join(missing_columns)}")
    else:
        print(f"Файл {csv_file} содержит все необходимые колонки.")

    # Оставляем только необходимые колонки
    df = df[required_columns]

    # Перезаписываем файл или сохраняем в новый файл
    output_path = f"preprocessed_csv/non_stroke/{csv_file}"
    df.to_csv(output_path, index=False)
    print(f"Файл {csv_file} обработан и сохранен в {output_path}.")

print("Обработка завершена.")
