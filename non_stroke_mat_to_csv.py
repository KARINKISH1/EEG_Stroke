import scipy.io
import pandas as pd
import os


def mat_to_dataframe(mat_file_path, channel_mapping=None, label=""):
    try:
        mat_data = scipy.io.loadmat(mat_file_path)

        eeg_struct = mat_data['eeg'][0, 0]

        noise_data = eeg_struct['noise'][0]
        noise_data = noise_data[0]

        noise_data = noise_data[:64, :]

        df = pd.DataFrame(noise_data.T)

        if channel_mapping:
            df.columns = [channel_mapping.get(i, f'channel_{i}') for i in range(noise_data.shape[0])]
        else:
            df.columns = [f'channel_{i}' for i in range(noise_data.shape[0])]

        df['label'] = label

        return df

    except Exception as e:
        print(f"Error processing {mat_file_path}: {e}")
        print(f"  - Exception type: {type(e)}")
        print(f"  - Exception message: {e}")
        return None


def save_dataframe_to_csv(df, csv_file_path):
    try:
        df.to_csv(csv_file_path, index=False)
        print(f"Saved to {csv_file_path}")
    except Exception as e:
        print(f"Error saving to {csv_file_path}: {e}")


mat_dir = "mat/non_stroke"
output_dir = "csv/non_stroke"
label = "non_stroke"

channel_mapping = {
    0: 'Fp1',
    1: 'Af7',
    2: 'Af3',
    3: 'F1',
    4: 'F3',
    5: 'F5',
    6: 'F7',
    7: 'Ft7',
    8: 'Fc5',
    9: 'FC3',
    10: 'Fc1',
    11: 'C1',
    12: 'C3',
    13: 'C5',
    14: 'T7',
    15: 'Tp7',
    16: 'Cp5',
    17: 'Cp3',
    18: 'Cp1',
    19: 'P1',
    20: 'P3',
    21: 'P5',
    22: 'P7',
    23: 'P9',
    24: 'Po7',
    25: 'Po3',
    26: 'O1',
    27: 'Iz',
    28: 'Oz',
    29: 'Poz',
    30: 'Pz',
    31: 'CPz',
    32: 'Fp7',
    33: 'Fp2',
    34: 'Af8',
    35: 'Af4',
    36: 'Afz',
    37: 'Fz',
    38: 'F2',
    39: 'F4',
    40: 'F6',
    41: 'F8',
    42: 'Ft8',
    43: 'Fc6',
    44: 'Fc4',
    45: 'Fc2',
    46: 'Fcz',
    47: 'Cz',
    48: 'C2',
    49: 'C4',
    50: 'C6',
    51: 'T8',
    52: 'Tp8',
    53: 'Cp6',
    54: 'Cp4',
    55: 'Cp2',
    56: 'P2',
    57: 'P4',
    58: 'P6',
    59: 'P8',
    60: 'P10',
    61: 'Po8',
    62: 'Po4',
    63: 'O2'
}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(mat_dir):
    if filename.endswith(".mat"):
        mat_file_path = os.path.join(mat_dir, filename)
        df = mat_to_dataframe(mat_file_path, channel_mapping, label)
        if df is not None:
            csv_file_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".csv")
            save_dataframe_to_csv(df, csv_file_path)

print("Batch conversion complete.")
