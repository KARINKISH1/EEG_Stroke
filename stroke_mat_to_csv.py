import scipy.io
import pandas as pd
import numpy as np
import os

channel_names = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Fcz', 'Fc3', 'Fc4', 'Ft7', 'Ft8', 'Cz', 'C3', 'C4', 'T3',
                 'T4', 'Cpz', 'Cp3', 'Cp4', 'Tp7', 'Tp8', 'Pz', 'P3', 'P4', 'T5', 'T6', 'Oz', 'O1', 'O2', 'Heol',
                 'Heor', 'Trig']


def process_mat_file(mat_file_path, label_value=""):
    try:
        mat_data = scipy.io.loadmat(mat_file_path)

        if 'eeg' in mat_data:
            eeg_struct = mat_data['eeg']
            if isinstance(eeg_struct, np.ndarray) and eeg_struct.size == 1:
                eeg_struct = eeg_struct[0, 0]

            if isinstance(eeg_struct, np.void):
                print(f"  - eeg structure fields: {eeg_struct.dtype.names}")

                if 'rawdata' in eeg_struct.dtype.names and 'label' in eeg_struct.dtype.names:
                    rawdata = eeg_struct['rawdata']
                    labels = eeg_struct['label']

                    print(f"  - rawdata shape: {rawdata.shape}")
                    print(f"  - labels shape: {labels.shape}")

                    num_trials, num_channels, num_timepoints = rawdata.shape

                    rawdata_transposed = rawdata.transpose(0, 2, 1)
                    rawdata_reshaped = rawdata_transposed.reshape(num_trials * num_timepoints, num_channels)

                    df = pd.DataFrame(rawdata_reshaped)
                    df.columns = channel_names

                    df['label'] = [label_value] * (num_trials * num_timepoints)

                    if 'Trig' in df.columns:
                        df = df.drop(columns=['Trig'])

                    return df
                else:
                    print(f"  - Warning: 'rawdata' or 'label' not found in 'eeg' structure of {mat_file_path}")
                    return None
            elif isinstance(eeg_struct, np.ndarray) and hasattr(eeg_struct, 'dtype') and eeg_struct.dtype.names:
                print(f"  - eeg structure fields: {eeg_struct.dtype.names}")

                if 'rawdata' in eeg_struct.dtype.names and 'label' in eeg_struct.dtype.names:
                    rawdata = eeg_struct['rawdata'][0, 0]
                    labels = eeg_struct['label'][0, 0]

                    print(f"  - rawdata shape: {rawdata.shape}")
                    print(f"  - labels shape: {labels.shape}")

                    num_trials, num_channels, num_timepoints = rawdata.shape
                    rawdata_flattened = rawdata.reshape(num_trials, num_channels * num_timepoints)

                    df = pd.DataFrame(rawdata_flattened)
                    df.columns = [f'channel_{i}' for i in range(rawdata_flattened.shape[1])]
                    df['label'] = [label_value] * num_trials
                    df['trial'] = range(1, num_trials + 1)
                    df['original_label'] = labels.flatten()

                    return df
                else:
                    print(f"  - Warning: 'rawdata' or 'label' not found in 'eeg' structure of {mat_file_path}")
                    return None
            else:
                print(f"  - Warning: Unexpected structure in 'eeg' of {mat_file_path}")
                print(f"  - Type of eeg_struct: {type(eeg_struct)}")
                if isinstance(eeg_struct, np.ndarray):
                    print(f"  - Shape of eeg_struct: {eeg_struct.shape}")
                    print(f"  - dtype of eeg_struct: {eeg_struct.dtype}")
                return None
        else:
            print(f"  - Warning: No 'eeg' key found in {mat_file_path}")
            return None

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


input_dir = "mat/stroke"
output_dir = "csv/stroke"
label_value = "stroke"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mat_files = [f for f in os.listdir(input_dir) if f.endswith(".mat")]

for mat_file in mat_files:
    mat_file_path = os.path.join(input_dir, mat_file)
    print(f"Processing: {mat_file_path}")

    df = process_mat_file(mat_file_path, label_value)

    if df is not None:
        csv_file_path = os.path.join(output_dir, os.path.splitext(mat_file)[0] + ".csv")
        save_dataframe_to_csv(df, csv_file_path)

print("\nBatch conversion complete.")
