import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(file_path):
    """Loads the NASA CMAPSS dataset and drops low-variance sensors."""
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
    
    # Drop constant/low-variance sensors for FD001: 1, 5, 6, 10, 16, 18, 19
    # Indices in sensor_names (s_1 is index 0)
    sensors_to_drop = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    df.drop(sensors_to_drop, axis=1, inplace=True)
    
    return df

def calculate_rul(df, cap=125):
    """Calculates Piecewise Linear RUL (capped at specified value)."""
    max_cycle = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_cycle']
    
    df = df.merge(max_cycle, on=['unit_nr'], how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    
    # Apply piecewise linear RUL cap
    if cap is not None:
        df['RUL'] = df['RUL'].clip(upper=cap)
    
    df.drop('max_cycle', axis=1, inplace=True)
    
    # Add binary classification label: 1 if RUL <= 30 (imminent failure), else 0
    df['label_bc'] = (df['RUL'] <= 30).astype(int)
    
    return df

def engine_train_test_split(df, train_ratio=0.8, val_ratio=0.1):
    """Splits data by engine units to prevent leakage."""
    units = df['unit_nr'].unique()
    np.random.shuffle(units)
    
    train_size = int(len(units) * train_ratio)
    val_size = int(len(units) * val_ratio)
    
    train_units = units[:train_size]
    val_units = units[train_size:train_size+val_size]
    test_units = units[train_size+val_size:]
    
    train_df = df[df['unit_nr'].isin(train_units)]
    val_df = df[df['unit_nr'].isin(val_units)]
    test_df = df[df['unit_nr'].isin(test_units)]
    
    return train_df, val_df, test_df

def process_test_data(test_df, rul_df, cap=125):
    """Adds RUL ground truth and classification labels to test data."""
    # rul_df has 1 column: RUL at last cycle
    rul_df.columns = ['RUL_end']
    rul_df['unit_nr'] = rul_df.index + 1
    
    max_cycle = test_df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_cycle']
    
    test_df = test_df.merge(max_cycle, on='unit_nr', how='left')
    test_df = test_df.merge(rul_df, on='unit_nr', how='left')
    
    # RUL = RUL_at_end + (max_cycle - current_cycle)
    test_df['RUL'] = test_df['RUL_end'] + (test_df['max_cycle'] - test_df['time_cycles'])
    
    if cap is not None:
        test_df['RUL'] = test_df['RUL'].clip(upper=cap)
    
    test_df['label_bc'] = (test_df['RUL'] <= 30).astype(int)
    
    test_df.drop(['max_cycle', 'RUL_end'], axis=1, inplace=True)
    return test_df

def gen_sequence(df, seq_length, seq_cols):
    """Generates sequences for LSTM input."""
    data_array = df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length + 1), range(seq_length, num_elements + 1)):
        yield data_array[start:stop, :]

def gen_labels(df, seq_length, label_cols):
    """Generates labels for sequences."""
    data_array = df[label_cols].values
    num_elements = data_array.shape[0]
    return data_array[seq_length - 1:num_elements, :]

if __name__ == "__main__":
    # Test loading and processing
    data_path = "data/"
    train_file = os.path.join(data_path, "train_FD001.txt")
    test_file = os.path.join(data_path, "test_FD001.txt")
    rul_file = os.path.join(data_path, "RUL_FD001.txt")
    
    if os.path.exists(train_file):
        print("Loading training data...")
        train_df = load_data(train_file)
        train_df = calculate_rul(train_df)
        print(f"Train Shape: {train_df.shape}")
        
        print("Normalizing data...")
        scaler = MinMaxScaler()
        sensor_cols = ['s_{}'.format(i) for i in range(1, 22)]
        cols_normalize = sensor_cols + ['setting_1', 'setting_2', 'setting_3']
        train_df[cols_normalize] = scaler.fit_transform(train_df[cols_normalize])
        
        print("Data Preprocessing Complete.")
    else:
        print("Dataset not found. Please ensure files are in 'data/' folder.")
