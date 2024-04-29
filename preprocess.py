import pandas as pd
from datetime import datetime


def preprocess(train_data_path, test_data_path):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    print(f'Train Dataset shape: {train_data.shape}')
    print(f'Test Dataset shape: {test_data.shape}')
    print(f'Columns: {train_data.columns}')
    utc_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(utc_time)
    if 'processed_text' not in train_data:
        train_data = review_preprocess(train_data, train_data_path, proc_id, devices)
    if 'processed_text' not in test_data:
        test_data = review_preprocess(test_data, test_data_path, proc_id, devices)
    utc_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(utc_time)
    return train_data, test_data

