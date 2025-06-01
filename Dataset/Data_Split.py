"""
RadioML Dataset Split Script

This script splits the RadioML dataset into training and test sets according to a specified ratio 
(currently 80% training/validation and 20% testing), and saves the resulting datasets in pickle format.

Usage:
    - Modify the `file_path`, `train_output`, and `test_output` variables as needed.
    - Run the script directly to perform the split and save the output.

Details:
    - The input dataset is expected to be a pickle file containing a dictionary.
      Each key is a tuple (modulation type, SNR), and each value is a list of signal samples.
    - The samples are randomly shuffled before splitting to avoid order bias.
    - The test set is held out completely from training and validation.

Author: Wenhao Liu
Date: 2025-06-01
"""


import pickle
import numpy as np

class RadioMLDataSplit:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            with open(self.file_path, 'rb') as file:
                return pickle.load(file, encoding='latin1')
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

    def split_data(self, train_ratio=0.8):
        raw_data = self.load_data()

        train_data, test_data = {}, {}

        for key in raw_data:
            samples = raw_data[key]
            n_samples = len(samples)

            if n_samples != 1000:
                print(f"Warning: Unexpected sample count {n_samples} for {key}")

            indices = np.random.permutation(n_samples)

            split_idx = int(n_samples * train_ratio)

            train_data[key] = [samples[i] for i in indices[:split_idx]]
            test_data[key] = [samples[i] for i in indices[split_idx:]]

            print(f"Modulation: {key[0]}, SNR: {key[1]} - "
                  f"Train: {len(train_data[key])}, Test: {len(test_data[key])}")

        return train_data, test_data

    def save_data(self, train_output, test_output):
        train_data, test_data = self.split_data()

        with open(train_output, 'wb') as f:
            pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(test_output, 'wb') as f:
            pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data_split = RadioMLDataSplit('Dataset/RML2016.10a_dict.pkl')

    train_output = 'Dataset/train_val_data.pkl'
    test_output = 'Dataset/test_data.pkl'

    data_split.save_data(train_output, test_output)
