# Description: This file contains the class RMLDataProcessor which is used to load and process the RadioML 2016.10A dataset.
# Date: 2024-9-20
# Author: Wenhao Liu
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class RMLDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        print(f"Loading data from: {self.file_path}")
        self.modulation_data = self.load_data()
        print("Data loaded successfully.")
        self.snr_levels, self.modulation_types = self.get_snr_and_modulation_types()
        print(f"Found SNR levels: {self.snr_levels}")
        print(f"Found modulation types: {self.modulation_types}")

    def load_data(self):
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print(f"Data contains {len(data)} modulation types at different SNR levels.")
        return data

    def get_snr_and_modulation_types(self):
        snr_levels = sorted(set(key[1] for key in self.modulation_data.keys()))
        modulation_types = sorted(set(key[0] for key in self.modulation_data.keys()))
        return snr_levels, modulation_types

    def get_data_by_snr(self, snr_levels):
        print(f"Fetching data for SNR levels: {snr_levels}")
        signal_data, label_data = [], []
        
        for snr_level in snr_levels:
            for modulation_type in self.modulation_types:
                signals = self.modulation_data.get((modulation_type, snr_level))
                if signals is not None:
                    signal_data.append(signals)
                    label_data.extend([modulation_type] * signals.shape[0])

        if signal_data:
            signal_data = np.vstack(signal_data)
            label_data = np.array([self.modulation_types.index(label) for label in label_data], dtype=np.int32)
            print(f"Data for SNR levels {snr_levels} retrieved: {signal_data.shape[0]} samples.")
        else:
            raise ValueError(f"No data found for given SNR levels: {snr_levels}")
        
        return signal_data, label_data

    def augment_data(self, signal_data, snr_level):
        print(f"Augmenting data for SNR level: {snr_level}")
        augmented_signals = []
        
        noise_std = 0.1 / np.sqrt(10 ** (snr_level / 10))
        
        for signal in signal_data:
            noise = np.random.normal(0, noise_std, signal.shape)
            noisy_signal = signal + noise
            
            if np.random.rand() < 0.5:
                noisy_signal = np.flip(noisy_signal, axis=1)
            
            shift = np.random.randint(-2, 3)
            noisy_signal = np.roll(noisy_signal, shift, axis=1)
            
            augmented_signals.append(noisy_signal)
        
        print(f"Augmented {len(augmented_signals)} signals.")
        return np.array(augmented_signals)

    def prepare_data_for_model(self, snr_levels, validation_size=0.2, augment=False):
        print(f"Preparing data for model with validation size: {validation_size}, augmentation: {augment}")
        signal_data, label_data = self.get_data_by_snr(snr_levels)

        if augment:
            augmented_signals, augmented_labels = [], []
            for snr_level in snr_levels:
                snr_signal_data, snr_label_data = self.get_data_by_snr([snr_level])
                augmented_signal = self.augment_data(snr_signal_data, snr_level)
                augmented_signals.append(augmented_signal)
                augmented_labels.append(snr_label_data)
            
            if augmented_signals:
                print(f"Augmenting signals with {len(augmented_signals)} sets of augmented data.")
                signal_data = np.vstack([signal_data] + augmented_signals)
                label_data = np.concatenate([label_data] + [np.array(labels) for labels in augmented_labels])

        print("Splitting data into train and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(signal_data, label_data, test_size=validation_size, random_state=42)
        
        print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
        
        train_mean = np.mean(X_train, axis=0, keepdims=True)
        train_std = np.std(X_train, axis=0, keepdims=True) + 1e-8
        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std

        print("Data preparation complete.")
        return X_train, X_val, y_train, y_val


def plot_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    snr_values, validation_accuracies = zip(*sorted((int(key.split('_')[1]), value['validation_accuracy'] / 100)
                                                     for key, value in data.items()))
    
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, validation_accuracies, marker='o', label="Validation Accuracy")
    plt.title("Accuracy vs SNR", fontsize=14)
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Recording Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)
    plt.legend(fontsize=10, loc='lower right')

    ax_inset = plt.gca().inset_axes([0.6, 0.4, 0.35, 0.35])
    last_six_snr, last_six_accuracies = snr_values[-6:], validation_accuracies[-6:]
    ax_inset.plot(last_six_snr, last_six_accuracies, marker='o', color='red')
    
    ax_inset.set_xlabel("SNR", fontsize=8)
    ax_inset.tick_params(axis='both', which='major', labelsize=8)
    ax_inset.grid(True, linestyle='--', alpha=0.2)
    ax_inset.set_ylim(0.87, 0.93)
    ax_inset.set_yticks(np.arange(0.87, 0.94, 0.01))
    
    plt.show()


if __name__ == "__main__":
    file_path = ""
    processor = RMLDataProcessor(file_path)
    X_train, X_val, y_train, y_val = processor.prepare_data_for_model([-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18], validation_size=0.2, augment=True)
    print("Training data shape:", X_train.shape)
    print("Validation data shape:", X_val.shape)
    print("Training labels shape:", y_train.shape)
    print("Validation labels shape:", y_val.shape)
