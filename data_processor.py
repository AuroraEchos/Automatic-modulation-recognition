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
        self.modulation_data = self.load_data()
        self.snr_levels, self.modulation_types = self.get_snr_and_modulation_types()

    def load_data(self):
        with open(self.file_path, 'rb') as f:
            return pickle.load(f, encoding='latin1')

    def get_snr_and_modulation_types(self):
        snr_levels = sorted(list(set(key[1] for key in self.modulation_data.keys())))
        modulation_types = sorted(list(set(key[0] for key in self.modulation_data.keys())))
        #print("Modulation Types:", modulation_types)
        return snr_levels, modulation_types

    def get_data_by_snr(self, snr_levels):
        signal_data, label_data = [], []

        for snr_level in snr_levels:
            for modulation_type in self.modulation_types:
                signals = self.modulation_data.get((modulation_type, snr_level))
                if signals is not None:
                    signal_data.append(signals)
                    label_data.extend([modulation_type] * signals.shape[0])

        if signal_data:
            signal_data = np.vstack(signal_data)
            label_data = np.array([self.modulation_types.index(label) for label in label_data])
        else:
            raise ValueError(f"No data found for given SNR levels: {snr_levels}")

        return signal_data, label_data

    def augment_data(self, signal_data, snr_level):
        """Apply data augmentation techniques."""
        augmented_signals = []
        
        for signal in signal_data:
            noise = np.random.normal(0, 0.1 * (1 / (1 + np.exp(snr_level))), signal.shape)
            noisy_signal = signal + noise

            if np.random.rand() < 0.5:
                noisy_signal = np.flip(noisy_signal, axis=1)

            shift = np.random.randint(-2, 3)
            noisy_signal = np.roll(noisy_signal, shift, axis=1)

            augmented_signals.append(noisy_signal)

        return np.array(augmented_signals)

    def prepare_data_for_model(self, snr_levels, validation_size=0.2, augment=False):
        signal_data, label_data = self.get_data_by_snr(snr_levels)

        if augment:
            augmented_signals = []
            augmented_labels = []

            for i, snr_level in enumerate(snr_levels):
                snr_indices = [idx for idx, label in enumerate(label_data) if label == i]

                augmented_signal = self.augment_data(signal_data[snr_indices], snr_level)
                augmented_signals.append(augmented_signal)
                
                augmented_labels.append([i] * len(augmented_signal))
            
            signal_data = np.vstack([signal_data] + augmented_signals)
            label_data = np.concatenate([label_data] + [np.array(labels) for labels in augmented_labels])

        #print("Final Signal Data Shape:", signal_data.shape)
        #print("Final Label Data Shape:", label_data.shape)

        X_train, X_val, y_train, y_val = train_test_split(signal_data, label_data, test_size=validation_size, random_state=42)

        train_mean = X_train.mean(axis=(0, 2), keepdims=True)
        train_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std

        return X_train, X_val, y_train, y_val
    

def plot_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    snr_values = []
    validation_accuracies = []

    for key, value in data.items():
        snr = int(key.split('_')[1])
        snr_values.append(snr)
        validation_accuracies.append(value['validation_accuracy'] / 100)

    snr_values, validation_accuracies = zip(*sorted(zip(snr_values, validation_accuracies)))

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
    last_six_snr = snr_values[-6:]
    last_six_accuracies = validation_accuracies[-6:]
    ax_inset.plot(last_six_snr, last_six_accuracies, marker='o', color='red', label='_nolegend_')

    ax_inset.set_xlabel("SNR", fontsize=8)
    ax_inset.tick_params(axis='both', which='major', labelsize=8)
    ax_inset.grid(True, linestyle='--', alpha=0.2)
    ax_inset.legend(fontsize=5)

    ax_inset.set_ylim(0.87, 0.93)
    step = 0.01
    ax_inset.set_yticks(np.arange(0.87, 0.93 + step, step))


    plt.show()


if __name__ == "__main__":
    plot_results("results.json")
