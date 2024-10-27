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
        print(modulation_types)
        return snr_levels, modulation_types

    def get_data(self):
        return self.get_data_by_snr(self.snr_levels)

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

    def prepare_data_for_model(self, snr_levels, validation_size=0.2):
        signal_data, label_data = self.get_data_by_snr(snr_levels)
        
        X_train, X_val, y_train, y_val = train_test_split(signal_data, label_data, test_size=validation_size, random_state=42)

        mean = np.mean(X_train, axis=0, keepdims=True)
        std = np.std(X_train, axis=0, keepdims=True) + 1e-8

        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        return X_train, X_val, y_train, y_val
    
    def plot_signal(self, modulation_type, snr_level):
        signal = self.modulation_data[(modulation_type, snr_level)][0]
        i_component = signal[0]  # I
        q_component = signal[1]  # Q

        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(i_component)
        plt.title(f"{modulation_type} Signal at SNR {snr_level} - I Component")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(q_component)
        plt.title(f"{modulation_type} Signal at SNR {snr_level} - Q Component")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

    def plot_signal_spectrum(self, modulation_type, snr_level):
        signal = self.modulation_data[(modulation_type, snr_level)][0]
        i_component = signal[0]
        q_component = signal[1]
        
        # 计算I/Q信号的频谱
        freqs = np.fft.fftfreq(len(i_component))
        i_spectrum = np.fft.fft(i_component)
        q_spectrum = np.fft.fft(q_component)
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(freqs, np.abs(i_spectrum))
        plt.title(f"{modulation_type} Signal Spectrum at SNR {snr_level} - I Component")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        
        plt.subplot(2, 1, 2)
        plt.plot(freqs, np.abs(q_spectrum))
        plt.title(f"{modulation_type} Signal Spectrum at SNR {snr_level} - Q Component")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        
        plt.tight_layout()
        plt.show()

    def plot_signal_histogram(self, modulation_type, snr_level):
        signal = self.modulation_data[(modulation_type, snr_level)][0]
        i_component = signal[0]
        q_component = signal[1]
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.hist(i_component, bins=30, alpha=0.7, label="I Component", color='blue')
        plt.title(f"{modulation_type} Signal Histogram at SNR {snr_level} - I Component")
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        
        plt.subplot(2, 1, 2)
        plt.hist(q_component, bins=30, alpha=0.7, label="Q Component", color='green')
        plt.title(f"{modulation_type} Signal Histogram at SNR {snr_level} - Q Component")
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()

    def plot_iq_phase_diagram(self, modulation_type, snr_level):
        signal = self.modulation_data[(modulation_type, snr_level)][0]
        i_component = signal[0]
        q_component = signal[1]
        
        plt.figure(figsize=(6, 6))
        plt.scatter(i_component, q_component, alpha=0.5)
        plt.title(f"I-Q Phase Diagram for {modulation_type} at SNR {snr_level}")
        plt.xlabel("I Component")
        plt.ylabel("Q Component")
        plt.grid(True)
        plt.show()

    def plot_multiple_signals(self, modulation_list, snr_level, title, fig_size=(10, 6), rows=2, cols=4):
        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        axes = axes.flatten()

        for i, modulation_type in enumerate(modulation_list):
            signal = self.modulation_data[(modulation_type, snr_level)][0]
            i_component = signal[0]  # I
            q_component = signal[1]  # Q

            axes[i].plot(i_component, label="I Component")
            axes[i].plot(q_component, label="Q Component", linestyle='--')
            axes[i].set_title(f"{modulation_type} - SNR {snr_level} dB")
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Amplitude")
            axes[i].legend()

        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


def plot_accuracy_from_json(files, batch_sizes):
    snr_values = range(-20, 20, 2)

    accuracies = {batch_size: [] for batch_size in batch_sizes}

    for file, batch_size in zip(files, batch_sizes):
        with open(file, 'r') as f:
            data = json.load(f)

        for snr in snr_values:
            snr_str = str(snr)
            if snr_str in data:
                accuracies[batch_size].append(data[snr_str]['best_val_accuracy'])
            else:
                accuracies[batch_size].append(None)
    
    plt.figure(figsize=(10, 6))

    for batch_size in batch_sizes:
        plt.plot(snr_values, accuracies[batch_size], label=f'Batch Size {batch_size}', marker='o')

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Best Validation Accuracy (%)', fontsize=12)
    plt.title('Validation Accuracy vs SNR for Different Batch Sizes', fontsize=14)

    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10), [f'{i}%' for i in range(0, 101, 10)])

    # Improving grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.minorticks_on()

    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    data_processor = RMLDataProcessor('RML\RML2016.10a_dict.pkl')

    # plot accuracy from json files
    json_files = ['training_results_64.json', 'training_results_128.json', 'training_results_256.json', 'training_results_1024.json']
    batch_sizes = [64, 128, 256, 1024]
    plot_accuracy_from_json(json_files, batch_sizes)