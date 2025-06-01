"""
Train a modulation recognition model on the RadioML2016.10A dataset.

This script performs preprocessing, model training, evaluation, and result saving.
The model is trained on IQ samples to recognize modulation types under various SNR conditions.

Usage:
    python train.py --train_data_path Dataset/train_data.pkl \
                    --train_result_path result/train_result.json \
                    --save_model_path result/best_model.pth \
                    --epochs 100 --batch_size 64 --lr 1e-4

Dependencies:
    - Python 3.9+
    - PyTorch
    - NumPy
    - scikit-learn
    - tqdm
    - logging
    - json
    - pickle

Author: Wenhao Liu
Date: 2025-06-01
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import argparse
import pickle
import numpy as np
import os
import logging
from datetime import datetime
from tqdm import tqdm
import json

from model import AMRModel  # Assuming AMCModel is defined in model.py

class RadioMLDataProcess:
    def __init__(self, file_path):
        self.file_path = file_path
 
    def load_data(self):
        try:
            with open(self.file_path, 'rb') as file:
                return pickle.load(file, encoding='latin1')
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
    def normalize_iq(self, iq_signals):
        power = np.mean(np.square(iq_signals), axis=0)
        return iq_signals / np.sqrt(power + 1e-8)
     
    def process(self):
        """
        Process the RadioML2016.10A dataset and convert IQ signals into a format suitable for model input.
         
        Returns:
            all_features (np.ndarray): Processed feature array with shape (N, 2, 128)
            all_labels (np.ndarray): Corresponding label array with shape (N,)
            all_snr (np.ndarray): Corresponding SNR array with shape (N,)
        """
        raw_data = self.load_data()
        snr = sorted(set(key[1] for key in raw_data.keys()))
        modulation = sorted(set(key[0] for key in raw_data.keys()))
 
        num_modulation = len(modulation)
        num_snr = len(snr)
        num_samples_per_snr = len(raw_data[(modulation[0], snr[0])])
        total_samples = num_modulation * num_snr * num_samples_per_snr
         
        all_features = np.zeros((total_samples, 2, 128))
        all_labels = np.zeros(total_samples, dtype=int)
        all_snr = np.zeros(total_samples)
 
        sample_idx = 0
        for mod_idx, mod_type in enumerate(modulation):
            for snr_idx, snr_type in enumerate(snr):
                key = (mod_type, snr_type)
                samples = np.array(raw_data[key])
                for i in range(samples.shape[0]):
                    iq_signal = samples[i].T
                    normalized_signal = self.normalize_iq(iq_signal).T
                    all_features[sample_idx] = normalized_signal
                    all_labels[sample_idx] = mod_idx
                    all_snr[sample_idx] = snr_type
                    sample_idx += 1
 
        return all_features, all_labels, all_snr

class RMLDataset(Dataset):
    def __init__(self, features, labels):
        assert features.shape[1:] == (2, 128), f"Expected features shape (N, 2, 128), got {features.shape}"
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def setup_logging():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "result")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_results_to_json(train_losses, val_losses, val_accs, file_path):
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)
    print(f"Results saved to {file_path}")

def train(args):
    logger = setup_logging()
    logger.info("Starting training process...")

    train_val_data_path = args.train_val_data_path
    best_model_path     = args.best_model_path
    train_result_path   = args.train_result_path
    batch_size          = args.batch_size
    epochs              = args.epochs
    lr                  = args.lr
    patience            = args.patience
    weight_decay        = args.weight_decay
    scheduler_factor    = args.scheduler_factor
    scheduler_patience  = args.scheduler_patience

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading and processing data...")
    data_processor = RadioMLDataProcess(train_val_data_path)
    features, labels, _ = data_processor.process()
    logger.info(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.25, random_state=2025, stratify=labels
        # Ensure the overall segmentation ratio is 6:2:2
    )

    train_dataset = RMLDataset(X_train, y_train)
    val_dataset = RMLDataset(X_val, y_val)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = AMRModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    logger.info("Start training...")
    train_losses, val_losses, val_accs = [], [], []
    best_acc = 0.0
    early_stop_counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_train_samples = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
 
        for inputs, labels in train_bar:
            batch_size = inputs.size(0)
            total_train_samples += batch_size
            inputs, labels = inputs.to(device), labels.to(device)
 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size
 
            train_bar.set_postfix({'loss': loss.item()})
 
        train_loss /= total_train_samples
        train_losses.append(train_loss)
 
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        correct = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
 
        with torch.no_grad():
            for inputs, labels in val_bar:
                batch_size = inputs.size(0)
                total_val_samples += batch_size
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
 
                val_bar.set_postfix({'loss': loss.item()})
 
        val_loss /= total_val_samples
        val_acc = correct / total_val_samples
        val_losses.append(val_loss)
        val_accs.append(val_acc)
 
        scheduler.step(val_loss)
 
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
 
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
 
    save_results_to_json(train_losses, val_losses, val_accs, train_result_path)
    logger.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an AMC model with RadioML dataset.")
    parser.add_argument("--train_val_data_path", type=str, default="Dataset/train_val_data.pkl")
    parser.add_argument("--best_model_path", type=str, default="result/best_model.pth")
    parser.add_argument("--train_result_path", type=str, default="result/train_result.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=5)

    args = parser.parse_args()

    train(args)
