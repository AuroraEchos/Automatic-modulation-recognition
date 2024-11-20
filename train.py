# Description: This script is used to train the BiLSTM model on the RML dataset for different SNR levels.
# Date: 2024-9-23
# Author: Wenhao Liu

import os
import json
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import AMRModel
from data_processor import RMLDataProcessor
from sklearn.utils.class_weight import compute_class_weight

def save_results_to_json(snr, val_acc, args, save_path='results.json'):
    results = {}
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            results = json.load(f)
    results[f"SNR_{snr}"] = {"validation_accuracy": val_acc}
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_path}")


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = to_device((inputs, labels), device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy, all_labels, all_predictions


def validate_model(model, val_loader, criterion, device):
    avg_val_loss, val_accuracy, _, _ = evaluate_model(model, val_loader, criterion, device)
    print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    return avg_val_loss, val_accuracy


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def train(args):
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")

    print(f"Loading RML dataset from {args.file_path}")
    data = RMLDataProcessor(args.file_path)

    best_acc_per_snr = {}

    for snr in args.snr_level:
        print(f"\nTraining for SNR level: {snr} dB")

        X_train, X_val, y_train, y_val = data.prepare_data_for_model([snr], augment=True)

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).long())
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).long())

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = AMRModel(args.num_classes).to(device)

        class_weights = compute_class_weight('balanced', classes=np.arange(args.num_classes), y=y_train)
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_val_acc = 0.0

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0

            with tqdm(total=len(train_loader), desc=f"Epoch [{epoch + 1}/{args.epochs}]", unit="batch") as pbar:
                for inputs, labels in train_loader:
                    inputs, labels = to_device((inputs, labels), device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())

            print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {total_loss / len(train_loader):.4f}')

            avg_val_loss, val_acc = validate_model(model, val_loader, criterion, device)
            scheduler.step(epoch + len(train_loader))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"best_model_snr_{snr}.pth")
                print(f"New best model saved for SNR {snr} with accuracy: {best_val_acc:.2f}%")

        best_acc_per_snr[snr] = best_val_acc
        print(f"Best validation accuracy for SNR {snr}: {best_val_acc:.2f}%")

        save_results_to_json(snr, best_val_acc, args)

    print("\nTraining completed successfully!")
    for snr, acc in best_acc_per_snr.items():
        print(f"Best validation accuracy for SNR {snr}: {acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RML Training')

    # Load parameters
    parser.add_argument('--file_path', type=str, default='RML/RML2016.10a_dict.pkl', help='The path of the RML dataset')
    parser.add_argument('--snr_level', type=int, nargs='+', default=[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18], 
                        help='The SNR level(s) for training, currently only training [-20, -10, 0, 10, 20] one by one')

    # Training parameters
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use CUDA for training')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size for training')
    parser.add_argument('--num_classes', type=int, default=11, help='The number of classes')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate for training')
    parser.add_argument('--epochs', type=int, default=50, help='The number of epochs for training')

    args = parser.parse_args()
    train(args)
