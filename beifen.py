import torch
from torch.utils.data import Dataset, DataLoader
 
import os
import yaml
import json
import importlib
import numpy as np
from tqdm import tqdm
import pickle
import argparse
 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
 
def load_config(config_path, experiment_name):
    config_path = os.path.abspath(config_path)
    project_root = os.path.abspath(os.path.join(os.path.dirname(config_path), ".."))
 
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
 
    exp_config = full_config['experiments'][experiment_name]
 
    for key, val in exp_config.items():
        if isinstance(val, str) and ("/" in val or "\\" in val):
            exp_config[key] = os.path.abspath(os.path.join(project_root, val))
 
    return exp_config
 
def load_model(model_name):
    module_name = f"model.{model_name}"
    module = importlib.import_module(module_name)
    return module.AMCModel
 
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
 
def test(experiment_name):
    experiment_name = experiment_name
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path, experiment_name)
    print("Loaded Experiment Configuration:")
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
 
    model_name = config["model_name"]
    test_data_path = config["test_data_path"]
    best_model_path = config["best_model_path"]
    test_result_path = config["test_result_path"]
    batch_size = config["batch_size"]
 
    os.makedirs(os.path.dirname(test_result_path), exist_ok=True)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"Using device: {device}")
 
    AMCModel = load_model(model_name)
    model = AMCModel().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {best_model_path}")
 
    print("Loading test data...")
    data_processor = RadioMLDataProcess(test_data_path)
    test_features, test_labels, test_snr = data_processor.process()
    print(f"Test features shape: {test_features.shape}, Test labels shape: {test_labels.shape}, Test SNR shape: {test_snr.shape}")
 
    test_dataset = RMLDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
    all_preds = []
    all_labels = []
    all_snr = test_snr
 
    print("Start testing...")
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing", leave=True)
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
 
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
 
    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Test Accuracy: {overall_acc:.4f}")
 
    # 定义SNR范围：-20到+18，间隔2
    expected_snr = np.arange(-20, 19, 2)
    snr_acc = {}
    for snr in expected_snr:
        mask = np.where(all_snr == snr)[0]
        if len(mask) > 0:
            snr_acc[snr] = accuracy_score(all_labels[mask], all_preds[mask])
            print(f"SNR {snr} dB Accuracy: {snr_acc[snr]:.4f}")
        else:
            snr_acc[snr] = 0.0
            print(f"SNR {snr} dB Accuracy: N/A (no data)")
 
    # 总体混淆矩阵
    raw_data = data_processor.load_data()
    modulation_types = sorted(set(key[0] for key in raw_data.keys()))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=modulation_types, yticklabels=modulation_types)
    plt.title("Confusion Matrix (Overall)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(os.path.dirname(test_result_path), "confusion_matrix.png"))
    plt.close()
 
    # 分 SNR 准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(list(snr_acc.keys()), list(snr_acc.values()), marker='o')
    plt.title("Accuracy vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(test_result_path), "snr_accuracy_curve.png"))
    plt.close()
 
    # 每个SNR的混淆矩阵
    num_snrs = len(expected_snr)
    rows = int(np.ceil(num_snrs / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(24, 6 * rows), constrained_layout=True)
    axes = axes.flatten()
 
    for idx, snr in enumerate(expected_snr):
        mask = np.where(all_snr == snr)[0]
        if len(mask) > 0:
            cm_snr = confusion_matrix(all_labels[mask], all_preds[mask])
            sns.heatmap(cm_snr, annot=True, fmt="d", cmap="Blues", xticklabels=modulation_types, yticklabels=modulation_types, ax=axes[idx])
            axes[idx].set_title(f"SNR = {snr} dB")
        else:
            axes[idx].text(0.5, 0.5, "No Data", fontsize=12, ha='center', va='center')
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            axes[idx].set_title(f"SNR = {snr} dB")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
 
    for idx in range(num_snrs, len(axes)):
        axes[idx].set_visible(False)
 
    plt.savefig(os.path.join(os.path.dirname(test_result_path), "snr_confusion_matrices.png"))
    plt.close()
 
    # 保存指定 SNR 下归一化混淆矩阵的数据
    target_snr = -8
    mask = np.where(all_snr == target_snr)[0]
    if len(mask) > 0:
        cm_target_norm = confusion_matrix(all_labels[mask], all_preds[mask], normalize='true')
        cm_data = {
            "target_snr": target_snr,
            "modulation_types": modulation_types,
            "normalized_confusion_matrix": cm_target_norm.tolist()
        }
        cm_json_path = os.path.join(os.path.dirname(test_result_path), f"normalized_confusion_matrix_2016_snr{target_snr}.json")
        with open(cm_json_path, "w") as f:
            json.dump(cm_data, f, indent=4)
        print(f"Normalized confusion matrix data for SNR = {target_snr} saved to {cm_json_path}")
    else:
        print(f"No data available for SNR = {target_snr}, normalized confusion matrix not saved.")
 
 
    results = {
        "overall_accuracy": float(overall_acc),
        "snr_accuracy": {str(k): float(v) for k, v in snr_acc.items()},
        "confusion_matrix": cm.tolist(),
        "modulation_types": modulation_types
    }
    with open(test_result_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Test results saved to {test_result_path}")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an AMC model with best model weights")
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment_1",
        help="Name of the experiment to run (default: experiment_1)"
    )
    args = parser.parse_args()
    experiment_name = args.experiment
    test(experiment_name)