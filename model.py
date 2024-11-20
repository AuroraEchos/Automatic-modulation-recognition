# Description: This file contains the model architecture for the AMR classification task.
# Date: 2024-9-22
# Author: Wenhao Liu

import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
    
    def forward(self, x):
        batch, channels, _ = x.size()
        se = x.mean(dim=2)
        se = torch.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(batch, channels, 1)
        return x * se

class ResidualBlockWithSE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockWithSE, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return torch.relu(out)

class AMRModel(nn.Module):
    def __init__(self, num_classes=11):
        super(AMRModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            ResidualBlockWithSE(2, 32),
            nn.MaxPool1d(2),
            ResidualBlockWithSE(32, 64),
            nn.MaxPool1d(2),
            ResidualBlockWithSE(64, 128),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_out, dim=1)
        return self.fc(pooled)



