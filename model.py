import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Reduce the number of residual blocks
        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)  # Slightly reduced dropout

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        
        x = self.res_block1(x)
        x = self.pool(x)
        
        x = self.res_block2(x)
        x = self.pool(x)
        
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        weighted_output = torch.sum(weights * lstm_output, dim=1)
        return weighted_output


class BiLSTMModel(nn.Module):
    def __init__(self, feature_extractor, num_classes=11):
        super(BiLSTMModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.lstm_input_size = 128
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = Attention(512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.fc(x)
        return x