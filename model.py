import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class PhaseAwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super(PhaseAwareConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn_real = nn.BatchNorm2d(out_channels)
        self.bn_imag = nn.BatchNorm2d(out_channels)

    def forward(self, x_real, x_imag):
        real = self.conv_real(x_real) - self.conv_imag(x_imag)
        imag = self.conv_imag(x_real) + self.conv_real(x_imag)
        real = self.bn_real(real)
        imag = self.bn_imag(imag)
        output = torch.cat((real, imag), dim=-2)
        return output

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_weights, V)
        return attn_output.mean(dim=1)

class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class AMRModel(nn.Module):
    def __init__(self, timesteps=128, classes=11):
        super(AMRModel, self).__init__()
        self.timesteps = timesteps

        # Part-A: Multi-Channel Feature Extraction
        self.conv1 = PhaseAwareConv2d(in_channels=1, out_channels=50, kernel_size=(1, 8))
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=8, padding=7)
        self.bn2 = nn.BatchNorm1d(50)
        self.se2 = SEBlock1D(channel=50)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=8, padding=7)
        self.bn3 = nn.BatchNorm1d(50)
        self.se3 = SEBlock1D(channel=50)
        self.conv4 = SeparableConv2d(in_channels=50, out_channels=50, kernel_size=(1, 8))
        self.bn4 = nn.BatchNorm2d(50)
        self.conv5 = SeparableConv2d(in_channels=100, out_channels=100, kernel_size=(2, 5), padding=0)
        self.bn5 = nn.BatchNorm2d(100)

        # Part-B: Temporal Characteristics Extraction
        self.GRU1 = nn.GRU(input_size=100, hidden_size=128, num_layers=1, batch_first=True)
        self.attention = SelfAttention(hidden_dim=128)

        # Part-C: Fully Connected Classifier
        self.fc1 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc_out = nn.Linear(128, classes)

    def extract_features(self, x):
        input_i = x[:, 0:1, :].unsqueeze(2)
        input_q = x[:, 1:2, :].unsqueeze(2)

        x1 = self.conv1(input_i, input_q)
        x1 = F.relu(x1)

        x2 = self.conv2(input_i.squeeze(2))
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.se2(x2).unsqueeze(2)

        x3 = self.conv3(input_q.squeeze(2))
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = self.se3(x3).unsqueeze(2)

        x = torch.cat((x2, x3), dim=2)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = torch.cat((x1, x), dim=1)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = x.squeeze(2).permute(0, 2, 1)

        x, _ = self.GRU1(x)
        x = self.attention(x)
        return x

    def forward(self, x):
        input_i = x[:, 0:1, :].unsqueeze(2)
        input_q = x[:, 1:2, :].unsqueeze(2)

        x1 = self.conv1(input_i, input_q)
        x1 = F.relu(x1)

        x2 = self.conv2(input_i.squeeze(2))
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.se2(x2).unsqueeze(2)

        x3 = self.conv3(input_q.squeeze(2))
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = self.se3(x3).unsqueeze(2)

        x = torch.cat((x2, x3), dim=2)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = torch.cat((x1, x), dim=1)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = x.squeeze(2).permute(0, 2, 1)

        x, _ = self.GRU1(x)
        x = self.attention(x)

        x = F.selu(self.fc1(x))
        x = self.dropout1(x)
        x = F.selu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc_out(x)

        return logits

if __name__ == "__main__":
    model = AMRModel()
    x = torch.randn(64, 2, 128)
    output = model(x)
    print(output.shape)  # (64, 11)
