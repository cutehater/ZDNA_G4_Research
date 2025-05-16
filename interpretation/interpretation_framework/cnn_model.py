import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCNN_12_layers(nn.Module):
    def __init__(self, width, features_count):
        super().__init__()
        self.width = width
        self.features_count = features_count

        self.seq = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.GroupNorm(2, 4),


            nn.Conv2d(4, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 8),


            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 16),


            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.GroupNorm(16, 32),


            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.GroupNorm(16, 64),

            nn.Conv2d(64, 128, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.GroupNorm(32, 128),


            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, 64),



            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.GroupNorm(16, 32),


            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 16),


            nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 8),


            nn.Conv2d(8, 4, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 4),


            nn.Conv2d(4, 1, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.GroupNorm(1, 1),



            nn.AlphaDropout(p = 0.2),
            nn.Linear(features_count + 4, 500),
            nn.AlphaDropout(p = 0.2),
            nn.SELU(),
            nn.Linear(500, 2)
    )

    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(batch, 1, self.width, self.features_count + 4)
        x = self.seq(x)
        x = torch.squeeze(x)
        x = F.log_softmax(x, dim=-1)
        return x
    
class DeepCNN_OptunaTuned(nn.Module):
    def __init__(self, width, omics_features_num):
        super().__init__()
        self.width = width
        self.omics_features_num = omics_features_num
        self.conv_layers = self._build_conv()
        self.flattened_size = self._get_flattened_size()
        self.fc_layers = self._build_fc()

    def _build_conv(self):
        n_layers = 8
        layers = []
        
        kernel_size = 3
        activation_layer = nn.LeakyReLU()
        
        channels_number = [1, 4, 32, 256, 512, 128, 16, 4, 1]
        dilations = [3, 2, 1, 1, 2, 2, 2, 3]
        strides = [1, 1, 1, 2, 1, 1, 1, 1]
        
        for i in range(1, n_layers, 2):
            block = ResidualConvBlock(
                channels_number=[channels_number[i-1], channels_number[i], channels_number[i+1]],
                kernel_size=kernel_size,
                strides=[strides[i-1], strides[i]],
                dilations=[dilations[i-1], dilations[i]],
                activation=activation_layer
            )
            layers.append(block)
            
        return nn.Sequential(*layers)

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.width, self.omics_features_num + 4)
            dummy_output = self.conv_layers(dummy_input)
            return dummy_output.shape[2] * dummy_output.shape[3]
        
    def _build_fc(self):
        linear_size = 1200
        return nn.Sequential(
            nn.AlphaDropout(p=0.4),
            nn.Flatten(-2, -1),
            nn.Linear(self.flattened_size, linear_size),
            nn.AlphaDropout(p=0.45),
            nn.ReLU(),
            nn.Linear(linear_size, self.width * 2),
            nn.Unflatten(-1, (self.width, 2)),
        )

    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(batch, 1, self.width, self.omics_features_num + 4)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = torch.squeeze(x)
        return F.log_softmax(x, dim=-1)
    
class ResidualConvBlock(nn.Module):
    def __init__(self, channels_number, kernel_size, strides, dilations, activation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_number[0], channels_number[1], kernel_size, 
                      stride=strides[0], dilation=dilations[0], padding=(kernel_size - 1) * dilations[0] // 2),
            nn.BatchNorm2d(channels_number[1]),
            activation,
            nn.Conv2d(channels_number[1], channels_number[2], kernel_size, 
                      stride=strides[1], dilation=dilations[1], padding=(kernel_size - 1) * dilations[1] // 2),
            nn.BatchNorm2d(channels_number[2]),
            activation,
        )
        
        self.shortcut = nn.Conv2d(
            channels_number[0],
            channels_number[2],
            kernel_size=1,
            stride=strides[0] * strides[1],
            padding=0
        )

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out