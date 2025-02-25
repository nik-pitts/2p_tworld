import torch.nn as nn


# FNC Network for behavior cloning
class BehaviorCloningModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(BehaviorCloningModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.fc(x)
