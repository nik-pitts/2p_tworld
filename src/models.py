import torch.nn as nn


# FNC Network for behavior cloning
class BehaviorCloningModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(BehaviorCloningModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.fc(x)


# FNC Network for behavior cloning
# FNC Network for behavior cloning
class BehaviorCloningModelLv2(nn.Module):
    def __init__(self, input_size, output_size):
        super(BehaviorCloningModelLv2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.fc(x)
