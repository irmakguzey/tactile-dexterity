import torch 
import torch.nn as nn

# Module to print out the shape of the conv layer - used to debug
class PrintSize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x

# Encoder where convolution is not used 
# only linear layers are used to encode tactile information
class TactileLinearEncoder(nn.Module):
    def __init__(
        self,
        input_dim = 48,
        hidden_dim = 128,
        output_dim = 64
    ):
        super().__init__() 
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Flatten the image
        x = torch.flatten(x,1)
        x = self.model(x)
        return x

# Encoder for single sensor design where we learn each sensor separately
# and concatenate embeddings for each pad in the end (during KNN)
class TactileSingleSensorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_dim # Final dimension of the representation
    ):
        super().__init__()
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4),
            nn.ReLU(),
        )
        self.final_layer = nn.Sequential(
           nn.Linear(in_features=32*7*7, out_features=512),
           nn.ReLU(), 
           nn.Linear(in_features=512, out_features=out_dim)
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.final_layer(x)
        return x

# Model for 16x3 RGB channelled images
class TactileStackedEncoder(nn.Module): 
    def __init__(
        self,
        in_channels,
        out_dim # Final dimension of the representation
    ):
        super().__init__()
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2),
            nn.ReLU(),
        )
        self.final_layer = nn.Sequential(
           nn.Linear(in_features=32*10*10, out_features=1024),
           nn.ReLU(), 
           nn.Linear(in_features=1024, out_features=out_dim)
        )
        
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.final_layer(x)
        return x

# Encoder for the whole tactile image (16x16 tactile image)
class TactileWholeHandEncoder(nn.Module): 
    def __init__(
        self,
        in_channels,
        out_dim # Final dimension of the representation
    ):
        super().__init__()
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2),
            nn.ReLU(),
        )

        self.final_layer = nn.Sequential(
           nn.Linear(in_features=16*10*10, out_features=1024),
           nn.ReLU(), 
           nn.Linear(in_features=1024, out_features=out_dim)
        )
        
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.final_layer(x)
        return x
