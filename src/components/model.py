from src.entities.configs import PredictConfig
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch import nn
from torchinfo import summary
import torch


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.config = PredictConfig()

        # Load the pretrained ResNet18 model
        # self.base_model = models.resnet18(pretrained=True) -> deprecated 
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Updated to use weights
        
        # Remove the fully connected layer from ResNet18
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])  # Keep the spatial dimensions (8x8)

        for param in self.base_model.parameters():
            param.requires_grad = False

        # Custom convolutional layers
        self.custom_layers = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv1-512
            nn.ReLU(),
            nn.Conv2d(512, 32, kernel_size=3, padding=1),   # Conv2-32
            nn.ReLU(),
            nn.Conv2d(32, 6, kernel_size=3, padding=1)      # Conv3-6
        )

        # Fully connected layer
        self.flatten = nn.Flatten()
        # Dynamically calculate the input size to the fully connected layer
        self.fc = nn.Linear(6 * 8 * 8, self.config.LABEL)

    def forward(self, x):
        x = self.base_model(x)  # Pass through ResNet18 (Output: [B, 512, 8, 8])
        x = self.custom_layers(x)  # Pass through custom conv layers (Output: [B, 6, 8, 8])
        x = self.flatten(x)  # Flatten (Output: [B, 6 * 8 * 8])
        x = self.fc(x)  # Fully connected layer (Output: [B, output_classes])
        return x    
    

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = NeuralNet().to(device)
    summary(net, input_size=[1, 3, 256, 256])