import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.avgpool(x)
        return self.projector(x)

class AudioClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, embedding_dim: int = 128):
        super().__init__()
        self.encoder = AudioEncoder(embedding_dim=embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, audio_mel: torch.Tensor) -> torch.Tensor:
        features = self.encoder(audio_mel)
        return self.classifier(features)
    
if __name__ == "__main__":
    model = AudioClassifier(num_classes=3)
    x = torch.randn(4, 1, 128, 173)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Ouput: {y.shape}")