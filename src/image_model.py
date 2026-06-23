import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, pretrained: bool = False):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = backbone.classifier[0].in_features
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.projector(x)
    
class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, embedding_dim: int = 128, pretrained: bool = False):
        super().__init__()
        self.encoder = ImageEncoder(embedding_dim=embedding_dim, pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        return self.classifier(features)
    
if __name__ == "__main__":
    model = ImageClassifier(num_classes=3, pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")