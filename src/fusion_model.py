import torch
import torch.nn as nn

try:
    from image_model import ImageClassifier
    from audio_model import AudioClassifier
except ImportError:
    from .image_model import ImageEncoder
    from .audio_model import AudioEncoder

class FusionClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, embedding_dim: int = 128, pretrained_image: bool = True):
        super().__init__()
        self.image_encoder = ImageEncoder(embedding_dim=embedding_dim, pretrained=pretrained_image)
        self.audio_encoder = AudioEncoder(embedding_dim=embedding_dim)
        fusion_dim = embedding_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, image: torch.Tensor, audio_mel: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        audio_features = self.audio_encoder(audio_mel)
        fused = torch.cat([image_features, audio_features], dim=1)
        return self.classifier(fused)
    
if __name__ == "__main__":
    model = FusionClassifier(num_classes=3)
    image = torch.randn(4, 3, 224, 224)
    audio = torch.randn(4, 1, 128, 173)
    out = model(image, audio)
    print(f"Image: {image.shape}")
    print(f"Audio: {audio.shape}")
    print(f"Output: {out.shape}")