from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Tuple
from PIL import Image
import torch
import librosa
import pandas as pd
import numpy as np
from config import (
    SAMPLE_RATE,
    AUDIO_DURATION,
    N_MELS,
    LABEL_MAP,
    PROJECT_ROOT,
    SPECTROGRAM_DIR,
    N_FFT,
    HOP_LENGTH,
    METADATA_PATH,
)

class DurianMultimodalDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_size: int = 224,
        sample_rate: int = SAMPLE_RATE,
        duration: float = AUDIO_DURATION,
        n_mels: int = N_MELS,
        label_map: str = LABEL_MAP,
        augment_image: bool = False,
        use_cached_spectrogram: bool = True,
        fallback_to_raw_audio: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.label_map = label_map if label_map is not None else LABEL_MAP
        self.use_cached_spectrogram = use_cached_spectrogram
        self.fallback_to_raw_audio = fallback_to_raw_audio
        
        self._validate_columns()
        
        if augment_image:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
    def _validate_columns(self) -> None:
        require_cols = {"fruit_id", "image_path", "audio_path", "label"}
        missing_cols = require_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"CSV thiếu các cột bắt buộc: {missing_cols}")
        
    def __len__(self) -> int:
        return len(self.df)
    
    @staticmethod
    def _resolve_path(path_value: str) -> Path:
        path = Path(str(path_value))
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path
    
    @staticmethod
    def _spectrogram_path(row: pd.Series) -> Path:
        audio_key = Path(str(row["audio_path"])).stem
        return SPECTROGRAM_DIR / f"{audio_key}_mel.npy" 
    
    def _load_image(self, image_path: Path) -> torch.Tensor:
        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.image_transform(image)
    
    def _audio_to_mel_from_raw(self, audio_path: Path) -> torch.Tensor:
        if not audio_path.exists():
            raise FileNotFoundError(f"Không tìm thấy audio {audio_path}")
        
        target_lenght = int(self.duration * self.sample_rate)
        y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        if len(y) > target_lenght:
            y = y[:target_lenght]
        else:
            y = np.pad(y, (0, target_lenght - len(y)), mode='constant')
            
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
    
    def _load_audio_as_mel(self, row: pd.Series) -> torch.Tensor:
        spectrogram_path = self._spectrogram_path(row)
        
        if self.use_cached_spectrogram and spectrogram_path.exists():
            mel_db = np.load(spectrogram_path).astype(np.float32)
            return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        
        if self.use_cached_spectrogram and not spectrogram_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy spectrogram cache: {spectrogram_path}."
                "Hãy chạy: python src/preprocess_audio.py --save_png"
            )
        
        audio_path = self._resolve_path(row["audio_path"])
        return self._audio_to_mel_from_raw(audio_path)
    
    def _encode_label(self, label_value: str) -> int:
        label_map = str(label_value).strip().lower()
        if label_value not in self.label_map:
            raise ValueError(f"Nhẫn {label_value} chưa có trong LABEL_MAP: {list(self.label_map.keys())}")
        return self.label_map[label_value]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        image = self._load_image(self._resolve_path(row["image_path"]))
        audio_mel = self._load_audio_as_mel(row)
        label = torch.tensor(self._encode_label(row["label"]), dtype=torch.long)
        fruit_id = str(row["fruit_id"])
        return image, audio_mel, label, fruit_id
    
if __name__ == "__main__":
    ds = DurianMultimodalDataset(str(METADATA_PATH), use_cached_spectrogram=True)
    print(f"Số mẫu: {len(ds)}")
    if len(ds) > 0:
        image, audio, label, fruit_id = ds[0]
        print(f"Image shape: {image.shape}")
        print(f"Audio shape: {audio.shape}")
        print(f"Label: {label.item()}")
        print(f"Fruit ID: {fruit_id}")