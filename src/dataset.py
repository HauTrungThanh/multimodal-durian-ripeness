from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_MAP: Dict[str, int] = {
    "unripe": 0,
    "ripe": 1,
    "overripe": 2,
}


class DurianMultimodalDataset(Dataset):
    """
    Dataset đọc dữ liệu sầu riêng đa phương thức:
    - image_path: đường dẫn ảnh RGB
    - audio_path: đường dẫn file âm thanh gõ
    - label: nhãn độ chín

    Mỗi dòng trong metadata.csv đại diện cho một cặp image-audio.
    Về sau khi có data thật, cần đảm bảo chia train/test theo fruit_id.
    """

    def __init__(
        self,
        csv_path: str,
        image_size: int = 224,
        sample_rate: int = 22050,
        duration: float = 2.0,
        n_mels: int = 128,
        label_map: Optional[Dict[str, int]] = None,
    ):
        self.csv_path = Path(csv_path)
        self.root_dir = Path.cwd()
        self.df = pd.read_csv(self.csv_path)

        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.label_map = label_map if label_map is not None else LABEL_MAP

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._validate_columns()

    def _validate_columns(self) -> None:
        required_cols = {"fruit_id", "image_path", "audio_path", "label"}
        current_cols = set(self.df.columns)

        missing_cols = required_cols - current_cols
        if missing_cols:
            raise ValueError(f"metadata.csv thiếu các cột: {missing_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)

        if path.is_absolute():
            return path

        return self.root_dir / path

    def _load_image(self, image_path: Path) -> torch.Tensor:
        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        return image

    def _load_audio_as_mel(self, audio_path: Path) -> torch.Tensor:
        if not audio_path.exists():
            raise FileNotFoundError(f"Không tìm thấy audio: {audio_path}")

        target_length = int(self.sample_rate * self.duration)

        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Cắt hoặc padding để audio có cùng độ dài
        if len(y) > target_length:
            y = y[:target_length]
        else:
            pad_length = target_length - len(y)
            y = np.pad(y, (0, pad_length), mode="constant")

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=1024,
            hop_length=256,
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Chuẩn hóa về khoảng gần 0 mean, 1 std
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        # Tensor shape: [1, n_mels, time]
        mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)

        return mel_tensor

    def _encode_label(self, label_value: str) -> int:
        label_value = str(label_value).strip().lower()

        if label_value not in self.label_map:
            raise ValueError(
                f"Nhãn '{label_value}' chưa có trong LABEL_MAP. "
                f"Các nhãn hiện có: {list(self.label_map.keys())}"
            )

        return self.label_map[label_value]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]

        image_path = self._resolve_path(row["image_path"])
        audio_path = self._resolve_path(row["audio_path"])

        image = self._load_image(image_path)
        audio_mel = self._load_audio_as_mel(audio_path)
        label = torch.tensor(self._encode_label(row["label"]), dtype=torch.long)
        fruit_id = str(row["fruit_id"])

        return image, audio_mel, label, fruit_id


if __name__ == "__main__":
    dataset = DurianMultimodalDataset("data/raw/metadata.csv")

    print("Số mẫu:", len(dataset))

    image, audio_mel, label, fruit_id = dataset[0]

    print("Fruit ID:", fruit_id)
    print("Image shape:", image.shape)
    print("Audio Mel shape:", audio_mel.shape)
    print("Label:", label.item())