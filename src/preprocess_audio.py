from pathlib import Path
from typing import Iterable
import argparse
import numpy as np
import pandas as pd
import librosa

from config import (
    PROJECT_ROOT, 
    AUDIO_DURATION, 
    SAMPLE_RATE, 
    N_MELS, 
    N_FFT, 
    HOP_LENGTH, 
    METADATA_PATH, 
    SPECTROGRAM_DIR
)

def resolve_path(path_value: str) -> None:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path

def audio_to_mel_db(audio_path: Path) -> np.ndarray:
    target = int(AUDIO_DURATION * SAMPLE_RATE)
    
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    if len(y) > target:
        y = y[:target]
    else:
        y = np.pad(y, (0, target - len(y)), mode='constant')
        
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    
    return mel_db
    
def unique_audio_rows(df: pd.DataFrame) -> Iterable[pd.Series]:
    if "audio_id" in df.columns:
        return (row for _, row in df.drop_duplicates(subset=['audio_id']).iterrows())
    return (row for _, row in df.drop_duplicates(subset=["audio_path"]).iterrows())

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default=str(METADATA_PATH))
    parser.add_argument("--output_dir", type=str, default=str(SPECTROGRAM_DIR))
    parser.add_argument("--save_png", action="store_true", help="Lưu thêm ảnh .png preview của spectrogram")
    args= parser.parse_args()
    
    metatdata_path = resolve_path(args.metadata)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not metatdata_path.exists():
        raise FileNotFoundError(f"Không tìm thấy metadata: {metatdata_path}")
    
    df = pd.read_csv(metatdata_path)
    if "audio_path" not in df.columns:
        raise ValueError("metadata.csv thiếu cột audio_path")
    
    success = 0
    skipped = 0
    
    for row in unique_audio_rows(df):
        audio_path = resolve_path(row["audio_path"])
        audio_id = str(row["audio_id"]) if "audio_id" in df.columns else audio_path.stem
        
        if not audio_path.exists():
            print(f"[SKIP] Không tìm thấy audio: {audio_path}")
            skipped += 1
            continue
        
        mel_db = audio_to_mel_db(audio_path)
        npy_path = output_dir / f"{audio_id}_mel.npy"
        np.save(npy_path, mel_db)
        
        if args.save_png:
            png_path = output_dir / f"{audio_id}_mel.py"
            print(f"[OK] {audio_path.name} -> {npy_path.name}, {png_path.name}")
        else:
            print(f"[OK] {audio_path.name} -> {npy_path.name}")
            
        success += 1
    
    print("\nHoàn tất preprocess audio")
    print(f"Thành công: {success}")
    print(f"Bỏ qua: {skipped}")
    print(f"Output: {output_dir}")

if __name__ == "__main__":
    main()
