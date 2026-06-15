from pathlib import Path

dirs = [
    Path("data/"),

    Path("data/raw/"),
    Path("data/raw/images/"),
    Path("data/raw/audios/"),

    Path("data/processed/"),
    Path("data/processed/images/"),
    Path("data/processed/spectrogram/"),
    
    Path("data/splits/"),

    Path("notebooks/"),
    Path("src/"),

    Path("experiments/"),
]

files = [
    Path("data/raw/metadata.csv"),

    Path("data/splits/train.csv"),
    Path("data/splits/val.csv"),
    Path("data/splits/test.csv"),

    Path("notebooks/01_check_image.ipynb"),
    Path("notebooks/02_check_audio.ipynb"),
    Path("notebooks/03_test_spectrogram.ipynb"),

    Path('src/dataset.py'),
    Path('src/image_model.py'),
    Path('src/audio_model.py'),
    Path('src/fusion_model.py'),
    Path('src/train_image.py'),
    Path('src/train_audio.py'),
    Path('src/train_fusion.py'),

    Path('requirements.txt'),
    Path('README.md'),
]

def main():
    for dir in dirs:
        dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        file.touch(exist_ok=True)

if __name__ == '__main__':
    main()