from pathlib import Path

dirs = [
    Path("data/"),

    Path("data/raw/"),
    Path("data/raw/images/"),
    Path("data/raw/audios/"),

    Path("data/processed/"),
    Path("data/processed/spectrogram"),

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
    
    Path('src/config.py'),
    Path('src/generate_dummy_data.py'),
    Path('src/data_audit.py'),

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