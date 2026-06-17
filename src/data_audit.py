import argparse
from pathlib import Path

import pandas as pd

from config import METADATA_PATH, PROJECT_ROOT


def resolve_path(path_value: str) -> Path:
    p = Path(str(path_value))
    return p if p.is_absolute() else PROJECT_ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default=str(METADATA_PATH))
    args = parser.parse_args()

    metadata = Path(args.metadata)
    if not metadata.exists():
        raise FileNotFoundError(f"Không tìm thấy metadata: {metadata}")

    df = pd.read_csv(metadata)
    print("=== Metadata preview ===")
    print(df.head())
    print("\n=== Basic stats ===")
    print("Số dòng:", len(df))
    print("Số fruit_id:", df["fruit_id"].nunique())
    print("Phân bố nhãn theo dòng:")
    print(df["label"].value_counts())
    print("\nPhân bố nhãn theo fruit_id:")
    fruit_labels = df.groupby("fruit_id")["label"].first()
    print(fruit_labels.value_counts())

    missing_images = []
    missing_audios = []
    for _, row in df.iterrows():
        if not resolve_path(row["image_path"]).exists():
            missing_images.append(row["image_path"])
        if not resolve_path(row["audio_path"]).exists():
            missing_audios.append(row["audio_path"])

    print("\n=== File check ===")
    print("Thiếu ảnh:", len(missing_images))
    if missing_images[:5]:
        print(missing_images[:5])
    print("Thiếu audio:", len(missing_audios))
    if missing_audios[:5]:
        print(missing_audios[:5])

    duplicated = df.duplicated(subset=["fruit_id", "image_path", "audio_path"]).sum()
    print("Dòng trùng fruit_id + image + audio:", duplicated)


if __name__ == "__main__":
    main()
