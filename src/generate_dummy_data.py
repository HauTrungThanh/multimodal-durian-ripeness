import argparse
import csv
import math
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import AUDIO_DIR, IMAGE_DIR, METADATA_PATH, SAMPLE_RATE

LABELS = ["unripe", "ripe", "overripe"]


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def make_knock_audio(label: str, duration: float = 2.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = np.random.default_rng()

    if label == "unripe":
        base_freq = rng.uniform(650, 900)
        decay = rng.uniform(8, 12)
    elif label == "ripe":
        base_freq = rng.uniform(350, 600)
        decay = rng.uniform(5, 8)
    else:
        base_freq = rng.uniform(180, 350)
        decay = rng.uniform(3, 6)

    knock_time = rng.uniform(0.15, 0.35)
    envelope = np.exp(-decay * np.maximum(0, t - knock_time))
    envelope[t < knock_time] = 0
    audio = 0.6 * np.sin(2 * math.pi * base_freq * t) * envelope
    audio += 0.2 * np.sin(2 * math.pi * base_freq * 1.8 * t) * envelope
    audio += rng.normal(0, 0.015, size=n)
    return audio.astype(np.float32)


def make_durian_like_image(label: str, path: Path, size: int = 320) -> None:
    rng = np.random.default_rng()
    if label == "unripe":
        bg = (235, 245, 225)
        fruit_color = (95, 145, 70)
        spike_color = (65, 110, 45)
    elif label == "ripe":
        bg = (245, 238, 210)
        fruit_color = (145, 130, 65)
        spike_color = (105, 95, 50)
    else:
        bg = (235, 220, 200)
        fruit_color = (120, 95, 55)
        spike_color = (85, 65, 40)

    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2 + int(rng.integers(-10, 10)), size // 2 + int(rng.integers(-10, 10))
    rx, ry = int(size * 0.33), int(size * 0.42)
    bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
    draw.ellipse(bbox, fill=fruit_color, outline=(60, 60, 45), width=3)

    for _ in range(110):
        angle = rng.uniform(0, 2 * math.pi)
        r = math.sqrt(rng.uniform(0, 1))
        x = cx + int(rx * r * math.cos(angle))
        y = cy + int(ry * r * math.sin(angle))
        if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
            length = int(rng.integers(8, 16))
            draw.line((x, y, x + int(length * math.cos(angle)), y + int(length * math.sin(angle))), fill=spike_color, width=2)

    draw.rectangle((cx - 15, cy - ry - 35, cx + 15, cy - ry + 5), fill=(90, 65, 35))
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0, 0.8))))
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_fruits", type=int, default=30)
    parser.add_argument("--images_per_fruit", type=int, default=3)
    parser.add_argument("--audios_per_fruit", type=int, default=3)
    args = parser.parse_args()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for fruit_idx in range(1, args.num_fruits + 1):
        fruit_id = f"D{fruit_idx:03d}"
        label = LABELS[(fruit_idx - 1) % len(LABELS)]
        variety = "Ri6"
        date_recorded = "2026-06-15"

        image_paths = []
        audio_paths = []

        for image_idx in range(1, args.images_per_fruit + 1):
            image_id = f"{fruit_id}_img_{image_idx:02d}"
            image_rel = Path("data/raw/images") / f"{image_id}.jpg"
            make_durian_like_image(label, IMAGE_DIR / f"{image_id}.jpg")
            image_paths.append((image_id, image_rel.as_posix()))

        for audio_idx in range(1, args.audios_per_fruit + 1):
            audio_id = f"{fruit_id}_knock_{audio_idx:02d}"
            audio_rel = Path("data/raw/audios") / f"{audio_id}.wav"
            audio = make_knock_audio(label)
            save_wav(AUDIO_DIR / f"{audio_id}.wav", audio, SAMPLE_RATE)
            audio_paths.append((audio_id, audio_rel.as_posix()))

        for image_id, image_path in image_paths:
            for audio_id, audio_path in audio_paths:
                rows.append(
                    {
                        "fruit_id": fruit_id,
                        "image_id": image_id,
                        "audio_id": audio_id,
                        "image_path": image_path,
                        "audio_path": audio_path,
                        "label": label,
                        "variety": variety,
                        "date_recorded": date_recorded,
                        "harvest_date": "",
                        "days_to_eat": "",
                        "note": "dummy synthetic sample - replace with real data later",
                    }
                )

    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fruit_id",
                "image_id",
                "audio_id",
                "image_path",
                "audio_path",
                "label",
                "variety",
                "date_recorded",
                "harvest_date",
                "days_to_eat",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã tạo dummy data: {args.num_fruits} trái, {len(rows)} cặp image-audio")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
