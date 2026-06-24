import torch
import json
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from image_model import ImageClassifier
from audio_model import AudioClassifier
from fusion_model import FusionClassifier
from dataset import DurianMultimodalDataset

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import NUM_CLASSES, TEST_CSV, ID_TO_LABEL

def build_model(mode: str, embedding_dim: int, pretrained_image: bool):
    if mode == "image":
        return ImageClassifier(num_classes=NUM_CLASSES, embedding_dim=embedding_dim, pretrained=pretrained_image)
    if mode == "audio":
        return AudioClassifier(num_classes=NUM_CLASSES, embedding_dim=embedding_dim)
    if mode == "fusion":
        return FusionClassifier(num_classes=NUM_CLASSES, embedding_dim=embedding_dim, pretrained_image=pretrained_image)
    raise ValueError(mode)

def forward_by_mode(model, mode, images, audio_mels):
    if mode == "image":
        return model(images)
    if mode == "audio":
        return model(audio_mels)
    if mode == "fusion":
        return model(images, audio_mels)
    raise ValueError(mode)

def save_confusion_matrix(cm, labels, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "audio", "fusion"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default=str(TEST_CSV))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    embedding_dim = int(checkpoint.get("embedding_dim", 128))
    pretrained_image = bool(checkpoint.get("pretrained_image", False))
    
    model = build_model(args.mode, embedding_dim, pretrained_image).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    dataset = DurianMultimodalDataset(args.test_csv, augment_image=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    y_true = []
    y_pred = []
    fruit_ids_all = []
    
    with torch.no_grad():
        for images, audio_mels, labels, fruits_ids in loader:
            images = images.to(device)
            audio_mels = audio_mels.to(device)
            outputs = forward_by_mode(model, args.mode, images, audio_mels)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
            fruit_ids_all.extend(list(fruits_ids))
            
    labels = [ID_TO_LABEL[i] for i in range (NUM_CLASSES)]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    
    output_dir = Path(args.checkpoint).parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
    }    
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        
    save_confusion_matrix(cm, labels, output_dir / "confusion_matrix.png")
    
    print(f"Accuracy: {acc}")
    print(f"Macro F1: {macro_f1}")
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    print("Đã lưu evaluation tại :", output_dir)
    
if __name__ == "__main__":
    main()
     