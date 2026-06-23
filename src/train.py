import json
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from config import NUM_CLASSES, TRAIN_CSV, VAL_CSV, EXPERIMENTS_DIR
from image_model import ImageClassifier
from audio_model import AudioClassifier
from fusion_model import FusionClassifier
from dataset import DurianMultimodalDataset
from torch.utils.data import DataLoader

def buildd_model(mode: str, pretrained_image: bool, embedding_dim: int):
    if mode == "image":
        return ImageClassifier(num_classes=NUM_CLASSES, embedding_dim=embedding_dim, pretrained=pretrained_image)
    if mode == "audio":
        return AudioClassifier(num_classes=NUM_CLASSES, embedding_dim=embedding_dim, )
    if mode == "fusion":
        return FusionClassifier(num_classes=NUM_CLASSES, embedding_dim=embedding_dim, pretrained_image=pretrained_image)
    raise ValueError(f"Mode không hợp lệ: {mode}")

def forward_by_mode(model, mode, images, audio_mels):
    if mode == "image":
        return model(images)
    if mode == "audio":
        return model(audio_mels)
    if mode == "fusion":
        return model(images, audio_mels)
    raise ValueError(mode)

def run_epoch(model, dataloader, criterion, optimizer, device, mode: str, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(dataloader, desc="train" if train else "val", leave=False)
    for images, audio_mels, labels, fruit_ids in loop:
        images = images.to(device)
        audio_mels = audio_mels.to(device)
        labels = labels.to(device)
        
        if train:
            optimizer.zero_grad()
            
        with torch.set_grad_enabled(train):
            outputs = forward_by_mode(model, mode, images, audio_mels)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
                
        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / max(total, 1), correct / max(total, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "audio", "fusion"], required=True)
    parser.add_argument("--train_csv", type=str, default=str(TRAIN_CSV))
    parser.add_argument("--val_csv", type=str, default=str(VAL_CSV))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--pretrained_image", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_ds = DurianMultimodalDataset(args.train_csv, augment_image=(args.mode in ["image", "fusion"]))
    val_ds = DurianMultimodalDataset(args.val_csv, augment_image=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = buildd_model(args.mode, args.pretrained_image, args.embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    exp_dir = EXPERIMENTS_DIR / args.mode 
    
    best_val_acc = -1.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, args.mode, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, args.mode, train=False)
        
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d}/{args.epochs} |"
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} |"
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": args.mode,
                    "embedding_dim": args.embedding_dim,
                    "pretrained_image": args.pretrained_image,
                    "num_classes": NUM_CLASSES,
                    "val_acc": val_acc
                },
                exp_dir / "best_model.pt",
            )
    
    with open(exp_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
        
    print("Đã lưu best model tại:", exp_dir / "best_model.pt")
    print("Best val acc:", best_val_acc)
    
if __name__ == "__main__":
    main()