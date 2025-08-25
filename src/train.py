import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from torch.optim import Adam
from utils import read_classes, load_model, make_transform, build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset_ready", help="Dataset folder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outdir", default="artifacts")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset ---
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), transform)
    val_ds   = datasets.ImageFolder(os.path.join(args.data, "val"), transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch)

    # --- Model ---
    model = build_model(num_classes=len(train_ds.classes)).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # --- Validation ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss:.4f} - Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.outdir, "best_model.pth"))
            with open(os.path.join(args.outdir, "classes.txt"), "w") as f:
                f.write("\n".join(train_ds.classes))
            print(f"✅ Saved new best model ({best_acc:.4f})")

if __name__ == "__main__":
    main()
