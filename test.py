import argparse
import os
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report
from src.loadData import GraphDataset
from src.models import ImprovedGAT as ImprovedNNConv
from src.utils import set_seed, add_node_features, train, evaluate
from collections import Counter
from torch.utils.data import WeightedRandomSampler


# IMPLEMENTED
def compute_class_weights(dataset, num_classes=6):
    labels = [data.y.item() for data in dataset if data.y is not None]
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    freqs = torch.tensor([label_counts.get(i, 0) / total for i in range(num_classes)], dtype=torch.float32)
    weights = 1.0 / (freqs + 1e-8)
    weights = weights / weights.sum()
    return weights

def make_balanced_sampler(dataset, num_classes=6):
    labels = [data.y.item() for data in dataset if data.y is not None]
    label_counts = Counter(labels)
    weights_per_class = {cls: 1.0 / count for cls, count in label_counts.items()}
    sample_weights = [weights_per_class[data.y.item()] for data in dataset]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

def add_node_features(data):
    row, col = data.edge_index
    deg = torch.bincount(row, minlength=data.num_nodes).float().view(-1, 1)
    deg = deg / (deg.max() + 1e-5)

    in_deg = torch.bincount(col, minlength=data.num_nodes).float().view(-1, 1)
    in_deg = in_deg / (in_deg.max() + 1e-5)

    out_deg = torch.bincount(row, minlength=data.num_nodes).float().view(-1, 1)
    out_deg = out_deg / (out_deg.max() + 1e-5)

    norm_node_id = torch.arange(data.num_nodes).float().view(-1, 1) / (data.num_nodes + 1e-5)
    data.x = torch.cat([deg, in_deg, out_deg, norm_node_id], dim=1)
    return data

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set_name = args.test_path.split("/")[-2]

    input_dim = 4
    hidden_dim = 64
    output_dim = 6
    model = ImprovedNNConv(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_dataset = GraphDataset(args.train_path, transform=add_node_features) if args.train_path else None
    test_dataset = GraphDataset(args.test_path, transform=add_node_features)
    batch_size = 32

    if train_dataset:
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        sampler = make_balanced_sampler(train_set)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2)

        weights = compute_class_weights(train_set)
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        train_loader = val_loader = None
        criterion = None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    # === Training ===
    best_val_acc = 0.0
    best_model_path = ""

    if train_loader:
        for epoch in range(args.epochs):
            train_loss, train_acc = train(train_loader, model, optimizer, criterion, device)

            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(val_loader, model, device, criterion, calculate_metrics=True)
                print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    checkpoints_dir = os.path.join("checkpoints", test_set_name)
                    os.makedirs(checkpoints_dir, exist_ok=True)
                    best_model_path = os.path.join(checkpoints_dir, f"best_model.pt")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"🟢 Best model updated! Saved to {best_model_path} (Acc: {best_val_acc:.4f})")
            else:
                val_loss, _ = evaluate(val_loader, model, device, criterion, calculate_metrics=False)
                print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # === Load best model ===
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"\n🔄 Loaded best model from {best_model_path}")

    # === Predict test ===
    print("Predicting on test set...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            logits = model(data)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())

    os.makedirs("submission", exist_ok=True)
    df = pd.DataFrame({"id": list(range(len(all_preds))), "pred": all_preds})
    df.to_csv(f"submission/testset_{test_set_name}.csv", index=False)
    print(f"Predictions saved to submission/testset_{test_set_name}.csv")


if __name__ == "_test_":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./datasets/A/train.json.gz")
    parser.add_argument("--test_path", type=str, default="./datasets/A/test.json.gz")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    main(args)
