import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from src.loadData import GraphDataset
from src.utils import set_seed, AddStructuralFeatures
from src.models import GNN
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# Set the random seed
set_seed(88)


def train(data_loader, model, optimizer, criterion, device, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader, desc="Training batches", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    
    if (current_epoch + 1) % 5 == 0: # Save checkpoint every 5 epochs
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / total, correct / total


def evaluate(data_loader, model, device, criterion, calculate_metrics=False):
    model.eval()
    total_loss = 0
    labels = []
    predictions = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_metrics:
                labels.extend(data.y.cpu().tolist())
            if criterion is not None:
                loss = criterion(output, data.y)
                total_loss += loss.item() * data.num_graphs

    avg_loss = total_loss / len(data_loader.dataset) if criterion is not None else None
    
    if calculate_metrics and labels:
        f1 = f1_score(labels, predictions, average='macro')
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        recall = recall_score(labels, predictions, average='macro', zero_division=0)
        accuracy = accuracy_score(labels, predictions)
        return avg_loss, accuracy, f1, precision, recall, predictions
    
    return avg_loss, labels, predictions


def class_weights(dataset, num_classes=6):
    labels = []
    for data in dataset:
        if data.y is not None:
            labels.append(data.y.item() if data.y.dim() == 0 else data.y.squeeze().item())
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-9)
    weights = weights * (len(labels) / weights.sum())
    return torch.tensor(weights, dtype=torch.float)


def save_predictions(predictions, test_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()


def main(args):
    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Set hyperparameters and model
    best_num_layers = 5
    best_emb_dim = 300
    best_drop_ratio = 0.3

    best_lr = 0.001

    num_epochs = 10

    model = GNN(gnn_type='gin', num_class=6, num_layer=best_num_layers, emb_dim=best_emb_dim,
                drop_ratio=best_drop_ratio, virtual_node=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    
    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well

    # Define checkpoint path relative to the script's directory
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # Load pre-trained model for inference
    if os.path.exists(checkpoint_path) and not args.train_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded best model from {checkpoint_path}")

    # Function to transfrorm data by adding information about the nodes
    transform = Compose([AddStructuralFeatures()])
    best_batch_size = 64

    if args.train_path:
        print("📊 Loading training dataset...")
        train_dataset = GraphDataset(args.train_path, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
        print(f"Training dataset loaded with {len(train_dataset)} graphs.")

        weights = class_weights(train_dataset)
        print(f"Class weights computed: {weights}")
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

        best_accuracy = 0.0
        train_losses = []
        train_accuracies = []

        for epoch in range(num_epochs):
            train_loss, train_acc = train(
                train_loader, model, optimizer, criterion, device,
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )
            train_loss_eval, train_acc_eval, _ , _ , _ , _ = evaluate(train_loader, model, device, calculate_metrics=True)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Epoch {epoch + 1}/{num_epochs}, Evaluation Loss: {train_loss_eval:.4f}, Evaluation Train Acc: {train_acc_eval:.4f}")
            
            # Save logs for training progress
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Save best model
            if train_acc > best_accuracy:
                best_accuracy = train_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")

        # Plot training progress in current directory
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))

        print("🏋Training completed. Best model saved.")

    print("📊 Loading test dataset...")
    test_dataset = GraphDataset(args.test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)
    print(f"Test dataset loaded with {len(test_dataset)} graphs.")




def research_main(args):
    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Hyperparameter search mode
    num_laers_search = [4, 5, 6]
    emb_dim_search = [64, 128, 256]
    drop_ratio_search = [0.2, 0.3, 0.5]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--hyper_research", type=bool, default=False, help="Whether to use hyperparameter research mode")
    
    args = parser.parse_args()
    if args.hyper_research:
        research_main(args)
    else:
        main(args)