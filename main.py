import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from src.loadData import GraphDataset
from src.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

from src.models import GNN 

# Set the random seed
set_seed(88)


def recon_loss(z, pos_edge_index, neg_edge_index):
    pos_loss = -torch.log(torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)) + 1e-15).mean()
    return pos_loss + neg_loss


def gcod_loss(z, y, temperature=0.5):
    device = z.device
    sim_matrix = torch.matmul(z, z.T) / temperature
    sim_matrix = torch.nn.functional.log_softmax(sim_matrix, dim=1)

    labels = y.unsqueeze(0) == y.unsqueeze(1)  # [N, N] bool matrix
    labels = labels.float().to(device)

    loss_per_sample = - (sim_matrix * labels).sum(dim=1) / labels.sum(dim=1).clamp(min=1)
    return loss_per_sample.mean()


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


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


def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, desc="Training batches", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        z = model(data)
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=z.size(0))
        if hasattr(data, 'y') and data.y is not None:
            loss = gcod_loss(z, data.y)
        else:
            loss = recon_loss(z, pos_edge_index, neg_edge_index)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader)


def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    losses = []
    predictions = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            z = model(data)
            pos_edge_index = data.edge_index
            neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=z.size(0))
            if hasattr(data, 'y') and data.y is not None:
                loss = gcod_loss(z, data.y)
            else:
                loss = recon_loss(z, pos_edge_index, neg_edge_index)
            losses.append(loss.item())
            pred = z.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    if calculate_accuracy:
        accuracy = correct / total # ERROR: Division By 0
        return accuracy, sum(losses)/len(losses), predictions
    return predictions


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
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3
    
    

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = None

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

    print("📊 Loading test dataset...")
    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_node_features)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test dataset loaded with {len(test_dataset)} graphs.")

    # If train_path is provided, train the model
    if args.train_path:
        print("📊 Loading training dataset...")
        train_dataset = GraphDataset(args.train_path, transform=add_node_features)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Training dataset loaded with {len(train_dataset)} graphs.")

        print("🏋 Start training the model...")
        # Training loop
        num_epochs = args.epochs
        best_accuracy = 0.0
        train_losses = []
        train_accuracies = []

        # Calculate intervals for saving checkpoints
        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        for epoch in range(num_epochs):
            train_loss = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )
            train_acc, eval_loss, _ = evaluate(train_loader, model, device, calculate_accuracy=True)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Eval Loss: {eval_loss:.4f}")
            
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

    # Generate predictions for the test set using the best model
    print("🏋Start evaluating the model on the test set...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    save_predictions(predictions, args.test_path)
    print("✅ Predictions saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.4, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=6, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=18, help='number of epochs to train (default: 10)')
    
    args = parser.parse_args()
    main(args)
