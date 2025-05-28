import torch
import random
import numpy as np
import tarfile
import os
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.utils import degree, to_networkx

def set_seed(seed=777):
    seed = seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def gzip_folder(folder_path, output_file):
    """
    Compresses an entire folder into a single .tar.gz file.
    
    Args:
        folder_path (str): Path to the folder to compress.
        output_file (str): Path to the output .tar.gz file.
    """
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    print(f"Folder '{folder_path}' has been compressed into '{output_file}'")


class AddStructuralFeatures(object):
    def __call__(self, data):
        row, col = data.edge_index
        num_nodes = data.num_nodes

        deg = degree(row, num_nodes=data.num_nodes)
        deg_in = degree(col, num_nodes=data.num_nodes)
        deg_out = degree(row, num_nodes=data.num_nodes)

        max_deg = 10
        deg_onehot = torch.nn.functional.one_hot(deg.clamp(max=max_deg).long(), num_classes=max_deg + 1).float()

        G = to_networkx(data, to_undirected=True)
        pagerank = nx.pagerank(G)
        clustering = nx.clustering(G)

        pr_values = torch.tensor([pagerank[i] for i in range(num_nodes)], dtype=torch.float).view(-1, 1)
        clust_values = torch.tensor([clustering[i] for i in range(num_nodes)], dtype=torch.float).view(-1, 1)

        # Combina tutto
        features = torch.cat([
            deg.view(-1, 1),
            deg_in.view(-1, 1),
            deg_out.view(-1, 1),
            deg_onehot,
            pr_values,
            clust_values
        ], dim=1)

        data.x = features
        return data