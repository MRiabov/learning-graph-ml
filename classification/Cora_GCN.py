import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
import json
import pandas
import numpy as np
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import torch_geometric.nn as gnn
from torch_geometric.transforms.normalize_features import NormalizeFeatures
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.datasets import Planetoid
import torch.nn.functional as F


class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="sum")

        self.lin1 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops to adjacency matrix
        edge_index, _ = add_self_loops(edge_index=edge_index)

        # apply linear transformation
        x = self.lin1(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # start propagating messages
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j: features of source nodes [num_edges, out channels]
        return norm.view(-1, 1) * x_j  # flatten and expand dims or something?


class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(num_features, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, hidden_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        out = self.conv2(x, edge_index)
        return out


if __name__ == "__main__":
    epochs = 200
    # train_size = 8192

    # note: the goal of the dataset is to predict the label which the graph has.
    batch_dataset = Planetoid("datasets/cora/", "Cora", transform=NormalizeFeatures())
    data = batch_dataset[0]  # note: Cora is a single graph, not many of them.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init model:
    model = GCN(
        num_features=data.num_features,
        hidden_dim=128,
        num_classes=batch_dataset.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data)
        loss = loss_fn(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # validation

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data)
                val_labels = data.y[val_mask]
                val_loss = loss_fn(val_out[val_mask], val_labels)
                val_pred = val_out[val_mask].argmax(
                    dim=1
                )  # highest score in prediction.
                val_acc = (val_pred == val_labels).sum().item() / val_mask.sum().item()

                print(
                    f"Epoch: {epoch}/{epochs}, loss: {loss.item()}, val_loss:{val_loss}, val_acc: {val_acc}"
                )

    # save net
    torch.save(model, "classification/checkpoints/cora_gcn.pt")

    # eval: # (here eval=test).
    model.eval()
    eval_out = model(data)
    eval_labels = data.y[test_mask]
    eval_loss = loss_fn(eval_out[test_mask], eval_labels)
    eval_pred = eval_out[test_mask].argmax(dim=1)  # highest score in prediction.
    eval_acc = (eval_pred == eval_labels).sum().item() / test_mask.sum().item()

    print("eval precision:", eval_acc)

# precision:
# 200epoch, 64 hidden_dim - .778 precision
# 200 epoch, 128 hidden_dim - .801 precision
# 200 epoch, 3 layers, 128 hidden_dim - .7803 - eval precision actually dropped.
