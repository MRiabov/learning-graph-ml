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



class GATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=2):
        super().__init__(aggr="sum")
        self.heads = heads
        self.out_channels = out_channels

        # linear transformation for node features
        self.lin = nn.Linear(in_channels, heads*out_channels, bias=False)

        # Attention mechanism parameters
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.att)

        # Optional bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.zeros_(self.bias)
        # ^note: # could be made nn.Linear but less efficient.

    def forward(self, x, edge_index):
        # Add self-loops to adjacency matrix
        edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=x.size(0))

        # apply linear transformation
        x = self.lin(x)  # [num_nodes, heads*out_channels]
        x = x.view(-1, self.heads, self.out_channels)  # num_nodes, heads, out_channels

        # start propagating messages (calls "message" and "aggregate")
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index):
        # x_i: target_node_features [num_edges]
        # x_j: features of source nodes [num_edges, out channels]
        x_cat = torch.cat([x_i, x_j], dim=-1)

        # Compute attention scores (e^T * [Wh_i || Wh_j])
        alpha = (x_cat * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        # ^ ensure that even highly negative relationships are weighted less, but are still included (leaky relu.).
        alpha = torch.exp(alpha - alpha.max())  # Numerical stability
        alpha = alpha / (alpha.sum(dim=0, keepdim=True) + 1e-10)  # Softmax # normalize over edges for each head

        # weight messages by attention.
        return x_j * alpha.view(-1, self.heads, 1)  # [num_edges, heads, out_channels]

    def update(self, aggr_out):
        # aggregate across heads (mean or sum)
        aggr_out = aggr_out.mean(dim=1)  # [num_nodes, out_channels]
        aggr_out = aggr_out + self.bias
        return aggr_out


class GAT(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, heads: int):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(num_features, hidden_dim, heads=heads)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, heads=heads)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        out = self.gat2(x, edge_index)
        return out


if __name__ == "__main__":
    epochs = 200
    # train_size = 8192

    # note: the goal of the dataset is to predict the label which the graph has.
    batch_dataset = Planetoid("datasets/cora/", "Cora", transform=NormalizeFeatures())
    data = batch_dataset[0]  # note: Cora is a single graph, not many of them.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init model:
    model = GAT(
        num_features=data.num_features,
        hidden_dim=128,
        num_classes=batch_dataset.num_classes,
        heads=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
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
    torch.save(model, "classification/checkpoints/cora_gat.pt")

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
