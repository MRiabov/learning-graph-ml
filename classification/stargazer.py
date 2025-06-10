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


class GRUGNN(MessagePassing):
    def __init__(self, hidden_dim, num_layers):
        super().__init__(aggr="mean")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node_encoder = torch.nn.Linear(1, hidden_dim)#how about identity/
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: Batch):
        x = torch.ones((data.num_nodes, 1))
        h = self.node_encoder(x)  # note: unnecessary since it's dummy anyway.
        for _ in range(self.num_layers):
            m = self.propagate(edge_index=data.edge_index, x=h)
            h = self.gru(m, h)

        out = gnn.global_mean_pool(h, data.batch)
        return self.readout(out)

    def message(self, x_j):
        return x_j


if __name__ == "__main__":
    batch_size = 128
    epochs = 16
    num_transitions = 2
    train_size = 8192
    eval_size = 2048

    # note: the graphs contain NO nodal or edge features, only graph connectivity information.
    # the goal is to predict whether a graph represents a machine learning or a web development community.
    with open("datasets/github_stargazers/git_edges.json") as f:
        git_edges = json.load(f)

    with open("datasets/github_stargazers/git_target.csv") as f:
        dataframe = pandas.read_csv(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preprocess
    labels = dataframe.to_numpy(dtype=np.float32)

    batch_labels = labels[:train_size, 1].reshape(
        train_size // batch_size, batch_size, 1
    )
    batch_labels = torch.from_numpy(batch_labels).to(device=device, dtype=torch.float32)

    datas = []
    # convert to connections to numpy array and then data.
    for i, (idx, edges) in enumerate(git_edges.items()):
        assert str(i) == idx, "data check: index must be equal to enumerate"
        edge_index = torch.tensor(edges).t().contiguous()
        num_nodes = edge_index.max().item() + 1
        x = torch.ones((num_nodes, 1))
        datas.append(Data(x=x, edge_index=edge_index, num_nodes=num_nodes))

    batches = []
    for batch_num in range(train_size // batch_size):
        start_id = batch_num * batch_size
        end_id = (batch_num + 1) * batch_size
        batch = Batch.from_data_list(datas[start_id:end_id])
        batches.append(batch)

    # init model:
    model = GRUGNN(hidden_dim=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        epoch_losses = torch.zeros((train_size // batch_size))
        for batch_num in range(train_size // batch_size):
            out_targets = batch_labels[batch_num]
            batch = batches[batch_num]
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, out_targets)
            loss.backward()
            optimizer.step()

            epoch_losses[batch_num] = loss.item()
        print("epoch losses mean:", epoch_losses.mean())

    # save net
    torch.save(model, "classification/checkpoints/stargazer.pt")

    # eval:
    model.eval()
    eval_datas = []
    # convert to connections to numpy array and then data.
    for i, (idx, edges) in enumerate(git_edges.items()):
        assert str(i) == idx, "data check: index must be equal to enumerate"
        edge_index = torch.tensor(edges).t().contiguous()
        num_nodes = edge_index.max().item() + 1
        x = torch.ones((num_nodes, 1))
        eval_datas.append(Data(x=x, edge_index=edge_index, num_nodes=num_nodes))

    eval_batches = []
    for batch_num in range(eval_size // batch_size):
        start_id = batch_num * batch_size
        end_id = (batch_num + 1) * batch_size
        batch = Batch.from_data_list(eval_datas[start_id:end_id])
        eval_batches.append(batch)

    eval_precision = torch.zeros((eval_size // batch_size))
    eval_labels = (
        torch.from_numpy(labels[:eval_size, 1])
        .reshape(eval_size // batch_size, batch_size, 1)
        .to(device=device, dtype=torch.float32)
    )
    with torch.no_grad():
        for batch_num in range(len(eval_labels)):
            out_targets = eval_labels[batch_num]
            batch = eval_batches[batch_num]
            out = model(batch)
            out = torch.round(out)
            eval_precision[batch_num] = (out_targets == out).float().mean()
    print("eval precision:", eval_precision)
    print("eval precision mean:", eval_precision.mean())
