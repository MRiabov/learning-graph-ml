import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms.normalize_features import NormalizeFeatures
from torch_geometric.utils import to_dense_batch


class GraphSAGELayer(MessagePassing):
    def __init__(self, in_channels, out_channels, lstm_hidden_dim):
        super().__init__(aggr=None)

        self.lin = torch.nn.Linear(in_channels + lstm_hidden_dim, out_channels)
        self.lstm = nn.LSTM(
            input_size=in_channels, hidden_size=lstm_hidden_dim, batch_first=True
        )
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        # save original node features to concatenate later
        self.x_self = x  # (num_nodes, in_channels)

        # GCN uses symmetric normalization with degree matrices; *GraphSAGE does not*.

        # start propagating messages
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j: features of source nodes [num_edges, out channels]
        return x_j  # neighbor messages

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # group inputs by index
        # index: destination node for each edge (size=num_edges)

        # group inputs by index
        # we use scatter logic to create a list of sequences per node

        # num_nodes = dim_size or int(index.max()) + 1
        # neighbors = [[] for _ in range(num_nodes)]

        # for i in range(index.size(0)):
        #     node_id = index[i].item()
        #     neighbors[node_id].append(inputs[i])

        # # pad to equal length and process with lstm
        # device = inputs.device
        # lstm_outputs = []
        # for seq in neighbors:
        #     if len(seq) == 0:
        #         # no neighbors: use zero
        #         lstm_outputs.append(torch.zeros(self.lstm.hidden_size, device=device))
        #     else:
        #         seq_tensor = torch.stack(seq).unsqueeze(0)
        #         _, (h_n, _) = self.lstm(seq_tensor)
        #         lstm_outputs.append(h_n.squeeze((0,1)))
        # out = torch.stack(lstm_outputs, dim=0)

        # note: this ^ is a manual implementation with a for loop (inefficient). the more real implementation would be:
        dense_inputs, mask = to_dense_batch(inputs, index, max_num_nodes=None)

        lengths = mask.sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            dense_inputs,
            lengths,  # lengths per node
            batch_first=True,
            enforce_sorted=False,
        )

        _, (h_n, _) = self.lstm(packed)

        out = h_n.squeeze(0)

        return out

    def update(self, aggr_out):
        # aggr_out: [num_nodes, in_channels]
        x_self = self.x_self
        concat = torch.cat([x_self, aggr_out], dim=1)  # concatenate self and neighbors.
        x = self.lin(concat)
        out = F.relu(x)
        out = self.norm(out)
        out = self.dropout(out)
        return out


class GraphSAGE(nn.Module):
    def __init__(self, num_features, hidden_dim, lstm_hidden_dim, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = GraphSAGELayer(num_features, hidden_dim, lstm_hidden_dim)
        self.conv2 = GraphSAGELayer(hidden_dim, hidden_dim, lstm_hidden_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        out = self.conv2(x, edge_index)
        return out


if __name__ == "__main__":
    # note: the goal of the dataset is to predict the label which the graph has.
    batch_dataset = Planetoid("datasets/cora/", "Cora", transform=NormalizeFeatures())
    data = batch_dataset[0]  # note: Cora is a single graph, not many of them.
    epochs = 120

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init model:
    model = GraphSAGE(
        num_features=data.num_features,
        hidden_dim=128,
        lstm_hidden_dim=128,
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
    torch.save(model, "classification/checkpoints/cora_graphSAGE_lstm.pt")

    # eval: # (here eval=test).
    model.eval()
    eval_out = model(data)
    eval_labels = data.y[test_mask]
    eval_loss = loss_fn(eval_out[test_mask], eval_labels)
    eval_pred = eval_out[test_mask].argmax(dim=1)  # highest score in prediction.
    eval_acc = (eval_pred == eval_labels).sum().item() / test_mask.sum().item()

    print("eval precision:", eval_acc)

# precision:
# 120 epoch,
#  hidden_dim, mean aggr, with self-loops - .308 precision  # note: it could be random initialization.
# 120 epoch, 128 hidden_dim, mean aggr, with self-loops - .458 precision
# 120 epoch, 128 hidden_dim, mean aggr, *no self loops* - .708 precision (not bad!)
# 120 epoch, 16 hidden_dim, mean aggr, no self-loops - very unstable, but .13~.504 (not enough features), avg~.350
# 120 epoch, 16 hidden_dim , LSTM aggr (python for loops), no self-loops - .167
# 120 epoch, 128 hidden_dim, LSTM aggr (python for loops), no self-loops - .167
# 120 epoch, 128 hidden dim, LSTM aggr, efficient impl, + norm and dropout - .755, but takes >15 minutes to train (long)


# note: 120 epoch,
#  hidden dim does not improve after 40 epoch.
