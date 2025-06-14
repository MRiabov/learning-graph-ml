import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms.normalize_features import NormalizeFeatures
from torch_geometric.utils import to_dense_batch
from torch.distributions import Normal


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
    def __init__(self, num_features, hidden_dim, out_size):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(num_features, hidden_dim)
        self.conv_mean = GCNLayer(hidden_dim, out_size)
        self.conv_stddev = GCNLayer(hidden_dim, out_size)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        out_mean = self.conv2(x, edge_index)
        out_stddev = self.conv2(x, edge_index)
        return out

#DIFFERENCE BETWEEN AE AND VAE:
#VAE SAMPLES MEAN AND STDDEV, AND SAMPLES A SAMPLE FROM THEM
#AE SIMPLY ENCODES FEATURES AND DECODES TO AND FROM LATENT.


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
    def forward 


class VGAE:
    def __init__(encoder, decoder, edge_detection_threshold=0.5):
        self.encoder=encoder
        self.decoder=decoder
    
    def forward(data: Data):
        latent = self.encoder(data)
        loc, stddev = latent.split(2)
        dist = Normal(loc, scale)
        sample = dist.sample()
        edge_likelihood = self.decoder(sample)
        edges = edge_detection_threshold(0.5)
        
        


if __name__ == "__main__":
    # note: the goal of the dataset is to predict the label which the graph has.
    batch_dataset = Planetoid("datasets/cora/", "Cora", transform=NormalizeFeatures())
    data = batch_dataset[0]  # note: Cora is a single graph, not many of them.
    epochs = 120

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_size = 128
    

    # init model:
    encoder = GCN(num_features = 128, hidden_dim = 128, out_size=128*2)
    decoder = Decoder()

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
    torch.save(model, "classification/checkpoints/cora_graphSAGE_pool.pt")

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
# 120 epoch, 128 hidden dim, LSTM aggr, efficient impl, + norm and dropout - .755, but takes >15 minutes to train on CPU (long)
# 120 epoch, 128 hidden dim, maxpool aggr - 0.638, .651, very quick to train (30 seconds)

# note: 120 epoch,
#  hidden dim does not improve after 40 epoch.
