import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms.normalize_features import NormalizeFeatures
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import (
    train_test_split_edges,
    negative_sampling,
    add_self_loops,
    degree,
)
from torch.distributions import Normal, kl_divergence
from sklearn.metrics import roc_auc_score, average_precision_score


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
        x = F.dropout(x, p=0.5, training=self.training)
        out_mean = self.conv_mean(x, edge_index)
        out_stddev = self.conv_stddev(x, edge_index)
        return out_mean, out_stddev


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, z, edge_index):
        # z: [num_nodes, latent_dim]
        z_i = z[edge_index[0]]  # target?
        z_j = z[edge_index[1]]  # source?
        return torch.sum(z_i * z_j, dim=-1)  # dot product


class VGAE(nn.Module):
    def __init__(self, encoder, decoder: Decoder):
        super(VGAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # note: edge detection threshold is unnecessary - it is only useful during inference time.
        # otherwise edge detection is optimized from e.g. sigmoid output - directly.

    def forward(self, data: Data):
        loc, scale = self.encoder(data)
        dist = Normal(
            loc, torch.exp(scale)
        )  # scale should be exponential for positivity
        z = dist.rsample()
        return z, loc, scale


def reconstruction_loss(decoder, z, pos_edge_index, neg_edge_index):
    # dot product decoder

    # note: decoder for a standard VGAE can be implicitly done via a simply the dot product:
    # pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim - 1)
    # neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim - 1)

    # however, using a `decoder` is more modular:
    pos_scores = decoder(z, pos_edge_index)
    neg_scores = decoder(z, neg_edge_index)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores)
    )
    return pos_loss + neg_loss


def compute_kl_divergence(loc, log_std):
    # KL divergence from N(mu, log_std) to N(0,I)
    # return -0.5 * torch.mean(
    #     torch.sum(1 + 2 * log_std - mu.pow(2) - (2 * log_std).exp(), dim=1)
    # )
    return (
        kl_divergence(
            Normal(loc, log_std.exp()),
            Normal(torch.zeros_like(loc), torch.ones_like(log_std)),
        )
        .sum(dim=1)
        .mean()
    )


# next: implement vgae.
if __name__ == "__main__":
    # note: the goal of the dataset is to predict the label which the graph has.
    batch_dataset = Planetoid(
        "datasets/pubmed/", "Pubmed", transform=NormalizeFeatures()
    )
    data = batch_dataset[0]  # note: Cora is a single graph, not many of them.
    epochs = 1
    beta = 5  # b-vae.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_size = 128

    # init model:
    encoder = GCN(
        num_features=batch_dataset.num_node_features,
        hidden_dim=latent_size,
        out_size=latent_size,
    )
    decoder = Decoder()

    model = VGAE(encoder, decoder)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    recon_loss_fn = nn.CrossEntropyLoss()

    train_data, val_data, test_data = RandomLinkSplit(
        is_undirected=True, split_labels=True
    )(data)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z, mu, log_std = model(train_data)

        # reconstruct edges
        # pos_edge_index = train_data.edge_index
        # neg_edge_index = negative_sampling(
        #     # negative sampling - where there is no edges between nodes.
        #     edge_index=pos_edge_index,
        #     num_nodes=train_data.num_nodes,
        #     num_neg_samples=pos_edge_index.size(1),
        # )
        pos_edge_index = train_data.pos_edge_label_index
        neg_edge_index = train_data.neg_edge_label_index
        recon_loss = reconstruction_loss(
            model.decoder, z, pos_edge_index, neg_edge_index
        )
        kl_loss = compute_kl_divergence(mu, log_std)
        loss = recon_loss + (1 / train_data.num_nodes) * kl_loss

        loss.backward()
        optimizer.step()

        # validation
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                z, mu, _ = model(val_data)

                # val_pos_edge_index = val_data.edge_index
                # val_neg_edge_index = negative_sampling(
                #     # negative sampling - where there is no edges between nodes.
                #     edge_index=val_pos_edge_index,
                #     num_nodes=val_data.num_nodes,
                #     num_neg_samples=val_pos_edge_index.size(1),
                # )

                # we can do this with split_labels=True
                pos_scores = model.decoder(mu, val_data.pos_edge_label_index)
                neg_scores = model.decoder(mu, val_data.neg_edge_label_index)

                scores = torch.cat([pos_scores, neg_scores], dim=0)
                labels = torch.cat(
                    [torch.ones(pos_scores.size(0)), torch.zeros(pos_scores.size(0))]
                )

                print(
                    f"epoch {epoch}/{epochs}, roc_auc: ",
                    roc_auc_score(labels.cpu(), scores.cpu()),
                    "average precision:",
                    average_precision_score(labels.cpu(), scores.cpu()),
                )

    # save net
    torch.save(model, "link_prediction/checkpoints/vgae_pubmed.pt")

    # eval: # (here eval=test).
    model.eval()
    z, mu, _log_std = model(test_data)

    pos_scores = model.decoder(mu, test_data.edge_index)
    # test_neg_edge_index = negative_sampling(
    #     # negative sampling - where there is no edges between nodes.
    #     edge_index=test_data.edge_index,
    #     num_nodes=test_data.num_nodes,
    #     num_neg_samples=test_data.edge_index.size(1),
    # )
    # neg_scores = model.decoder(mu, val_neg_edge_index)
    pos_scores = model.decoder(mu, test_data.pos_edge_label_index)
    neg_scores = model.decoder(mu, test_data.neg_edge_label_index)

    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat(
        [torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]
    )
    print(
        "eval precision:",
        average_precision_score(labels.detach().cpu(), scores.detach().cpu()),
    )

# precision:
# 110 epoch, 16 latent size, lr=0.01, correct KL regularization - .859~.862 - as in the tutorial, much better values here (better quality and more data) - training is slower due to more data though.
# 200 epoch, 16 latent size, lr=0.01, correct KL regularization - .905~.922 - more epoch on bigger dataset yields better values.
# 1 epoch, 16 latent size, lr=0.01, correct KL regularization - .851~.855 - 1 epoch and not bad. (3 tests)

# each was tested for 3 runs.
