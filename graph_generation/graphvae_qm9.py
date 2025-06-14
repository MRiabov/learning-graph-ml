from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.distributions import Normal, kl_divergence
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import Planetoid, QM9, qm9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.transforms.normalize_features import NormalizeFeatures
from torch_geometric.utils import (
    add_self_loops,
    degree,
    negative_sampling,
    subgraph,
    to_dense_adj,
    to_dense_batch,
)
from torch_scatter import scatter

from graph_gen_utils import build_affinity_matrix, mpm_batch, mpm_to_perm_batch


class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="sum")

        self.lin1 = nn.Linear(in_channels, out_channels)  # 1

    def forward(self, x, edge_index):
        # Add self-loops to adjacency matrix
        edge_index, _ = add_self_loops(edge_index=edge_index)

        # apply linear transformation
        x = self.lin1(x)  # add dim to node class

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
        self.conv1 = GCNLayer(num_features, 32)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.conv2 = GCNLayer(32, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.lin_1 = nn.Linear(64, 128)  # as in paper...

    def forward(self, x, edge_index):
        # block A
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # x = x + h  # residual connection (as in GraphVAE paper)
        # block 2
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # h = x + h  # residual connection (as in GraphVAE paper)

        # FIXME: res con won't work! they are of different shape!
        # hmm. I don't know if the original paper did it because there is no implementation!

        x = self.lin_1(x)
        x = F.relu(x)
        # out_mean = self.out_mean(x)
        # out_stddev = self.out_stddev(x)

        return x


class GraphEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim):
        super(GraphEncoder, self).__init__()
        self.gcn = GCN(
            num_features=num_features, hidden_dim=hidden_dim, out_size=hidden_dim
        )
        self.gated_pool_lin = nn.Linear(
            hidden_dim, 1
        )  # attention-like mechanism with single query.
        self.out_mean = nn.Linear(hidden_dim, latent_dim)  # worked over params
        self.out_std = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch_idx):
        x = self.gcn(x, edge_index)  # all node features?

        # multiply each node by gated pool lin
        attn = self.gated_pool_lin(x)
        x = x * attn
        x = scatter(
            x, batch_idx, dim=0, reduce="sum"
        )  # NOTE: changed from -1 to 0... because that's how it should work?

        # note: batchnorm here?
        mean = self.out_mean(x)
        log_std = self.out_std(x)

        return mean, log_std


class Decoder(nn.Module):
    def __init__(self, max_nodes=9, node_classes=4, edge_classes=4):
        super(Decoder, self).__init__()
        # nodes decoders:
        self.lin1_nodes = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.lin2_nodes = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.lin3_nodes = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        # nodes decoder
        self.adj_logits_lin = nn.Linear(512, max_nodes * max_nodes)
        self.node_features_lin = nn.Linear(512, max_nodes * node_classes)
        self.edge_features_lin = nn.Linear(512, max_nodes * max_nodes * edge_classes)

        self.max_nodes = max_nodes
        self.node_classes = node_classes
        self.edge_classes = edge_classes

    def forward(self, z):
        # shared mlp:
        x = self.lin1_nodes(z)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.lin2_nodes(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.lin3_nodes(x)
        x = self.bn3(x)
        x = F.relu(x)

        adj_logits = self.adj_logits_lin(x).view(-1, self.max_nodes, self.max_nodes)
        node_logits = self.node_features_lin(x).view(
            -1, self.max_nodes, self.node_classes
        )
        edge_logits = self.edge_features_lin(x).view(
            -1, self.max_nodes, self.max_nodes, self.edge_classes
        )

        # enforce symmetry in the decoder:
        # sum up the transpose of the decoder. It makes the inductive bias (?) valid and reduces the search space of decoder.
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        # node feature transposing is unnecessary.

        return adj_logits, node_logits, edge_logits  # node_features, edge_features

        # NEXT: finish implementing graphVAE.


class GraphVAE(nn.Module):
    def __init__(self, encoder: GraphEncoder, decoder: Decoder):
        super(GraphVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # note: edge detection threshold is unnecessary - it is only useful during inference time.
        # otherwise edge detection is optimized from e.g. sigmoid output - directly.

    def forward(self, batch: Data):
        loc, scale = self.encoder(
            batch.z.float().unsqueeze(-1), batch.edge_index, batch.batch
        )
        # ^ the "batch.z" above is atom ID. The "z" above and below are different "z".
        dist = Normal(
            loc, torch.exp(scale)
        )  # scale should be exponential for positivity
        # note:
        z = dist.rsample()  # latent `z`
        adj_probs, node_features, edge_features = self.decoder(z)
        return adj_probs, node_features, edge_features, loc, scale


def reconstruction_loss(adj_probs, node_logits, edge_logits, batch_y: Batch):
    B, N, node_C = node_logits.shape
    edge_classes_y = to_dense_adj(
        batch_y.edge_index,
        batch.batch,
        max_num_nodes=9,
        batch_size=batch_y.num_graphs,
        edge_attr=batch_y.edge_attr,
    )
    edge_classes_y = edge_classes_y.argmax(dim=-1)
    edge_mask = edge_classes_y.bool().float()

    node_classes_y, _node_mask = to_dense_batch(
        batch.z, batch.batch, max_num_nodes=9, batch_size=batch.num_graphs
    )

    # --- Graph matching ---
    K = build_affinity_matrix(node_logits, node_classes_y)  # (B, N*N)
    x = mpm_batch(K)
    perm = mpm_to_perm_batch(x, N)

    # --- apply perm matrix ---
    perm_t = perm.transpose(1, 2)
    adj_probs_perm = torch.bmm(torch.bmm(perm_t, adj_probs), perm)
    node_logits_perm = torch.bmm(perm.transpose(1, 2), node_logits)  # (B, N, C)
    edge_logits_perm = torch.matmul(
        torch.matmul(perm_t.unsqueeze(-1), edge_logits), perm.unsqueeze(1)
    )
    # ^ tldr: this applies permutation matrix. It could also be done with einsum, but this is also OK.

    adj_loss = F.binary_cross_entropy_with_logits(adj_probs_perm, edge_mask)
    # node_logits = node_logits_perm.permute(0, 2, 1)  # put the class logits onto 2nd dim.
    # `F.cross_entropy` demands a 2 or 3d array. so reshape both to 2d (although not necessary in node feat). Could use the above line^
    node_feat_loss = F.cross_entropy(
        node_logits_perm.reshape(-1, node_logits_perm.shape[-1]),
        node_classes_y.reshape(-1),
    )
    # `F.cross_entropy` demands a 2 or 3d array. so reshape it to 2d and back up
    edge_feat_loss = F.cross_entropy(
        edge_logits_perm.reshape(-1, edge_logits_perm.shape[-1]),
        edge_classes_y.reshape(-1),
    )
    recon_loss = adj_loss + node_feat_loss + edge_feat_loss
    return recon_loss


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


# filtering criteria
MAX_HEAVY_ATOMS = 9
MAX_DISTINCT_ATOMIC_NUMS = 4
MAX_BOND_TYPES = 4  # depends on encoding in edge_attr


def qm9_pre_filter(data: Data) -> bool:
    heavy_atoms = (data.z != 1).sum().item()
    num_distinct_atoms = len(torch.unique(data.z))
    bond_types = torch.unique(data.edge_attr, dim=0)
    return (
        heavy_atoms <= 9
        and num_distinct_atoms <= MAX_DISTINCT_ATOMIC_NUMS
        and len(bond_types) <= MAX_BOND_TYPES
    )


z_to_class = {0: 0, 6: 1, 7: 2, 8: 3, 9: 4}  # 0-edge, C, N, O, F
z_to_class_map = torch.full((max(z_to_class) + 1,), -1, dtype=torch.long)
for z, cls_ in z_to_class.items():
    # using data.z as a long slice lookup, get values from map...
    z_to_class_map[z] = cls_


def qm9_pre_transform(data: Data) -> Data:
    # filter out all hydrogen values:
    hydrogen_mask = data.z != 1
    # data.x = data.x[hydrogen_mask]
    # data.z = data.z[hydrogen_mask]
    # filter out using subgraph instead.
    non_h_indices = hydrogen_mask.nonzero(as_tuple=False).view(-1)
    data.edge_index, data.edge_attr = subgraph(
        non_h_indices, data.edge_index, data.edge_attr, relabel_nodes=True
    )
    data.z = data.z[non_h_indices]
    data.x = data.x[non_h_indices]

    # map all atoms to indices.
    data.z = z_to_class_map[data.z]
    return data


if __name__ == "__main__":
    # note: the goal of the dataset is to predict the label which the graph has.
    batch_dataset = QM9(
        "datasets/QM9/",
        pre_filter=qm9_pre_filter,
        pre_transform=qm9_pre_transform,  # , force_reload=True
    )  # transform=NormalizeFeatures()
    assert len(batch_dataset) > 100_000, (
        f"Batch dataset is expected to be ~134k, now {batch_dataset}"
    )
    train_dataset, val_dataset, test_dataset = random_split(
        batch_dataset, [0.8, 0.1, 0.1]
    )  # note: it's a huge dataset (130k graphs). Maybe reduce it to 50k graphs total.
    epochs = 20
    batch_size = 512  # because graphs are tiny.
    beta = 5
    latent_size = 128

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(
        torch.unique(batch_dataset.z)[1:]
    )  # exclude hydrogen, which is on [0].

    # init model:
    encoder = GraphEncoder(  # num_features=1 - a label of node class (atom type).
        num_features=1, hidden_dim=latent_size, latent_dim=latent_size
    )
    decoder = Decoder(
        max_nodes=MAX_HEAVY_ATOMS,
        node_classes=MAX_DISTINCT_ATOMIC_NUMS,
        edge_classes=MAX_BOND_TYPES,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphVAE(encoder, decoder)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # I'll experiment.

    for epoch in range(epochs):
        model.train()

        for batch in train_dataloader:
            optimizer.zero_grad()

            adj_probs, node_features, edge_features, loc, scale = model(batch)

            recon_loss = reconstruction_loss(
                adj_probs, node_features, edge_features, batch
            )
            kl_loss = compute_kl_divergence(loc, scale)
            loss = recon_loss + (1 / batch.num_nodes) * kl_loss

            loss.backward()
            optimizer.step()

        # validation
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                # note: get your own code from chatgpt and write the eval yourself.  (for learning!)
                # this is a stub for momentary speed.
                # ground truth
                val_batch = next(iter(val_dataloader))

                # FIXME: does not implement graph matching.

                # NEXT: learn components of GraphVAE. There is still a lot to learn.

                # # Convert to dense adjacency and node feature tensors
                # edge_classes_y = to_dense_adj(
                #     val_batch.edge_index,
                #     val_batch.batch,
                #     max_num_nodes=9,
                #     batch_size=val_batch.num_graphs,
                #     edge_attr=val_batch.edge_attr,
                # ).argmax(dim=-1)  # (B, N, N)
                # edge_mask = edge_classes_y != 0  # Assume class 0 means "no edge"

                # # FIXME: I don't include 0 edge class!
                # dense_node_classes_y, node_mask = to_dense_batch(
                #     val_batch.z,  # ground-truth node classes
                #     val_batch.batch,
                #     max_num_nodes=9,
                #     batch_size=val_batch.num_graphs,
                # )

                # # Forward pass
                # adj_logits, node_logits, edge_logits, loc, scale = model(val_batch)
                # pred_edge_presence = torch.sigmoid(adj_logits) > 0.5
                # pred_edge_classes = edge_logits.argmax(dim=-1)
                # pred_node_classes = node_logits.argmax(dim=-1)

                # # Node classification accuracy
                # node_accuracy = (
                #     (pred_node_classes == dense_node_classes_y)[node_mask]
                #     .float()
                #     .mean()
                # )

                # # Edge classification accuracy (only for present edges)
                # edge_accuracy = (
                #     (pred_edge_classes == edge_classes_y)[edge_mask].float().mean()
                # )

                # # Link prediction metrics
                # link_labels = edge_mask.float().view(-1)  # binary: edge present or not
                # link_scores = torch.sigmoid(adj_logits).view(-1)

                # roc_auc = roc_auc_score(link_labels.cpu(), link_scores.cpu())
                # ap_score = average_precision_score(link_labels.cpu(), link_scores.cpu())

                # print(
                #     f"Epoch {epoch}/{epochs} | "
                #     f"Link ROC-AUC: {roc_auc:.4f} | "
                #     f"Link AP: {ap_score:.4f} | "
                #     f"Node Acc: {node_accuracy:.4f} | "  # node classes are predicted well?
                #     f"Edge Acc: {edge_accuracy:.4f}"  # however edge features are not at all.
                # )

    # save net
    torch.save(model, "graph_generation/checkpoints/graphvae_qm9.pt")

    # eval: # (here eval=test).
    model.eval()
    z, mu, _log_std = model(test_dataset)

    pos_scores = model.decoder(mu, test_dataset.edge_index)
    # test_neg_edge_index = negative_sampling(
    #     # negative sampling - where there is no edges between nodes.
    #     edge_index=test_data.edge_index,
    #     num_nodes=test_data.num_nodes,
    #     num_neg_samples=test_data.edge_index.size(1),
    # )
    # neg_scores = model.decoder(mu, val_neg_edge_index)
    pos_scores = model.decoder(mu, test_dataset.pos_edge_label_index)
    neg_scores = model.decoder(mu, test_dataset.neg_edge_label_index)

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
