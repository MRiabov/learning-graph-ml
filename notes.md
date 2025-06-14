# Theory
**Spatial** graph convolution operations - a type of graph convolutions where neigboring vertex features are aggregated to produce embeddings; are called so because spatial connectivity is used to retrieve neighborhoods in this process.

GraphSAGE has demonstrated that learned aggregators can outperform simpler aggregation functions (such as taking the mean of embeddings), and thus create more discriminative, powerful vertex embeddings.
"GraphSAGE works by computing embeddings based on the central vertex and an aggregation of its neighborhood. By including the central vertex, it ensures that vertices with near identical neighborhoods have different embeddings"
graphsage: h^k(i) = relu(W * concat(h^(k-1)(j), aggregate(h^(k-1)(j) * Vprev(j) from N(v)(i)))
(graphsage was since outperformed, but it is useful to explore the concept of learned aggregators.)

**Message Passing Neural Networks (MPNN)** compute *directional* messages with a message function that is dependant on the source vertex, the destination vertex, and the edge connecting them. Rather than aggregate the neighbor's features and concatenating them with the central vertex's features as in GraphSAGE, MPNNs sum the incoming messages and pass the result to a readout function alongside the central vertex's features. (here readout - operation that produces the final prediction.)

## GraphSAGE
GraphSAGE operates on a simple assumption: vertices with similar neighborhoods should have similar embeddings. In this way, when calculating a vertex's embeddings, GraphSAGE considers its neighborhood's embeddings.
The function which produces the embeddings from the neighborhoods is learned rather than the embeddings being learned directly.

*Transductive* methods - embeddings of the graph are being learned directly
*Inductive* methods - generate general rules which can be applied to unseen vertices, rather than reasoning from specific training cases to specific test cases.  

GraphSAGE is an *inductive* method.

GraphSAGE loss function is unsupervised, and uses two distinct terms to ensure that neighboring vertices have similar embeddings and distant or disconnected vertices have embeddings which are numerically far apart.
This ensures that the calculated vertex embeddings are highly discriminative.

(experimentally) LSTM aggregation in GraphSAGE is the most powerful aggregator, however it introduces high complexity during training - >30 times increase in training time.

# Spectral approaches
Spectral approaches are expensive and generally not SOTA. They are also a specific case of other network, and classifying them as spectral/spatial is limiting.
Spectral approaches are grounded in signal processing theory.
GCN and GAT and GraphSAGE are spatial approaches.

# Graph Autoencoders
(As usual,) autoencoders encode into a latent space and then decode this compressed representation to reconstruct the original input data. They are trained to minimize the *reconstruction loss*

Difference between AEs to GAEs is that encoders and decoders take in and put out graph structures respectively. One of the most common methods is to replace the encoder with a CGNN and replace the decoder with a method that can reconstruct the graph structure of the input.


Once trained, GAEs (like AEs) can be split into their component networks to perform specific downstream tasks. A popular usecase for the encoder is to generating robust embeddings for supervised downstream tasks (e.g. classification, visualization, regression, clustering), and a use for the decoder is to generate new graph instances that include properties from the original dataset. This allows the generation of large synthetic datasets

## VGAE
Rather than representing inputs as single points in the latent space, variational autoencoders learn to encode inputs as probability distributions in the latent space, and they sample a distribution from them rather than getting them directly.

Unlike in GAEs - where the loss is the mean squared error between the input and the reconstructed output, a VGAE's loss imposes an additional penalty which ensures that the latent distributions are normalised. More specifically, this term regularizes the latent space distributions by ensuring that they do not diverge significantly from some prior distributions with desirable properties. E.g., we use the normal distribution `N(0,1)`. This divergence is quantified in our case using **Kullback-Leiber** (denoted as "KL") divergence is used as the normal distribution, though other similarity metrics (e.g. Wasserstein space distance or ranking loss) can be used successfully. Without this loss penalty, the VGAE encoder might generate distributions with small variances or high magnitude means: both of which would make it harder to sample from the distribution effectively.

(next: implement VGAE.. how to generate graph structures?)

(##### note: there was very interesting work on generation of 3d models that used sparsity in reconstruction... how? they actually used diffusion.)

# GAdvT (Graph Adversarial Networks)

The robustness of graph neural networks can be improved with Graph Adversarial Techniques. Here, the AI model acts as an adversary to another during training to mutually improve the performance of both models in tandem. (Generative Adversarial Network adopted to graphs description follows). 

As with traditional adversarial techniques, common goals for GAdvTs include:
* Improving the robustness, regularisation, or distribution of learned embeddings.
* Improving the robustness of models to targeted attacks.
* Training generative AI models.

(Note: GATs can be split to be used as both the discriminator and the generator, which is useful for both.)

# GraphVAE

Encoder:
GCN with gated pooling - a paradigm specifically for graphs. 
Gated pooling operates over all nodes simultaneously, multiplying a single learnable parameter over all of them and summing all features.

Flipping a single pixel in MNIST generation is of no issue, however adding a single atom or bond makes the prediction invalid, thus making the prediction more challenging.

GraphVAE uses dense generation for all atoms, and connections, so the decoder uses:
1. 3 shared MLPs,
2. Followed by:
    - A dense net for link prediction (num_atoms * num_atoms)
    - A dense net for edge feature prediction (num_atoms * num_atoms * num_edge_features)
    - A dense net for edge feature prediction (num_atoms * num_atom_features)

Not convolutional or recurrent layers (Which would make generation simpler.)

GraphVAE discards hydrogen atoms on QM9 to make prediction for the model easier.

GraphVAE reconstructs only the upper triangle of the graph since it is enough for generation, and mirrors (?) the bottom half.
On the other hand, we can use a "trick" for undirected edges to sum up the outputs of a decoder with itself, which results in a perfectly aligned output edge and edge feature prediction matrix. Normally, a triu mask would be used, but this is more stable because it reduces search space.

~~In link prediction:
 - **Accuracy** - of all labels, how many were actually accurate?
 - **Precision** - of all labels that were positive, how many were actually positive? true_positives/(true_positives/false_positives)~~

### graph matching
Because the ground truth is assumed to have non-unique node ordering, we first must align the generated graph with the ground truth for supervised reconstruction loss. GraphVAE uses Max Pooling Matching (MPM) which is used to align predicted and ground truth graphs before computing the reconstruction loss.
