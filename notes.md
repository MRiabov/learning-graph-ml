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


