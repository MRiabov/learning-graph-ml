# Learning list:

(shorten for brevity)

## basics:
## classification
[x] - RGNN
[x] - GCN
[x] - GAT
[x] - GraphSAGE & variants 
[?] - GATv2 - an improvement over GAT by operation restructuring (though uses 2x memory.)
*note*: classification is both graph-level and node classification.
[ ] - explore more and better for classification (SOTA.)

### DAGs
[ ] - D-vae - a variational autoencoder for directed acyclic graphs (e.g. electronics.) - used in a NeurlIPS publication 


## link prediction
[x] - VGAE for link prediction (rep: 1)
[?] - AE for link prediction
[ ] - explore more and better (sota.)

## Graph generation
[W] - GraphVAE
[ ] - GraphRNN - Generating Realistic Graphs with Deep Auto-regressive Model
[?] - GRAN - Graph Recurrent Attention Networks - Efficient Graph Generation with Graph Recurrent Attention Networks


## node feature reconstruction (with link predition?)

**At last, learn SOTA for classification, prediction and reconstruction.**


%% notation: 
[x] - done
[ ] - todo
[W] - WIP.
[?] - maybe.


# SOTA:
[ ] - An end-to-end attention-based approach for learning on graphs - although how are they not overly expensive?


%% todo: send to chatgpt: I have implemented a GVAE, and now I want to implement a D-VAE - directed acyclical graph autoencoder. What are the differences? Basically, encode the 