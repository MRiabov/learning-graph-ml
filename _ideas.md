# encoding of FEA data

Encoding of FEA mesh data can be done via representing the tetras as graph nodes, faces as edges... masses are encoded as node features (via calculating average of mesh node mass). geometry position can be a parameter mapped as implicit, since implicits have proven effective as encoding of geometry structure (or have they? in this graph at least..). Geometry will have to follow explicit rules of delaunay meshes.
(note: just ideas. there are probably much better ways to do it, and more to it.)

assemblies with constraints, as usual, can be encoded as graph nodes with edges.

create a 3d tensor representing embeddings (bins), create embeddings between embeddings (2d faces between bins), generate inside embeddings, generate connections between bins... Make a loss based on divergence between bins, and divergence between face between bins (the assumption behind the face modelling is that it will stabilize generation between bins.)
Predict a graph in each.

Ideally, use sparsity. There is no need to 

%% maybe also model the bins shifted by a half-pos in each direction, generate, and penalize using the graph divergence between them to make the training definitely stable. Faces are then unnecessary.

%% did you even need modelling of 3d meshes for problemologist? if yes, why? -> to encode stress information after the fact of modelling, that is, to show the sim for the data.
%%> however you don't need the model-based methods because were proven more effective than model-based? so why?
%%> well, it is a cool contribution to the science.
%%>oh yes! I need to encode the starting position for the model. How could I forget... To be clear though, there are plenty of ways to encode such 3d information.

Just an idea: since each tetra face can connect to only one face, we can make an attention mask on those tetras have all connections already. Of course, we can also filter by distance. Of course, we can also filter by angles and intersecting geometry.

Though sparse voxel encoding is not bad at all either...

Hmm. If initial shape is known, a better way to reconstruct it would be shape -> distribute equally spaced points (with more near the edges), and make a diffusion on their position. Then link prediction, and then property of tetras.
And *then* update the shape with predicted stresses/strains. But that's reconstruction, not encoding.



# electronics encoding:
There is no dataset or much less model for encoding and reconstructing electronics. It is a great opening for a paper.
1. Collect dataset from public PCB repos
2. Data augmentation for custom generation (code-cads for pcbs)
3. Use a generative model to generate more pcbs.    

Electronics is a directed heterogenous graph. So it can and should be doable with graphs. 
The big advantage of doing this is that it will allow e.g. future work on predicting autocompletion for PCBs. Better yet, it will open up better research on Repairs (and of course, Problemologist.)

%% Note: SPICE netlist is exactly a graph. 

-> Oh yes, exactly this was done by (Circuit design completion using graph neural networks, 2023). Authors also have create a SPICE dataset available at https://github.com/symbench/spice-datasets/tree/master (1k SPICE netlists. There is also a bigger Open Circuit Benchmark that collects 50k circuits https://github.com/zehao-dong/CktGNN.

Electronics seems to be a "Directed Acyclic Graph". There are DAGNN networks and D-VAE which reconstruct them. A new Flow-attention Graph NN Claims to improve over both.

Overall, there is work on prediction, electronics graph representation in multi-stage (Versatile Multi-stage Graph Neural Network for Circuit Representation, 2022), which records information about netlist- and geometric stages; there is also research on routing optimization. Most electronics encoding is done via D-VAE - a Directed Acyclical Graph neural where data flows only in one direction (case for electronics). 

There is work on circuit completion (link prediction and ), and this means generation. There is similarly work on reconstructing the necessary graph using D-VAE, and this means generation can be done (though not autoregressively.)

## nets: 
1. DAGNN - Directed Acyclic Graph neural network
2. D-VAE - D-VAE

## datasets:
OpenCircuitBenchmark - https://github.com/zehao-dong/CktGNN, though only 1k samples.


# NOTE!!!!!!! 
Before starting on other projects, finish your current one first (Teaching Robots to Repair.)