# encoding of FEA data

Encoding of FEA mesh data can be done via representing the tetras as graph nodes, faces as edges... masses are encoded as node features (via calculating average of mesh node mass). geometry position can be a parameter mapped as implicit, since implicits have proven effective as encoding of geometry structure (or have they? in this graph at least..). Geometry will have to follow explicit rules of delaunay meshes.
(note: just ideas. there are probably much better ways to do it, and more to it.)

assemblies with constraints, as usual, can be encoded as graph nodes with edges.

# electronics encoding:
There is no dataset or much less model for encoding and reconstructing electronics. It is a great opening for a paper.
1. Collect dataset from public PCB repos
2. Data augmentation for custom generation (code-cads for pcbs)
3. Use a generative model to generate more pcbs.    

Electronics is a directed heterogenous graph. So it can and should be doable with graphs. 
The big advantage of doing this is that it will allow e.g. future work on predicting autocompletion for PCBs. Better yet, it will open up better research on Repairs (and of course, Problemologist.)

%% Note: SPICE netlist is exactly a graph. 