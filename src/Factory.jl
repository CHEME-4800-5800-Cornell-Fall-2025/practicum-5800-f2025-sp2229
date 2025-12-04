
# Simple factory helper for creating a classical Hopfield network model.
# This file provides a single public function `build(...)` that takes a
# named tuple containing memory vectors (each column is a memory) and
# builds a `MyClassicalHopfieldNetworkModel` instance.
#
# 1. Initialize weight matrix `W` and bias vector `b`.
# 2. For each memory vector, compute the outer product (Hebbian rule)
#    and add it to `W`.
# 3. Remove self-connections by zeroing the diagonal entries of `W`.
# 4. Apply Hebbian scaling by dividing by the number of memories.
# 5. Compute and store the energy of each stored memory (using `_energy`).
# 6. Populate and return the model instance.
# -- PUBLIC METHODS BELOW HERE ---------------------------------------------------------------------------------------- #
"""
    build(modeltype::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple) -> MyClassicalHopfieldNetworkModel

Factory method for building a Hopfield network model. 

### Arguments
- `modeltype::Type{MyClassicalHopfieldNetworkModel}`: the type of the model to be built.
- `data::NamedTuple`: a named tuple containing the data for the model.

The named tuple should contain the following fields:
- `memories`: a matrix of memories (each column is a memory).

### Returns
- `model::MyClassicalHopfieldNetworkModel`: the built Hopfield network model with the following fields populated:
    - `W`: the weight matrix.
    - `b`: the bias vector.
    - `energy`: a dictionary of energies for each memory.
"""
function build(modeltype::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple)::MyClassicalHopfieldNetworkModel

    # initialize -
    #user provides a set of memory vectors
    model = modeltype();
    linearimagecollection = data.memories; #load the memories into the linearimagecollection
    number_of_rows, number_of_cols = size(linearimagecollection);
    W = zeros(Float32, number_of_rows, number_of_rows); #set initial partial memory weight matrix to zero 
    b = zeros(Float32, number_of_rows); # zero bias for classical Hopfield

    # compute the weight matrix W -
    # For each memory vector (column j) compute the outer product
    # s * s^T and add it to W. This is the Hebbian outer-product rule.
    for j ∈ 1:number_of_cols # for each image memory
        Y = ⊗(linearimagecollection[:,j], linearimagecollection[:,j]); # compute the outer product -
        W += Y; # accumulate into the weight matrix
    end
    
    # no self-coupling and Hebbian scaling -
    # Remove self-connections: set diagonal to zero
    for i ∈ 1:number_of_rows
        W[i,i] = 0.0f0; # no self-coupling in a classical Hopfield network
    end
    # Scale by number of memories (average the outer products)
    WN = (1/number_of_cols)*W; # Hebbian scaling by number of memories stored
    
    # compute the energy dictionary -
    energy = Dict{Int64, Float32}();
    # Compute the energy for each stored memory using the network weights
    for i ∈ 1:number_of_cols
        energy[i] = _energy(linearimagecollection[:,i], WN, b);
    end

    # add data to the model -
    model.W = WN;
    model.b = b;
    model.energy = energy;

    # return -
    return model;
end
# --- PUBLIC METHODS ABOVE HERE --------------------------------------------------------------------------------------- #
