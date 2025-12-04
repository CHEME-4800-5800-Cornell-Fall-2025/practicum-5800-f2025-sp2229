#!/usr/bin/env julia
# Types.jl
# Definitions for Hopfield model types used by the practicum.
# This file defines an abstract model type and a concrete mutable
# struct `MyClassicalHopfieldNetworkModel` that stores the weight
# matrix `W`, the bias vector `b`, and a dictionary of per-memory
# energies.

# Abstract type to allow future model variants to share an interface.
abstract type AbstractlHopfieldNetworkModel end

"""
    MyClassicalHopfieldNetworkModel <: AbstractlHopfieldNetworkModel

A mutable struct representing a classical Hopfield network model.

### Fields
- `W::Array{<:Number, 2}`: weight matrix (N x N).
- `b::Array{<:Number, 1}`: bias vector (length N). Classical Hopfield uses zero bias.
- `energy::Dict{Int64, Float32}`: energy values for stored memories.
"""
mutable struct MyClassicalHopfieldNetworkModel <: AbstractlHopfieldNetworkModel

    # The synaptic weight matrix (square). Each entry w[i,j] is the
    # coupling between neuron i and neuron j. In a classical Hopfield
    # network this matrix is symmetric with zero diagonal.
    W::Array{<:Number, 2} # weight matrix

    # The bias (threshold) vector. For the classical Hebbian Hopfield
    # network in this practicum, `b` is initialized to all zeros.
    b::Array{<:Number, 1} # bias vector

    # A small lookup table storing the energy of each stored memory
    # (indexed by memory id). This is helpful for diagnostics and
    # plotting during retrieval experiments.
    energy::Dict{Int64, Float32} # energy of the states

    # Empty/default constructor - creates an instance with uninitialized
    # fields; the `build(...)` factory fills these fields before use.
    MyClassicalHopfieldNetworkModel() = new();
end