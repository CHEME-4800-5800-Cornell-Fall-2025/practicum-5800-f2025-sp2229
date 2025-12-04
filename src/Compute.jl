
# Utility functions for running the Hopfield network simulation:
# - `_energy(...)` computes the scalar energy of a network state
#   according to the classical Hopfield energy function.
# - `⊗(...)` computes an outer product (used when building weights).
# - `recover(...)` runs the asynchronous update rule until convergence
#   (or max iterations) and records visited states and energies.

"""
    _energy(s::Array{<:Number,1}, W::Array{<:Number,2}, b::Array{<:Number,1}) -> Float32

Compute the Hopfield network energy for state `s` with weights `W`
and bias `b` using the standard energy function:

    E(s) = -1/2 * sum_{ij} W[i,j] s[i] s[j] - sum_i b[i] s[i]

The function returns a Float32 energy value.
"""
function _energy(s::Array{<: Number,1}, W::Array{<:Number,2}, b::Array{<:Number,1})::Float32
    # Accumulate the pairwise term sum_{ij} W[i,j] s[i] s[j]
    tmp_energy_state = 0.0;
    number_of_states = length(s);

    # bias contribution: b^T * s (scalar)
    tmp = transpose(b)*s;

    # double loop over pairs to compute the quadratic term
    for i ∈ 1:number_of_states
        for j ∈ 1:number_of_states
            tmp_energy_state += W[i,j]*s[i]*s[j];
        end
    end

    # Combine terms: note the -1/2 factor on the pairwise sum
    energy_state = -(1/2)*tmp_energy_state + tmp;

    # return energy (Float32)
    return energy_state;
end

"""
    ⊗(a::Array{Float64,1},b::Array{Float64,1}) -> Array{Float64,2}

Compute the outer product of two vectors `a` and `b` and returns a matrix.

### Arguments
- `a::Array{Float64,1}`: a vector of length `m`.
- `b::Array{Float64,1}`: a vector of length `n`.

### Returns
- `Y::Array{Float64,2}`: a matrix of size `m x n` such that `Y[i,j] = a[i]*b[j]`.
"""
function ⊗(a::Array{T,1}, b::Array{T,1})::Array{T,2} where T <: Number
    # Outer product helper: returns matrix Y where Y[i,j] = a[i] * b[j].
    # This is used when building the Hebbian weight matrix as s * s^T.

    # dimensions
    m = length(a)
    n = length(b)
    Y = zeros(m,n)

    # compute the outer product explicitly (keeps types simple and clear)
    for i ∈ 1:m
        for j ∈ 1:n
            Y[i,j] = a[i]*b[j]
        end
    end

    return Y
end


"""
    recover(model::MyClassicalHopfieldNetworkModel, sₒ::Array{Int32,1};
        maxiterations::Int = 1000, patience::Union{Int,Nothing} = nothing,
        miniterations_before_convergence::Union{Int,Nothing} = nothing) -> Tuple{Dict{Int64, Array{Int32,1}}, Dict{Int64, Float32}}

Run asynchronous Hopfield updates starting from `sₒ` until convergence (or `maxiterations`) and
collect the visited states and their energies.

### Arguments
- `model::MyClassicalHopfieldNetworkModel`: a Hopfield network model.
- `sₒ::Array{Int32,1}`: initial state (±1 spins encoded as `Int32`).
- `maxiterations::Int`: maximum number of updates.
- `patience::Union{Int,Nothing}`: number of consecutive identical states required to declare convergence. If `nothing`, defaults to `max(5, round(Int, 0.01*N))` where `N` is number of pixels.
- `miniterations_before_convergence::Union{Int,Nothing}`: minimum updates to run before checking convergence. If `nothing`, defaults to `patience`.

### Returns
Tuple of dictionaries:
- `frames::Dict{Int64, Array{Int32,1}}`: state at each iteration (starting at key 0).
- `energydictionary::Dict{Int64, Float32}`: energy at each iteration (starting at key 0).
"""
function recover(model::MyClassicalHopfieldNetworkModel, sₒ::Array{Int32,1}, trueenergyvalue::Float32;
    maxiterations::Int = 1000, patience::Union{Int,Nothing} = nothing,
    miniterations_before_convergence::Union{Int,Nothing} = nothing)::Tuple{Dict{Int64, Array{Int32,1}}, Dict{Int64, Float32}}

    # initialize - extract network parameters and prepare bookkeeping
    W = model.W; # weights matrix (N x N)
    b = model.b; # bias vector (length N)
    number_of_pixels = length(sₒ); # number of neurons/pixels

    # Determine patience: how many identical consecutive states required
    # before we declare convergence. Default scales with problem size.
    patience_val = isnothing(patience) ? max(5, Int(round(0.1 * number_of_pixels))) : patience;

    # Minimum iterations to perform before checking for convergence
    min_iterations = max(isnothing(miniterations_before_convergence) ? patience_val : miniterations_before_convergence, patience_val);

    # Circular buffer to hold the last `patience_val` states for stability checks
    S = CircularBuffer{Array{Int32,1}}(patience_val);

    # Storage for frames (states) and their energies recorded during the run
    frames = Dict{Int64, Array{Int32,1}}();
    energydictionary = Dict{Int64, Float32}();
    has_converged = false; # convergence flag

    # Record initial state and its energy
    frames[0] = copy(sₒ);
    energydictionary[0] = _energy(sₒ,W, b);
    s = copy(sₒ); # working copy of the state vector
    iteration_counter = 1;
    while (has_converged == false)
        # classical hopfield models have guaranteed convergence 
        # Asynchronous update: pick a random neuron j and update its spin
        j = rand(1:number_of_pixels);
        w = W[j,:]; # weights connecting neuron j to all others
        h = dot(w,s) - b[j]; # local field at neuron j

        # Apply sign activation: if local field is zero, break the tie randomly
        if h == 0
            s[j] = rand() < 0.5 ? Int32(-1) : Int32(1);
        else
            s[j] = h > 0 ? Int32(1) : Int32(-1);
        end

        # Record energy after the update and snapshot the state
        energydictionary[iteration_counter] = _energy(s, W, b);
        state_snapshot = copy(s);
        frames[iteration_counter] = state_snapshot;
        
        # check for convergence -
        # Push the new state into the circular buffer and check stability
        push!(S, state_snapshot);
        if (length(S) == patience_val) && (iteration_counter >= min_iterations)
            # If all stored states in the buffer are identical (Hamming distance 0)
            # we consider the network to have converged to a stable attractor.
            all_equal = true;
            first_state = S[1];
            for state ∈ S
                if (hamming(first_state, state) != 0)
                    all_equal = false;
                    break;
                end
            end
            if (all_equal == true)
                has_converged = true;
            end
        end
        
        # If the current energy is less than or equal to the true (target) energy,
        # we can stop early: the network has reached a configuration as good as
        # the stored memory's energy (or better).
        current_energy = energydictionary[iteration_counter];
        if (current_energy ≤ trueenergyvalue)
            has_converged = true;
            @info "Energy value lower than true. Stopping"
        end

        # update counter, and check max iterations -
        iteration_counter += 1;
        if (iteration_counter > maxiterations && has_converged == false)
            has_converged = true; # we have reached the maximum number of iterations
            @warn "Maximum iterations reached without convergence."
        end

        
    end
            
    # return 
    frames, energydictionary
end


"""
    decode(simulationstate::Array{T,1}; number_of_rows::Int64 = 28, number_of_cols::Int64 = 28) -> Array{T,2}

Reshape a flattened Hopfield state vector into an image matrix, mapping spins to pixel intensities.

- `simulationstate`: length `number_of_rows * number_of_cols` vector containing ±1 spin values.
- `number_of_rows`: output image height; defaults to 28 for MNIST-style digits.
- `number_of_cols`: output image width; defaults to 28 for MNIST-style digits.

Returns a `number_of_rows x number_of_cols` `Int32` array where `-1` becomes `0` and any other value becomes `1`. A `BoundsError` will be thrown if the provided vector is shorter than the requested shape.
"""
function decode(simulationstate::Array{T,1}; 
    number_of_rows::Int64 = 28, number_of_cols::Int64 = 28)::Array{T,2} where T <: Number
    
    # initialize -
    reconstructed_image = Array{Int32,2}(undef, number_of_rows, number_of_cols);
    linearindex = 1;
    for row ∈ 1:number_of_rows
        for col ∈ 1:number_of_cols
            s = simulationstate[linearindex];
            if (s == -1)
                reconstructed_image[row,col] = 0;
            else
                reconstructed_image[row,col] = 1;
            end
            linearindex+=1;
        end
    end
    
    # return 
    return reconstructed_image
end
