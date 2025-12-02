#throw(ErrorException("Oppps! No methods defined in src/Compute.jl. What should you do here?"))
"""
    _energy(W::Array{T,2}, α::Array{T,1}, s::Array{T,1}) -> T where T <: Number
"""
function _energy(s::Array{<: Number,1}, W::Array{<:Number,2}, b::Array{<:Number,1})::Float32
    
    # initialize -
    tmp_energy_state = 0.0;
    number_of_states = length(s);

    # main loop -
    tmp = transpose(b)*s; # alias for the bias term
    for i ∈ 1:number_of_states
        for j ∈ 1:number_of_states
            tmp_energy_state += W[i,j]*s[i]*s[j];
        end
    end
    energy_state = -(1/2)*tmp_energy_state + tmp;

    # return -
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

    # initialize -
    m = length(a)
    n = length(b)
    Y = zeros(m,n)

    # main loop 
    for i ∈ 1:m
        for j ∈ 1:n
            Y[i,j] = a[i]*b[j]
        end
    end

    # return 
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

    # initialize -
    W = model.W; # get the weights
    b = model.b; # get the biases
    number_of_pixels = length(sₒ); # number of pixels
    patience_val = isnothing(patience) ? max(5, Int(round(0.1 * number_of_pixels))) : patience; # scale patience with problem size
    min_iterations = max(isnothing(miniterations_before_convergence) ? patience_val : miniterations_before_convergence, patience_val); # floor before declaring convergence
    S = CircularBuffer{Array{Int32,1}}(patience_val); # buffer to check for convergence
    
    # initialize -
    frames = Dict{Int64, Array{Int32,1}}(); # dictionary to hold frames
    energydictionary = Dict{Int64, Float32}(); # dictionary to hold energies
    has_converged = false; # convergence flag

    # setup -
    frames[0] = copy(sₒ); # copy the initial random state
    energydictionary[0] = _energy(sₒ,W, b); # initial energy
    s = copy(sₒ); # initial state
    iteration_counter = 1;
    while (has_converged == false)
        
        j = rand(1:number_of_pixels); # select a random pixel
        w = W[j,:]; # get the weights
        h = dot(w,s) - b[j]; # state at node j
        
        # Edge case: if h == 0, we have a tie, so we randomly assign ±1
        if h == 0
            s[j] = rand() < 0.5 ? Int32(-1) : Int32(1); # random tie-break to avoid bias
        else
            s[j] = h > 0 ? Int32(1) : Int32(-1); # map sign to ±1 spins
        end

        energydictionary[iteration_counter] = _energy(s, W, b);
        state_snapshot = copy(s); # single snapshot reused for storage and convergence checks
        frames[iteration_counter] = state_snapshot;
        
        # check for convergence -
        push!(S, state_snapshot); # push the current state to the buffer
        if (length(S) == patience_val) && (iteration_counter >= min_iterations)
            all_equal = true;
            first_state = S[1]; # look at the oldest state in the buffer
            for state ∈ S
                if (hamming(first_state, state) != 0)
                    all_equal = false;
                    break;
                end
            end
            if (all_equal == true)
                has_converged = true; # we have converged
            end
        end
        
        # is energy below the true value?
        current_energy = energydictionary[iteration_counter];
        if (current_energy ≤ trueenergyvalue)
            has_converged = true; # stop
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
