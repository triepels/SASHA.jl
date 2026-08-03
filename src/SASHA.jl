module SASHA

using Random: AbstractRNG, default_rng, SamplerTrivial

import Random: rand

export fit!, loss, sasha!, sasha, Space, space

struct Space{names, T}
    iters::T
end

"""
    Space{names}(iters::Tuple)

Constructs a `Space` with the given `names` (a tuple of Symbols) from a tuple of iterators.
"""
function Space{names}(iters::T) where {names, T<:Tuple}
    names isa Tuple || throw(ArgumentError("names must be a tuple"))
    all(name -> name isa Symbol, names) || throw(ArgumentError("names must be a tuple of Symbols"))
    length(names) == length(iters) || throw(ArgumentError("names and iterators must have matching lengths"))
    return Space{names, typeof(iters)}(iters)
end

Base.eltype(::Type{Space{names, T}}) where {names, T} = NamedTuple{names,
    Tuple{map(eltype, fieldtypes(T))...}}

Base.firstindex(s::Space) = 1
Base.keys(s::Space) = Base.OneTo(length(s))
Base.lastindex(s::Space) = length(s)
Base.length(s::Space) = length(s.iters) == 0 ? 0 : prod(length, s.iters)
Base.size(s::Space) = length(s.iters) == 0 ? (0,) : map(length, s.iters)

function Base.show(io::IO, s::Space{names}) where names
    print(io, "$(length(s))-element Space{$names}")
end

@inline function Base.getindex(s::Space{names}, i::Int) where names
    @boundscheck 1 ≤ i ≤ length(s) || throw(BoundsError(s, i))
    strides = (1, cumprod(map(length, Base.front(s.iters)))...)
    return NamedTuple{names}(map(getindex, s.iters, mod.((i - 1) .÷ strides, size(s)) .+ 1))
end

@inline function Base.getindex(s::Space{names}, I::Vararg{Int}) where names
    @boundscheck length(I) == length(s.iters) && all(1 .≤ I .≤ size(s)) || throw(BoundsError(s, I))
    return NamedTuple{names}(map(getindex, s.iters, I))
end

@inline function Base.getindex(s::Space, inds)
    return [s[i] for i in inds]
end

Base.@propagate_inbounds function Base.iterate(s::Space, state::Int=1)
    state > length(s) && return nothing
    return s[state], state + 1
end

@inline function rand(rng::AbstractRNG, s::SamplerTrivial{T}) where {names, T<:Space{names}}
    return NamedTuple{names}(map(var -> rand(rng, var), s[].iters))
end

"""
    space(; iters...)

Returns an iterator over the product of several named iterators. Each generated
element is a named tuple whose ith element is taken from the ith iterator. The first
iterator changes the fastest.

# Examples
```jldoctest
julia> collect(space(a=1:3, b=4:5))
6-element Vector{@NamedTuple{a::Int64, b::Int64}}:
(a = 1, b = 4)
(a = 2, b = 4)
(a = 3, b = 4)
(a = 1, b = 5)
(a = 2, b = 5)
(a = 3, b = 5)
```
"""
space(; iters...) = space(keys(iters), values(values(iters)))
space(names::NTuple{N, Symbol}, iters::NTuple{N, Any}) where N = Space{names, typeof(iters)}(iters)

_fit!(model::Any, data::Union{Tuple, NamedTuple}, kwargs::NamedTuple) = fit!(model, data...; kwargs...)
_fit!(model::Any, data::Any, kwargs::NamedTuple) = fit!(model, data; kwargs...)

"""
    fit!(model, data; kwargs)

Fits `model` on `data` based on any additional provided keyword arguments and returns
the fitted model. Must be implemented for any custom model type before using the SASHA
optimization functions.

See also [`sasha`](@ref), [`sasha!`](@ref).
"""
fit!(model::Any, data::Any; kwargs...) = throw(MethodError(fit!, (model, data)))

_loss(model::Any, data::Union{Tuple, NamedTuple}) = loss(model, data...)::Real
_loss(model::Any, data::Any) = loss(model, data)::Real

"""
    loss(model, data)

Evaluates the loss of `model` on `data`. Should return a concrete subtype of a Real. Must
be implemented for any custom model type before using the SASHA optimization functions.

See also [`sasha`](@ref), [`sasha!`](@ref).
"""
loss(model::Any, data::Any) = throw(MethodError(loss, (model, data)))

"""
    progress(state)

A ready-made callback function for [`sasha`](@ref) and [`sasha!`](@ref) to log the process
of the optimization at the end of each round.
"""
function progress(state)
    @info "SASHA round $(state.round)" arms=length(state.arms) best=state.best elapsed=state.elapsed
    return false
end

"""
    sasha!([rng=default_rng()], arms, train, val; kwargs)

Performs hyperparameter optimization using the SASHA optimizer on `arms` by modifying
the vector of arms in-place. Each arm is fitted on `train` by iteratively calling
`fit!`(`model`, `train`; `kwargs`...). The loss of each arm is evaluated on `val` by
calling `loss`(`model`, `val`). These functions must be implemented for the type of the
model to be optimized. If `train` or `val` are tuples, the arguments are splatted in the
respective function calls. Returns the model corresponding to the winning arm.

Multiple threads will be used (if available) to fit all remaining arms each round. The
number of available threads can be determined by calling `Threads.nthreads()`.

# Keyword Arguments
- `p::Real=0.8`: acceptance probability of the worst arm at the start of the
  optimization.
- `maximize::Bool=false`: whether the loss of the arms should be maximized.
- `fit_kwargs::NamedTuple=NamedTuple())`: optional keyword arguments that are passed on to
  `fit!` when fitting an arm.
- `callback::Function=(state) -> false`: callback function that is called at the end of
  each round. Can be used to monitor the optimization progress or terminate the process
  based on custom stopping criterion. Should return a Bool to indicate whether the process
  must terminate at the current round. See the state arguments below for more details on 
  the objects passed on to the callback function via `state`.

# State Arguments
- `round`: current round number.
- `arms`: all remaining arms. 
- `loss`: loss of all remaining arms.
- `best`: loss of the winning arm.
- `temp`: current temperature.
- `elapsed`: time elapsed since the start of the optimization in seconds.

# Notes
The annealing process differs from the one described in (Triepels, 2023) in that no initial
temperature has to be provided. Instead, the initial temperature is determined before the
first round such that the worst arm has an acceptance probability of `p`. In this way, the
initial temperature is better calibrated to the scale of the loss function.

# References
Triepels, R. (2023). SASHA: Hyperparameter Optimization by Simulated Annealing and
Successive Halving.

See also [`sasha`](@ref)
"""
function sasha!(rng::AbstractRNG, arms::Vector, train::Any, val::Any; p::Real=0.8,
        maximize::Bool=false, fit_kwargs::NamedTuple=NamedTuple(), callback::Function=(state) -> false)
    !isempty(arms) || throw(ArgumentError("no arms to optimize"))
    0 < p < 1 || throw(ArgumentError("p must be in (0,1)"))

    keep = Vector{Bool}(undef, length(arms))
    loss = map(arm -> _loss(arm, val), arms)
    diff = Vector{eltype(loss)}(undef, length(arms))

    if maximize
        best = maximum(loss)
        for i in eachindex(diff)
            diff[i] = loss[i] - best
        end
    else
        best = minimum(loss)
        for i in eachindex(diff)
            diff[i] = best - loss[i]
        end
    end

    temp = minimum(diff) / log(p)
    if iszero(temp)
        throw(ArgumentError("Cannot calibrate annealing schedule. The inital loss of all \
        arms are identical."))
    end

    round = 1
    t = time()
    while length(arms) > 1
        @sync for i in eachindex(arms)
            Threads.@spawn arms[i] = _fit!(arms[i], train, fit_kwargs)
        end
        
        for i in eachindex(arms)
            loss[i] = _loss(arms[i], val)
        end

        if all(isequal(first(loss)), loss)
            @warn "Unable to determine best arm. Terminating prematurely."
            keepat!(arms, 1)
            return first(arms)
        end

        if maximize
            best = maximum(loss)
            for i in eachindex(diff)
                diff[i] = loss[i] - best
            end
        else
            best = minimum(loss)
            for i in eachindex(diff)
                diff[i] = best - loss[i]
            end
        end

        for i in eachindex(keep)
            keep[i] = rand(rng) ≤ exp(round * diff[i] / temp)
        end

        keepat!(arms, keep)
        keepat!(loss, keep)
        keepat!(diff, keep)
        keepat!(keep, keep)

        state = (round=round, arms=arms, loss=loss, best=best, temp=temp, elapsed=time()-t)
        if callback(state)
            ind = maximize ? argmax(loss) : argmin(loss)
            keepat!(arms, ind)
            return first(arms)
        end

        round += 1
    end

    return first(arms)
end

sasha!(arms::Vector, train::Any, val::Any; kwargs...) = 
    sasha!(default_rng(), arms, train, val; kwargs...)

"""
    sasha([rng=default_rng()], T, space, train, val; kwargs)
    sasha([rng=default_rng()], f, space, train, val; kwargs)

Optimizes the hyperparameters of a model using the SASHA optimizer. Initially, an arm
is created for each configuration in `space` by calling the constructor of type `T`
with model parameters provided as named arguments. Alternatively, if `T` cannot have a
constructor with named arguments, function `f` can be provided to create each arm
instead. `f` should take a named tuple with model parameters as input and initialize a
new model based on the provided parameter configuration. Each arm is fitted on `train`
by iteratively calling `fit`!(`model`, `train`; `kwargs`...). The loss of each arm is
evaluated on `val` by calling `loss`(`model`, `val`). These functions must be implemented
for the type of the model to be optimized. If `train` or `val` are tuples, the arguments
are splatted in the respective function calls. Returns the model corresponding to the
winning arm.

Multiple threads will be used (if available) to fit all remaining arms each round. The
number of available threads can be determined by calling `Threads.nthreads()`.

# Keyword Arguments
- `p::Real=0.8`: acceptance probability of the worst arm at the start of the
  optimization.
- `maximize::Bool=false`: whether the loss of the arms should be maximized.
- `fit_kwargs::NamedTuple=NamedTuple())`: optional keyword arguments that are passed on to
  `fit!` when fitting an arm.
- `callback::Function=(state) -> false`: callback function that is called at the end of
  each round. Can be used to monitor the optimization progress or terminate the process
  based on custom stopping criterion. Should return a Bool to indicate whether the process
  must terminate at the current round. See the state arguments below for more details on 
  the objects passed on to the callback function via `state`.

# State Arguments
- `round`: current round number.
- `arms`: all remaining arms. 
- `loss`: loss of all remaining arms.
- `best`: loss of the winning arm.
- `temp`: current temperature.
- `elapsed`: time elapsed since the start of the optimization in seconds.

# Notes
The annealing process differs from the one described in (Triepels, 2023) in that no initial
temperature has to be provided. Instead, the initial temperature is determined before the
first round such that the worst arm has an acceptance probability of `p`. In this way, the
initial temperature is better calibrated to the scale of the loss function.

# References
Triepels, R. (2023). SASHA: Hyperparameter Optimization by Simulated Annealing and
Successive Halving.

See also [`sasha!`](@ref)
"""
function sasha(rng::AbstractRNG, T::Type, space::Union{Space, Vector{V}}, train::Any, val::Any;
        p::Real=0.8, maximize::Bool=false, fit_kwargs::NamedTuple=NamedTuple(), callback::Function=(state) -> false) where V<:NamedTuple
    arms = map(x -> T(; x...), space)
    return sasha!(rng, arms, train, val, p=p, maximize=maximize, fit_kwargs=fit_kwargs, callback=callback)
end

sasha(T::Type, space::Union{Space, Vector{V}}, train::Any, val::Any; kwargs...) where V<:NamedTuple =
    sasha(default_rng(), T, space, train, val; kwargs...)

function sasha(rng::AbstractRNG, f::Function, space::Union{Space, Vector{V}}, train::Any, val::Any;
        p::Real=0.8, maximize::Bool=false, fit_kwargs::NamedTuple=NamedTuple(), callback::Function=(state) -> false) where V<:NamedTuple
    arms = map(f, space)
    return sasha!(rng, arms, train, val, p=p, maximize=maximize, fit_kwargs=fit_kwargs, callback=callback)
end

sasha(f::Function, space::Union{Space, Vector{V}}, train::Any, val::Any; kwargs...) where V<:NamedTuple =
    sasha(default_rng(), f, space, train, val; kwargs...)

end
