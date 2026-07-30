module SASHA

using Base: @propagate_inbounds
using Random: GLOBAL_RNG, AbstractRNG, SamplerTrivial

import Random: rand

export fit!, loss, sasha!, sasha, Space, space

struct Space{names, T}
    vars::T
end

Base.eltype(::Type{Space{names, T}}) where {names, T} = NamedTuple{names,
    Tuple{map(eltype, fieldtypes(T))...}}

Base.firstindex(s::Space) = 1
Base.keys(s::Space) = Base.OneTo(length(s))
Base.lastindex(s::Space) = length(s)
Base.length(s::Space) = length(s.vars) == 0 ? 0 : prod(length, s.vars)
Base.size(s::Space) = length(s.vars) == 0 ? (0,) : map(length, s.vars)

function Base.show(io::IO, s::Space)
    print(io, "$(length(s))-element Space")
end

@inline function Base.getindex(s::Space{names}, i::Int) where names
    @boundscheck 1 ≤ i ≤ length(s) || throw(BoundsError(s, i))
    strides = (1, cumprod(map(length, Base.front(s.vars)))...)
    return NamedTuple{names}(map(getindex, s.vars, mod.((i - 1) .÷ strides, size(s)) .+ 1))
end

@inline function Base.getindex(s::Space{names}, I::Vararg{Int}) where names
    @boundscheck length(I) == length(s.vars) && all(1 .≤ I .≤ size(s)) || throw(BoundsError(s, I))
    return NamedTuple{names}(map(getindex, s.vars, I))
end

@inline function Base.getindex(s::Space, inds)
    return [s[i] for i in inds]
end

@propagate_inbounds function Base.iterate(s::Space, state::Int=1)
    state > length(s) && return nothing
    return s[state], state + 1
end

@inline function rand(rng::AbstractRNG, s::SamplerTrivial{T}) where T<:Space{names} where names
    return NamedTuple{names}(map(var -> rand(rng, var), s[].vars))
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
space(; vars...) = space(keys(vars), values(values(vars)))
space(names::NTuple{N, Symbol}, vars::NTuple{N, Any}) where N = Space{names, typeof(vars)}(vars)

_fit!(model::Any, data::Union{Tuple, NamedTuple}, args::NamedTuple) = fit!(model, data...; args...)
_fit!(model::Any, data::Any, args::NamedTuple) = fit!(model, data; args...)

"""
    fit!(model, data; <keyword arguments)

Fits `model` on `data` based on any additional provided keyword arguments and returns
the fitted model. Must be implemented for any custom model type before using the SASHA
optimization functions.

See also [`sasha`](@ref), [`sasha!`](@ref).
"""
fit!(model::Any, data::Any; args...) = throw(MethodError(fit!, (model, data)))

_loss(model::Any, data::Union{Tuple, NamedTuple}) = loss(model, data...)
_loss(model::Any, data::Any) = loss(model, data)

"""
    loss(model, data)

Evaluates the loss of `model` on `data`. Must be implemented for any custom model type
before using the SASHA optimization functions.

See also [`sasha`](@ref), [`sasha!`](@ref).
"""
loss(model::Any, data::Any) = throw(MethodError(loss, (model, data)))

"""
    sasha!([rng=GLOBAL_RNG], arms, train, val; <keyword arguments>)

Performs hyperparameter optimization using the SASHA optimizer on `arms` by modifying
the vector of arms in-place. Each arm is fitted on `train` by iteratively calling
`fit!`(`model`, `train`; `args`...). The loss of each arm is evaluated on `val` by
calling `loss`(`model`, `val`). These functions must be implemented for the type of the
model to be optimized. If `train` or `val` are (named) tuples, the arguments are
splatted in the respective function calls. Returns a tuple with the winning arm and its
final loss.

Multiple threads will be used (if available) to fit all remaining arms each round. The
number of available threads can be determined by calling `Threads.nthreads()`.

# Keyword Arguments
- `p::Real=0.8`: average acceptance probability of the arms at the start of the
  optimization (i.e., before any arms are fitted).
- `nmax::Int=typemax(Int)`: maximum number of rounds.
- `maximize::Bool=false`: whether the loss of the arms should be maximized.
- `args::NamedTuple=NamedTuple())`: optional named arguments that are passed on to
  `fit!` when fitting an arm.

# References
Triepels, R. (2023). SASHA: Hyperparameter Optimization by Simulated Annealing and
Successive Halving.

See also [`sasha`](@ref)
"""
function sasha!(rng::AbstractRNG, arms::Vector, train::Any, val::Any; p::Real=0.8,
        nmax::Int=typemax(Int), maximize::Bool=false, args::NamedTuple=NamedTuple())
    !isempty(arms) || throw(ArgumentError("no arms to optimze"))
    0 < p < 1 || throw(ArgumentError("p must be in [0,1]"))

    loss = map(arm -> _loss(arm, val), arms)
    temp = -(sum(loss) / length(loss)) / log(p)

    n = 1
    while length(arms) > 1
        @sync for i in eachindex(arms)
            Threads.@spawn arms[i] = _fit!(arms[i], train, args)
        end
        
        for i in eachindex(arms)
            loss[i] = _loss(arms[i], val)
        end

        if n == nmax
            if maximize
                best = argmax(loss)
            else
                best = argmin(loss)
            end
            keepat!(arms, best)
            keepat!(loss, best)
            break
        end

        if maximize
            prob = exp.(n .* (loss .- maximum(loss)) ./ temp)
        else
            prob = exp.(-n .* (loss .- minimum(loss)) ./ temp)
        end

        if all(isequal(first(prob)), prob)
            @warn "Unable to determine best arm. Terminating prematurely."
            keepat!(arms, 1)
            keepat!(loss, 1)
            break
        end

        discard = findall(<(rand(rng)), prob)
        deleteat!(arms, discard)
        deleteat!(loss, discard)

        n += 1
    end

    return first(arms), first(loss)
end

sasha!(arms::Vector, train::Any, val::Any; p::Real=0.8, nmax::Int=typemax(Int),
        maximize::Bool=false, args::NamedTuple=NamedTuple()) =
    sasha!(GLOBAL_RNG, arms, train, val, p=p, nmax=nmax, maximize=maximize, args=args)

"""
    sasha([rng=GLOBAL_RNG], T, space, train, val; <keyword arguments>)
    sasha([rng=GLOBAL_RNG], f, space, train, val; <keyword arguments>)

Optimizes the hyperparameters of a model using the SASHA optimizer. Initially, an arm
is created for each configuration in `space` by calling the constructor of type `T`
with model parameters provided as named arguments. Alternatively, if `T` cannot have a
constructor with named arguments, function `f` can be provided to create each arm
instead. `f` should take a named tuple with model parameters as input and initialize a
new model based on the provided parameter configuration. Each arm is fitted on `train`
by iteratively calling `fit`!(`model`, `train`; `args`...). The loss of each arm is
evaluated on `val` by calling `loss`(`model`, `val`). These functions must be
implemented for the type of the model to be optimized. If `train` or `val` are (named)
tuples, the arguments are splatted in the respective function calls. Returns a tuple with
the winning arm and its final loss.

Multiple threads will be used (if available) to fit all remaining arms each round. The
number of available threads can be determined by calling `Threads.nthreads()`.

# Keyword Arguments
- `p::Real=0.8`: average acceptance probability of the arms at the start of the
  optimization (i.e., before any arms are fitted).
- `nmax::Int=typemax(Int)`: maximum number of rounds.
- `maximize::Bool=false`: whether the loss of the arms should be maximized.
- `args::NamedTuple=NamedTuple())`: optional named arguments that are passed on to
  `fit!` when fitting an arm.

# References
Triepels, R. (2023). SASHA: Hyperparameter Optimization by Simulated Annealing and
Successive Halving.

See also [`sasha!`](@ref)
"""
function sasha(rng::AbstractRNG, T::Type, space::Union{Space, Vector{V}}, train::Any, val::Any;
        p::Real=0.8, nmax::Int=typemax(Int), maximize::Bool=false, args::NamedTuple=NamedTuple()) where V<:NamedTuple
    arms = map(x -> T(; x...), space)
    return sasha!(rng, arms, train, val, p=p, nmax=nmax, maximize=maximize, args=args)
end

sasha(T::Type, space::Union{Space, Vector{V}}, train::Any, val::Any; p::Real=0.8, nmax::Int=typemax(Int),
        maximize::Bool=false, args::NamedTuple=NamedTuple()) where V<:NamedTuple =
    sasha(GLOBAL_RNG, T, space, train, val, p=p, nmax=nmax, maximize=maximize, args=args)

function sasha(rng::AbstractRNG, f::Function, space::Union{Space, Vector{V}}, train::Any, val::Any;
            p::Real=0.8, nmax::Int=typemax(Int), maximize::Bool=false, args::NamedTuple=NamedTuple()) where V<:NamedTuple
    arms = map(f, space)
    return sasha!(rng, arms, train, val, p=p, nmax=nmax, maximize=maximize, args=args)
end

sasha(f::Function, space::Union{Space, Vector{V}}, train::Any, val::Any; p::Real=0.8, nmax::Int=typemax(Int), 
        maximize::Bool=false, args::NamedTuple=NamedTuple()) where V<:NamedTuple =
    sasha(GLOBAL_RNG, f, space, train, val, p=p, nmax=nmax, maximize=maximize, args=args)

end
