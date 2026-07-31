using SASHA

import SASHA: fit!, loss

mutable struct MyModel
    const η::Float64
    n::Int
    MyModel(; η::Float64) = new(η, 0)
end

function fit!(model::MyModel, x::AbstractArray; epochs::Int = 1)
    model.n += epochs
    return model
end

function loss(model::MyModel, x::AbstractArray)
    return model.η + (1 - model.η)^(model.n + 1) + 0.1 * rand()
end

train = rand(2, 10)
val = rand(2, 10)

sp = space(η=logrange(1e-3, 1e-1, length=20))

kwargs = (p=0.9, nmax=20, maximize=false, fit_kwargs=(epochs=100,))

sasha(MyModel, sp, train, val; kwargs...)
sasha(MyModel, rand(sp, 10), train, val; kwargs...)
sasha((x) -> MyModel(; x...), sp, train, val; kwargs...)
sasha((x) -> MyModel(; x...), rand(sp, 10), train, val; kwargs...)
