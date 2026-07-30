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
    return model.η + (1 - model.η)^(model.n + 1) + 1e-3 * rand()
end

train = rand(2, 10)
val = rand(2, 10)

sp = space(η=logrange(1e-3, 1e-1, length=20))

args = (p=0.8, nmax=20, maximize=false, args=(epochs=100,))

sasha(MyModel, sp, train, val; args...)
sasha(MyModel, rand(sp, 10), train, val; args...)
sasha((args) -> MyModel(; args...), sp, train, val; args...)
sasha((args) -> MyModel(; args...), rand(sp, 10), train, val; args...)
