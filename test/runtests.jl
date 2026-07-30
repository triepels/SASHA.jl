using SASHA

import SASHA: fit!, loss

mutable struct MyModel
    const β::Float64
    n::Int
    MyModel(; β::Float64) = new(β, 0)
end

function fit!(model::MyModel, x::AbstractArray; epochs::Int = 1)
    model.n += epochs
    return model
end

function loss(model::MyModel, x::AbstractArray)
    return -model.β * log(model.n / 100 + 1e-4)
end

train = rand(2, 10)
val = rand(2, 10)

sp = space(β=0.05:0.05:1.0)

sasha(MyModel, sp, train, val, p=0.8, maximize=false, args=(epochs=10,))
sasha(MyModel, rand(sp, 10), train, val, p=0.8, maximize=false, args=(epochs=10,))
sasha((args) -> MyModel(; args...), sp, train, val, p=0.8, maximize=false, args=(epochs=10,))
sasha((args) -> MyModel(; args...), rand(sp, 10), train, val, p=0.8, maximize=false, args=(epochs=10,))
