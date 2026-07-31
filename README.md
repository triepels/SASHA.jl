# Description
Implements the SASHA optimizer for hyperparameter parameter optimization in Julia.

# Installation
Run the following code to install the package:
```
] add https://github.com/triepels/SASHA.jl
```

# Quick Start
Suppose we have the following model with two hyperparameters `a` and `b`:

```julia
struct MyModel
    a::Float64
    b::Float64
    MyModel(; a::Float64, b::Float64) = new(a, b)
end
```

We need to implement functions `fit!` and `loss` for this model type.

```julia
julia> import SASHA: fit!, loss
```

Function `fit!` takes the model and fits it on data based on some optional keyword arguments:

```julia
julia> function fit!(model::MyModel, data; args)
           # Code to fit model...
       end
```

Function `loss` estimates how well the model performs on (out-of-sample) data:

```julia
julia> function loss(model::MyModel, data)
           # Code to evalute loss of the model...
       end
```

Accordingly, we need to create a space (i.e., grid) of configurations over which want to optimize the hyperparameters:

```julia
julia> sp = space(a=0.0:0.5:1.0, b=0.0:0.5:1.0)
```

Finally, we can call the SASHA optimizer:

```julia
julia> sasha(MyModel, sp, train, val)
```

Here, `train` is the training set on which the model is fitted and `val` is the validation set that is used to estimate the out-of-sample loss of the model.

An alternative way to call the optimizer is:

```julia
julia> sasha((x)->MyModel(; x...), sp, train, val)
```

This makes it possible to optimize a model that cannot have a constructor with named arguments.

# Reference
If you use the SASHA optimizer in your research, please cite the following paper:

Triepels, R. (2023). SASHA: Hyperparameter Optimization by Simulated Annealing and Successive Halving. In IFIP International Conference on Artificial Intelligence Applications and Innovations (pp. 491-502). Cham: Springer Nature Switzerland.