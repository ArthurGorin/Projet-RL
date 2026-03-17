include(joinpath(@__DIR__, "..", "src", "train.jl"))

using .Train

train!()
