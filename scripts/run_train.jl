include(joinpath(@__DIR__, "..", "src", "train.jl"))

using .Train

Train.train!()
