module Critic





#------------------IL EST FINI/COMMENTÉ------------#
using Flux

using ..Utils: to_device

export build_critic, to_batch, to_batch_states, q_values, evaluate_action, greedy_value

const INPUT_HEIGHT = 84
const INPUT_WIDTH = 84
const INPUT_CHANNELS = 4

# Ajoute une dimension de batch a un etat unique de forme (84, 84, 4).
function to_batch(x::AbstractArray{<:Real,3})
    size(x) == (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS) ||
        throw(ArgumentError("expected input shape (84, 84, 4), got $(size(x))"))

    batch = reshape(Float32.(x), INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, 1)
    return to_device(batch)
end

#convertit un batch de forme (84, 84, 4, B).
function to_batch_states(x::AbstractArray{<:Real,4})
    size(x)[1:3] == (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS) ||
        throw(ArgumentError("expected batch shape (84, 84, 4, B), got $(size(x))"))

    batch = Float32.(x)
    return to_device(batch)
end

# Calcule automatiquement la taille du vecteur aplati apres les convolutions.
function conv_output_size()
    dummy = rand(Float32, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, 1)
    features = Chain(
        Conv((8, 8), INPUT_CHANNELS => 32, stride=4, relu),
        Conv((4, 4), 32 => 64, stride=2, relu),
        Conv((3, 3), 64 => 64, stride=1, relu),
        Flux.flatten,
    )

    return size(features(dummy), 1)
end

const FLAT_SIZE = conv_output_size()

# Construit un modèle de critic Q avec la meme base convolutive que l'acteur.
function build_critic(n_actions::Integer)
    n_actions > 0 || throw(ArgumentError("n_actions must be positive, got $n_actions"))

    return Chain(
        Conv((8, 8), INPUT_CHANNELS => 32, stride=4, relu), # (84,84,4,B) -> (20,20,32,B)
        Conv((4, 4), 32 => 64, stride=2, relu),             # -> (9,9,64,B)
        Conv((3, 3), 64 => 64, stride=1, relu),             # -> (7,7,64,B)
        Flux.flatten,                                        # -> (FLAT_SIZE,B)
        Dense(FLAT_SIZE, 512, relu),                         # -> (512,B)
        Dense(512, n_actions),                               # -> (n_actions,B)
    )
end

#Renvoie une Q-value (estimation par le critic de la qualité d’une action dans un état donné.)  
#par action pour un etat unique.
function q_values(critic, stack::AbstractArray{<:Real,3})
    return vec(critic(to_batch(stack)))
end

# Renvoie une Q-value par action pour un batch d'etat.
function q_values(critic, stacks::AbstractArray{<:Real,4})
    return critic(to_batch_states(stacks))
end

# Extrait la Q-value d'une action pour un etat unique.
function evaluate_action(critic, stack::AbstractArray{<:Real,3}, action::Integer)
    values = q_values(critic, stack)
    1 <= action <= length(values) || throw(BoundsError(values, action))
    return values[action]
end

#Renvoie la meilleure valeur estimee par le critique pour un etat unique.
function greedy_value(critic, stack::AbstractArray{<:Real,3})
    return maximum(q_values(critic, stack))
end

end
