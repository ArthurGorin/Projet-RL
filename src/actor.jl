module Actor




#------------------IL EST FINI/COMMENTÉ------------#
using Flux
using Random

using ..Utils: to_device

export build_actor, to_batch, to_batch_states, action_scores, action_probs, sample_action, select_action

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

# Verifie ou convertit un lot d'etats de forme (84, 84, 4, B).
function to_batch_states(x::AbstractArray{<:Real,4})
    size(x)[1:3] == (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS) ||
        throw(ArgumentError("expected batch shape (84, 84, 4, B), got $(size(x))"))

    batch = Float32.(x)
    return to_device(batch)
end

#------Réseaux de neuronnes : CNN----#
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

# Construit le reseau de scores de l'acteur.
function build_actor(n_actions::Integer)
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

# Renvoie les scores bruts pour un etat unique.
function action_scores(actor, stack::AbstractArray{<:Real,3})
    return vec(actor(to_batch(stack)))
end

# Renvoie les scores bruts pour un batch.
function action_scores(actor, stacks::AbstractArray{<:Real,4})
    return actor(to_batch_states(stacks))
end

#-----------C0-LAYER--------------#
#L'espace d'action étant petit la CO-Layer est triviale :

#---entrainement---#
# Ici, on peut prendre directement le softmax comme argmax régularisée pour qu'il soit différentiable
# lors de l'entrainement
function action_probs(actor, stack::AbstractArray{<:Real,3}; temperature::Real=1.0f0)
    scaled_scores = action_scores(actor, stack) ./ Float32(temperature)
    return Flux.softmax(scaled_scores)
end
#version batch
function action_probs(actor, stacks::AbstractArray{<:Real,4}; temperature::Real=1.0f0)
    scaled_scores = action_scores(actor, stacks) ./ Float32(temperature)
    return Flux.softmax(scaled_scores; dims=1)
end
   
#Exploration : Tire une action a partir de la distribution produite par la co layer
function sample_action(actor, stack::AbstractArray{<:Real,3}; rng=Random.default_rng(), temperature::Real=1.0f0)
    probs = action_probs(actor, stack; temperature=temperature)
    threshold = rand(rng)
    cumulative = 0.0f0

    for (i, p) in enumerate(probs)
        cumulative += p
        threshold <= cumulative && return i
    end

    return length(probs)
end

#----Eval/inférence------#
#choisit directement l'argmax selon les scores, plus besoin de régularisation car pas de backpropagation
function select_action(actor, stack::AbstractArray{<:Real,3})
    return argmax(action_scores(actor, stack))
end

end
