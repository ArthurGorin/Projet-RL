module Env





#------------------IL EST FINI/COMMENTÉ------------#

using ReinforcementLearning
using ArcadeLearningEnvironment
using ImageTransformations

export make_env, preprocess_frame, init_state_stack, update_state_stack, reset_env!, step_env!

const FRAME_HEIGHT = 84
const FRAME_WIDTH = 84
const STACK_SIZE = 4

function make_env(game::AbstractString="breakout")
    return AtariEnv(game)
end

# Convertit une frame brute en Float32 normalise et redimensionne en 84x84.
function preprocess_frame(frame::AbstractMatrix{UInt8})
    x = Float32.(frame) ./ 255f0
    x = imresize(x, (FRAME_HEIGHT, FRAME_WIDTH))
    return Float32.(x)
end

# Initialise l'etat empile en dupliquant la premiere frame 4 fois.
function init_state_stack(frame::AbstractMatrix{UInt8})
    x = preprocess_frame(frame)
    return cat(ntuple(_ -> x, STACK_SIZE)...; dims=3)
end

# Met a jour la stack en conservant les 3 dernieres frames.
function update_state_stack(stack::AbstractArray{<:Real,3}, frame::AbstractMatrix{UInt8})
    size(stack) == (FRAME_HEIGHT, FRAME_WIDTH, STACK_SIZE) ||
        throw(ArgumentError("expected stack shape (84, 84, 4), got $(size(stack))"))

    x = preprocess_frame(frame)
    return cat(Float32.(stack[:, :, 2:STACK_SIZE]), reshape(x, FRAME_HEIGHT, FRAME_WIDTH, 1); dims=3)
end

# Reinitialise l'environnement et construit la stack initiale.
function reset_env!(env)
    reset!(env)
    return init_state_stack(state(env))
end

# Execute une action dans l'environnement et renvoie l'etat suivant.
function step_env!(env, stack::AbstractArray{<:Real,3}, action::Int)
    act!(env, action)
    new_stack = update_state_stack(stack, state(env))
    return new_stack, reward(env), is_terminated(env)
end

end
