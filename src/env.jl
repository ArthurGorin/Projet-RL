module Env





#------------------IL EST FINI/COMMENTÉ------------#

using ReinforcementLearning
using ArcadeLearningEnvironment

export make_env, preprocess_frame, init_state_stack, update_state_stack, reset_env!, step_env!

const FRAME_HEIGHT = 84
const FRAME_WIDTH = 84
const STACK_SIZE = 4

function make_env(game::AbstractString="breakout")
    return AtariEnv(game)
end

# Recadre au centre et complete avec des zeros si la frame est trop petite.
function center_crop_or_pad(x::AbstractMatrix{Float32}, out_h::Int, out_w::Int)
    h, w = size(x)
    y0 = max(1, fld(h - out_h, 2) + 1)
    x0 = max(1, fld(w - out_w, 2) + 1)
    y1 = min(h, y0 + out_h - 1)
    x1 = min(w, x0 + out_w - 1)

    cropped = @view x[y0:y1, x0:x1]
    out = zeros(Float32, out_h, out_w)
    copy_h, copy_w = size(cropped)
    dst_y0 = fld(out_h - copy_h, 2) + 1
    dst_x0 = fld(out_w - copy_w, 2) + 1
    out[dst_y0:dst_y0 + copy_h - 1, dst_x0:dst_x0 + copy_w - 1] .= cropped
    return out
end

# Convertit une frame brute en Float32 normalise, puis sous-echantillonne
# par strides entiers avant un recadrage centre en 84x84.
function preprocess_frame(frame::AbstractMatrix{UInt8})
    h, w = size(frame)
    stride_h = max(1, fld(h, FRAME_HEIGHT))
    stride_w = max(1, fld(w, FRAME_WIDTH))
    sampled = @view frame[1:stride_h:end, 1:stride_w:end]
    x = Float32.(sampled) ./ 255f0
    return center_crop_or_pad(x, FRAME_HEIGHT, FRAME_WIDTH)
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
