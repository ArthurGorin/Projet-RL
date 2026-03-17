module Replay


#------------------IL EST FINI/COMMENTÉ------------#
#c'est la mémoire d'éxpérience de l'agent, utilisée pour entrainer les réseaux à partir d'anciens exemples. 
#au lieu d’apprendre uniquement en ligne sur le dernier step.
#Le replay buffer aide parce qu’on échantillonne aléatoirement des transitions venant de moments différents


using Random

export ReplayBuffer, push_transition!, can_sample, sample_batch, buffer_size

mutable struct ReplayBuffer
    capacity::Int
    position::Int
    size::Int
    states::Vector{Array{Float32,3}}
    actions::Vector{Int}
    rewards::Vector{Float32}
    next_states::Vector{Array{Float32,3}}
    dones::Vector{Bool}
end

# Construit un buffer circulaire de transitions pour casser la correlation temporelle.
function ReplayBuffer(capacity::Integer)
    capacity > 0 || throw(ArgumentError("capacity must be positive, got $capacity"))

    return ReplayBuffer(
        Int(capacity),
        1,
        0,
        Vector{Array{Float32,3}}(undef, capacity),
        Vector{Int}(undef, capacity),
        Vector{Float32}(undef, capacity),
        Vector{Array{Float32,3}}(undef, capacity),
        Vector{Bool}(undef, capacity),
    )
end

# Donne le nombre actuel de transitions stockees.
buffer_size(buffer::ReplayBuffer) = buffer.size

# Indique si le buffer contient assez d'exemples pour un mini-batch.
can_sample(buffer::ReplayBuffer, batch_size::Integer) = buffer.size >= batch_size

# Ajoute une transition et ecrase les plus anciennes quand le buffer est plein.
function push_transition!(
    buffer::ReplayBuffer,
    state::AbstractArray{<:Real,3},
    action::Integer,
    reward::Real,
    next_state::AbstractArray{<:Real,3},
    done::Bool,
)
    idx = buffer.position
    buffer.states[idx] = Float32.(state)
    buffer.actions[idx] = Int(action)
    buffer.rewards[idx] = Float32(reward)
    buffer.next_states[idx] = Float32.(next_state)
    buffer.dones[idx] = done

    buffer.position = idx == buffer.capacity ? 1 : idx + 1
    buffer.size = min(buffer.size + 1, buffer.capacity)

    return buffer
end

# Tire aleatoirement un mini-batch et l'empile en tenseurs (84, 84, 4, B).
function sample_batch(buffer::ReplayBuffer, batch_size::Integer; rng=Random.default_rng())
    can_sample(buffer, batch_size) ||
        throw(ArgumentError("cannot sample $batch_size elements from a buffer of size $(buffer.size)"))

    indices = rand(rng, 1:buffer.size, batch_size)

    state_batch = cat((buffer.states[i] for i in indices)...; dims=4)
    next_state_batch = cat((buffer.next_states[i] for i in indices)...; dims=4)
    action_batch = Int[buffer.actions[i] for i in indices]
    reward_batch = Float32[buffer.rewards[i] for i in indices]
    done_batch = Bool[buffer.dones[i] for i in indices]

    return (
        states=state_batch,
        actions=action_batch,
        rewards=reward_batch,
        next_states=next_state_batch,
        dones=done_batch,
    )
end

end
