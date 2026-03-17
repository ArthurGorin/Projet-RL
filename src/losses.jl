module Losses








#------------------IL EST FINI/COMMENTÉ------------#
using Flux
using Statistics

export regularized_prediction, critic_td_targets, critic_mse_loss, q_advantage_targets, fenchel_young_loss

const FY_EPS = 1.0f-8


# Applique le regularized argmax entropique, qui coincide ici avec un softmax.
function regularized_prediction(scores::AbstractVector{<:Real}; temperature::Real=1.0f0)
    return Flux.softmax(Float32.(scores) ./ Float32(temperature))
end
#Version batch.
function regularized_prediction(scores::AbstractMatrix{<:Real}; temperature::Real=1.0f0)
    return Flux.softmax(Float32.(scores) ./ Float32(temperature); dims=1)
end

#-------Loss pour le critic--------#
# Construit la valeur cible du critique a partir de celle du target critic: y = r + gamma * max_a' Q_target(s', a') * (1 - done).
function critic_td_targets(next_q_values::AbstractMatrix{<:Real}, rewards::AbstractVector{<:Real}, dones::AbstractVector{Bool}; gamma::Real=0.99f0)
    max_next_q = vec(maximum(Float32.(next_q_values); dims=1))
    done_mask = 1.0f0 .- Float32.(dones)
    return Float32.(rewards) .+ Float32(gamma) .* max_next_q .* done_mask
end

# Calcule la MSE des Qvalues predicted du critic (avec les Qvalues du target critic : target) sur les actions effectivement choisies du batch.
function critic_mse_loss(current_q_values::AbstractMatrix{<:Real}, actions::AbstractVector{<:Integer}, targets::AbstractVector{<:Real})
    batch_size = length(actions)
    predicted = Float32.(current_q_values[CartesianIndex.(actions, 1:batch_size)])

    diff = predicted .- Float32.(targets)
    return mean(diff .^ 2), predicted
end

#--------Loss pour l'acteur---------#

#Création de l'action cible pour la loss de l'acteur.
function q_advantage_targets(q_values_batch::AbstractMatrix{<:Real}; temperature::Real=1.0f0)
    centered = Float32.(q_values_batch) .- mean(Float32.(q_values_batch); dims=1) #on centre 
    return regularized_prediction(centered; temperature=temperature)
end


# Calcule la perte Fenchel-Young associee au regularized argmax entropique.
# Ici:
# - prediction map régularisée (CO-layer en entraînement): y_hat(theta) = softmax(theta / tau)
# - fonction de regularisation pour la loss: Omega(p) = tau * sum_i p_i log(p_i)
# - conjugate: Omega*(theta) = tau * log(sum_i exp(theta_i / tau))
# - perte FY: L(theta, y) = Omega*(theta) - <theta, y> + Omega(y)
#
# Avec y provenant du critique, le gradient en theta vaut y_hat(theta) - y.
function fenchel_young_loss(scores::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real}; temperature::Real=1.0f0)
    size(scores) == size(targets) ||
        throw(ArgumentError("scores and targets must have the same shape, got $(size(scores)) and $(size(targets))"))

    tau = Float32(temperature)
    scaled_scores = Float32.(scores) ./ tau

    max_scores = maximum(scaled_scores; dims=1)
    shifted = scaled_scores .- max_scores
    log_partition = vec(log.(sum(exp.(shifted); dims=1)) .+ FY_EPS) .+ vec(max_scores)

    linear_term = vec(sum(Float32.(scores) .* Float32.(targets); dims=1))
    entropy_term = tau .* vec(sum(Float32.(targets) .* log.(Float32.(targets) .+ FY_EPS); dims=1))

    losses = tau .* log_partition .- linear_term .+ entropy_term
    return mean(losses)
end

end
