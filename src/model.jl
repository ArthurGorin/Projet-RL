module Model





#------------------IL EST FINI/COMMENTÉ------------#

using BSON: @save, @load
using Flux

export copy_model!, save_checkpoint, load_checkpoint

# Copie integralement les poids d'un modele source vers un modele cible (utile pour target critic).
function copy_model!(target, source)
    Flux.loadmodel!(target, source)
    return target
end

# Sauvegarde les modeles (actor et critics) et quelques metadonnees utiles pour reprise/evaluation.
function save_checkpoint(path::AbstractString; actor, critic, target_critic=nothing, metadata=Dict{String,Any}())
    mkpath(dirname(path))
    actor = Flux.cpu(actor)
    critic = Flux.cpu(critic)
    target_critic = isnothing(target_critic) ? nothing : Flux.cpu(target_critic)
    @save path actor critic target_critic metadata
    return path
end

# Recharge un checkpoint BSON avec les modeles et les metadonnees associees.
function load_checkpoint(path::AbstractString)
    actor = nothing
    critic = nothing
    target_critic = nothing
    metadata = Dict{String,Any}()
    @load path actor critic target_critic metadata
    return (; actor, critic, target_critic, metadata)
end

end
