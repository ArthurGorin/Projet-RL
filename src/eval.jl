module Eval

include("env.jl")
include("utils.jl")
include("actor.jl")
include("model.jl")

using ReinforcementLearning

using .Env
using .Actor
using .Model
using .Utils

export evaluate_agent

# Evalue un acteur en mode glouton sur plusieurs episodes.
function evaluate_agent(;
    episodes::Integer=3,
    max_steps_per_episode::Integer=300,
    checkpoint_path::Union{Nothing,AbstractString}=joinpath(@__DIR__, "..", "models", "srl_breakout_demo.bson"),
    actor=nothing,
)
    env = make_env()
    n_actions = length(action_space(env))

    policy = actor
    if policy === nothing
        if checkpoint_path !== nothing && isfile(checkpoint_path)
            checkpoint = load_checkpoint(checkpoint_path)
            policy = to_device(checkpoint.actor)
            log_message("checkpoint charge depuis $checkpoint_path")
        else
            policy = to_device(build_actor(n_actions))
            log_message("aucun checkpoint trouve, evaluation d'un acteur non entraine")
        end
    else
        policy = to_device(policy)
    end

    returns = Float32[]

    for episode in 1:episodes
        state = reset_env!(env)
        episode_return = 0.0f0
        steps = 0

        for step in 1:max_steps_per_episode
            action = select_action(policy, state)
            state, reward, done = step_env!(env, state, action)

            episode_return += reward
            steps = step
            done && break
        end

        push!(returns, episode_return)
        log_message("eval episode=$episode return=$(round(episode_return, digits=3)) steps=$steps")
    end

    mean_return = isempty(returns) ? 0.0f0 : sum(returns) / length(returns)
    log_message("evaluation terminee mean_return=$(round(mean_return, digits=3))")

    return (; actor=policy, returns, mean_return)
end

end
