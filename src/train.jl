module Train
# L'espace d'actions de Breakout est petit et discret, donc la couche
# structuree n'est pas un solveur combinatoire riche comme dans le papier SRL.
#
# Ce qui reste fidele a l'esprit SRL:
# - un reseau de scores pour l'acteur
# - une couche de decision structuree regularisee (softmax)
# - une perte d'acteur de style Fenchel-Young (régularisation entropique)
# - un critique Q(s, a) pour evaluer explicitement les actions



include("env.jl")
include("utils.jl")
include("actor.jl")
include("critic.jl")
include("losses.jl")
include("replay.jl")
include("model.jl")

using Flux
using Random
using ReinforcementLearning
using Statistics

using .Env
using .Actor
using .Critic
using .Losses
using .Replay
using .Model
using .Utils

export collect_transition!, update_critic!, update_actor!, train!

# Collecte une transition complete (s, a, r, s', done) a partir de l'acteur.
function collect_transition!(buffer::ReplayBuffer, env, actor, state; rng=Random.default_rng(), actor_temperature::Real=1.0f0)
    action = sample_action(actor, state; rng=rng, temperature=actor_temperature)
    next_state, reward, done = step_env!(env, state, action)
    push_transition!(buffer, state, action, reward, next_state, done)
    return next_state, Float32(reward), done, action
end

# Met a jour le critique par regression TD sur les actions observees.
function update_critic!(critic, target_critic, critic_state, batch; gamma::Real=0.99f0)
    critic_loss_fn = function (critic_model)
        current_q = q_values(critic_model, batch.states)
        next_q = q_values(target_critic, batch.next_states)
        targets = critic_td_targets(next_q, batch.rewards, batch.dones; gamma=gamma)
        critic_loss, _ = critic_mse_loss(current_q, batch.actions, targets)
        return critic_loss
    end

    loss, grads = Flux.withgradient(critic_loss_fn, critic)

    Flux.update!(critic_state, critic, grads[1])

    current_q = q_values(critic, batch.states)
    next_q = q_values(target_critic, batch.next_states)
    targets = critic_td_targets(next_q, batch.rewards, batch.dones; gamma=gamma)
    critic_loss, predicted_q = critic_mse_loss(current_q, batch.actions, targets)

    return (
        critic_loss=Float32(critic_loss),
        mean_q=Float32(mean(predicted_q)),
        mean_target=Float32(mean(targets)),
    )
end


# Met a jour l'acteur via une perte Fenchel-Young construite depuis les Q-values du critique.
function update_actor!(actor, critic, actor_state, batch; actor_temperature::Real=1.0f0, target_temperature::Real=1.0f0)
    critic_q = q_values(critic, batch.states)                         # (n_actions, B)
    target_distribution = q_advantage_targets(critic_q; temperature=target_temperature)

    loss, grads = Flux.withgradient(actor) do actor_model
        scores = action_scores(actor_model, batch.states)             # (n_actions, B)
        return fenchel_young_loss(scores, target_distribution; temperature=actor_temperature)
    end

    Flux.update!(actor_state, actor, grads[1])

    scores = action_scores(actor, batch.states)
    predicted_distribution = regularized_prediction(scores; temperature=actor_temperature)

    return (
        actor_loss=Float32(loss),
        alignment=Float32(mean(sum(predicted_distribution .* target_distribution; dims=1))),
    )
end

#Boucle d'entrainement globale
function train!(;
    episodes::Integer=100,
    max_steps_per_episode::Integer=500,
    replay_capacity::Integer=20_000,
    batch_size::Integer=32,
    warmup_steps::Integer=1_000,
    gamma::Real=0.99f0,
    actor_lr::Real=1.0f-4,
    critic_lr::Real=1.0f-4,
    actor_temperature::Real=1.0f0,
    target_temperature::Real=1.0f0,
    target_update_interval::Integer=250,
    updates_per_step::Integer=1,
    checkpoint_path::AbstractString=joinpath(@__DIR__, "..", "models", "srl_breakout_demo.bson"),
    checkpoint_interval::Integer=5,
    rng=Random.default_rng(),
)
    env = make_env()
    n_actions = length(action_space(env))

    actor = to_device(build_actor(n_actions))
    critic = to_device(build_critic(n_actions))
    target_critic = to_device(build_critic(n_actions))
    copy_model!(target_critic, critic)

    actor_state = Flux.setup(Flux.Adam(Float32(actor_lr)), actor)
    critic_state = Flux.setup(Flux.Adam(Float32(critic_lr)), critic)
    replay_buffer = ReplayBuffer(replay_capacity)

    total_env_steps = 0
    total_updates = 0

    log_message(
        "debut entrainement SRL demo episodes=$episodes batch_size=$batch_size " *
        "warmup_steps=$warmup_steps n_actions=$n_actions gpu=$(gpu_available())"
    )

    for episode in 1:episodes
        state = reset_env!(env)                                       # (84,84,4)
        episode_return = 0.0f0
        actor_loss_sum = 0.0f0
        critic_loss_sum = 0.0f0
        updates_this_episode = 0

        for step in 1:max_steps_per_episode
            next_state, reward, done, action = collect_transition!(
                replay_buffer,
                env,
                actor,
                state;
                rng=rng,
                actor_temperature=actor_temperature,
            )

            total_env_steps += 1
            episode_return += reward
            state = next_state

            if total_env_steps >= warmup_steps && can_sample(replay_buffer, batch_size)
                for _ in 1:updates_per_step
                    batch = sample_batch(replay_buffer, batch_size; rng=rng)

                    critic_stats = update_critic!(critic, target_critic, critic_state, batch; gamma=gamma)
                    actor_stats = update_actor!(
                        actor,
                        critic,
                        actor_state,
                        batch;
                        actor_temperature=actor_temperature,
                        target_temperature=target_temperature,
                    )

                    critic_loss_sum += critic_stats.critic_loss
                    actor_loss_sum += actor_stats.actor_loss
                    updates_this_episode += 1
                    total_updates += 1

                    if total_updates % target_update_interval == 0
                        copy_model!(target_critic, critic)
                    end
                end
            end

            done && break
        end

        mean_actor_loss = updates_this_episode == 0 ? NaN32 : actor_loss_sum / updates_this_episode
        mean_critic_loss = updates_this_episode == 0 ? NaN32 : critic_loss_sum / updates_this_episode

        if episode % checkpoint_interval == 0 || episode == episodes
            metadata = Dict(
                "episode" => episode,
                "total_env_steps" => total_env_steps,
                "total_updates" => total_updates,
                "gamma" => Float32(gamma),
                "actor_temperature" => Float32(actor_temperature),
                "target_temperature" => Float32(target_temperature),
                "n_actions" => n_actions,
            )

            save_checkpoint(
                checkpoint_path;
                actor=actor,
                critic=critic,
                target_critic=target_critic,
                metadata=metadata,
            )
        end

        log_message(
            "ep=$episode episode_reward=$(round(episode_return, digits=3)) " *
            "updates=$updates_this_episode actor_loss=$(round(mean_actor_loss, digits=5)) " *
            "critic_loss=$(round(mean_critic_loss, digits=5))"
        )
    end

    return (
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        replay_buffer=replay_buffer,
    )
end

end
