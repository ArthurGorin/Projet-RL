# srl_breakout_julia

Base de projet Julia pour un agent SRL sur Breakout.

## Structure

- `src/`: logique principale de l'environnement, du modele, de l'acteur et du critique
- `scripts/`: points d'entree pour l'entrainement et l'evaluation
- `models/`: stockage des checkpoints
- `logs/`: sorties d'execution

## Lancement

```bash
julia --project=. scripts/run_train.jl
julia --project=. scripts/run_eval.jl
```
