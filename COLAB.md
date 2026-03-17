# Google Colab

Ce projet peut etre lance sur Google Colab avec le runtime Julia.

Liens utiles :
- Nouveau notebook Julia Colab : https://colab.research.google.com/notebook#create=true&language=julia
- Colab annonce Julia : https://github.com/googlecolab/colabtools/issues/5151
- Notes de version Colab : https://developers.google.com/colab/release-notes

## Chemin simple

1. Ouvrir Colab en runtime `Julia`
2. Uploader ce repo sur GitHub ou en archive zip
3. Ouvrir `colab/Breakout_Train_Colab.ipynb`
4. Modifier `repo_url` dans la premiere cellule si besoin
5. Executer les cellules dans l'ordre

## Ce que fait le notebook

- clone le repo dans `/content`
- active le projet Julia local
- installe les dependances avec `Pkg.instantiate()`
- lance `scripts/run_train.jl`

## Points importants

- Le premier lancement sera lent a cause de l'installation et de la precompilation Julia.
- Le code actuel du repo est configure en CPU. Si tu veux exploiter un GPU NVIDIA Colab avec `CUDA.jl`, il faudra reintroduire un chemin GPU dans le projet.
- `ArcadeLearningEnvironment` peut etre la partie la plus fragile sur Colab selon les dependances systeme et la disponibilite de l'environnement Atari.
