# Detection de tweets suspects

Projet realise dans le cadre du cours "Construction de modeles et leur Deploiement"
(Master Fouilles de Donnees & Intelligence Artificielle). Le but est de construire un
modele capable de dire si un tweet est suspect ou non, en suivant tout le cycle de vie
d'un projet de machine learning : exploration des donnees, pretraitement, entrainement,
evaluation, et deploiement.

## Donnees

Le dataset utilise contient environ 60 000 tweets avec une etiquette `label` :
- `1` = tweet suspect
- `0` = tweet non suspect

Les classes sont desequilibrees (environ 90% / 10%), ce qui est pris en compte lors
de l'entrainement du modele (voir plus bas).

Le fichier brut est versionne avec **DVC** et n'est pas stocke directement dans Git.

## Structure du projet

Rapide description des dossiers :

- `data/raw` : les tweets bruts, suivis par DVC
- `data/processed` : les donnees nettoyees, generees par le pipeline
- `notebooks/01_eda.ipynb` : exploration des donnees et nettoyage du texte
- `notebooks/02_modeles.ipynb` : representation TF-IDF, comparaison des modeles, evaluation
- `src/` : les scripts du pipeline (text_cleaning.py, preprocess.py, train.py, evaluate.py)
- `models/` : le modele et le vectoriseur entraines, generes par le pipeline
- `metrics/` : les metriques et graphiques d'evaluation, generes par le pipeline
- `reports/figures` : les graphiques du notebook d'exploration
- `app/` : l'application de deploiement (Streamlit)
- `params.yaml` et `dvc.yaml` : la config et la definition du pipeline DVC

## Installation

Le projet utilise `uv` pour gerer l'environnement Python.

```bash
uv venv
source .venv/Scripts/activate   # Windows
uv pip install -r requirements.txt
```

## Recuperer les donnees et reproduire le pipeline

```bash
dvc pull      # recupere le dataset depuis le stockage DVC
dvc repro     # relance le pipeline (pretraitement -> entrainement -> evaluation)
```

Le pipeline est defini dans `dvc.yaml` et les parametres (taille du split,
modele choisi, strategie de reequilibrage des classes, etc.) sont dans `params.yaml`.

## Pipeline

1. **Pretraitement** (`src/preprocess.py`) : nettoyage du texte (minuscules, suppression
   des URLs/mentions/caracteres speciaux/stop words, lemmatisation) puis separation
   train/test.
2. **Entrainement** (`src/train.py`) : transformation du texte en TF-IDF, gestion du
   desequilibre des classes (class weight ou SMOTE selon `params.yaml`), entrainement
   du modele choisi.
3. **Evaluation** (`src/evaluate.py`) : calcul des metriques (accuracy, precision,
   recall, F1, ROC AUC), matrice de confusion et courbe ROC.

## Deploiement

Une application Streamlit permet de saisir un tweet et d'obtenir la prediction du
modele avec la probabilite associee :

```bash
streamlit run app/streamlit_app.py
```

## Rapport

Le rapport complet (methodologie, resultats, discussion) se trouve dans `reports/`.

## Ou trouver quoi par rapport au sujet

Pour aider a s'y retrouver vu que le sujet a plusieurs parties :

- Partie 1 (exploration et pretraitement) -> `notebooks/01_eda.ipynb`
- Partie 2 (DVC) -> `dvc.yaml`, `.dvc/`, `params.yaml`, dossier `data/`
- Partie 3 (representation) -> `notebooks/02_modeles.ipynb` et `src/train.py`
- Partie 4 (comparaison des modeles) -> `notebooks/02_modeles.ipynb`
- Partie 5 (validation) -> `notebooks/02_modeles.ipynb` et `src/preprocess.py`
- Partie 6 (evaluation et optimisation) -> `notebooks/02_modeles.ipynb`, `src/evaluate.py`, dossier `metrics/`
- Partie 7 (deploiement) -> `app/streamlit_app.py`
- Partie 8 (documentation) -> ce README et le rapport dans `reports/`
