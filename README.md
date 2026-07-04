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

```
tweet-suspect-detection/
├── data/
│   ├── raw/           tweets bruts (suivi par DVC)
│   └── processed/     donnees nettoyees, sorties du pipeline
├── notebooks/
│   └── 01_eda.ipynb   exploration des donnees et prise en main du nettoyage
├── src/
│   ├── text_cleaning.py   fonctions de nettoyage du texte
│   ├── preprocess.py      etape 1 du pipeline DVC
│   ├── train.py           etape 2 du pipeline DVC
│   └── evaluate.py        etape 3 du pipeline DVC
├── models/            modele et vectoriseur entraines (sortie du pipeline)
├── metrics/            metriques et graphiques d'evaluation (sortie du pipeline)
├── reports/figures/    graphiques produits dans le notebook d'exploration
├── app/                application de deploiement (Streamlit)
├── params.yaml         parametres du pipeline
└── dvc.yaml            definition du pipeline DVC
```

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
