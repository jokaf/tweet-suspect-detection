# Détection de tweets suspects

Projet réalisé dans le cadre du cours "Construction de modèles et leur Déploiement"
(Master Fouilles de Données & Intelligence Artificielle). Le but est de construire un
modèle capable de dire si un tweet est suspect ou non, en suivant tout le cycle de vie
d'un projet de machine learning : exploration des données, prétraitement, entraînement,
évaluation, et déploiement.

## Données

Le dataset utilisé contient environ 60 000 tweets avec une étiquette `label` :
- `1` = tweet suspect
- `0` = tweet non suspect

Les classes sont déséquilibrées (environ 90% / 10%), ce qui est pris en compte lors
de l'entraînement du modèle (voir plus bas).

Le fichier brut est versionné avec **DVC** et n'est pas stocké directement dans Git.

## Structure du projet

Rapide description des dossiers :

- `data/raw` : les tweets bruts, suivis par DVC
- `data/processed` : les données nettoyées, générées par le pipeline
- `notebooks/01_eda.ipynb` : exploration des données et nettoyage du texte
- `notebooks/02_modeles.ipynb` : représentation TF-IDF, comparaison des modèles, évaluation
- `src/` : les scripts du pipeline (text_cleaning.py, preprocess.py, train.py, evaluate.py)
- `models/` : le modèle et le vectoriseur entraînés, générés par le pipeline
- `metrics/` : les métriques et graphiques d'évaluation, générés par le pipeline
- `reports/figures` : les graphiques du notebook d'exploration
- `reports/captures` : les captures d'écran demandées par le sujet
- `reports/rapport.pdf` : le rapport final
- `app/` : l'application de déploiement (Streamlit)
- `params.yaml` et `dvc.yaml` : la config et la définition du pipeline DVC

## Installation

Le projet utilise `uv` pour gérer l'environnement Python.

```bash
uv venv
source .venv/Scripts/activate   # Windows
uv pip install -r requirements.txt
```

Il faut aussi télécharger les données NLTK utilisées pour le nettoyage du texte
(stop words et lemmatisation), sinon le prétraitement plante :

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Récupérer les données et reproduire le pipeline

```bash
dvc pull      # récupère le dataset depuis le stockage DVC
dvc repro     # relance le pipeline (prétraitement -> entraînement -> évaluation)
```

Le pipeline est défini dans `dvc.yaml` et les paramètres (taille du split,
modèle choisi, stratégie de rééquilibrage des classes, etc.) sont dans `params.yaml`.

## Pipeline

1. **Prétraitement** (`src/preprocess.py`) : nettoyage du texte (minuscules, suppression
   des URLs/mentions/caractères spéciaux/stop words, lemmatisation) puis séparation
   train/test.
2. **Entraînement** (`src/train.py`) : transformation du texte en TF-IDF, gestion du
   déséquilibre des classes (class weight ou SMOTE selon `params.yaml`), entraînement
   du modèle choisi.
3. **Évaluation** (`src/evaluate.py`) : calcul des métriques (accuracy, precision,
   recall, F1, ROC AUC), matrice de confusion et courbe ROC.

## Déploiement

Une application Streamlit permet de saisir un tweet et d'obtenir la prédiction du
modèle avec la probabilité associée :

```bash
streamlit run app/streamlit_app.py
```

## Résultats

Trois modèles ont été comparés (Logistic Regression, Random Forest, Naive Bayes)
avec du TF-IDF et une gestion du déséquilibre par class weight. Random Forest
donne le meilleur résultat global, avec une accuracy autour de 0.98 et un
F1-score autour de 0.99. Naive Bayes a un très bon recall (il détecte presque
tous les tweets suspects) mais une précision plus faible, ce qui veut dire
qu'il se trompe plus souvent en signalant des tweets normaux comme suspects.
Le détail des métriques se trouve dans `notebooks/02_modeles.ipynb`.

## Rapport

Le rapport complet (méthodologie, résultats, discussion) se trouve dans
`reports/rapport.pdf`, avec les captures d'écran dans `reports/captures/`.

## Où trouver quoi par rapport au sujet

Pour aider à s'y retrouver vu que le sujet a plusieurs parties :

- Partie 1 (exploration et prétraitement) -> `notebooks/01_eda.ipynb`
- Partie 2 (DVC) -> `dvc.yaml`, `.dvc/`, `params.yaml`, dossier `data/`
- Partie 3 (représentation) -> `notebooks/02_modeles.ipynb` et `src/train.py`
- Partie 4 (comparaison des modèles) -> `notebooks/02_modeles.ipynb`
- Partie 5 (validation) -> `notebooks/02_modeles.ipynb` et `src/preprocess.py`
- Partie 6 (évaluation et optimisation) -> `notebooks/02_modeles.ipynb`, `src/evaluate.py`, dossier `metrics/`
- Partie 7 (déploiement) -> `app/streamlit_app.py`
- Partie 8 (documentation) -> ce README et le rapport dans `reports/`
