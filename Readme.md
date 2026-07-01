# Détection de fraude bancaire — Écosystème Hadoop


## Sujet

Détecter les transactions frauduleuses dans un jeu de données de paiements
mobiles (**PaySim**), à l'aide de **Spark** et **HDFS**


## Stack technique

- **Hadoop 3.5** (HDFS + YARN) — installation native
- **Spark 4 / PySpark** — traitement et Machine Learning (MLlib)
- **Python 3** + Jupyter Notebook
- Modèle : **Random Forest**

## Le dataset

Le dataset complet :

**https://www.kaggle.com/datasets/ealaxi/paysim1**

Une fois téléchargé, placer le fichier CSV dans `data/raw/` :
```
data/raw/fraud.csv
```


## Lancement

1. Démarrer Hadoop :
```bash
start-dfs.sh
start-yarn.sh
```

2. Importer le dataset dans HDFS :
```bash
hdfs dfs -mkdir -p /data/raw
hdfs dfs -put data/raw/fraud.csv /data/raw/
```

3. Lancer Jupyter et ouvrir le notebook :
```bash
jupyter notebook
```
Ouvrir `notebooks/projet_fraude_natif.ipynb` et exécuter les cellules.

> Astuce : `MODE_TEST = True` alors se base sur une partie du dataset
> (`False`) pour utiliser tout le dataset 

## La pipeline (étapes)

1. **Importation** du dataset depuis HDFS
2. **Parsing** et découverte des données
3. **Nettoyage** et feature engineering (colonne `typeEchange`, suppression des colonnes inutiles)
4. **Statistiques descriptives**
5. **Machine Learning** : entraînement d'un Random Forest
6. **Évaluation** (AUC, recall, matrice de confusion)
7. **Data Viz** : importance des variables

## Principaux résultats (dataset complet)

- 6 362 620 transactions, dont 8 213 fraudes (0,13 % — dataset très déséquilibré)
- La fraude n'apparaît que sur les transactions **TRANSFER** et **CASH_OUT**, entre particuliers
- **AUC = 0,98**
- Le modèle détecte ~48 % des fraudes avec quasiment aucune fausse alerte
- À titre de comparaison, l'ancien système de la banque (`isFlaggedFraud`) n'en détectait que 0,2 %

> Note : l'accuracy (99,9 %) est trompeuse à cause du fort déséquilibre du dataset ;
> l'indicateur pertinent est le **recall sur la classe fraude**.

## Structure du dépôt

```
├── README.md
├── requirements.txt            librairies Python
├── .gitignore
├── notebooks/
│   └── projet_fraude_natif.ipynb
└── data/
    ├── raw/                    (dataset complet à placer ici, non versionné)
    └── samples/                petit échantillon de test
```
