# Détection de fraude bancaire — Écosystème Hadoop


## Sujet

Détecter les transactions frauduleuses dans un jeu de données de paiements
mobiles (PaySim), à l'aide de Spark** et **HDFS


## requirements

- **Hadoop 3.5** (HDFS + YARN) — installation native
- **Spark 4 / PySpark** — traitement et Machine Learning (MLlib)
- **Python 3** + Jupyter Notebook
- Modèle : **Random Forest**

##  dataset

**https://www.kaggle.com/datasets/ealaxi/paysim1**

placer le fichier CSV dans `data/raw/` :
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
Ouvrir `notebooks/projet_fraude_v3.ipynb` et exécuter les cellules.

> `MODE_TEST = True`  se base sur une partie du dataset
> (`False`) pour tout le dataset 



## Structure du dépôt

```
├── README.md
├── requirements.txt            librairies Python
├── .gitignore
├── notebooks/
│   └── projet_fraude_natif.ipynb
└── data/
    ├── raw/                    
    └── samples/               
```
