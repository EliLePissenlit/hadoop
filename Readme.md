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

## Dashboard Streamlit

Un dashboard interactif (`streamlit_app.py`) reprend la même analyse (exploration
+ Random Forest) sous forme de graphes interactifs (Plotly), toujours via WSL
et Spark/HDFS.

1. Démarrer Hadoop :
```bash
wsl
start-dfs.sh
start-yarn.sh
```

2. Lancer le dashboard :
```bash
streamlit run streamlit_app.py
```

3. Ouvrir l'URL affichée (ex. `http://localhost:8501`) dans le navigateur —
WSL2 redirige automatiquement `localhost` vers la distribution Linux.

Dans la barre latérale : chemin HDFS, mode test / taille d'échantillon, nombre
d'arbres du Random Forest, puis bouton **Lancer / relancer l'analyse**.

## Structure du dépôt

```
├── README.md
├── requirements.txt            librairies Python
├── streamlit_app.py            dashboard interactif (graphes)
├── .gitignore
├── notebooks/
│   └── projet_fraude_natif.ipynb
└── data/
    ├── raw/                    
    └── samples/               
```
