"""
Dashboard Streamlit — Détection de fraude bancaire (Hadoop/HDFS + Spark)

Reprend l'analyse du notebook `notebook/projet_fraude_v3.ipynb` sous forme
d'un dashboard interactif. Doit être lancé depuis WSL, avec HDFS (NameNode)
et Spark disponibles (mêmes prérequis que le notebook).

Lancement :
    wsl
    start-dfs.sh
    start-yarn.sh
    streamlit run streamlit_app.py
"""

import plotly.express as px
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array

st.set_page_config(page_title="Détection de fraude bancaire", layout="wide")


# ---------------------------------------------------------------------------
# Spark : une seule SparkSession, réutilisée entre les reruns Streamlit
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Démarrage de Spark...")
def get_spark():
    return SparkSession.builder \
        .appName("DetectionFraude-Streamlit") \
        .config("spark.driver.memory", "6g") \
        .getOrCreate()


# ---------------------------------------------------------------------------
# Pipeline complet : lecture HDFS -> exploration -> ML
# Mis en cache par (chemin, mode_test, fraction, nb_arbres) : ne retourne que
# des objets "légers" (dict / listes / nombres), pas de DataFrame Spark, pour
# rester compatible avec le cache Streamlit.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Lecture HDFS + calcul Spark en cours...")
def charger_et_analyser(_spark, chemin_hdfs: str, mode_test: bool, fraction: float, num_trees: int):
    df = _spark.read.csv(chemin_hdfs, header=True, inferSchema=True)

    if mode_test:
        df = df.sample(fraction=fraction, seed=67)

    nb_lignes = df.count()

    # --- 1. Système anti-fraude de la banque -------------------------------
    fraudes_oui = df.filter((df.isFraud == 1) & (df.isFlaggedFraud == 1)).count()
    fraudes_non = df.filter((df.isFraud == 1) & (df.isFlaggedFraud == 0)).count()
    total_f = fraudes_oui + fraudes_non
    recall_banque = fraudes_oui / total_f if total_f else 0.0

    # --- 2. Répartition fraude / normal --------------------------------------
    nb_fraude = df.filter(df.isFraud == 1).count()
    nb_normal = df.filter(df.isFraud == 0).count()

    # --- 3. Fraudes par type de transaction ----------------------------------
    lignes_type = df.filter(df.isFraud == 1).groupBy("type").count().collect()
    fraudes_par_type = {"type": [row["type"] for row in lignes_type], "count": [row["count"] for row in lignes_type]}

    # --- 4. Fraudes par type d'échange (CC, CM...) ---------------------------
    df = df.withColumn(
        "typeEchange",
        F.concat(F.substring("nameOrig", 1, 1), F.substring("nameDest", 1, 1)),
    )
    lignes_echange = df.filter(df.isFraud == 1).groupBy("typeEchange").count().collect()
    fraudes_par_echange = {"typeEchange": [row["typeEchange"] for row in lignes_echange], "count": [row["count"] for row in lignes_echange]}

    # --- 5. Montants ---------------------------------------------------------
    moy_fraude = df.filter(df.isFraud == 1).agg(F.avg("amount")).collect()[0][0] or 0.0
    moy_non_fraude = df.filter(df.isFraud == 0).agg(F.avg("amount")).collect()[0][0] or 0.0

    # --- 6. Anomalies par personne : a-t-on un historique par emetteur ? -----
    nb_emetteurs_uniques = df.select("nameOrig").distinct().count()

    # --- 7. Nettoyage ---------------------------------------------------------
    df = df.dropDuplicates()
    df = df.select(
        "step", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "typeEchange", "isFraud",
    )
    df = df.dropna(subset=["isFraud"])

    train, test = df.randomSplit([0.8, 0.2], seed=67)

    indexeur_type = StringIndexer(inputCol="type", outputCol="type_num")
    indexeur_echange = StringIndexer(inputCol="typeEchange", outputCol="typeEchange_num")

    def entrainer_evaluer(colonnes, nom):
        assembleur = VectorAssembler(inputCols=colonnes, outputCol="features")
        rf = RandomForestClassifier(featuresCol="features", labelCol="isFraud", numTrees=num_trees)
        pipeline = Pipeline(stages=[indexeur_type, indexeur_echange, assembleur, rf])

        modele = pipeline.fit(train)
        predictions = modele.transform(test)

        evaluateur = BinaryClassificationEvaluator(
            labelCol="isFraud", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        )
        auc = evaluateur.evaluate(predictions)

        vp = predictions.filter((predictions.isFraud == 1) & (predictions.prediction == 1)).count()
        fn = predictions.filter((predictions.isFraud == 1) & (predictions.prediction == 0)).count()
        fp = predictions.filter((predictions.isFraud == 0) & (predictions.prediction == 1)).count()
        recall = vp / (vp + fn) if (vp + fn) else 0.0

        rf_entraine = modele.stages[-1]
        paires = sorted(
            zip(colonnes, (float(v) for v in rf_entraine.featureImportances)),
            key=lambda p: p[1], reverse=True,
        )
        importances = {"variable": [p[0] for p in paires], "importance": [p[1] for p in paires]}

        return {
            "nom": nom, "auc": auc, "recall": recall,
            "vp": vp, "fn": fn, "fp": fp, "importances": importances,
        }, predictions

    col_avec_soldes = [
        "step", "type_num", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "typeEchange_num",
    ]
    col_sans_soldes = ["step", "type_num", "amount", "typeEchange_num"]

    modele_1, _ = entrainer_evaluer(col_avec_soldes, "Modèle 1 (avec soldes)")
    modele_2, predictions_2 = entrainer_evaluer(col_sans_soldes, "Modèle 2 (sans soldes)")

    # --- Ajustement du seuil (modele 2) : le seuil par defaut (0.5) est trop
    # haut pour un dataset aussi desequilibre -> on teste plusieurs seuils.
    pred_2 = predictions_2.withColumn("proba_fraude", vector_to_array("probability")[1])
    pred_2.cache()
    pred_2.count()

    seuils = [0.5, 0.3, 0.2, 0.1, 0.05]
    recalls_seuil, alertes_seuil = [], []
    for s in seuils:
        vp_s = pred_2.filter((pred_2.isFraud == 1) & (pred_2.proba_fraude >= s)).count()
        fn_s = pred_2.filter((pred_2.isFraud == 1) & (pred_2.proba_fraude < s)).count()
        fp_s = pred_2.filter((pred_2.isFraud == 0) & (pred_2.proba_fraude >= s)).count()
        recalls_seuil.append(100 * vp_s / (vp_s + fn_s) if (vp_s + fn_s) else 0.0)
        alertes_seuil.append(fp_s)

    return {
        "nb_lignes": nb_lignes,
        "fraudes_oui": fraudes_oui, "fraudes_non": fraudes_non,
        "total_f": total_f, "recall_banque": recall_banque,
        "nb_fraude": nb_fraude, "nb_normal": nb_normal,
        "fraudes_par_type": fraudes_par_type,
        "fraudes_par_echange": fraudes_par_echange,
        "moy_fraude": moy_fraude, "moy_non_fraude": moy_non_fraude,
        "nb_emetteurs_uniques": nb_emetteurs_uniques,
        "modele_1": modele_1, "modele_2": modele_2,
        "seuils": seuils, "recalls_seuil": recalls_seuil, "alertes_seuil": alertes_seuil,
    }


# ---------------------------------------------------------------------------
# Barre latérale : paramètres
# ---------------------------------------------------------------------------
st.sidebar.header("Paramètres")
chemin_hdfs = st.sidebar.text_input("Chemin HDFS", "hdfs://localhost:9000/data/raw/fraud.csv")
mode_test = st.sidebar.checkbox("Mode test (échantillon)", value=True)
fraction = st.sidebar.slider("Taille de l'échantillon", 0.01, 1.0, 0.05, step=0.01, disabled=not mode_test)
num_trees = st.sidebar.slider("Nombre d'arbres (Random Forest)", 10, 200, 50, step=10)
lancer = st.sidebar.button("Lancer / relancer l'analyse", type="primary")

st.title("Détection de fraude bancaire — Hadoop / Spark / Streamlit")
st.caption("HDFS : localhost:9000 — Dataset PaySim — Random Forest (PySpark MLlib)")

if not lancer and "resultats" not in st.session_state:
    st.info("Configurez les paramètres dans la barre latérale puis cliquez sur **Lancer / relancer l'analyse**.")
    st.stop()

if lancer:
    spark = get_spark()
    st.session_state["resultats"] = charger_et_analyser(spark, chemin_hdfs, mode_test, fraction, num_trees)

r = st.session_state["resultats"]

tab_banque, tab_explo, tab_modeles, tab_comparaison, tab_seuil, tab_limites = st.tabs(
    ["1. Système banque", "2. Exploration", "3. Modèles Random Forest", "4. Comparaison",
     "5. Ajustement du seuil", "6. Limites (anomalies)"]
)

# --- Onglet 1 : système anti-fraude de la banque ----------------------------
with tab_banque:
    st.subheader("Le système anti-fraude de la banque (`isFlaggedFraud`) est-il efficace ?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes analysées", f"{r['nb_lignes']:,}")
    c2.metric("Fraudes détectées par la banque", r["fraudes_oui"])
    c3.metric("Fraudes ratées par la banque", r["fraudes_non"])

    fig = px.bar(
        x=["ratées", "détectées"], y=[r["fraudes_non"], r["fraudes_oui"]],
        color=["ratées", "détectées"], color_discrete_map={"ratées": "red", "détectées": "green"},
        labels={"x": "", "y": "nb fraudes"}, title="Fraudes vues par le système de la banque",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Taux de détection de la banque", f"{100 * r['recall_banque']:.2f} %")
    st.caption("Conclusion : le système de la banque ne détecte quasiment rien. C'est notre référence à battre.")

# --- Onglet 2 : exploration --------------------------------------------------
with tab_explo:
    st.subheader("Répartition des transactions")
    fig = px.bar(
        x=["Normal", "Fraude"], y=[r["nb_normal"], r["nb_fraude"]],
        color=["Normal", "Fraude"], color_discrete_map={"Normal": "blue", "Fraude": "red"},
        labels={"x": "", "y": "nombre"}, title="Répartition des transactions",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Pourcentage de fraude : {100 * r['nb_fraude'] / (r['nb_fraude'] + r['nb_normal']):.3f} % — dataset très déséquilibré.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Fraudes par type de transaction")
        fig = px.bar(r["fraudes_par_type"], x="type", y="count", color_discrete_sequence=["red"],
                     labels={"count": "nb fraudes"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Uniquement sur TRANSFER et CASH_OUT : le fraudeur transfère puis retire l'argent volé.")
    with col_b:
        st.subheader("Fraudes par type d'échange")
        fig = px.bar(r["fraudes_par_echange"], x="typeEchange", y="count", color_discrete_sequence=["red"],
                     labels={"count": "nb fraudes"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Toutes les fraudes sont de type CC (particulier vers particulier).")

    st.subheader("Montant moyen d'une transaction")
    fig = px.bar(
        x=["normal", "fraude"], y=[r["moy_non_fraude"], r["moy_fraude"]],
        color=["normal", "fraude"], color_discrete_map={"normal": "blue", "fraude": "red"},
        labels={"x": "", "y": "montant moyen"},
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Onglet 3 : modèles ------------------------------------------------------
with tab_modeles:
    for cle in ("modele_1", "modele_2"):
        m = r[cle]
        st.subheader(m["nom"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", f"{m['auc']:.4f}")
        c2.metric("Recall", f"{100 * m['recall']:.1f} %")
        c3.metric("Détectées", m["vp"])
        c4.metric("Ratées", m["fn"])

        fig = px.bar(
            m["importances"], x="importance", y="variable", orientation="h",
            title=f"{m['nom']} : importance des variables",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

# --- Onglet 4 : comparaison ---------------------------------------------------
with tab_comparaison:
    m1, m2 = r["modele_1"], r["modele_2"]

    st.subheader("Comparaison des AUC")
    fig = px.bar(
        x=["Modèle 1 (avec soldes)", "Modèle 2 (sans soldes)"], y=[m1["auc"], m2["auc"]],
        color=["Modèle 1", "Modèle 2"], color_discrete_map={"Modèle 1": "teal", "Modèle 2": "orange"},
        labels={"x": "", "y": "AUC"}, range_y=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("% de fraudes détectées (recall)")
    fig = px.bar(
        x=["Banque", "Modèle 1 (avec soldes)", "Modèle 2 (sans soldes)"],
        y=[100 * r["recall_banque"], 100 * m1["recall"], 100 * m2["recall"]],
        color=["Banque", "Modèle 1", "Modèle 2"],
        color_discrete_map={"Banque": "gray", "Modèle 1": "teal", "Modèle 2": "orange"},
        labels={"x": "", "y": "recall (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Le modèle 1 (AUC très élevée) se base en grande partie sur les colonnes de soldes, "
        "qui sont modifiées APRÈS la fraude (fuite de données). Le modèle 2, sans ces colonnes, "
        "est plus honnête même si ses scores sont plus bas."
    )

# --- Onglet 5 : ajustement du seuil (modèle 2) --------------------------------
with tab_seuil:
    st.subheader("Le modèle 2 rate beaucoup de fraudes : et si on baissait le seuil ?")
    st.caption(
        "Par défaut, une transaction est classée « fraude » si sa probabilité dépasse 0.5. "
        "Sur un dataset aussi déséquilibré (~0.13% de fraudes), une transaction dépasse rarement "
        "ce seuil : on le baisse pour attraper plus de fraudes, au prix de plus de fausses alertes."
    )

    fig = px.line(
        x=r["seuils"], y=r["recalls_seuil"], markers=True,
        labels={"x": "seuil", "y": "recall (%)"}, title="Recall selon le seuil (modèle 2)",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        x=r["seuils"], y=r["alertes_seuil"], markers=True,
        labels={"x": "seuil", "y": "nb fausses alertes"}, title="Fausses alertes selon le seuil (modèle 2)",
    )
    fig.update_traces(line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    st.info("Plus on baisse le seuil, plus on attrape de fraudes, mais plus on a de fausses alertes : c'est un compromis à choisir selon le coût de chaque erreur.")

# --- Onglet 6 : limites (anomalies par personne) -------------------------------
with tab_limites:
    st.subheader("Peut-on détecter les fraudes via l'historique d'une personne ?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{r['nb_lignes']:,}")
    c2.metric("Émetteurs uniques (nameOrig)", f"{r['nb_emetteurs_uniques']:,}")
    c3.metric("Transactions / personne", round(r["nb_lignes"] / r["nb_emetteurs_uniques"], 2) if r["nb_emetteurs_uniques"] else 0)

    st.warning(
        "Le nombre d'émetteurs uniques est quasiment égal au nombre de transactions : chaque "
        "personne n'apparaît (quasiment) qu'une seule fois. Il n'y a donc PAS d'historique par "
        "personne dans PaySim, et on ne peut pas faire de détection d'anomalie par comportement "
        "(d'où le choix de la classification globale plutôt qu'une détection d'anomalie individuelle)."
    )

    st.markdown(
        """
**Colonnes qui manquent pour une vraie détection d'anomalie par personne :**

| Colonne | À quoi ça sert |
|---|---|
| id client récurrent | Suivre une personne dans le temps |
| date + heure précise | Voir les habitudes horaires |
| lieu / pays | Repérer une transaction à l'étranger inhabituelle |
| appareil / IP | Détecter une connexion depuis un nouvel appareil |
| catégorie du marchand | Voir si l'achat sort des habitudes |
| solde moyen habituel | Comparer à la normale de la personne |
"""
    )
