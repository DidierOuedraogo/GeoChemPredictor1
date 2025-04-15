import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import seaborn as sns
from io import BytesIO

# Vérifier si TensorFlow est disponible
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="GeoChem Predictor",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1ABC9C;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #566573;
    }
    .highlight {
        background-color: #E8F8F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.markdown("<h1 class='main-header'>GeoChem Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-header'>Analyse Géochimique Automatisée</h3>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Développé par Didier Ouedraogo, P.Geo</p>", unsafe_allow_html=True)

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une fonction :", 
                        ["Accueil", 
                         "Prédiction de Minéralisation", 
                         "Détection d'Anomalies", 
                         "Recommandation de Cibles"])

# Fonction pour charger les données
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            st.error("Format de fichier non supporté. Veuillez télécharger un fichier CSV ou Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Fonction pour télécharger un fichier
def download_file(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
        
    # Créer un lien de téléchargement
    b64 = BytesIO()
    if isinstance(object_to_download, str):
        b64.write(object_to_download.encode())
    else:
        b64.write(object_to_download)
    b64.seek(0)
    return st.download_button(
        label=download_link_text,
        data=b64,
        file_name=download_filename,
        mime="text/csv"
    )

# Page d'accueil
if page == "Accueil":
    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("""
    # Bienvenue dans GeoChem Predictor
    
    GeoChem Predictor est un outil d'analyse géochimique automatisé qui offre trois fonctionnalités principales :
    
    1. **Prédiction de Minéralisation** - Prédisez la teneur en minéralisation à partir de données géochimiques (As, Sb, Cu, etc.) via des modèles de régression avancés.
    
    2. **Détection d'Anomalies** - Identifiez les anomalies géochimiques grâce à l'apprentissage non supervisé.
    
    3. **Recommandation de Cibles** - Obtenez des suggestions pour les emplacements des prochains prélèvements.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Comment utiliser cette application :
    
    1. Sélectionnez une fonction dans le menu de gauche
    2. Téléchargez vos données géochimiques (format CSV ou Excel)
    3. Configurez les paramètres selon vos besoins
    4. Analysez les résultats et téléchargez-les si nécessaire
    
    ### Format requis pour vos données :
    
    Vos données doivent inclure des coordonnées (X, Y) et des valeurs géochimiques pour différents éléments.
    Un exemple de format attendu :
    """)
    
    # Exemple de données
    example_data = pd.DataFrame({
        'X': [350245, 350255, 350265, 350275],
        'Y': [7652360, 7652370, 7652380, 7652390],
        'As_ppm': [12.5, 23.7, 45.2, 8.3],
        'Sb_ppm': [1.2, 2.3, 4.5, 0.8],
        'Cu_ppm': [256, 312, 489, 145],
        'Au_ppb': [25, 35, 125, 10]
    })
    
    st.dataframe(example_data)
    
    # Téléchargement des données exemple
    csv = example_data.to_csv(index=False)
    st.download_button(
        label="Télécharger les données exemple (CSV)",
        data=csv,
        file_name="exemple_donnees_geochimiques.csv",
        mime="text/csv"
    )

# Page de prédiction de minéralisation
elif page == "Prédiction de Minéralisation":
    st.markdown("<h2 class='sub-header'>Prédiction de Minéralisation</h2>", unsafe_allow_html=True)
    
    st.info("Cette fonction utilise des algorithmes d'apprentissage automatique (XGBoost, LightGBM) pour prédire la teneur en minéralisation à partir de données géochimiques.")
    
    # Téléchargement du fichier
    uploaded_file = st.file_uploader("Télécharger vos données géochimiques (CSV ou Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("Aperçu des données:")
            st.dataframe(df.head())
            
            # Sélection des colonnes
            st.subheader("Configuration du modèle")
            
            # Sélection des caractéristiques et de la cible
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                features = st.multiselect("Sélectionner les caractéristiques (éléments géochimiques)", 
                                         numeric_columns, 
                                         default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns)
            
            with col2:
                target = st.selectbox("Sélectionner la cible à prédire", 
                                     numeric_columns, 
                                     index=min(3, len(numeric_columns)-1) if len(numeric_columns) > 3 else 0)
            
            # Exclusion de la cible des caractéristiques
            if target in features:
                features.remove(target)
            
            # Sélection du modèle
            model_options = ["XGBoost", "LightGBM"]
            if TENSORFLOW_AVAILABLE:
                model_options.append("Réseau de Neurones")
                
            model_type = st.radio("Sélectionner le modèle de régression", model_options)
            
            # Paramètres du modèle
            with st.expander("Paramètres avancés du modèle"):
                if model_type == "XGBoost" or model_type == "LightGBM":
                    n_estimators = st.slider("Nombre d'estimateurs", 50, 500, 100, 10)
                    learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, 0.01)
                    max_depth = st.slider("Profondeur maximale", 3, 10, 6, 1)
                elif model_type == "Réseau de Neurones" and TENSORFLOW_AVAILABLE:
                    epochs = st.slider("Nombre d'époques", 10, 200, 50, 5)
                    batch_size = st.slider("Taille du batch", 8, 128, 32, 8)
                    dropout_rate = st.slider("Taux de dropout", 0.0, 0.5, 0.2, 0.05)
            
            # Entraînement du modèle
            if st.button("Entraîner le modèle"):
                # Préparation des données
                X = df[features].copy()
                y = df[target].copy()
                
                # Vérification des valeurs manquantes
                if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                    st.warning("Des valeurs manquantes ont été détectées dans vos données. Elles seront remplacées par la médiane.")
                    X = X.fillna(X.median())
                    y = y.fillna(y.median())
                
                # Division train/test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scaling des données
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Entraînement du modèle
                with st.spinner('Entraînement du modèle en cours...'):
                    if model_type == "XGBoost":
                        model = xgb.XGBRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        
                    elif model_type == "LightGBM":
                        model = lgb.LGBMRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        
                    elif model_type == "Réseau de Neurones" and TENSORFLOW_AVAILABLE:
                        # Définition du modèle
                        model = Sequential()
                        model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
                        model.add(Dropout(dropout_rate))
                        model.add(Dense(64, activation='relu'))
                        model.add(Dropout(dropout_rate))
                        model.add(Dense(32, activation='relu'))
                        model.add(Dense(1))
                        
                        # Compilation
                        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                        
                        # Entraînement
                        history = model.fit(
                            X_train_scaled, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.2,
                            verbose=0
                        )
                
                # Évaluation du modèle
                with st.spinner('Évaluation du modèle en cours...'):
                    if model_type == "Réseau de Neurones" and TENSORFLOW_AVAILABLE:
                        y_pred = model.predict(X_test_scaled).flatten()
                    else:
                        y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    st.success('Modèle entraîné avec succès!')
                    
                    # Affichage des métriques
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MSE", f"{mse:.4f}")
                    col2.metric("RMSE", f"{rmse:.4f}")
                    col3.metric("R²", f"{r2:.4f}")
                    
                    # Visualisation des résultats
                    st.subheader("Visualisation des performances")
                    
                    # Graphique prédictions vs réalité
                    fig = px.scatter(
                        x=y_test, y=y_pred,
                        labels={"x": f"{target} réel", "y": f"{target} prédit"},
                        title="Prédictions vs Valeurs réelles"
                    )
                    
                    # Ajout de la ligne parfaite (y=x)
                    fig.add_trace(
                        go.Scatter(
                            x=[y_test.min(), y_test.max()],
                            y=[y_test.min(), y_test.max()],
                            mode="lines",
                            line=dict(color="red", dash="dash"),
                            name="Prédiction parfaite"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Importance des caractéristiques pour XGBoost et LightGBM
                    if model_type in ["XGBoost", "LightGBM"]:
                        st.subheader("Importance des caractéristiques")
                        
                        importance = model.feature_importances_
                        feature_importance = pd.DataFrame({
                            'Feature': features,
                            'Importance': importance
                        }).sort_values(by='Importance', ascending=False)
                        
                        fig = px.bar(
                            feature_importance,
                            x='Importance', y='Feature',
                            orientation='h',
                            title="Importance des éléments géochimiques"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Histogramme d'erreurs
                    st.subheader("Distribution des erreurs")
                    errors = y_pred - y_test
                    fig = px.histogram(
                        errors,
                        nbins=20,
                        title="Distribution des erreurs de prédiction"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prédiction sur l'ensemble des données
                    st.subheader("Prédictions sur l'ensemble des données")
                    
                    if model_type == "Réseau de Neurones" and TENSORFLOW_AVAILABLE:
                        X_scaled = scaler.transform(X)
                        predictions = model.predict(X_scaled).flatten()
                    else:
                        predictions = model.predict(X)
                    
                    result_df = df.copy()
                    result_df[f'{target}_prédit'] = predictions
                    result_df[f'Erreur_{target}'] = result_df[f'{target}_prédit'] - result_df[target]
                    
                    st.dataframe(result_df)
                    
                    # Téléchargement des résultats
                    csv_result = result_df.to_csv(index=False)
                    st.download_button(
                        label="Télécharger les résultats (CSV)",
                        data=csv_result,
                        file_name="resultats_prediction.csv",
                        mime="text/csv"
                    )
                    
                    # Carte de prédiction si coordonnées disponibles
                    if 'X' in df.columns and 'Y' in df.columns:
                        st.subheader("Carte de prédiction")
                        
                        fig = px.scatter(
                            result_df,
                            x='X', y='Y',
                            color=f'{target}_prédit',
                            size=f'{target}_prédit',
                            color_continuous_scale='Viridis',
                            size_max=15,
                            title=f"Carte de prédiction de {target}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

# Page de détection d'anomalies
elif page == "Détection d'Anomalies":
    st.markdown("<h2 class='sub-header'>Détection d'Anomalies Géochimiques</h2>", unsafe_allow_html=True)
    
    st.info("Cette fonction utilise des algorithmes d'apprentissage non supervisé pour détecter les anomalies géochimiques dans vos données.")
    
    # Téléchargement du fichier
    uploaded_file = st.file_uploader("Télécharger vos données géochimiques (CSV ou Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("Aperçu des données:")
            st.dataframe(df.head())
            
            # Sélection des caractéristiques pour la détection d'anomalies
            st.subheader("Configuration de la détection d'anomalies")
            
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            features = st.multiselect("Sélectionner les éléments géochimiques pour la détection d'anomalies", 
                                     numeric_columns, 
                                     default=numeric_columns[:5] if len(numeric_columns) >= 5 else numeric_columns)
            
            # Sélection de l'algorithme
            algo_options = ["Isolation Forest"]
            if TENSORFLOW_AVAILABLE:
                algo_options.append("Autoencoder")
                
            algo_type = st.radio("Sélectionner l'algorithme de détection d'anomalies", algo_options)
            
            # Paramètres avancés
            with st.expander("Paramètres avancés de détection"):
                if algo_type == "Isolation Forest":
                    contamination = st.slider("Taux de contamination estimé", 0.01, 0.5, 0.1, 0.01)
                    n_estimators = st.slider("Nombre d'estimateurs", 50, 500, 100, 10)
                elif algo_type == "Autoencoder" and TENSORFLOW_AVAILABLE:
                    epochs = st.slider("Nombre d'époques", 10, 200, 50, 5)
                    threshold_percentile = st.slider("Percentile pour le seuil d'erreur", 90, 99, 95, 1)
            
            # Lancement de la détection d'anomalies
            if st.button("Détecter les anomalies"):
                # Préparation des données
                X = df[features].copy()
                
                # Vérification des valeurs manquantes
                if X.isnull().sum().sum() > 0:
                    st.warning("Des valeurs manquantes ont été détectées dans vos données. Elles seront remplacées par la médiane.")
                    X = X.fillna(X.median())
                
                # Scaling des données
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Détection d'anomalies
                with st.spinner('Détection d\'anomalies en cours...'):
                    if algo_type == "Isolation Forest":
                        model = IsolationForest(
                            n_estimators=n_estimators,
                            contamination=contamination,
                            random_state=42
                        )
                        anomaly_scores = model.fit_predict(X_scaled)
                        # Conversion des scores (-1 pour anomalie, 1 pour normal) en binaire (1 pour anomalie, 0 pour normal)
                        anomalies = np.where(anomaly_scores == -1, 1, 0)
                        # Calcul des scores d'anomalie
                        scores = -model.score_samples(X_scaled)
                        
                    elif algo_type == "Autoencoder" and TENSORFLOW_AVAILABLE:
                        # Définition du modèle
                        input_dim = X_scaled.shape[1]
                        encoding_dim = max(1, input_dim // 2)
                        
                        model = Sequential([
                            Dense(encoding_dim * 2, activation='relu', input_shape=(input_dim,)),
                            Dense(encoding_dim, activation='relu'),
                            Dense(encoding_dim * 2, activation='relu'),
                            Dense(input_dim, activation='linear')
                        ])
                        
                        model.compile(optimizer='adam', loss='mse')
                        
                        # Entraînement
                        history = model.fit(
                            X_scaled, X_scaled,
                            epochs=epochs,
                            batch_size=32,
                            validation_split=0.2,
                            verbose=0
                        )
                        
                        # Calcul des erreurs de reconstruction
                        reconstructed = model.predict(X_scaled)
                        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
                        scores = mse
                        
                        # Détermination du seuil d'anomalie
                        threshold = np.percentile(mse, threshold_percentile)
                        anomalies = np.where(mse > threshold, 1, 0)
                
                st.success('Détection d\'anomalies terminée!')
                
                # Ajout des résultats au dataframe
                result_df = df.copy()
                result_df['Anomalie'] = anomalies
                result_df['Score_Anomalie'] = scores
                
                # Affichage des résultats
                st.subheader("Résultats de la détection d'anomalies")
                
                num_anomalies = np.sum(anomalies)
                anomaly_percent = (num_anomalies / len(df)) * 100
                
                col1, col2 = st.columns(2)
                col1.metric("Nombre d'anomalies détectées", f"{num_anomalies}")
                col2.metric("Pourcentage d'anomalies", f"{anomaly_percent:.2f}%")
                
                # Filtrer pour voir uniquement les anomalies
                show_anomalies_only = st.checkbox("Afficher uniquement les anomalies")
                if show_anomalies_only:
                    filtered_df = result_df[result_df['Anomalie'] == 1]
                else:
                    filtered_df = result_df
                
                st.dataframe(filtered_df)
                
                # Téléchargement des résultats
                csv_result = result_df.to_csv(index=False)
                st.download_button(
                    label="Télécharger les résultats (CSV)",
                    data=csv_result,
                    file_name="resultats_anomalies.csv",
                    mime="text/csv"
                )
                
                # Visualisations
                st.subheader("Visualisations des anomalies")
                
                # Distribution des scores d'anomalies
                fig = px.histogram(
                    result_df, 
                    x='Score_Anomalie',
                    color='Anomalie',
                    nbins=50,
                    title="Distribution des scores d'anomalies"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # PCA pour visualisation 2D ou 3D
                if len(features) >= 2:
                    st.subheader("Visualisation par réduction dimensionnelle (PCA)")
                    
                    pca_df = pd.DataFrame(
                        PCA(n_components=min(3, len(features))).fit_transform(X_scaled),
                        columns=[f'PC{i+1}' for i in range(min(3, len(features)))]
                    )
                    pca_df['Anomalie'] = anomalies
                    pca_df['Score_Anomalie'] = result_df['Score_Anomalie']
                    
                    if len(features) >= 3:
                        fig = px.scatter_3d(
                            pca_df, 
                            x='PC1', y='PC2', z='PC3',
                            color='Anomalie',
                            size='Score_Anomalie',
                            opacity=0.7,
                            title="Visualisation 3D des anomalies (PCA)"
                        )
                    else:
                        fig = px.scatter(
                            pca_df, 
                            x='PC1', y='PC2',
                            color='Anomalie',
                            size='Score_Anomalie',
                            opacity=0.7,
                            title="Visualisation 2D des anomalies (PCA)"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Carte des anomalies si coordonnées disponibles
                if 'X' in df.columns and 'Y' in df.columns:
                    st.subheader("Carte des anomalies")
                    
                    fig = px.scatter(
                        result_df,
                        x='X', y='Y',
                        color='Anomalie',
                        size='Score_Anomalie',
                        color_discrete_sequence=['blue', 'red'],
                        size_max=15,
                        title="Carte spatiale des anomalies"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Heatmap de corrélation entre les variables et les anomalies
                st.subheader("Corrélation entre éléments et anomalies")
                
                corr_features = features + ['Anomalie', 'Score_Anomalie']
                corr_matrix = result_df[corr_features].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    title="Matrice de corrélation"
                )
                st.plotly_chart(fig, use_container_width=True)

# Page de recommandation de cibles
elif page == "Recommandation de Cibles":
    st.markdown("<h2 class='sub-header'>Recommandation de Cibles pour Prélèvements</h2>", unsafe_allow_html=True)
    
    st.info("Cette fonction recommande des emplacements optimaux pour de futurs prélèvements d'échantillons, basés sur les données existantes et les zones d'intérêt potentielles.")
    
    # Téléchargement du fichier
    uploaded_file = st.file_uploader("Télécharger vos données géochimiques (CSV ou Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            # Vérification des colonnes de coordonnées
            if 'X' not in df.columns or 'Y' not in df.columns:
                st.error("Vos données doivent contenir des colonnes 'X' et 'Y' pour les coordonnées spatiales.")
            else:
                st.write("Aperçu des données:")
                st.dataframe(df.head())
                
                # Configuration de la recommandation
                st.subheader("Configuration de la recommandation")
                
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                numeric_columns = [col for col in numeric_columns if col not in ['X', 'Y']]
                
                col1, col2 = st.columns(2)
                with col1:
                    target_elements = st.multiselect(
                        "Éléments d'intérêt pour la recommandation", 
                        numeric_columns, 
                        default=numeric_columns[:2] if len(numeric_columns) >= 2 else numeric_columns
                    )
                
                with col2:
                    num_recommendations = st.slider(
                        "Nombre de recommandations", 
                        5, 50, 10, 1
                    )
                
                # Paramètres avancés
                with st.expander("Paramètres avancés"):
                    method = st.radio(
                        "Méthode de recommandation",
                        ["Zones de haute valeur", "Zones sous-échantillonnées", "Hybride"]
                    )
                    
                    if method == "Hybride":
                        exploration_weight = st.slider(
                            "Balance exploration/exploitation", 
                            0.0, 1.0, 0.5, 0.1,
                            help="0 = uniquement zones de haute valeur, 1 = uniquement zones sous-échantillonnées"
                        )
                    
                    min_distance = st.slider(
                        "Distance minimale entre recommandations (m)", 
                        10, 1000, 100, 10
                    )
                
                # Lancement de la recommandation
                if st.button("Générer les recommandations"):
                    # Préparation des données
                    X = df[['X', 'Y']].copy()
                    
                    # Valeurs manquantes dans les coordonnées
                    if X.isnull().sum().sum() > 0:
                        st.error("Des valeurs manquantes ont été détectées dans les coordonnées X, Y. Veuillez les corriger avant de continuer.")
                    else:
                        with st.spinner('Génération des recommandations en cours...'):
                            # Définition de la zone d'étude (limites)
                            x_min, x_max = df['X'].min(), df['X'].max()
                            y_min, y_max = df['Y'].min(), df['Y'].max()
                            
                            # Valeurs pour les éléments d'intérêt
                            element_values = df[target_elements].copy()
                            
                            # Normalisation des valeurs
                            element_values = (element_values - element_values.min()) / (element_values.max() - element_values.min())
                            
                            # Score combiné pour les éléments d'intérêt
                            df['interest_score'] = element_values.mean(axis=1)
                            
                            # Création d'une grille de points potentiels
                            grid_spacing = min_distance / 2
                            x_grid = np.arange(x_min, x_max + grid_spacing, grid_spacing)
                            y_grid = np.arange(y_min, y_max + grid_spacing, grid_spacing)
                            xx, yy = np.meshgrid(x_grid, y_grid)
                            grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
                            
                            # Calcul des distances aux points existants
                            existing_points = df[['X', 'Y']].values
                            
                            # Fonction pour calculer la distance minimale à tous les points existants
                            def min_distance_to_existing(point):
                                distances = np.sqrt(np.sum((existing_points - point)**2, axis=1))
                                return np.min(distances)
                            
                            # Calcul des scores pour chaque point de la grille
                            grid_scores = []
                            for point in grid_points:
                                dist = min_distance_to_existing(point)
                                
                                # Ignorer les points trop proches des existants
                                if dist < min_distance / 2:
                                    grid_scores.append(-np.inf)
                                    continue
                                
                                # Calculer le score d'intérêt interpolé
                                if len(existing_points) > 0:
                                    weights = 1 / (np.sqrt(np.sum((existing_points - point)**2, axis=1)) + 1e-6)
                                    interest_interpolated = np.average(df['interest_score'], weights=weights)
                                else:
                                    interest_interpolated = 0.5
                                
                                # Calculer le score d'exploration (plus grande distance = meilleur)
                                exploration_score = dist / min_distance
                                
                                # Score final selon la méthode choisie
                                if method == "Zones de haute valeur":
                                    score = interest_interpolated
                                elif method == "Zones sous-échantillonnées":
                                    score = exploration_score
                                else:  # Hybride
                                    score = (1 - exploration_weight) * interest_interpolated + exploration_weight * exploration_score
                                
                                grid_scores.append(score)
                            
                            # Conversion en array
                            grid_scores = np.array(grid_scores)
                            
                            # Sélection des meilleures recommandations (greedy)
                            recommendations = []
                            remaining_grid = grid_points.copy()
                            remaining_scores = grid_scores.copy()
                            
                            for _ in range(num_recommendations):
                                if len(remaining_grid) == 0 or np.all(np.isneginf(remaining_scores)):
                                    break
                                
                                # Trouver le meilleur point
                                best_idx = np.argmax(remaining_scores)
                                best_point = remaining_grid[best_idx]
                                recommendations.append(best_point)
                                
                                # Mettre à jour les scores (réduire le score des points proches)
                                for i, point in enumerate(remaining_grid):
                                    dist = np.sqrt(np.sum((point - best_point)**2))
                                    if dist < min_distance:
                                        remaining_scores[i] = -np.inf
                            
                            # Création du dataframe des recommandations
                            rec_df = pd.DataFrame(recommendations, columns=['X', 'Y'])
                            rec_df['Rang'] = range(1, len(rec_df) + 1)
                            
                            # Estimation des valeurs pour les éléments d'intérêt (par interpolation)
                            for element in target_elements:
                                rec_df[f"{element}_estimé"] = np.nan
                                
                                for i, point in rec_df.iterrows():
                                    distances = np.sqrt(np.sum((existing_points - [point['X'], point['Y']])**2, axis=1))
                                    weights = 1 / (distances + 1e-6)
                                    rec_df.loc[i, f"{element}_estimé"] = np.average(df[element], weights=weights)
                        
                        st.success(f'{len(rec_df)} recommandations générées avec succès!')
                        
                        # Affichage des résultats
                        st.subheader("Recommandations de cibles")
                        st.dataframe(rec_df)
                        
                        # Téléchargement des recommandations
                        csv_result = rec_df.to_csv(index=False)
                        st.download_button(
                            label="Télécharger les recommandations (CSV)",
                            data=csv_result,
                            file_name="recommandations_cibles.csv",
                            mime="text/csv"
                        )
                        
                        # Carte des recommandations
                        st.subheader("Carte des recommandations")
                        
                        # Combinaison des données existantes et des recommandations pour la carte
                        df_map = df.copy()
                        df_map['Type'] = 'Existant'
                        
                        rec_df_map = rec_df.copy()
                        rec_df_map['Type'] = 'Recommandation'
                        
                        # Merger les colonnes nécessaires
                        common_cols = ['X', 'Y', 'Type']
                        for element in target_elements:
                            if element in df_map.columns:
                                common_cols.append(element)
                                rec_df_map[element] = rec_df_map[f"{element}_estimé"]
                        
                        map_df = pd.concat([df_map[common_cols], rec_df_map[common_cols]], ignore_index=True)
                        
                        # Création de la carte
                        fig = px.scatter(
                            map_df,
                            x='X', y='Y',
                            color='Type',
                            hover_data=target_elements,
                            color_discrete_sequence=['blue', 'red'],
                            title="Carte des recommandations de cibles"
                        )
                        
                        # Ajout d'annotations pour les rangs des recommandations
                        for i, row in rec_df.iterrows():
                            fig.add_annotation(
                                x=row['X'],
                                y=row['Y'],
                                text=str(row['Rang']),
                                showarrow=True,
                                arrowhead=1
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Carte de chaleur pour faciliter la visualisation
                        st.subheader("Carte de chaleur d'intérêt")
                        
                        # Reconstruction de la grille pour la carte de chaleur
                        grid_scores_2d = np.reshape(grid_scores, (len(y_grid), len(x_grid)))
                        
                        # Filtrer les valeurs -inf
                        grid_scores_2d[np.isneginf(grid_scores_2d)] = np.nan
                        
                        # Création de la carte de chaleur
                        fig = go.Figure(data=go.Heatmap(
                            z=grid_scores_2d,
                            x=x_grid,
                            y=y_grid,
                            colorscale='Viridis',
                            showscale=True
                        ))
                        
                        # Ajout des points existants
                        fig.add_trace(go.Scatter(
                            x=df['X'],
                            y=df['Y'],
                            mode='markers',
                            marker=dict(color='blue', size=8),
                            name='Points existants'
                        ))
                        
                        # Ajout des recommandations
                        fig.add_trace(go.Scatter(
                            x=rec_df['X'],
                            y=rec_df['Y'],
                            mode='markers+text',
                            marker=dict(color='red', size=10, symbol='star'),
                            text=rec_df['Rang'],
                            textposition='top center',
                            name='Recommandations'
                        ))
                        
                        fig.update_layout(
                            title="Carte de chaleur d'intérêt pour l'exploration",
                            xaxis_title="Coordonnée X",
                            yaxis_title="Coordonnée Y"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)