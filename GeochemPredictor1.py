import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from scipy.interpolate import griddata
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import time
import uuid

# Configuration de base de l'application
st.set_page_config(
    page_title="GeoChem Predictor",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thème et styles CSS personnalisés
st.markdown("""
<style>
    /* Variables de couleur */
    :root {
        --primary: #1E88E5;
        --secondary: #26A69A;
        --background: #FAFAFA;
        --accent: #FF7043;
        --text: #37474F;
        --light-grey: #ECEFF1;
        --dark-grey: #607D8B;
    }
    
    /* Styles globaux */
    .app-container {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }
    
    /* En-têtes */
    .title {
        color: var(--primary);
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: var(--dark-grey);
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        color: var(--primary);
        font-weight: 700;
        font-size: 1.8rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--light-grey);
    }
    
    .subsection-header {
        color: var(--secondary);
        font-weight: 600;
        font-size: 1.4rem;
        margin: 1rem 0 0.5rem 0;
    }
    
    /* Cartes et conteneurs */
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .info-card {
        background-color: #E3F2FD;
        border-left: 5px solid var(--primary);
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
    
    .success-card {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
    
    .warning-card {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
    
    /* Tableaux */
    .dataframe-container {
        border-radius: 5px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    /* Boutons */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1976D2;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Indicateurs de progression */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--dark-grey);
    }
    
    /* Logo animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .logo-animation {
        animation: pulse 2s infinite ease-in-out;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--light-grey);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--dark-grey);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #455A64;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div[data-baseweb="select"] > div {
        background-color: white;
        border-radius: 5px;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        color: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# Fonctions utilitaires
def generate_logo():
    """Génère un logo symbolique pour l'application."""
    fig = go.Figure()
    
    # Cercle principal
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(
            color='#1E88E5',
            size=25,
            line=dict(
                color='white',
                width=2
            )
        ),
        showlegend=False
    ))
    
    # Points représentant des éléments géochimiques
    x = np.random.normal(0, 0.8, 20)
    y = np.random.normal(0, 0.8, 20)
    sizes = np.random.uniform(5, 15, 20)
    colors = np.random.uniform(0, 1, 20)
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(
            color=colors,
            colorscale='Viridis',
            size=sizes,
            line=dict(width=1, color='white')
        ),
        showlegend=False
    ))
    
    # Mise en page du logo
    fig.update_layout(
        width=100,
        height=100,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            range=[-1.2, 1.2]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            range=[-1.2, 1.2]
        )
    )
    
    return fig

@st.cache_data
def load_data(file):
    """Charge les données depuis un fichier CSV ou Excel."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return None, "Format de fichier non supporté. Veuillez télécharger un fichier CSV ou Excel."
        
        # Vérifications de base sur les données
        if df.empty:
            return None, "Le fichier est vide."
        
        # Statistiques sur les données
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=['float64', 'int64']).columns),
            "missing_values": df.isnull().sum().sum(),
            "missing_percent": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
        }
        
        return df, stats
    except Exception as e:
        return None, f"Erreur lors du chargement des données: {e}"

def get_data_summary(df):
    """Génère un résumé statistique des données."""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    summary = pd.DataFrame({
        'Min': numeric_df.min(),
        'Max': numeric_df.max(),
        'Moyenne': numeric_df.mean(),
        'Médiane': numeric_df.median(),
        'Écart-type': numeric_df.std(),
        'Valeurs manquantes': numeric_df.isnull().sum(),
        '% Valeurs manquantes': (numeric_df.isnull().sum() / len(df) * 100).round(2)
    })
    
    return summary

def plot_correlation_matrix(df, columns):
    """Génère une matrice de corrélation pour les colonnes sélectionnées."""
    corr_matrix = df[columns].corr()
    
    fig = px.imshow(
        corr_matrix,
        x=columns,
        y=columns,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title="Matrice de Corrélation"
    )
    
    fig.update_layout(
        height=600,
        width=700,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        coloraxis_colorbar=dict(
            title="Coefficient de Corrélation",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=400,
            yanchor="top", y=1,
            ticks="outside"
        )
    )
    
    return fig

def plot_histogram(df, column, bins=30):
    """Génère un histogramme pour une colonne spécifique."""
    fig = px.histogram(
        df, x=column,
        nbins=bins,
        marginal="box",
        title=f"Distribution de {column}",
        color_discrete_sequence=['#1E88E5']
    )
    
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Fréquence",
        bargap=0.1
    )
    
    return fig

def plot_scatter_map(df, x_col, y_col, color_col=None, size_col=None, hover_cols=None):
    """Génère une carte des points d'échantillonnage."""
    if hover_cols is None:
        hover_cols = []
    
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        size=size_col,
        hover_data=hover_cols,
        title="Carte des Échantillons",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=600
    )
    
    return fig

def train_model(X, y, model_type, params=None):
    """Entraîne un modèle de régression avec les paramètres spécifiés."""
    if params is None:
        params = {}
    
    if model_type == "XGBoost":
        model = xgb.XGBRegressor(objective='reg:squarederror', **params)
    elif model_type == "LightGBM":
        model = lgb.LGBMRegressor(**params)
    elif model_type == "RandomForest":
        model = RandomForestRegressor(**params)
    elif model_type == "GradientBoosting":
        model = GradientBoostingRegressor(**params)
    elif model_type == "KNN":
        model = KNeighborsRegressor(**params)
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")
    
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """Évalue un modèle de régression et renvoie les métriques de performance."""
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "Prédictions": y_pred
    }

def cross_validate_model(model, X, y, cv=5):
    """Effectue une validation croisée et renvoie les métriques moyennes."""
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    cv_rmse = -cv_scores
    
    cv_r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    return {
        "CV RMSE": cv_rmse.mean(),
        "CV RMSE Std": cv_rmse.std(),
        "CV R²": cv_r2.mean(),
        "CV R² Std": cv_r2.std()
    }

def detect_anomalies(X, contamination=0.1, method="IsolationForest", **kwargs):
    """Détecte les anomalies dans les données en utilisant différentes méthodes."""
    if method == "IsolationForest":
        model = IsolationForest(contamination=contamination, **kwargs)
        scores = model.fit_predict(X)
        # Conversion des scores (-1 pour anomalie, 1 pour normal) en binaire (1 pour anomalie, 0 pour normal)
        anomalies = np.where(scores == -1, 1, 0)
        anomaly_scores = -model.score_samples(X)
    
    elif method == "DBSCAN":
        model = DBSCAN(**kwargs)
        labels = model.fit_predict(X)
        # Les points avec label -1 sont des anomalies
        anomalies = np.where(labels == -1, 1, 0)
        
        # Calcul des scores d'anomalie basés sur la distance au voisin le plus proche
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        anomaly_scores = distances[:, 1]  # Distance au voisin le plus proche
    
    elif method == "KMeans":
        model = KMeans(**kwargs)
        labels = model.fit_predict(X)
        distances = np.min(np.sqrt(np.sum((X - model.cluster_centers_[labels])**2, axis=1)), axis=0)
        
        # Définir un seuil pour les anomalies (par exemple, les 10% points les plus éloignés)
        threshold = np.percentile(distances, 100 - contamination * 100)
        anomalies = np.where(distances > threshold, 1, 0)
        anomaly_scores = distances
    
    else:
        raise ValueError(f"Méthode de détection d'anomalies non supportée: {method}")
    
    return anomalies, anomaly_scores

def generate_sampling_recommendations(df, target_cols, num_recommendations=10, min_distance=100, method="hybrid", exploration_weight=0.5):
    """Génère des recommandations pour de nouveaux points d'échantillonnage."""
    if 'X' not in df.columns or 'Y' not in df.columns:
        return None, "Les colonnes X et Y sont requises pour générer des recommandations."
    
    # Définition de la zone d'étude
    x_min, x_max = df['X'].min(), df['X'].max()
    y_min, y_max = df['Y'].min(), df['Y'].max()
    
    # Valeurs d'intérêt
    target_values = df[target_cols].copy()
    
    # Normalisation
    target_values = (target_values - target_values.min()) / (target_values.max() - target_values.min())
    
    # Score combiné
    df['interest_score'] = target_values.mean(axis=1)
    
    # Création d'une grille de points potentiels
    grid_spacing = min_distance / 2
    x_grid = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_grid = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # Points existants
    existing_points = df[['X', 'Y']].values
    
    # Fonction pour calculer la distance minimale
    def min_distance_to_existing(point):
        distances = np.sqrt(np.sum((existing_points - point)**2, axis=1))
        return np.min(distances)
    
    # Calcul des scores
    grid_scores = []
    for point in grid_points:
        dist = min_distance_to_existing(point)
        
        # Ignorer les points trop proches
        if dist < min_distance / 2:
            grid_scores.append(-np.inf)
            continue
        
        # Score d'intérêt
        if len(existing_points) > 0:
            weights = 1 / (np.sqrt(np.sum((existing_points - point)**2, axis=1)) + 1e-6)
            interest_interpolated = np.average(df['interest_score'], weights=weights)
        else:
            interest_interpolated = 0.5
        
        # Score d'exploration
        exploration_score = dist / min_distance
        
        # Score final
        if method == "value":
            score = interest_interpolated
        elif method == "exploration":
            score = exploration_score
        else:  # hybrid
            score = (1 - exploration_weight) * interest_interpolated + exploration_weight * exploration_score
        
        grid_scores.append(score)
    
    # Conversion en array
    grid_scores = np.array(grid_scores)
    
    # Sélection des recommandations
    recommendations = []
    remaining_grid = grid_points.copy()
    remaining_scores = grid_scores.copy()
    
    for _ in range(num_recommendations):
        if len(remaining_grid) == 0 or np.all(np.isneginf(remaining_scores)):
            break
        
        # Meilleur point
        best_idx = np.argmax(remaining_scores)
        best_point = remaining_grid[best_idx]
        recommendations.append(best_point)
        
        # Mise à jour des scores
        for i, point in enumerate(remaining_grid):
            dist = np.sqrt(np.sum((point - best_point)**2))
            if dist < min_distance:
                remaining_scores[i] = -np.inf
    
    # Création du DataFrame
    rec_df = pd.DataFrame(recommendations, columns=['X', 'Y'])
    rec_df['Rang'] = range(1, len(rec_df) + 1)
    
    # Estimation des valeurs pour les éléments d'intérêt
    for element in target_cols:
        rec_df[f"{element}_estimé"] = np.nan
        
        for i, point in rec_df.iterrows():
            distances = np.sqrt(np.sum((existing_points - [point['X'], point['Y']])**2, axis=1))
            weights = 1 / (distances + 1e-6)
            rec_df.loc[i, f"{element}_estimé"] = np.average(df[element], weights=weights)
    
    # Grille pour la heatmap
    grid_scores_2d = np.full((len(y_grid), len(x_grid)), np.nan)
    valid_indices = ~np.isinf(grid_scores)
    grid_scores_2d.flat[valid_indices] = grid_scores[valid_indices]
    
    return rec_df, {
        "grid_x": x_grid,
        "grid_y": y_grid,
        "grid_scores": grid_scores_2d
    }

def interpolate_values(df, value_col, grid_size=100, method='linear'):
    """Crée une grille d'interpolation pour visualiser la distribution spatiale des valeurs."""
    if 'X' not in df.columns or 'Y' not in df.columns:
        return None
    
    # Définition de la grille
    x_min, x_max = df['X'].min(), df['X'].max()
    y_min, y_max = df['Y'].min(), df['Y'].max()
    
    # Extension de la grille de 5% pour une meilleure visualisation
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_min -= 0.05 * x_range
    x_max += 0.05 * x_range
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range
    
    # Création de la grille
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Points et valeurs pour l'interpolation
    points = df[['X', 'Y']].values
    values = df[value_col].values
    
    # Interpolation
    zi = griddata(points, values, (xi_grid, yi_grid), method=method)
    
    return {
        "x": xi,
        "y": yi,
        "z": zi
    }

def plot_contour_map(interp_data, df, value_col, colorscale='Viridis'):
    """Génère une carte de contour interpolée avec les points d'échantillonnage."""
    # Création de la figure de base
    fig = go.Figure()
    
    # Ajout du contour
    fig.add_trace(go.Contour(
        z=interp_data["z"],
        x=interp_data["x"],
        y=interp_data["y"],
        colorscale=colorscale,
        colorbar=dict(title=value_col),
        line=dict(width=0.5),
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color='white')
        )
    ))
    
    # Ajout des points d'échantillonnage
    fig.add_trace(go.Scatter(
        x=df['X'],
        y=df['Y'],
        mode='markers',
        marker=dict(
            color=df[value_col],
            colorscale=colorscale,
            size=8,
            line=dict(color='black', width=1),
            showscale=False
        ),
        name='Points d\'échantillonnage'
    ))
    
    # Mise en page
    fig.update_layout(
        title=f"Carte de Contour de {value_col}",
        xaxis_title='X',
        yaxis_title='Y',
        height=600,
        width=800
    )
    
    return fig

def plot_3d_surface(interp_data, value_col, colorscale='Viridis'):
    """Génère une surface 3D pour visualiser la distribution spatiale des valeurs."""
    fig = go.Figure(data=[go.Surface(
        z=interp_data["z"],
        x=interp_data["x"],
        y=interp_data["y"],
        colorscale=colorscale,
        colorbar=dict(title=value_col)
    )])
    
    fig.update_layout(
        title=f"Surface 3D de {value_col}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title=value_col,
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        height=700,
        width=800
    )
    
    return fig

def create_voronoi_diagram(df, value_col, colorscale='Viridis'):
    """Crée un diagramme de Voronoi coloré selon les valeurs d'une colonne."""
    # Points pour le diagramme de Voronoi
    points = df[['X', 'Y']].values
    
    # Calcul du diagramme de Voronoi
    vor = Voronoi(points)
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Tracer le diagramme de Voronoi
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', line_width=1, line_alpha=0.5, point_size=0)
    
    # Colorier les points selon leurs valeurs
    scatter = ax.scatter(points[:,0], points[:,1], c=df[value_col], cmap=colorscale, s=50, alpha=0.8, edgecolors='black')
    
    # Ajouter une barre de couleur
    plt.colorbar(scatter, label=value_col)
    
    # Ajustements
    ax.set_title(f"Diagramme de Voronoi - {value_col}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Convertir la figure en image pour Streamlit
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return f'<img src="data:image/png;base64,{img_str}" alt="Diagramme de Voronoi">'

# Personnaliser le sidebar
def customize_sidebar():
    logo_fig = generate_logo()
    
    with st.sidebar:
        # Logo et titre
        col1, col2 = st.columns([1, 3])
        with col1:
            st.plotly_chart(logo_fig, use_container_width=True)
        with col2:
            st.markdown("<h1 style='color:#1E88E5; margin-bottom:0; font-size:1.8em;'>GeoChem</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='color:#26A69A; margin-top:0; font-size:1.4em;'>Predictor</h2>", unsafe_allow_html=True)
        
        st.markdown("""---""")
        
        # Navigation
        st.subheader("🧭 Navigation")
        page = st.radio(
            label="",
            options=["📊 Tableau de Bord", "🔍 Analyse Exploratoire", "🔮 Prédiction de Minéralisation", 
                     "🛑 Détection d'Anomalies", "🎯 Recommandation de Cibles"],
            label_visibility="collapsed"
        )
        
        st.markdown("""---""")
        
        # Informations
        with st.expander("ℹ️ À propos"):
            st.markdown("""
            **GeoChem Predictor** est un outil d'analyse géochimique automatisé développé par Didier Ouedraogo, P.Geo.
            
            Cette application permet d'analyser des données géochimiques, de prédire la minéralisation, de détecter des anomalies et de recommander des cibles pour de nouveaux prélèvements.
            
            Version: 2.0.1
            """)
        
        # Crédit
        st.markdown("""
        <div style='position:fixed; bottom:10px; left:20px; font-size:0.8em; color:#607D8B;'>
            Développé par<br/>Didier Ouedraogo, P.Geo
        </div>
        """, unsafe_allow_html=True)
    
    return page

# Fonction pour enregistrer l'état de session
def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_stats' not in st.session_state:
        st.session_state.data_stats = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_features' not in st.session_state:
        st.session_state.model_features = None
    if 'model_target' not in st.session_state:
        st.session_state.model_target = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

# Composant pour télécharger des données
def upload_data_component():
    st.markdown("<h2 class='section-header'>📤 Charger des Données</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Télécharger un fichier CSV ou Excel contenant vos données géochimiques",
            type=["csv", "xlsx", "xls"],
            help="Vos données doivent contenir des coordonnées (X, Y) et des valeurs géochimiques pour différents éléments"
        )
    
    with col2:
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        use_example = st.checkbox("Utiliser des données d'exemple", value=False)
    
    if use_example:
        # Générer des données d'exemple
        np.random.seed(42)
        n_samples = 100
        x = np.random.uniform(low=350000, high=355000, size=n_samples)
        y = np.random.uniform(low=7650000, high=7655000, size=n_samples)
        
        # Simuler une zone minéralisée au centre
        center_x, center_y = 352500, 7652500
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.max(distance)
        
        # Valeurs d'or qui diminuent avec la distance du centre
        au_base = 10 * np.exp(-3 * distance / max_dist)
        au = au_base + np.random.lognormal(mean=0, sigma=0.5, size=n_samples)
        
        # Valeurs d'arsenic corrélées avec l'or
        as_base = 50 * np.exp(-2 * distance / max_dist)
        as_values = as_base + np.random.lognormal(mean=0, sigma=0.7, size=n_samples)
        
        # Valeurs de cuivre avec une autre zone d'intérêt
        center2_x, center2_y = 354000, 7653500
        distance2 = np.sqrt((x - center2_x)**2 + (y - center2_y)**2)
        cu_base = 200 * np.exp(-4 * distance2 / max_dist)
        cu = cu_base + np.random.lognormal(mean=0, sigma=0.6, size=n_samples)
        
        # Valeurs d'antimoine faiblement corrélées avec l'or
        sb_base = 2 * np.exp(-2.5 * distance / max_dist)
        sb = sb_base + np.random.lognormal(mean=0, sigma=0.8, size=n_samples)
        
        # Zinc aléatoire (non corrélé)
        zn = np.random.lognormal(mean=3.5, sigma=0.4, size=n_samples)
        
        # Créer le DataFrame
        example_data = pd.DataFrame({
            'X': x,
            'Y': y,
            'Au_ppb': au,
            'As_ppm': as_values,
            'Cu_ppm': cu,
            'Sb_ppm': sb,
            'Zn_ppm': zn
        })
        
        st.session_state.data = example_data
        st.session_state.data_stats = {
            "rows": len(example_data),
            "columns": len(example_data.columns),
            "numeric_columns": len(example_data.select_dtypes(include=['float64', 'int64']).columns),
            "missing_values": 0,
            "missing_percent": 0
        }
        st.session_state.uploaded_file_name = "donnees_exemple.csv"
        
        st.success("✅ Données d'exemple chargées avec succès!")
        
    elif uploaded_file is not None:
        data, result = load_data(uploaded_file)
        
        if isinstance(result, dict):
            # Chargement réussi
            st.session_state.data = data
            st.session_state.data_stats = result
            st.session_state.uploaded_file_name = uploaded_file.name
            
            st.success(f"✅ Fichier '{uploaded_file.name}' chargé avec succès!")
        else:
            # Erreur lors du chargement
            st.error(result)
    
    # Afficher un aperçu des données si disponibles
    if st.session_state.data is not None:
        with st.expander("📋 Aperçu des Données", expanded=True):
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
            
            # Afficher les statistiques du jeu de données
            stats = st.session_state.data_stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Échantillons", stats["rows"])
            
            with col2:
                st.metric("Variables", stats["columns"])
            
            with col3:
                st.metric("Variables numériques", stats["numeric_columns"])
            
            with col4:
                st.metric("Valeurs manquantes", f"{stats['missing_percent']}%")
                
        # Option pour réinitialiser les données
        if st.button("🗑️ Réinitialiser les données"):
            st.session_state.data = None
            st.session_state.data_stats = None
            st.session_state.uploaded_file_name = None
            st.session_state.model = None
            st.session_state.model_features = None
            st.session_state.model_target = None
            st.session_state.model_metrics = None
            st.rerun()

# Interface principale
def main():
    # Initialiser l'état de la session
    initialize_session_state()
    
    # Personnaliser la barre latérale et obtenir la page sélectionnée
    page = customize_sidebar()
    
    # Interface principale
    if page == "📊 Tableau de Bord":
        show_dashboard()
    elif page == "🔍 Analyse Exploratoire":
        show_exploratory_analysis()
    elif page == "🔮 Prédiction de Minéralisation":
        show_prediction()
    elif page == "🛑 Détection d'Anomalies":
        show_anomaly_detection()
    elif page == "🎯 Recommandation de Cibles":
        show_recommendation()

# Pages de l'application
def show_dashboard():
    st.markdown("<h1 class='title'>Tableau de Bord</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Vue d'ensemble des données géochimiques et outils d'analyse</p>", unsafe_allow_html=True)
    
    # Composant pour télécharger des données
    upload_data_component()
    
    # Si des données sont chargées, afficher les analyses de base
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Résumé statistique
        st.markdown("<h2 class='section-header'>📊 Résumé Statistique</h2>", unsafe_allow_html=True)
        
        # Obtenir le résumé des données
        summary = get_data_summary(df)
        
        # Afficher le résumé dans un tableau interactif
        st.dataframe(summary, use_container_width=True)
        
        # Visualisations rapides
        st.markdown("<h2 class='section-header'>🔍 Visualisations Rapides</h2>", unsafe_allow_html=True)
        
        # Obtenir les colonnes numériques
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Sélection de visualisation
        viz_type = st.selectbox(
            "Sélectionner le type de visualisation",
            ["Carte des échantillons", "Matrice de corrélation", "Histogrammes", "Boîtes à moustaches"]
        )
        
        if viz_type == "Carte des échantillons":
            if 'X' in df.columns and 'Y' in df.columns:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    color_col = st.selectbox("Colorer par", [None] + numeric_cols)
                    size_col = st.selectbox("Dimensionner par", [None] + numeric_cols)
                    
                with col1:
                    fig = plot_scatter_map(df, 'X', 'Y', color_col, size_col)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Les colonnes X et Y sont nécessaires pour afficher la carte des échantillons.")
        
        elif viz_type == "Matrice de corrélation":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            # Sélection des variables pour la matrice de corrélation
            selected_cols = st.multiselect(
                "Sélectionner les variables pour la matrice de corrélation",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if len(selected_cols) > 1:
                fig = plot_correlation_matrix(df, selected_cols)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sélectionnez au moins deux variables pour afficher la matrice de corrélation.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif viz_type == "Histogrammes":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                hist_col = st.selectbox("Variable à visualiser", numeric_cols)
                bins = st.slider("Nombre de bins", 5, 100, 30)
                
            with col1:
                fig = plot_histogram(df, hist_col, bins=bins)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif viz_type == "Boîtes à moustaches":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            selected_cols = st.multiselect(
                "Sélectionner les variables pour les boîtes à moustaches",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if selected_cols:
                fig = px.box(df, y=selected_cols, title="Distribution des Variables")
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Résumé des éléments-clés
        if 'X' in df.columns and 'Y' in df.columns and len(numeric_cols) > 2:
            st.markdown("<h2 class='section-header'>💡 Aperçu Rapide</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("🔝 Valeurs maximales")
                
                # Trouver l'emplacement des valeurs maximales pour chaque variable
                max_values = pd.DataFrame(columns=['Variable', 'Valeur Max', 'X', 'Y'])
                
                for col in numeric_cols[:5]:  # Limiter à 5 variables pour la lisibilité
                    if col not in ['X', 'Y']:
                        max_idx = df[col].idxmax()
                        max_values = pd.concat([max_values, pd.DataFrame({
                            'Variable': [col],
                            'Valeur Max': [df.loc[max_idx, col]],
                            'X': [df.loc[max_idx, 'X']],
                            'Y': [df.loc[max_idx, 'Y']]
                        })])
                
                st.dataframe(max_values, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("🔄 Corrélations principales")
                
                # Calculer les corrélations
                corr_matrix = df[numeric_cols].corr()
                
                # Obtenir les paires de variables avec les corrélations les plus fortes (en valeur absolue)
                corr_pairs = []
                
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        col1_name = numeric_cols[i]
                        col2_name = numeric_cols[j]
                        
                        if col1_name != 'X' and col1_name != 'Y' and col2_name != 'X' and col2_name != 'Y':
                            corr_value = corr_matrix.loc[col1_name, col2_name]
                            corr_pairs.append((col1_name, col2_name, corr_value))
                
                # Trier par valeur absolue de corrélation
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Afficher les 5 premières paires
                corr_df = pd.DataFrame(
                    [(f"{pair[0]} - {pair[1]}", f"{pair[2]:.3f}") for pair in corr_pairs[:5]],
                    columns=["Paire de variables", "Corrélation"]
                )
                
                st.dataframe(corr_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Si aucune donnée n'est chargée, afficher un message et des exemples
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("""
        ### 👋 Bienvenue dans GeoChem Predictor!
        
        Pour commencer, téléchargez vos données géochimiques ou utilisez les données d'exemple.
        
        Cette application vous permet de:
        - Analyser vos données géochimiques
        - Prédire la minéralisation
        - Détecter des anomalies
        - Générer des recommandations pour de futurs échantillonnages
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Afficher quelques exemples de ce que l'application peut faire
        st.markdown("<h2 class='section-header'>✨ Fonctionnalités</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🔮 Prédiction")
            st.image("https://raw.githubusercontent.com/streamlit/streamlit/master/examples/data/bike_rentals_visualization.jpg")
            st.markdown("""
            Prédisez la minéralisation à partir de données géochimiques grâce à des modèles de régression avancés.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🛑 Anomalies")
            st.image("https://raw.githubusercontent.com/streamlit/streamlit/master/examples/data/chart_data_anomalies.jpg")
            st.markdown("""
            Identifiez automatiquement les anomalies géochimiques à l'aide de l'apprentissage non supervisé.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🎯 Recommandations")
            st.image("https://raw.githubusercontent.com/streamlit/streamlit/master/examples/data/map_data.jpg")
            st.markdown("""
            Obtenez des suggestions pour les emplacements optimaux des prochains prélèvements.
            """)
            st.markdown("</div>", unsafe_allow_html=True)

def show_exploratory_analysis():
    st.markdown("<h1 class='title'>Analyse Exploratoire</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Explorez et visualisez vos données géochimiques en profondeur</p>", unsafe_allow_html=True)
    
    # Composant pour télécharger des données si nécessaire
    if st.session_state.data is None:
        upload_data_component()
    else:
        # Bouton pour changer de jeu de données
        if st.button("📤 Changer de jeu de données"):
            upload_data_component()
    
    # Si des données sont chargées, afficher les outils d'analyse exploratoire
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Informations sur le jeu de données
        st.markdown(f"<div class='info-card'>Jeu de données actuel: <strong>{st.session_state.uploaded_file_name}</strong> | {len(df)} échantillons | {len(df.columns)} variables</div>", unsafe_allow_html=True)
        
        # Navigation par onglets pour différentes visualisations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "🗺️ Cartes", "🔄 Relations", "📈 Tendances"])
        
        # Obtenir les colonnes numériques
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        with tab1:
            st.markdown("<h2 class='subsection-header'>Distribution des Variables</h2>", unsafe_allow_html=True)
            
            # Sélection de variable et type de graphique
            col1, col2 = st.columns([1, 3])
            
            with col1:
                dist_var = st.selectbox("Sélectionnez une variable", numeric_cols)
                dist_type = st.radio("Type de visualisation", ["Histogramme", "Boîte à moustaches", "Violin Plot", "ECDF"])
                
                # Options avancées
                with st.expander("Options avancées"):
                    if dist_type == "Histogramme":
                        bins = st.slider("Nombre de bins", 5, 100, 30)
                        use_log = st.checkbox("Échelle logarithmique", value=False)
                    
                    norm_test = st.checkbox("Test de normalité", value=False)
            
            with col2:
                if dist_type == "Histogramme":
                    fig = px.histogram(
                        df, x=dist_var,
                        nbins=bins,
                        marginal="box",
                        title=f"Distribution de {dist_var}",
                        log_x=use_log,
                        color_discrete_sequence=['#1E88E5']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif dist_type == "Boîte à moustaches":
                    fig = px.box(
                        df, y=dist_var,
                        title=f"Boîte à moustaches de {dist_var}",
                        points="all",
                        color_discrete_sequence=['#1E88E5']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif dist_type == "Violin Plot":
                    fig = px.violin(
                        df, y=dist_var,
                        title=f"Violin Plot de {dist_var}",
                        box=True,
                        points="all",
                        color_discrete_sequence=['#1E88E5']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif dist_type == "ECDF":
                    # Fonction de distribution cumulative empirique
                    sorted_data = np.sort(df[dist_var].dropna())
                    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    
                    fig = px.line(
                        x=sorted_data, y=y,
                        title=f"Distribution cumulative de {dist_var}",
                        labels={"x": dist_var, "y": "Probabilité cumulée"}
                    )
                    
                    fig.update_traces(mode='lines', line=dict(color='#1E88E5', width=2))
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tests statistiques
            if norm_test:
                from scipy import stats
                
                # Test de normalité (Shapiro-Wilk pour petits échantillons, D'Agostino-Pearson pour grands échantillons)
                data_no_na = df[dist_var].dropna()
                
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Tests de Normalité")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(data_no_na) <= 5000:  # Shapiro-Wilk est limité à ~5000 échantillons
                        shapiro_stat, shapiro_p = stats.shapiro(data_no_na)
                        st.metric("Test de Shapiro-Wilk (p-value)", f"{shapiro_p:.6f}")
                        
                        if shapiro_p < 0.05:
                            st.markdown("❌ Les données ne suivent pas une distribution normale.")
                        else:
                            st.markdown("✅ Les données suivent une distribution normale.")
                
                with col2:
                    # Test D'Agostino-Pearson
                    k2, p_value = stats.normaltest(data_no_na)
                    st.metric("Test D'Agostino-Pearson (p-value)", f"{p_value:.6f}")
                    
                    if p_value < 0.05:
                        st.markdown("❌ Les données ne suivent pas une distribution normale.")
                    else:
                        st.markdown("✅ Les données suivent une distribution normale.")
                
                # Statistiques descriptives
                st.markdown("#### Statistiques descriptives")
                
                stats_df = pd.DataFrame({
                    'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max', 'Q1 (25%)', 'Q3 (75%)', 'Skewness', 'Kurtosis'],
                    'Valeur': [
                        f"{data_no_na.mean():.4f}",
                        f"{data_no_na.median():.4f}",
                        f"{data_no_na.std():.4f}",
                        f"{data_no_na.min():.4f}",
                        f"{data_no_na.max():.4f}",
                        f"{data_no_na.quantile(0.25):.4f}",
                        f"{data_no_na.quantile(0.75):.4f}",
                        f"{stats.skew(data_no_na):.4f}",
                        f"{stats.kurtosis(data_no_na):.4f}"
                    ]
                })
                
                st.dataframe(stats_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            if 'X' in df.columns and 'Y' in df.columns:
                st.markdown("<h2 class='subsection-header'>Cartes et Visualisations Spatiales</h2>", unsafe_allow_html=True)
                
                # Sélection du type de carte
                map_type = st.selectbox(
                    "Type de carte",
                    ["Carte des points", "Carte de chaleur", "Carte de contour", "Surface 3D", "Diagramme de Voronoi"]
                )
                
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    # Options communes à toutes les cartes
                    map_var = st.selectbox("Variable à visualiser", [col for col in numeric_cols if col not in ['X', 'Y']])
                    colorscale = st.selectbox("Palette de couleurs", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "RdBu", "Spectral"])
                    
                    # Options spécifiques au type de carte
                    if map_type in ["Carte de contour", "Surface 3D"]:
                        interp_method = st.selectbox("Méthode d'interpolation", ["linear", "cubic", "nearest"])
                    
                    if map_type == "Carte de chaleur":
                        resolution = st.slider("Résolution", 50, 200, 100)
                
                with col1:
                    if map_type == "Carte des points":
                        fig = px.scatter(
                            df, x='X', y='Y',
                            color=map_var,
                            size=map_var,
                            hover_data=[map_var],
                            color_continuous_scale=colorscale,
                            title=f"Carte des échantillons - {map_var}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif map_type == "Carte de chaleur":
                        # Créer une grille régulière
                        x_range = np.linspace(df['X'].min(), df['X'].max(), resolution)
                        y_range = np.linspace(df['Y'].min(), df['Y'].max(), resolution)
                        xx, yy = np.meshgrid(x_range, y_range)
                        
                        # Interpolation
                        from scipy.interpolate import griddata
                        z = griddata((df['X'], df['Y']), df[map_var], (xx, yy), method='linear')
                        
                        # Créer la carte de chaleur
                        fig = go.Figure(data=go.Heatmap(
                            z=z,
                            x=x_range,
                            y=y_range,
                            colorscale=colorscale,
                            colorbar=dict(title=map_var)
                        ))
                        
                        # Ajouter les points d'échantillonnage
                        fig.add_trace(go.Scatter(
                            x=df['X'],
                            y=df['Y'],
                            mode='markers',
                            marker=dict(color='black', size=4),
                            showlegend=False
                        ))
                        
                        fig.update_layout(
                            title=f"Carte de chaleur - {map_var}",
                            xaxis_title='X',
                            yaxis_title='Y'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif map_type == "Carte de contour":
                        # Interpolation
                        interp_data = interpolate_values(df, map_var, grid_size=100, method=interp_method)
                        
                        if interp_data:
                            fig = plot_contour_map(interp_data, df, map_var, colorscale=colorscale)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif map_type == "Surface 3D":
                        # Interpolation
                        interp_data = interpolate_values(df, map_var, grid_size=100, method=interp_method)
                        
                        if interp_data:
                            fig = plot_3d_surface(interp_data, map_var, colorscale=colorscale)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif map_type == "Diagramme de Voronoi":
                        # Générer le diagramme de Voronoi
                        voronoi_html = create_voronoi_diagram(df, map_var, colorscale=colorscale.lower())
                        st.markdown(voronoi_html, unsafe_allow_html=True)
            else:
                st.warning("Les colonnes X et Y sont nécessaires pour les visualisations spatiales.")
        
        with tab3:
            st.markdown("<h2 class='subsection-header'>Relations entre Variables</h2>", unsafe_allow_html=True)
            
            # Type de visualisation
            relation_type = st.selectbox(
                "Type de visualisation",
                ["Nuage de points", "Matrice de corrélation", "Pairplot", "Heatmap"]
            )
            
            if relation_type == "Nuage de points":
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    x_var = st.selectbox("Variable X", numeric_cols, index=0)
                    y_var = st.selectbox("Variable Y", numeric_cols, index=min(1, len(numeric_cols)-1))
                    
                    color_var = st.selectbox("Colorer par", [None] + numeric_cols)
                    trendline = st.checkbox("Ajouter une ligne de tendance", value=True)
                
                with col1:
                    if trendline:
                        fig = px.scatter(
                            df, x=x_var, y=y_var,
                            color=color_var,
                            trendline="ols",
                            trendline_color_override="red",
                            title=f"Relation entre {x_var} et {y_var}"
                        )
                    else:
                        fig = px.scatter(
                            df, x=x_var, y=y_var,
                            color=color_var,
                            title=f"Relation entre {x_var} et {y_var}"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques de corrélation
                if trendline:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("#### Analyse de corrélation")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pearson_r = df[x_var].corr(df[y_var], method='pearson')
                        st.metric("Coefficient de Pearson", f"{pearson_r:.4f}")
                    
                    with col2:
                        spearman_r = df[x_var].corr(df[y_var], method='spearman')
                        st.metric("Rho de Spearman", f"{spearman_r:.4f}")
                    
                    with col3:
                        kendall_tau = df[x_var].corr(df[y_var], method='kendall')
                        st.metric("Tau de Kendall", f"{kendall_tau:.4f}")
                    
                    # Interprétation automatique
                    st.markdown("#### Interprétation")
                    
                    abs_r = abs(pearson_r)
                    if abs_r < 0.3:
                        st.markdown("📊 **Corrélation faible** entre les variables.")
                    elif abs_r < 0.7:
                        st.markdown("📊 **Corrélation modérée** entre les variables.")
                    else:
                        st.markdown("📊 **Corrélation forte** entre les variables.")
                    
                    if pearson_r > 0:
                        st.markdown("📈 La relation est **positive** : quand une variable augmente, l'autre tend à augmenter aussi.")
                    else:
                        st.markdown("📉 La relation est **négative** : quand une variable augmente, l'autre tend à diminuer.")
                    
                    # Différence Pearson vs Spearman
                    diff = abs(pearson_r) - abs(spearman_r)
                    if abs(diff) > 0.1:
                        st.markdown("⚠️ **Écart notable** entre les corrélations de Pearson et de Spearman, suggérant une relation non linéaire ou la présence de valeurs aberrantes.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            elif relation_type == "Matrice de corrélation":
                # Sélection des variables
                selected_vars = st.multiselect(
                    "Sélectionner les variables pour la matrice de corrélation",
                    numeric_cols,
                    default=numeric_cols[:min(6, len(numeric_cols))]
                )
                
                if len(selected_vars) > 1:
                    corr_method = st.radio("Méthode de corrélation", ["pearson", "spearman", "kendall"], horizontal=True)
                    
                    # Calculer la matrice de corrélation
                    corr_matrix = df[selected_vars].corr(method=corr_method)
                    
                    # Créer la heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        title=f"Matrice de corrélation ({corr_method.capitalize()})"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Analyse des corrélations principales
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("#### Corrélations principales")
                    
                    # Mettre en forme les résultats pour une meilleure lisibilité
                    corr_pairs = []
                    
                    for i in range(len(selected_vars)):
                        for j in range(i+1, len(selected_vars)):
                            var1 = selected_vars[i]
                            var2 = selected_vars[j]
                            corr_value = corr_matrix.loc[var1, var2]
                            corr_pairs.append((var1, var2, corr_value))
                    
                    # Trier par valeur absolue de corrélation
                    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    
                    # Créer un DataFrame pour l'affichage
                    strongest_corr = pd.DataFrame(
                        [(p[0], p[1], p[2]) for p in corr_pairs],
                        columns=["Variable 1", "Variable 2", "Corrélation"]
                    )
                    
                    # Styliser le DataFrame
                    def color_corr(val):
                        color = 'red' if val < 0 else 'green'
                        return f'color: {color}'
                    
                    styled_corr = strongest_corr.style.format({'Corrélation': '{:.4f}'}).applymap(color_corr, subset=['Corrélation'])
                    st.dataframe(styled_corr, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("Sélectionnez au moins deux variables pour afficher la matrice de corrélation.")
            
            elif relation_type == "Pairplot":
                # Sélection des variables
                selected_vars = st.multiselect(
                    "Sélectionner les variables pour le pairplot",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))]
                )
                
                if len(selected_vars) > 1:
                    # Créer le pairplot avec plotly
                    fig = px.scatter_matrix(
                        df,
                        dimensions=selected_vars,
                        title="Matrice de nuages de points"
                    )
                    
                    # Ajustements
                    fig.update_traces(diagonal_visible=False)
                    fig.update_layout(height=800)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Sélectionnez au moins deux variables pour afficher le pairplot.")
            
            elif relation_type == "Heatmap":
                # Sélection des variables pour les axes
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    x_var = st.selectbox("Variable X (horizontal)", numeric_cols)
                    y_var = st.selectbox("Variable Y (vertical)", numeric_cols, index=min(1, len(numeric_cols)-1))
                    z_var = st.selectbox("Variable Z (couleur)", numeric_cols, index=min(2, len(numeric_cols)-1))
                    
                    # Options
                    n_bins_x = st.slider("Nombre de bins (X)", 5, 50, 20)
                    n_bins_y = st.slider("Nombre de bins (Y)", 5, 50, 20)
                    aggregation = st.selectbox("Agrégation", ["mean", "median", "sum", "min", "max", "count"])
                
                with col2:
                    # Créer des bins pour les variables x et y
                    df['x_bin'] = pd.cut(df[x_var], bins=n_bins_x)
                    df['y_bin'] = pd.cut(df[y_var], bins=n_bins_y)
                    
                    # Agréger les données
                    heatmap_data = df.groupby(['x_bin', 'y_bin'])[z_var].agg(aggregation).reset_index()
                    
                    # Convertir les bins en points centraux pour la visualisation
                    heatmap_data['x_center'] = heatmap_data['x_bin'].apply(lambda x: x.mid)
                    heatmap_data['y_center'] = heatmap_data['y_bin'].apply(lambda x: x.mid)
                    
                    # Créer un pivottage pour la heatmap
                    pivot_data = heatmap_data.pivot(index='y_center', columns='x_center', values=z_var)
                    
                    # Créer la heatmap
                    fig = px.imshow(
                        pivot_data,
                        x=pivot_data.columns,
                        y=pivot_data.index,
                        color_continuous_scale="Viridis",
                        labels=dict(x=x_var, y=y_var, color=f"{z_var} ({aggregation})"),
                        title=f"Heatmap de {z_var} par {x_var} et {y_var}"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("<h2 class='subsection-header'>Analyse de Tendances</h2>", unsafe_allow_html=True)
            
            trend_type = st.selectbox(
                "Type d'analyse",
                ["Régression", "Groupement", "Décomposition en composantes principales (PCA)"]
            )
            
            if trend_type == "Régression":
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    x_var = st.selectbox("Variable indépendante (X)", numeric_cols)
                    y_var = st.selectbox("Variable dépendante (Y)", numeric_cols, index=min(1, len(numeric_cols)-1))
                    
                    reg_type = st.selectbox("Type de régression", ["linéaire", "polynomiale", "lowess"])
                    
                    if reg_type == "polynomiale":
                        degree = st.slider("Degré du polynôme", 1, 10, 2)
                    
                    show_formula = st.checkbox("Afficher l'équation", value=True)
                    show_r2 = st.checkbox("Afficher R²", value=True)
                
                with col1:
                    if reg_type == "linéaire":
                        fig = px.scatter(
                            df, x=x_var, y=y_var,
                            trendline="ols",
                            trendline_color_override="red",
                            labels={x_var: x_var, y_var: y_var},
                            title=f"Régression linéaire: {y_var} vs {x_var}"
                        )
                        
                        if show_formula or show_r2:
                            import statsmodels.api as sm
                            
                            X = sm.add_constant(df[x_var])
                            model = sm.OLS(df[y_var], X).fit()
                            
                            if show_formula:
                                formula = f"y = {model.params[1]:.4f}x + {model.params[0]:.4f}"
                                fig.add_annotation(
                                    x=0.05, y=0.95,
                                    xref="paper", yref="paper",
                                    text=formula,
                                    showarrow=False,
                                    font=dict(size=14),
                                    bgcolor="white",
                                    bordercolor="black",
                                    borderwidth=1
                                )
                            
                            if show_r2:
                                r2_text = f"R² = {model.rsquared:.4f}"
                                fig.add_annotation(
                                    x=0.05, y=0.85,
                                    xref="paper", yref="paper",
                                    text=r2_text,
                                    showarrow=False,
                                    font=dict(size=14),
                                    bgcolor="white",
                                    bordercolor="black",
                                    borderwidth=1
                                )
                    
                    elif reg_type == "polynomiale":
                        # Créer des variables polynomiales
                        import numpy as np
                        
                        fig = px.scatter(
                            df, x=x_var, y=y_var,
                            labels={x_var: x_var, y_var: y_var},
                            title=f"Régression polynomiale (degré {degree}): {y_var} vs {x_var}"
                        )
                        
                        # Calculer la régression polynomiale
                        from numpy.polynomial.polynomial import Polynomial
                        
                        x = df[x_var].values
                        y = df[y_var].values
                        
                        # Tri des points pour un traçage correct de la courbe
                        sort_idx = np.argsort(x)
                        x_sorted = x[sort_idx]
                        y_sorted = y[sort_idx]
                        
                        # Ajustement polynomial
                        coeffs = np.polyfit(x, y, degree)
                        p = np.poly1d(coeffs)
                        
                        # Tracer la courbe polynomiale
                        x_range = np.linspace(min(x), max(x), 100)
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=p(x_range),
                            mode='lines',
                            line=dict(color='red', width=2),
                            name=f'Polynôme degré {degree}'
                        ))
                        
                        if show_formula or show_r2:
                            if show_formula:
                                formula = "y = "
                                for i, coef in enumerate(reversed(coeffs)):
                                    if i == 0:
                                        formula += f"{coef:.4f}"
                                    else:
                                        formula += f" + {coef:.4f}x^{i}"
                                
                                fig.add_annotation(
                                    x=0.05, y=0.95,
                                    xref="paper", yref="paper",
                                    text=formula,
                                    showarrow=False,
                                    font=dict(size=14),
                                    bgcolor="white",
                                    bordercolor="black",
                                    borderwidth=1
                                )
                            
                            if show_r2:
                                # Calculer R²
                                y_pred = p(x)
                                r2 = r2_score(y, y_pred)
                                
                                r2_text = f"R² = {r2:.4f}"
                                fig.add_annotation(
                                    x=0.05, y=0.85,
                                    xref="paper", yref="paper",
                                    text=r2_text,
                                    showarrow=False,
                                    font=dict(size=14),
                                    bgcolor="white",
                                    bordercolor="black",
                                    borderwidth=1
                                )
                    
                    elif reg_type == "lowess":
                        fig = px.scatter(
                            df, x=x_var, y=y_var,
                            trendline="lowess",
                            trendline_color_override="red",
                            labels={x_var: x_var, y_var: y_var},
                            title=f"Régression LOWESS: {y_var} vs {x_var}"
                        )
                        
                        # Pas de formule pour LOWESS, mais on peut afficher une info contextuelle
                        fig.add_annotation(
                            x=0.05, y=0.95,
                            xref="paper", yref="paper",
                            text="Régression non paramétrique LOWESS",
                            showarrow=False,
                            font=dict(size=14),
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques supplémentaires
                with st.expander("Statistiques détaillées"):
                    if reg_type == "linéaire":
                        import statsmodels.api as sm
                        
                        X = sm.add_constant(df[x_var])
                        model = sm.OLS(df[y_var], X).fit()
                        
                        st.write(model.summary())
                    
                    elif reg_type == "polynomiale":
                        from sklearn.linear_model import LinearRegression
                        from sklearn.preprocessing import PolynomialFeatures
                        from sklearn.pipeline import make_pipeline
                        
                        X = df[x_var].values.reshape(-1, 1)
                        y = df[y_var].values
                        
                        model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
                        model.fit(X, y)
                        
                        y_pred = model.predict(X)
                        mse = mean_squared_error(y, y_pred)
                        r2 = r2_score(y, y_pred)
                        
                        st.write(f"Erreur quadratique moyenne (MSE): {mse:.4f}")
                        st.write(f"Coefficient de détermination (R²): {r2:.4f}")
                        
                        # Coefficients du modèle
                        coefs = model.named_steps['linearregression'].coef_
                        intercept = model.named_steps['linearregression'].intercept_
                        
                        st.write("Coefficients du modèle:")
                        for i, coef in enumerate(coefs):
                            if i == 0:
                                st.write(f"Constante: {intercept:.4f}")
                            else:
                                st.write(f"x^{i}: {coef:.4f}")
            
            elif trend_type == "Groupement":
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    # Variables pour le groupement
                    group_var = st.selectbox("Variable de groupement", [col for col in df.columns if col not in ['X', 'Y']])
                    agg_var = st.selectbox("Variable à agréger", numeric_cols)
                    
                    # Méthode de groupement
                    if df[group_var].dtype.name in ['object', 'category']:
                        # Variable catégorielle
                        group_method = "category"
                    else:
                        # Variable numérique, proposer des méthodes de binning
                        group_method = st.selectbox("Méthode de groupement", ["quantiles", "equal_width", "custom"])
                        
                        if group_method == "quantiles":
                            n_groups = st.slider("Nombre de quantiles", 2, 10, 4)
                        elif group_method == "equal_width":
                            n_groups = st.slider("Nombre d'intervalles", 2, 10, 4)
                        elif group_method == "custom":
                            bin_edges = st.text_input("Limites des intervalles (séparés par des virgules)", 
                                                      value=",".join(str(round(x, 2)) for x in [df[group_var].min(), df[group_var].median(), df[group_var].max()]))
                            try:
                                bin_edges = [float(x.strip()) for x in bin_edges.split(",")]
                            except:
                                st.error("Format invalide. Utilisez des nombres séparés par des virgules.")
                                bin_edges = [df[group_var].min(), df[group_var].median(), df[group_var].max()]
                    
                    # Méthode d'agrégation
                    agg_method = st.selectbox("Méthode d'agrégation", ["mean", "median", "sum", "min", "max", "count", "std"])
                
                with col1:
                    # Regrouper les données selon la méthode choisie
                    if group_method == "category":
                        grouped = df.groupby(group_var)[agg_var].agg(agg_method).reset_index()
                        x_title = group_var
                    elif group_method == "quantiles":
                        df['group'] = pd.qcut(df[group_var], n_groups, duplicates='drop')
                        grouped = df.groupby('group')[agg_var].agg(agg_method).reset_index()
                        grouped['group_label'] = grouped['group'].apply(lambda x: f"{x.left:.2f} - {x.right:.2f}")
                        x_title = f"{group_var} (quantiles)"
                    elif group_method == "equal_width":
                        df['group'] = pd.cut(df[group_var], n_groups)
                        grouped = df.groupby('group')[agg_var].agg(agg_method).reset_index()
                        grouped['group_label'] = grouped['group'].apply(lambda x: f"{x.left:.2f} - {x.right:.2f}")
                        x_title = f"{group_var} (intervalles égaux)"
                    elif group_method == "custom":
                        df['group'] = pd.cut(df[group_var], bin_edges)
                        grouped = df.groupby('group')[agg_var].agg(agg_method).reset_index()
                        grouped['group_label'] = grouped['group'].apply(lambda x: f"{x.left:.2f} - {x.right:.2f}" if not pd.isna(x) else "NA")
                        x_title = f"{group_var} (intervalles personnalisés)"
                    
                    # Créer le graphique
                    if group_method == "category":
                        fig = px.bar(
                            grouped,
                            x=group_var,
                            y=agg_var,
                            title=f"{agg_method.capitalize()} de {agg_var} par {group_var}",
                            labels={group_var: x_title, agg_var: f"{agg_method.capitalize()} de {agg_var}"}
                        )
                    else:
                        fig = px.bar(
                            grouped,
                            x='group_label',
                            y=agg_var,
                            title=f"{agg_method.capitalize()} de {agg_var} par {group_var}",
                            labels={'group_label': x_title, agg_var: f"{agg_method.capitalize()} de {agg_var}"}
                        )
                        fig.update_xaxes(tickangle=45)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des statistiques par groupe
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Statistiques détaillées par groupe")
                
                if group_method == "category":
                    detailed_stats = df.groupby(group_var)[agg_var].agg(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']).reset_index()
                else:
                    detailed_stats = df.groupby('group')[agg_var].agg(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']).reset_index()
                    detailed_stats['group'] = detailed_stats['group'].astype(str)
                
                st.dataframe(detailed_stats, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            elif trend_type == "Décomposition en composantes principales (PCA)":
                # Sélection des variables pour la PCA
                selected_vars = st.multiselect(
                    "Sélectionner les variables pour la PCA",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if len(selected_vars) > 1:
                    col1, col2 = st.columns([3, 1])
                    
                    with col2:
                        n_components = st.slider("Nombre de composantes", 2, min(len(selected_vars), 10), 2)
                        scale_data = st.checkbox("Standardiser les données", value=True)
                        
                        if n_components > 2:
                            plot_type = st.selectbox("Type de visualisation", ["2D", "3D"])
                        else:
                            plot_type = "2D"
                        
                        color_by = st.selectbox("Colorer par", [None] + numeric_cols)
                    
                    with col1:
                        # Préparation des données
                        X = df[selected_vars].copy()
                        
                        # Gestion des valeurs manquantes
                        if X.isnull().sum().sum() > 0:
                            X = X.fillna(X.mean())
                        
                        # Standardisation si demandée
                        if scale_data:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                        else:
                            X_scaled = X.values
                        
                        # Calcul de la PCA
                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(X_scaled)
                        
                        # Créer un DataFrame avec les résultats
                        pca_df = pd.DataFrame(
                            data=pca_result,
                            columns=[f'PC{i+1}' for i in range(n_components)]
                        )
                        
                        # Ajouter la variable de couleur si demandée
                        if color_by:
                            pca_df['color'] = df[color_by]
                        
                        # Visualisation
                        if plot_type == "2D":
                            if color_by:
                                fig = px.scatter(
                                    pca_df, x='PC1', y='PC2',
                                    color='color',
                                    title="Analyse en Composantes Principales (ACP)",
                                    labels={'color': color_by}
                                )
                            else:
                                fig = px.scatter(
                                    pca_df, x='PC1', y='PC2',
                                    title="Analyse en Composantes Principales (ACP)"
                                )
                            
                            # Ajouter des annotations pour le pourcentage de variance expliquée
                            explained_variance = pca.explained_variance_ratio_
                            
                            fig.update_xaxes(title=f"PC1 ({explained_variance[0]:.1%} de variance expliquée)")
                            fig.update_yaxes(title=f"PC2 ({explained_variance[1]:.1%} de variance expliquée)")
                        
                        elif plot_type == "3D":
                            if color_by:
                                fig = px.scatter_3d(
                                    pca_df, x='PC1', y='PC2', z='PC3',
                                    color='color',
                                    title="Analyse en Composantes Principales (ACP) - 3D",
                                    labels={'color': color_by}
                                )
                            else:
                                fig = px.scatter_3d(
                                    pca_df, x='PC1', y='PC2', z='PC3',
                                    title="Analyse en Composantes Principales (ACP) - 3D"
                                )
                            
                            # Variance expliquée
                            explained_variance = pca.explained_variance_ratio_
                            
                            fig.update_scenes(
                                xaxis_title=f"PC1 ({explained_variance[0]:.1%})",
                                yaxis_title=f"PC2 ({explained_variance[1]:.1%})",
                                zaxis_title=f"PC3 ({explained_variance[2]:.1%})"
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Analyse détaillée de la PCA
                    with st.expander("Analyse détaillée de la PCA"):
                        # Variance expliquée par chaque composante
                        explained_variance = pca.explained_variance_ratio_
                        cum_explained_variance = np.cumsum(explained_variance)
                        
                        # Graphique de la variance expliquée
                        var_df = pd.DataFrame({
                            'Composante': [f'PC{i+1}' for i in range(n_components)],
                            'Variance Expliquée (%)': explained_variance * 100,
                            'Variance Cumulée (%)': cum_explained_variance * 100
                        })
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=var_df['Composante'],
                            y=var_df['Variance Expliquée (%)'],
                            name='Variance Expliquée'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=var_df['Composante'],
                            y=var_df['Variance Cumulée (%)'],
                            mode='lines+markers',
                            name='Variance Cumulée',
                            line=dict(color='red')
                        ))
                        
                        fig.update_layout(
                            title="Variance expliquée par composante",
                            xaxis_title="Composante",
                            yaxis_title="Variance Expliquée (%)",
                            legend=dict(x=0.7, y=0.9)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Contributions des variables aux composantes
                        loadings = pca.components_
                        loading_df = pd.DataFrame(
                            loadings.T,
                            columns=[f'PC{i+1}' for i in range(n_components)],
                            index=selected_vars
                        )
                        
                        st.markdown("#### Contributions des variables aux composantes principales")
                        st.dataframe(loading_df, use_container_width=True)
                        
                        # Graphique des contributions (biplot pour les 2 premières composantes)
                        st.markdown("#### Biplot (PC1 vs PC2)")
                        
                        # Standardiser les loadings pour la visualisation
                        loading_scale = 5  # Facteur d'échelle pour les flèches
                        pcs = pca.components_
                        n = pcs.shape[1]
                        
                        # Créer le biplot
                        fig = go.Figure()
                        
                        # Tracer les observations (projections)
                        fig.add_trace(go.Scatter(
                            x=pca_df['PC1'],
                            y=pca_df['PC2'],
                            mode='markers',
                            marker=dict(
                                color='blue',
                                size=8,
                                opacity=0.5
                            ),
                            name='Observations'
                        ))
                        
                        # Tracer les flèches des variables
                        for i, var in enumerate(selected_vars):
                            fig.add_trace(go.Scatter(
                                x=[0, pcs[0, i] * loading_scale],
                                y=[0, pcs[1, i] * loading_scale],
                                mode='lines+markers+text',
                                line=dict(color='red', width=2),
                                marker=dict(size=5, color='red'),
                                text=['', var],
                                textposition='top center',
                                name=var
                            ))
                        
                        # Mise en page
                        fig.update_layout(
                            title="Biplot PCA",
                            xaxis=dict(
                                title=f"PC1 ({explained_variance[0]:.1%} de variance expliquée)",
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='black'
                            ),
                            yaxis=dict(
                                title=f"PC2 ({explained_variance[1]:.1%} de variance expliquée)",
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='black'
                            ),
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Sélectionnez au moins deux variables pour effectuer une ACP.")

def show_prediction():
    st.markdown("<h1 class='title'>Prédiction de Minéralisation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Prédisez la teneur en minéralisation à partir de données géochimiques</p>", unsafe_allow_html=True)
    
    # Composant pour télécharger des données si nécessaire
    if st.session_state.data is None:
        upload_data_component()
    else:
        # Bouton pour changer de jeu de données
        if st.button("📤 Changer de jeu de données"):
            upload_data_component()
    
    # Si des données sont chargées, afficher l'interface de prédiction
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Informations sur le jeu de données
        st.markdown(f"<div class='info-card'>Jeu de données actuel: <strong>{st.session_state.uploaded_file_name}</strong> | {len(df)} échantillons | {len(df.columns)} variables</div>", unsafe_allow_html=True)
        
        # Configuration du modèle
        st.markdown("<h2 class='section-header'>⚙️ Configuration du Modèle</h2>", unsafe_allow_html=True)
        
        # Colonnes numériques
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sélection des caractéristiques
            features = st.multiselect(
                "Sélectionner les caractéristiques (variables explicatives)",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
        
        with col2:
            # Sélection de la cible
            available_targets = [col for col in numeric_cols if col not in features]
            
            if available_targets:
                target = st.selectbox("Sélectionner la variable cible à prédire", available_targets)
            else:
                st.error("Veuillez d'abord sélectionner des caractéristiques.")
                target = None
        
        if features and target:
            # Exclure la cible des caractéristiques si elle y est
            if target in features:
                features.remove(target)
            
            # Choix du modèle
            model_options = ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting", "KNN"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox("Sélectionner le type de modèle", model_options)
            
            with col2:
                # Options d'évaluation
                test_size = st.slider("Taille de l'ensemble de test (%)", 10, 50, 20) / 100
                cv_folds = st.slider("Nombre de plis pour la validation croisée", 2, 10, 5)
            
            # Paramètres avancés par type de modèle
            with st.expander("Paramètres avancés du modèle"):
                if model_type == "XGBoost":
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        n_estimators = st.slider("Nombre d'estimateurs", 50, 1000, 100)
                        learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, 0.01)
                    
                    with col2:
                        max_depth = st.slider("Profondeur maximale", 3, 15, 6)
                        gamma = st.slider("Gamma", 0.0, 1.0, 0.0, 0.1)
                    
                    with col3:
                        subsample = st.slider("Subsample", 0.5, 1.0, 1.0, 0.1)
                        colsample_bytree = st.slider("Colsample Bytree", 0.5, 1.0, 1.0, 0.1)
                    
                    model_params = {
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "gamma": gamma,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree,
                        "random_state": 42
                    }
                
                elif model_type == "LightGBM":
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        n_estimators = st.slider("Nombre d'estimateurs", 50, 1000, 100)
                        learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, 0.01)
                    
                    with col2:
                        max_depth = st.slider("Profondeur maximale", 3, 15, 6)
                        num_leaves = st.slider("Nombre de feuilles", 10, 100, 31)
                    
                    with col3:
                        subsample = st.slider("Subsample", 0.5, 1.0, 1.0, 0.1)
                        colsample_bytree = st.slider("Colsample Bytree", 0.5, 1.0, 1.0, 0.1)
                    
                    model_params = {
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "num_leaves": num_leaves,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree,
                        "random_state": 42
                    }
                
                elif model_type == "RandomForest":
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        n_estimators = st.slider("Nombre d'estimateurs", 50, 1000, 100)
                        max_depth = st.slider("Profondeur maximale", 3, 15, 6)
                    
                    with col2:
                        min_samples_split = st.slider("Échantillons minimum pour la division", 2, 20, 2)
                        min_samples_leaf = st.slider("Échantillons minimum par feuille", 1, 20, 1)
                    
                    with col3:
                        max_features = st.selectbox("Caractéristiques maximum", ["auto", "sqrt", "log2"])
                        bootstrap = st.checkbox("Bootstrap", value=True)
                    
                    model_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "max_features": max_features,
                        "bootstrap": bootstrap,
                        "random_state": 42
                    }
                
                elif model_type == "GradientBoosting":
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        n_estimators = st.slider("Nombre d'estimateurs", 50, 1000, 100)
                        learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, 0.01)
                    
                    with col2:
                        max_depth = st.slider("Profondeur maximale", 3, 15, 3)
                        min_samples_split = st.slider("Échantillons minimum pour la division", 2, 20, 2)
                    
                    with col3:
                        min_samples_leaf = st.slider("Échantillons minimum par feuille", 1, 20, 1)
                        subsample = st.slider("Subsample", 0.5, 1.0, 1.0, 0.1)
                    
                    model_params = {
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "subsample": subsample,
                        "random_state": 42
                    }
                
                elif model_type == "KNN":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        n_neighbors = st.slider("Nombre de voisins", 1, 20, 5)
                        weights = st.selectbox("Pondération", ["uniform", "distance"])
                    
                    with col2:
                        algorithm = st.selectbox("Algorithme", ["auto", "ball_tree", "kd_tree", "brute"])
                        leaf_size = st.slider("Taille des feuilles", 10, 100, 30)
                    
                    model_params = {
                        "n_neighbors": n_neighbors,
                        "weights": weights,
                        "algorithm": algorithm,
                        "leaf_size": leaf_size
                    }
            
            # Entraînement du modèle
            train_button = st.button("🔥 Entraîner le modèle")
            
            if train_button:
                # Préparation des données
                X = df[features].copy()
                y = df[target].copy()
                
                # Gestion des valeurs manquantes
                if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                    st.warning("Des valeurs manquantes ont été détectées. Elles seront remplacées par la médiane.")
                    X = X.fillna(X.median())
                    y = y.fillna(y.median())
                
                # Division train/test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Standardisation des données
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Affichage d'une barre de progression pendant l'entraînement
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Étape 1: Entraînement du modèle
                status_text.text("Entraînement du modèle en cours...")
                progress_bar.progress(10)
                
                # Entraîner le modèle
                model = train_model(X_train, y_train, model_type, model_params)
                progress_bar.progress(40)
                
                # Étape 2: Évaluation sur l'ensemble de test
                status_text.text("Évaluation du modèle sur l'ensemble de test...")
                progress_bar.progress(60)
                
                # Évaluer le modèle
                test_metrics = evaluate_model(model, X_test, y_test)
                progress_bar.progress(80)
                
                # Étape 3: Validation croisée
                status_text.text("Validation croisée en cours...")
                cv_metrics = cross_validate_model(model, X, y, cv=cv_folds)
                progress_bar.progress(100)
                
                # Effacer la barre de progression et le texte de statut
                progress_bar.empty()
                status_text.empty()
                
                # Sauvegarder le modèle en session
                st.session_state.model = model
                st.session_state.model_features = features
                st.session_state.model_target = target
                st.session_state.model_metrics = {
                    "test": test_metrics,
                    "cv": cv_metrics
                }
                
                # Afficher un message de succès
                st.success("✅ Modèle entraîné avec succès!")
            
            # Afficher les résultats si un modèle a été entraîné
            if st.session_state.model is not None and st.session_state.model_features == features and st.session_state.model_target == target:
                st.markdown("<h2 class='section-header'>📊 Résultats du Modèle</h2>", unsafe_allow_html=True)
                
                # Récupérer les métriques
                test_metrics = st.session_state.model_metrics["test"]
                cv_metrics = st.session_state.model_metrics["cv"]
                
                # Afficher les métriques principales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{test_metrics['R²']:.4f}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>R² (Test)</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{test_metrics['RMSE']:.4f}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>RMSE (Test)</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{cv_metrics['CV R²']:.4f}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>R² (CV)</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col4:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{cv_metrics['CV RMSE']:.4f}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>RMSE (CV)</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Visualisations des résultats
                st.markdown("<h3 class='subsection-header'>Visualisations</h3>", unsafe_allow_html=True)
                
                tab1, tab2, tab3 = st.tabs(["Prédictions vs Réalité", "Importance des Variables", "Carte de Prédiction"])
                
                with tab1:
                    # Scatter plot des prédictions vs réalité
                    X_test = df[features].sample(frac=test_size, random_state=42)
                    y_test = df[target].loc[X_test.index]
                    y_pred = test_metrics["Prédictions"]
                    
                    fig = px.scatter(
                        x=y_test,
                        y=y_pred,
                        trendline="ols",
                        trendline_color_override="red",
                        labels={"x": f"{target} réel", "y": f"{target} prédit"},
                        title="Prédictions vs Valeurs réelles"
                    )
                    
                    # Ligne y=x (prédiction parfaite)
                    fig.add_trace(go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode="lines",
                        line=dict(color="black", dash="dash"),
                        name="Prédiction parfaite"
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Histogramme des erreurs
                    errors = y_pred - y_test
                    
                    fig = px.histogram(
                        errors,
                        nbins=20,
                        marginal="box",
                        title="Distribution des erreurs de prédiction",
                        color_discrete_sequence=['#FF7043']
                    )
                    
                    fig.add_vline(x=0, line_dash="dash", line_color="red")
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Importance des variables
                    if model_type in ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting"]:
                        importances = st.session_state.model.feature_importances_
                        importance_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': importances
                        }).sort_values(by='Importance', ascending=False)
                        
                        fig = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Importance des Variables",
                            color='Importance',
                            color_continuous_scale='Blues'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Table d'importance des variables
                        st.dataframe(importance_df.style.bar(subset=['Importance'], color='#71a4f9'), use_container_width=True)
                    else:
                        st.info("L'importance des variables n'est pas disponible pour ce type de modèle.")
                
                with tab3:
                    # Carte de prédiction si les coordonnées sont disponibles
                    if 'X' in df.columns and 'Y' in df.columns:
                        # Prédire sur l'ensemble complet des données
                        X_full = df[features]
                        predictions = st.session_state.model.predict(X_full)
                        
                        # Ajouter les prédictions au DataFrame
                        result_df = df.copy()
                        result_df[f'{target}_prédit'] = predictions
                        result_df[f'Erreur_{target}'] = result_df[f'{target}_prédit'] - result_df[target]
                        
                        # Sélection du type de carte
                        map_type = st.selectbox(
                            "Type de visualisation",
                            ["Valeurs prédites", "Erreurs de prédiction", "Comparaison prédictions vs réalité"]
                        )
                        
                        if map_type == "Valeurs prédites":
                            # Carte des valeurs prédites
                            fig = px.scatter(
                                result_df,
                                x='X', y='Y',
                                color=f'{target}_prédit',
                                size=f'{target}_prédit',
                                color_continuous_scale='Viridis',
                                title=f"Carte des valeurs prédites de {target}"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Carte de contour
                            with st.expander("Carte de contour des prédictions"):
                                interp_data = interpolate_values(result_df, f'{target}_prédit', grid_size=100, method='linear')
                                
                                if interp_data:
                                    fig = plot_contour_map(interp_data, result_df, f'{target}_prédit', colorscale='Viridis')
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        elif map_type == "Erreurs de prédiction":
                            # Carte des erreurs de prédiction
                            fig = px.scatter(
                                result_df,
                                x='X', y='Y',
                                color=f'Erreur_{target}',
                                size=abs(result_df[f'Erreur_{target}']),
                                color_continuous_scale='RdBu_r',
                                title=f"Carte des erreurs de prédiction de {target}"
                            )
                            
                            # Ajuster l'échelle de couleurs pour être centrée sur 0
                            max_abs_error = max(abs(result_df[f'Erreur_{target}'].max()), abs(result_df[f'Erreur_{target}'].min()))
                            fig.update_layout(coloraxis_colorbar=dict(title="Erreur"))
                            fig.update_traces(marker=dict(cmin=-max_abs_error, cmax=max_abs_error, colorbar=dict(title="Erreur")))
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif map_type == "Comparaison prédictions vs réalité":
                            # Deux cartes côte à côte: réalité et prédictions
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig1 = px.scatter(
                                    result_df,
                                    x='X', y='Y',
                                    color=target,
                                    title=f"{target} (Valeurs réelles)",
                                    color_continuous_scale='Viridis'
                                )
                                
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            with col2:
                                fig2 = px.scatter(
                                    result_df,
                                    x='X', y='Y',
                                    color=f'{target}_prédit',
                                    title=f"{target} (Valeurs prédites)",
                                    color_continuous_scale='Viridis'
                                )
                                
                                st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("Les colonnes X et Y sont nécessaires pour afficher la carte de prédiction.")
                
                # Téléchargement des résultats
                st.markdown("<h3 class='subsection-header'>Téléchargement des Résultats</h3>", unsafe_allow_html=True)
                
                # Prédire sur l'ensemble complet des données
                X_full = df[features]
                predictions = st.session_state.model.predict(X_full)
                
                # Ajouter les prédictions au DataFrame
                result_df = df.copy()
                result_df[f'{target}_prédit'] = predictions
                result_df[f'Erreur_{target}'] = result_df[f'{target}_prédit'] - result_df[target]
                
                # Afficher un aperçu des résultats
                st.dataframe(result_df, use_container_width=True)
                
                # Bouton de téléchargement
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger les résultats (CSV)",
                    data=csv,
                    file_name=f"resultats_prediction_{target}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Veuillez sélectionner au moins une caractéristique et une variable cible.")

def show_anomaly_detection():
    st.markdown("<h1 class='title'>Détection d'Anomalies</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Identifiez les anomalies géochimiques dans vos données</p>", unsafe_allow_html=True)
    
    # Composant pour télécharger des données si nécessaire
    if st.session_state.data is None:
        upload_data_component()
    else:
        # Bouton pour changer de jeu de données
        if st.button("📤 Changer de jeu de données"):
            upload_data_component()
    
    # Si des données sont chargées, afficher l'interface de détection d'anomalies
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Informations sur le jeu de données
        st.markdown(f"<div class='info-card'>Jeu de données actuel: <strong>{st.session_state.uploaded_file_name}</strong> | {len(df)} échantillons | {len(df.columns)} variables</div>", unsafe_allow_html=True)
        
        # Configuration de la détection d'anomalies
        st.markdown("<h2 class='section-header'>⚙️ Configuration de la Détection d'Anomalies</h2>", unsafe_allow_html=True)
        
        # Colonnes numériques
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Sélection des variables
        variables = st.multiselect(
            "Sélectionner les variables pour la détection d'anomalies",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )
        
        if variables:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Méthode de détection
                method = st.selectbox(
                    "Méthode de détection",
                    ["IsolationForest", "DBSCAN", "KMeans"]
                )
            
            with col2:
                # Paramètres communs
                if method == "IsolationForest":
                    contamination = st.slider("Contamination estimée (%)", 1, 20, 10) / 100
                    scaling = st.checkbox("Standardiser les données", value=True)
                elif method == "DBSCAN":
                    eps = st.slider("Epsilon (distance maximum)", 0.1, 2.0, 0.5, 0.1)
                    min_samples = st.slider("Nombre minimum d'échantillons", 2, 10, 5)
                    scaling = st.checkbox("Standardiser les données", value=True)
                elif method == "KMeans":
                    n_clusters = st.slider("Nombre de clusters", 2, 10, 5)
                    contamination = st.slider("Contamination estimée (%)", 1, 20, 10) / 100
                    scaling = st.checkbox("Standardiser les données", value=True)
            
            with col3:
                # Paramètres additionnels
                if method == "IsolationForest":
                    n_estimators = st.slider("Nombre d'estimateurs", 50, 200, 100)
                    max_samples = st.selectbox("Taille maximum d'échantillons", ["auto", 100, 0.5, 0.1])
                    if max_samples == "auto":
                        max_samples_param = "auto"
                    elif max_samples == 0.5 or max_samples == 0.1:
                        max_samples_param = float(max_samples)
                    else:
                        max_samples_param = int(max_samples)
                elif method == "DBSCAN":
                    algorithm = st.selectbox("Algorithme", ["auto", "ball_tree", "kd_tree", "brute"])
                    leaf_size = st.slider("Taille des feuilles", 10, 100, 30)
                elif method == "KMeans":
                    init = st.selectbox("Initialisation", ["k-means++", "random"])
                    n_init = st.slider("Nombre d'initialisations", 1, 20, 10)
            
            # Bouton pour lancer la détection
            detect_button = st.button("🔍 Détecter les anomalies")
            
            if detect_button:
                # Préparation des données
                X = df[variables].copy()
                
                # Gestion des valeurs manquantes
                if X.isnull().sum().sum() > 0:
                    st.warning("Des valeurs manquantes ont été détectées. Elles seront remplacées par la médiane.")
                    X = X.fillna(X.median())
                
                # Standardisation des données si demandée
                if scaling:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = X.values
                
                # Affichage d'une barre de progression
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Détection des anomalies
                status_text.text("Détection d'anomalies en cours...")
                progress_bar.progress(30)
                
                # Paramètres spécifiques à la méthode
                if method == "IsolationForest":
                    kwargs = {
                        "n_estimators": n_estimators,
                        "max_samples": max_samples_param,
                        "random_state": 42
                    }
                    anomalies, anomaly_scores = detect_anomalies(X_scaled, contamination, method, **kwargs)
                
                elif method == "DBSCAN":
                    kwargs = {
                        "eps": eps,
                        "min_samples": min_samples,
                        "algorithm": algorithm,
                        "leaf_size": leaf_size
                    }
                    anomalies, anomaly_scores = detect_anomalies(X_scaled, None, method, **kwargs)
                
                elif method == "KMeans":
                    kwargs = {
                        "n_clusters": n_clusters,
                        "init": init,
                        "n_init": n_init,
                        "random_state": 42
                    }
                    anomalies, anomaly_scores = detect_anomalies(X_scaled, contamination, method,
                   elif method == "KMeans":
                    kwargs = {
                        "n_clusters": n_clusters,
                        "init": init,
                        "n_init": n_init,
                        "random_state": 42
                    }
                    anomalies, anomaly_scores = detect_anomalies(X_scaled, contamination, method, **kwargs)
                
                progress_bar.progress(70)
                
                # Ajout des résultats au dataframe
                result_df = df.copy()
                result_df["Anomalie"] = anomalies
                result_df["Score_Anomalie"] = anomaly_scores
                
                progress_bar.progress(100)
                progress_bar.empty()
                status_text.empty()
                
                # Afficher les résultats
                st.success(f"✅ Détection d'anomalies terminée! {np.sum(anomalies)} anomalies détectées ({np.sum(anomalies) / len(df) * 100:.1f}%).")
                
                # Onglets pour différentes visualisations des résultats
                st.markdown("<h2 class='section-header'>📊 Résultats</h2>", unsafe_allow_html=True)
                
                # Affichage des statistiques principales
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{np.sum(anomalies)}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Anomalies détectées</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{np.sum(anomalies) / len(df) * 100:.1f}%</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Pourcentage d'anomalies</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    # Score moyen des anomalies
                    if np.sum(anomalies) > 0:
                        avg_score = np.mean(anomaly_scores[anomalies == 1])
                    else:
                        avg_score = 0
                    
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{avg_score:.3f}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Score moyen des anomalies</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Onglets pour les différentes visualisations
                tab1, tab2, tab3, tab4 = st.tabs(["Tableau de données", "Visualisations 2D/3D", "Carte des anomalies", "Statistiques comparatives"])
                
                with tab1:
                    # Option pour filtrer les anomalies
                    show_only_anomalies = st.checkbox("Afficher uniquement les anomalies", value=False)
                    
                    if show_only_anomalies:
                        filtered_df = result_df[result_df["Anomalie"] == 1]
                    else:
                        filtered_df = result_df
                    
                    # Tri des données par score d'anomalie
                    sorted_df = filtered_df.sort_values(by="Score_Anomalie", ascending=False)
                    
                    # Affichage du tableau
                    st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
                    st.dataframe(sorted_df, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Téléchargement des résultats
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger les résultats (CSV)",
                        data=csv,
                        file_name="resultats_anomalies.csv",
                        mime="text/csv"
                    )
                
                with tab2:
                    # Sélection de visualisation
                    viz_type = st.radio("Type de visualisation", ["2D (PCA)", "3D (PCA)"], horizontal=True)
                    
                    # Appliquer PCA pour la visualisation
                    if len(variables) >= 2:
                        n_components = min(3, len(variables))
                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(X_scaled)
                        
                        # Créer un dataframe pour la visualisation
                        pca_df = pd.DataFrame(data=pca_result, columns=[f"PC{i+1}" for i in range(n_components)])
                        pca_df["Anomalie"] = anomalies
                        pca_df["Score_Anomalie"] = anomaly_scores
                        
                        # Variance expliquée
                        explained_var = pca.explained_variance_ratio_
                        
                        if viz_type == "2D (PCA)":
                            fig = px.scatter(
                                pca_df,
                                x="PC1",
                                y="PC2",
                                color="Anomalie",
                                size="Score_Anomalie",
                                hover_data=["Score_Anomalie"],
                                color_discrete_sequence=["#1E88E5", "#FF5252"],
                                title="Visualisation 2D des anomalies (PCA)"
                            )
                            
                            fig.update_layout(
                                xaxis_title=f"PC1 ({explained_var[0]:.1%} variance expliquée)",
                                yaxis_title=f"PC2 ({explained_var[1]:.1%} variance expliquée)"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Histogramme des scores d'anomalie
                            fig = px.histogram(
                                pca_df,
                                x="Score_Anomalie",
                                color="Anomalie",
                                title="Distribution des scores d'anomalie",
                                color_discrete_sequence=["#1E88E5", "#FF5252"],
                                nbins=50
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == "3D (PCA)" and n_components >= 3:
                            fig = px.scatter_3d(
                                pca_df,
                                x="PC1",
                                y="PC2",
                                z="PC3",
                                color="Anomalie",
                                size="Score_Anomalie",
                                hover_data=["Score_Anomalie"],
                                color_discrete_sequence=["#1E88E5", "#FF5252"],
                                title="Visualisation 3D des anomalies (PCA)"
                            )
                            
                            fig.update_layout(
                                scene=dict(
                                    xaxis_title=f"PC1 ({explained_var[0]:.1%})",
                                    yaxis_title=f"PC2 ({explained_var[1]:.1%})",
                                    zaxis_title=f"PC3 ({explained_var[2]:.1%})"
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Au moins deux variables sont nécessaires pour la visualisation PCA.")
                
                with tab3:
                    # Carte des anomalies
                    if 'X' in df.columns and 'Y' in df.columns:
                        # Sélection du type de carte
                        map_type = st.selectbox(
                            "Type de carte",
                            ["Carte des points", "Carte de chaleur", "Carte de contour"]
                        )
                        
                        if map_type == "Carte des points":
                            fig = px.scatter(
                                result_df,
                                x="X",
                                y="Y",
                                color="Anomalie",
                                size="Score_Anomalie",
                                hover_data=variables + ["Score_Anomalie"],
                                color_discrete_sequence=["#1E88E5", "#FF5252"],
                                title="Carte des anomalies"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif map_type == "Carte de chaleur":
                            # Créer une grille régulière
                            x_range = np.linspace(result_df['X'].min(), result_df['X'].max(), 100)
                            y_range = np.linspace(result_df['Y'].min(), result_df['Y'].max(), 100)
                            xx, yy = np.meshgrid(x_range, y_range)
                            
                            # Interpolation des scores d'anomalie
                            from scipy.interpolate import griddata
                            z = griddata((result_df['X'], result_df['Y']), result_df["Score_Anomalie"], (xx, yy), method='linear')
                            
                            # Créer la carte de chaleur
                            fig = go.Figure(data=go.Heatmap(
                                z=z,
                                x=x_range,
                                y=y_range,
                                colorscale="Viridis",
                                colorbar=dict(title="Score d'anomalie")
                            ))
                            
                            # Ajouter les points des anomalies
                            anomalies_df = result_df[result_df["Anomalie"] == 1]
                            
                            fig.add_trace(go.Scatter(
                                x=anomalies_df['X'],
                                y=anomalies_df['Y'],
                                mode='markers',
                                marker=dict(
                                    color='red',
                                    size=8,
                                    symbol='x',
                                    line=dict(color='black', width=1)
                                ),
                                name='Anomalies'
                            ))
                            
                            fig.update_layout(
                                title="Carte de chaleur des scores d'anomalie",
                                xaxis_title='X',
                                yaxis_title='Y'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif map_type == "Carte de contour":
                            # Interpolation des scores d'anomalie
                            interp_data = interpolate_values(result_df, "Score_Anomalie", grid_size=100, method='linear')
                            
                            if interp_data:
                                # Créer la figure de base
                                fig = go.Figure()
                                
                                # Ajouter le contour
                                fig.add_trace(go.Contour(
                                    z=interp_data["z"],
                                    x=interp_data["x"],
                                    y=interp_data["y"],
                                    colorscale="Viridis",
                                    colorbar=dict(title="Score d'anomalie"),
                                    line=dict(width=0.5),
                                    contours=dict(
                                        showlabels=True,
                                        labelfont=dict(size=10, color='white')
                                    )
                                ))
                                
                                # Ajouter tous les points d'échantillonnage
                                fig.add_trace(go.Scatter(
                                    x=result_df[result_df["Anomalie"] == 0]['X'],
                                    y=result_df[result_df["Anomalie"] == 0]['Y'],
                                    mode='markers',
                                    marker=dict(
                                        color='blue',
                                        size=6,
                                        opacity=0.5
                                    ),
                                    name='Points normaux'
                                ))
                                
                                # Ajouter les anomalies
                                fig.add_trace(go.Scatter(
                                    x=result_df[result_df["Anomalie"] == 1]['X'],
                                    y=result_df[result_df["Anomalie"] == 1]['Y'],
                                    mode='markers',
                                    marker=dict(
                                        color='red',
                                        size=8,
                                        symbol='x',
                                        line=dict(color='black', width=1)
                                    ),
                                    name='Anomalies'
                                ))
                                
                                # Mise en page
                                fig.update_layout(
                                    title="Carte de contour des scores d'anomalie",
                                    xaxis_title='X',
                                    yaxis_title='Y'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Les colonnes X et Y sont nécessaires pour afficher la carte des anomalies.")
                
                with tab4:
                    # Statistiques comparatives
                    st.markdown("<h3 class='subsection-header'>Comparaison des statistiques</h3>", unsafe_allow_html=True)
                    
                    # Diviser le dataframe en deux: normal et anomalies
                    normal_df = result_df[result_df["Anomalie"] == 0]
                    anomaly_df = result_df[result_df["Anomalie"] == 1]
                    
                    # Calculer les statistiques pour chaque groupe
                    stats_columns = variables
                    
                    # Créer un tableau de comparaison
                    comparison_data = []
                    
                    for col in stats_columns:
                        normal_mean = normal_df[col].mean()
                        normal_std = normal_df[col].std()
                        normal_min = normal_df[col].min()
                        normal_max = normal_df[col].max()
                        
                        if len(anomaly_df) > 0:
                            anomaly_mean = anomaly_df[col].mean()
                            anomaly_std = anomaly_df[col].std()
                            anomaly_min = anomaly_df[col].min()
                            anomaly_max = anomaly_df[col].max()
                            
                            # Différence relative en pourcentage
                            if normal_mean != 0:
                                diff_percent = (anomaly_mean - normal_mean) / abs(normal_mean) * 100
                            else:
                                diff_percent = float('inf') if anomaly_mean > 0 else float('-inf') if anomaly_mean < 0 else 0
                        else:
                            anomaly_mean = np.nan
                            anomaly_std = np.nan
                            anomaly_min = np.nan
                            anomaly_max = np.nan
                            diff_percent = np.nan
                        
                        comparison_data.append({
                            "Variable": col,
                            "Moyenne (Normal)": normal_mean,
                            "Écart-type (Normal)": normal_std,
                            "Min (Normal)": normal_min,
                            "Max (Normal)": normal_max,
                            "Moyenne (Anomalie)": anomaly_mean,
                            "Écart-type (Anomalie)": anomaly_std,
                            "Min (Anomalie)": anomaly_min,
                            "Max (Anomalie)": anomaly_max,
                            "Différence (%)": diff_percent
                        })
                    
                    # Créer le dataframe
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Afficher le tableau de comparaison
                    st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
                    
                    # Formater les nombres pour une meilleure lisibilité
                    formatted_df = comparison_df.copy()
                    for col in formatted_df.columns:
                        if col != "Variable":
                            formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2f}" if not pd.isna(x) else "-")
                    
                    st.dataframe(formatted_df, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Visualisations comparatives
                    st.markdown("<h3 class='subsection-header'>Visualisations comparatives</h3>", unsafe_allow_html=True)
                    
                    # Sélectionner une variable pour la visualisation
                    selected_var = st.selectbox("Sélectionner une variable pour la comparaison", variables)
                    
                    # Créer des histogrammes comparatifs
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=normal_df[selected_var],
                        name='Normal',
                        opacity=0.7,
                        marker_color='#1E88E5'
                    ))
                    
                    if len(anomaly_df) > 0:
                        fig.add_trace(go.Histogram(
                            x=anomaly_df[selected_var],
                            name='Anomalie',
                            opacity=0.7,
                            marker_color='#FF5252'
                        ))
                    
                    fig.update_layout(
                        title=f"Distribution de {selected_var} par groupe",
                        xaxis_title=selected_var,
                        yaxis_title='Fréquence',
                        barmode='overlay'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Boxplots comparatifs pour toutes les variables
                    st.markdown("<h3 class='subsection-header'>Boxplots comparatifs</h3>", unsafe_allow_html=True)
                    
                    # Préparer les données pour les boxplots
                    boxplot_data = []
                    
                    for var in variables:
                        for val in normal_df[var]:
                            boxplot_data.append({
                                "Variable": var,
                                "Valeur": val,
                                "Groupe": "Normal"
                            })
                        
                        if len(anomaly_df) > 0:
                            for val in anomaly_df[var]:
                                boxplot_data.append({
                                    "Variable": var,
                                    "Valeur": val,
                                    "Groupe": "Anomalie"
                                })
                    
                    boxplot_df = pd.DataFrame(boxplot_data)
                    
                    # Créer les boxplots
                    fig = px.box(
                        boxplot_df,
                        x="Variable",
                        y="Valeur",
                        color="Groupe",
                        title="Comparaison des distributions par groupe",
                        color_discrete_map={"Normal": "#1E88E5", "Anomalie": "#FF5252"}
                    )
                    
                    fig.update_layout(
                        xaxis_title="",
                        yaxis_title="Valeur",
                        boxmode="group"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Veuillez sélectionner au moins une variable pour la détection d'anomalies.")

def show_recommendation():
    st.markdown("<h1 class='title'>Recommandation de Cibles</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Identifiez les emplacements optimaux pour de futurs prélèvements</p>", unsafe_allow_html=True)
    
    # Composant pour télécharger des données si nécessaire
    if st.session_state.data is None:
        upload_data_component()
    else:
        # Bouton pour changer de jeu de données
        if st.button("📤 Changer de jeu de données"):
            upload_data_component()
    
    # Si des données sont chargées, afficher l'interface de recommandation
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Vérifier si les coordonnées X et Y sont présentes
        if 'X' not in df.columns or 'Y' not in df.columns:
            st.error("Les colonnes X et Y sont nécessaires pour générer des recommandations de cibles.")
        else:
            # Informations sur le jeu de données
            st.markdown(f"<div class='info-card'>Jeu de données actuel: <strong>{st.session_state.uploaded_file_name}</strong> | {len(df)} échantillons | {len(df.columns)} variables</div>", unsafe_allow_html=True)
            
            # Configuration de la recommandation
            st.markdown("<h2 class='section-header'>⚙️ Configuration des Recommandations</h2>", unsafe_allow_html=True)
            
            # Colonnes numériques (exclure X et Y)
            numeric_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col not in ['X', 'Y']]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sélection des variables d'intérêt
                target_cols = st.multiselect(
                    "Sélectionner les variables d'intérêt",
                    numeric_cols,
                    default=numeric_cols[:min(2, len(numeric_cols))]
                )
                
                # Nombre de recommandations
                num_reco = st.slider("Nombre de recommandations", 3, 50, 10)
            
            with col2:
                # Méthode de recommandation
                method = st.selectbox(
                    "Méthode de recommandation",
                    ["hybrid", "value", "exploration"],
                    format_func=lambda x: {
                        "hybrid": "Hybride (valeur + exploration)",
                        "value": "Zones de haute valeur",
                        "exploration": "Zones sous-échantillonnées"
                    }.get(x)
                )
                
                # Si hybride, ajouter le paramètre de balance
                if method == "hybrid":
                    exploration_weight = st.slider(
                        "Balance exploration/exploitation",
                        0.0, 1.0, 0.5, 0.1,
                        help="0 = uniquement zones de haute valeur, 1 = uniquement zones sous-échantillonnées"
                    )
                else:
                    exploration_weight = 0.5  # Valeur par défaut
                
                # Distance minimale entre les recommandations
                min_distance = st.slider(
                    "Distance minimale entre recommandations (unités)",
                    10, 1000, 100
                )
            
            # Paramètres avancés
            with st.expander("Paramètres avancés"):
                # Contraintes spatiales
                enable_constraints = st.checkbox("Ajouter des contraintes spatiales", value=False)
                
                if enable_constraints:
                    constraint_type = st.selectbox(
                        "Type de contrainte",
                        ["rectangle", "circle", "polygon"]
                    )
                    
                    if constraint_type == "rectangle":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            x_min = st.number_input("X minimum", value=float(df['X'].min()))
                            y_min = st.number_input("Y minimum", value=float(df['Y'].min()))
                        
                        with col2:
                            x_max = st.number_input("X maximum", value=float(df['X'].max()))
                            y_max = st.number_input("Y maximum", value=float(df['Y'].max()))
                        
                        constraint_params = {
                            "type": "rectangle",
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max
                        }
                    
                    elif constraint_type == "circle":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            center_x = st.number_input("Centre X", value=float(df['X'].mean()))
                            center_y = st.number_input("Centre Y", value=float(df['Y'].mean()))
                        
                        with col2:
                            radius = st.number_input("Rayon", value=float(min(df['X'].max() - df['X'].min(), df['Y'].max() - df['Y'].min()) / 4))
                        
                        constraint_params = {
                            "type": "circle",
                            "center_x": center_x,
                            "center_y": center_y,
                            "radius": radius
                        }
                    
                    elif constraint_type == "polygon":
                        st.info("Pour définir un polygone, entrez les coordonnées des sommets (une paire X,Y par ligne).")
                        
                        polygon_coords = st.text_area(
                            "Coordonnées des sommets (format: X,Y)",
                            value=f"{df['X'].min()},{df['Y'].min()}\n{df['X'].max()},{df['Y'].min()}\n{df['X'].max()},{df['Y'].max()}\n{df['X'].min()},{df['Y'].max()}"
                        )
                        
                        # Parsing des coordonnées
                        try:
                            polygon_points = []
                            for line in polygon_coords.strip().split('\n'):
                                x, y = line.split(',')
                                polygon_points.append((float(x.strip()), float(y.strip())))
                            
                            constraint_params = {
                                "type": "polygon",
                                "points": polygon_points
                            }
                        except:
                            st.error("Format de coordonnées invalide. Utilisez le format 'X,Y' avec une paire par ligne.")
                            constraint_params = None
                else:
                    constraint_params = None
            
            # Vérification des variables sélectionnées
            if not target_cols:
                st.warning("Veuillez sélectionner au moins une variable d'intérêt.")
            else:
                # Bouton pour générer les recommandations
                gen_button = st.button("🎯 Générer les recommandations")
                
                if gen_button:
                    # Affichage d'une barre de progression
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Génération des recommandations
                    status_text.text("Génération des recommandations en cours...")
                    progress_bar.progress(30)
                    
                    # Appliquer les contraintes spatiales si nécessaire
                    if enable_constraints and constraint_params:
                        # Fonction pour vérifier si un point est dans la contrainte
                        def is_in_constraint(point, params):
                            x, y = point
                            
                            if params["type"] == "rectangle":
                                return (params["x_min"] <= x <= params["x_max"] and
                                        params["y_min"] <= y <= params["y_max"])
                            
                            elif params["type"] == "circle":
                                dx = x - params["center_x"]
                                dy = y - params["center_y"]
                                return dx**2 + dy**2 <= params["radius"]**2
                            
                            elif params["type"] == "polygon":
                                from matplotlib.path import Path
                                polygon_path = Path(params["points"])
                                return polygon_path.contains_point((x, y))
                            
                            return True  # Par défaut, pas de contrainte
                        
                        # Créer une copie du dataframe pour appliquer les contraintes
                        constrained_df = df.copy()
                        
                        # Filtrer les points en dehors des contraintes
                        mask = constrained_df.apply(lambda row: is_in_constraint((row['X'], row['Y']), constraint_params), axis=1)
                        constrained_df = constrained_df[mask]
                        
                        if len(constrained_df) == 0:
                            st.error("Aucun point ne satisfait les contraintes spatiales. Veuillez ajuster les paramètres.")
                            progress_bar.empty()
                            status_text.empty()
                        else:
                            # Générer les recommandations avec le dataframe filtré
                            if method == "hybrid":
                                recommendations, grid_info = generate_sampling_recommendations(
                                    constrained_df, target_cols, num_reco, min_distance, method, exploration_weight
                                )
                            else:
                                recommendations, grid_info = generate_sampling_recommendations(
                                    constrained_df, target_cols, num_reco, min_distance, method
                                )
                            
                            progress_bar.progress(100)
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Afficher les résultats
                            if recommendations is not None:
                                st.success(f"✅ {len(recommendations)} recommandations générées avec succès!")
                                
                                # Afficher les résultats
                                show_recommendation_results(recommendations, grid_info, constrained_df, target_cols)
                            else:
                                st.error("Une erreur s'est produite lors de la génération des recommandations.")
                    else:
                        # Générer les recommandations sans contraintes
                        if method == "hybrid":
                            recommendations, grid_info = generate_sampling_recommendations(
                                df, target_cols, num_reco, min_distance, method, exploration_weight
                            )
                        else:
                            recommendations, grid_info = generate_sampling_recommendations(
                                df, target_cols, num_reco, min_distance, method
                            )
                        
                        progress_bar.progress(100)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Afficher les résultats
                        if recommendations is not None:
                            st.success(f"✅ {len(recommendations)} recommandations générées avec succès!")
                            
                            # Afficher les résultats
                            show_recommendation_results(recommendations, grid_info, df, target_cols)
                        else:
                            st.error("Une erreur s'est produite lors de la génération des recommandations.")

# Fonction pour afficher les résultats des recommandations
def show_recommendation_results(recommendations, grid_info, df, target_cols):
    """Affiche les résultats des recommandations de cibles."""
    st.markdown("<h2 class='section-header'>📊 Résultats</h2>", unsafe_allow_html=True)
    
    # Onglets pour différentes visualisations
    tab1, tab2, tab3 = st.tabs(["Carte des recommandations", "Tableau des résultats", "Carte de chaleur d'intérêt"])
    
    with tab1:
        st.markdown("<h3 class='subsection-header'>Carte des recommandations</h3>", unsafe_allow_html=True)
        
        # Préparer les données pour la carte
        existing_df = df.copy()
        existing_df["Type"] = "Existant"
        
        reco_df = recommendations.copy()
        reco_df["Type"] = "Recommandation"
        
        # Fusionner les données pour la visualisation
        plot_data = pd.concat([
            existing_df[["X", "Y", "Type"]],
            reco_df[["X", "Y", "Type", "Rang"]]
        ])
        
        # Créer la carte
        fig = px.scatter(
            plot_data,
            x="X", y="Y",
            color="Type",
            symbol="Type",
            color_discrete_map={"Existant": "#1E88E5", "Recommandation": "#FF5252"},
            symbol_map={"Existant": "circle", "Recommandation": "star"},
            title="Carte des recommandations de cibles",
            labels={"X": "Coordonnée X", "Y": "Coordonnée Y"}
        )
        
        # Ajouter des annotations pour les rangs des recommandations
        for _, row in reco_df.iterrows():
            fig.add_annotation(
                x=row["X"],
                y=row["Y"],
                text=str(int(row["Rang"])),
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#FF5252",
                font=dict(size=12, color="white"),
                bgcolor="#FF5252",
                bordercolor="#FF5252",
                borderwidth=2,
                borderpad=4,
                ax=20,
                ay=-30
            )
        
        # Mise en page de la figure
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Afficher la carte
        st.plotly_chart(fig, use_container_width=True)
        
        # Afficher les distances entre les recommandations et les points existants
        with st.expander("Analyse des distances"):
            st.markdown("### Distances des recommandations aux points existants")
            
            # Calculer les distances minimales de chaque recommandation aux points existants
            existing_points = df[["X", "Y"]].values
            
            distance_data = []
            
            for idx, row in recommendations.iterrows():
                point = np.array([row["X"], row["Y"]])
                distances = np.sqrt(np.sum((existing_points - point)**2, axis=1))
                min_dist = np.min(distances)
                
                distance_data.append({
                    "Rang": int(row["Rang"]),
                    "X": row["X"],
                    "Y": row["Y"],
                    "Distance min.": min_dist,
                    "Point le plus proche": np.argmin(distances)
                })
            
            # Créer un DataFrame avec les distances
            distance_df = pd.DataFrame(distance_data)
            
            # Afficher le tableau
            st.dataframe(distance_df.sort_values(by="Rang"), use_container_width=True)
            
            # Histogramme des distances
            fig = px.histogram(
                distance_df,
                x="Distance min.",
                title="Distribution des distances minimales aux points existants",
                color_discrete_sequence=["#FF7043"]
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("<h3 class='subsection-header'>Tableau des recommandations</h3>", unsafe_allow_html=True)
        
        # Afficher le tableau des recommandations
        st.dataframe(recommendations, use_container_width=True)
        
        # Téléchargement des résultats
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les recommandations (CSV)",
            data=csv,
            file_name="recommandations_cibles.csv",
            mime="text/csv"
        )
        
        # Statistiques sur les valeurs estimées
        if len(target_cols) > 0:
            st.markdown("<h3 class='subsection-header'>Valeurs estimées</h3>", unsafe_allow_html=True)
            
            # Calculer des statistiques sur les valeurs estimées
            stats_data = []
            
            for col in target_cols:
                col_data = {
                    "Variable": col,
                    "Moyenne (existant)": df[col].mean(),
                    "Médiane (existant)": df[col].median(),
                    "Min (existant)": df[col].min(),
                    "Max (existant)": df[col].max(),
                    "Moyenne (estimée)": recommendations[f"{col}_estimé"].mean(),
                    "Médiane (estimée)": recommendations[f"{col}_estimé"].median(),
                    "Min (estimée)": recommendations[f"{col}_estimé"].min(),
                    "Max (estimée)": recommendations[f"{col}_estimé"].max()
                }
                
                stats_data.append(col_data)
            
            # Créer un DataFrame avec les statistiques
            stats_df = pd.DataFrame(stats_data)
            
            # Afficher le tableau
            st.dataframe(stats_df, use_container_width=True)
            
            # Graphique comparatif des distributions
            comparison_data = []
            
            for col in target_cols:
                for val in df[col]:
                    comparison_data.append({
                        "Variable": col,
                        "Valeur": val,
                        "Type": "Existant"
                    })
                
                for val in recommendations[f"{col}_estimé"]:
                    comparison_data.append({
                        "Variable": col,
                        "Valeur": val,
                        "Type": "Estimé"
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Créer les boxplots comparatifs
            fig = px.box(
                comparison_df,
                x="Variable",
                y="Valeur",
                color="Type",
                title="Comparaison des distributions: valeurs existantes vs estimées",
                color_discrete_map={"Existant": "#1E88E5", "Estimé": "#FF7043"}
            )
            
            fig.update_layout(boxmode="group")
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("<h3 class='subsection-header'>Carte de chaleur d'intérêt</h3>", unsafe_allow_html=True)
        
        # Récupérer les données de la grille pour la carte de chaleur
        grid_x = grid_info["grid_x"]
        grid_y = grid_info["grid_y"]
        grid_scores = grid_info["grid_scores"]
        
        # Créer la carte de chaleur
        fig = go.Figure()
        
        # Ajouter la heatmap
        fig.add_trace(go.Heatmap(
            z=grid_scores,
            x=grid_x,
            y=grid_y,
            colorscale="Viridis",
            colorbar=dict(title="Score d'intérêt")
        ))
        
        # Ajouter les points existants
        fig.add_trace(go.Scatter(
            x=df["X"],
            y=df["Y"],
            mode="markers",
            marker=dict(
                color="white",
                size=8,
                line=dict(
                    color="black",
                    width=1
                )
            ),
            name="Points existants"
        ))
        
        # Ajouter les recommandations
        fig.add_trace(go.Scatter(
            x=recommendations["X"],
            y=recommendations["Y"],
            mode="markers+text",
            marker=dict(
                color="red",
                size=10,
                symbol="star",
                line=dict(
                    color="white",
                    width=1
                )
            ),
            text=recommendations["Rang"],
            textposition="top center",
            name="Recommandations"
        ))
        
        # Mise en page
        fig.update_layout(
            title="Carte de chaleur d'intérêt pour l'exploration",
            xaxis_title="Coordonnée X",
            yaxis_title="Coordonnée Y"
        )
        
        # Afficher la carte
        st.plotly_chart(fig, use_container_width=True)
        
        # Explication des scores
        with st.expander("Comprendre les scores d'intérêt"):
            st.markdown("""
            ### Interprétation des scores d'intérêt
            
            Le score d'intérêt est calculé en fonction de la méthode sélectionnée:
            
            - **Zones de haute valeur**: Le score est basé uniquement sur les valeurs interpolées des variables d'intérêt. Les zones ayant des valeurs élevées ont des scores plus élevés.
            
            - **Zones sous-échantillonnées**: Le score est basé uniquement sur la distance aux points existants. Les zones éloignées des points existants ont des scores plus élevés.
            
            - **Hybride**: Le score est une combinaison pondérée des deux approches précédentes, contrôlée par le paramètre de balance exploration/exploitation.
            
            Les recommandations sont les points ayant les scores les plus élevés, tout en respectant la distance minimale entre elles.
            """)

# Lancement de l'application
if __name__ == "__main__":
    main()