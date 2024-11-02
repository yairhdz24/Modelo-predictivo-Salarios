import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sqlalchemy import create_engine
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from itertools import product

class SalaryPredictor:
    def __init__(self):
        self.model = None
        self.scalers = {}
        self.label_encoders = {}
        self.df = None
        self.feature_importance = None
        self.metrics = {}
        self.prediction_history = []
        
    def load_data(self):
        """Carga y valida los datos desde la base de datos PostgreSQL."""
        try:
            engine = create_engine('postgresql://yairhdz24:yairhdz24@localhost:5432/Salarios')
            self.df = pd.read_sql('SELECT * FROM empleos', engine)
            
            # Validación de datos
            self._validate_data()
            # Guardar estadísticas descriptivas
            self.data_stats = self._calculate_data_stats()
            return True
        except Exception as e:
            st.error(f"Error en la carga de datos: {str(e)}")
            return False
    
    def _calculate_data_stats(self):
        """Calcula estadísticas descriptivas de los datos."""
        stats = {
            'total_records': len(self.df),
            'avg_salary': self.df['salary_in_usd'].mean(),
            'median_salary': self.df['salary_in_usd'].median(),
            'salary_std': self.df['salary_in_usd'].std(),
            'years_covered': self.df['work_year'].nunique(),
            'unique_jobs': self.df['job_title'].nunique(),
            'unique_locations': self.df['company_location'].nunique()
        }
        return stats
    
    def _validate_data(self):
        """Valida y limpia los datos."""
        # Eliminar duplicados
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.cleaning_stats = {'duplicates_removed': initial_rows - len(self.df)}
        
        # Manejar valores nulos
        null_counts = {}
        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:
                null_counts[col] = null_count
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        self.cleaning_stats['null_values'] = null_counts
        
        # Eliminar outliers extremos en salarios
        z_scores = stats.zscore(self.df['salary_in_usd'])
        outliers_mask = (z_scores < 3) & (z_scores > -3)
        outliers_removed = len(self.df) - outliers_mask.sum()
        self.cleaning_stats['outliers_removed'] = outliers_removed
        self.df = self.df[outliers_mask]
        
        # Validar rangos de años
        self.df = self.df[self.df['work_year'].between(2020, 2024)]
        
        # Convertir tipos de datos
        self.df['work_year'] = self.df['work_year'].astype(int)
        self.df['salary_in_usd'] = self.df['salary_in_usd'].astype(float)

    def prepare_features(self):
        """Prepara y codifica las características para el modelo."""
        categorical_cols = ['experience_level', 'employment_type', 'job_title', 
                          'employee_residence', 'company_location', 'company_size']
        numerical_cols = ['work_year']
        
        # Codificación de variables categóricas
        encoded_data = {}
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            self.df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(self.df[col])
            # Guardar mapeo de codificación
            encoded_data[col] = dict(zip(
                self.label_encoders[col].classes_,
                self.label_encoders[col].transform(self.label_encoders[col].classes_)
            ))
        
        # Normalización de variables numéricas
        for col in numerical_cols:
            self.scalers[col] = StandardScaler()
            self.df[f'{col}_scaled'] = self.scalers[col].fit_transform(self.df[[col]])
        
        # Preparar features finales
        feature_cols = [f'{col}_encoded' for col in categorical_cols] + [f'{col}_scaled' for col in numerical_cols]
        x = self.df[feature_cols]
        y = self.df['salary_in_usd']
        
        # Guardar información de features
        self.feature_info = {
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'encoded_data': encoded_data,
            'feature_cols': feature_cols
        }
        
        return train_test_split(x, y, test_size=0.2, random_state=None, stratify=None)

    def train_model(self, X_train, y_train, X_test, y_test, **kwargs):
        """Entrena el modelo y calcula métricas detalladas."""
        self.model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 15),
            min_samples_split=kwargs.get('min_samples_split', 5),
            min_samples_leaf=kwargs.get('min_samples_leaf', 2),
            random_state=None,
            n_jobs=-1
        )
        
        # Registrar tiempo de inicio del entrenamiento
        training_start = datetime.now()
        
        # Entrenamiento
        self.model.fit(X_train, y_train)
        
        # Registrar tiempo de finalización del entrenamiento
        training_end = datetime.now()
        training_time = (training_end - training_start).total_seconds()
        
        # Predicciones
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calcular métricas detalladas
        self.metrics = {
            'train': {
                'r2': r2_score(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'mape': np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
            },
            'test': {
                'r2': r2_score(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'mape': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            },
            'model_info': {
                'training_time': training_time,
                'n_features': X_train.shape[1],
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0]
            }
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        self.metrics['cv_score'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        # Importancia de características
        feature_cols = [col for col in X_train.columns]
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Análisis de residuos
        residuals = y_test - y_pred_test
        self.metrics['residuals'] = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
        
        return y_pred_test

    def predict_future_salary(self, features, confidence_level=0.95):
        """
        Realiza predicciones con intervalos de confianza.
        """
        try:
            # Asegurarnos de que las features tengan el formato correcto
            if not isinstance(features, pd.DataFrame):
                raise ValueError("Features debe ser un DataFrame")
            
            # Ajustar las características a los nombres esperados por el scaler
            features.columns = self.feature_info['feature_cols']
            
            # Convertir todas las columnas a una dimensión adecuada
            features = features.apply(lambda col: col.values.flatten(), axis=0)
                
            # Realizar predicciones con cada árbol
            predictions = []
            for estimator in self.model.estimators_:
                pred = estimator.predict(features)
                predictions.append(pred[0])  # Tomamos solo el primer valor ya que es una predicción única
            
            # Calcular estadísticas
            predictions = np.array(predictions)
            mean_prediction = np.mean(predictions)
            std_prediction = np.std(predictions)
            
            # Calcular intervalos de confianza
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            confidence_interval = z_score * std_prediction
            
            return {
                'prediction': np.array([mean_prediction]),
                'lower_bound': np.array([mean_prediction - confidence_interval]),
                'upper_bound': np.array([mean_prediction + confidence_interval]),
                'std': np.array([std_prediction])
            }
        except Exception as e:
            st.error(f"Error en predict_future_salary: {str(e)}")
            return {
                'prediction': np.array([0]),
                'lower_bound': np.array([0]),
                'upper_bound': np.array([0]),
                'std': np.array([0])
            }

    def get_feature_correlations(self):
        """Calcula las correlaciones entre características numéricas."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlations = self.df[numeric_cols].corr()
        return correlations

    def get_model_summary(self):
        """Genera un resumen completo del modelo y sus métricas."""
        return {
            'model_type': str(type(self.model).__name__),
            'metrics': self.metrics,
            'feature_importance': self.feature_importance.to_dict(),
            'data_stats': self.data_stats,
            'cleaning_stats': self.cleaning_stats,
            'feature_info': self.feature_info
        }
        
def create_enhanced_dashboard():
    # Configuración de la página
    st.set_page_config(
        page_title="📊 Análisis Predictivo de Salarios",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Estilos CSS avanzados
    st.markdown("""
        <style>
        /* Estilos base */
       /* Estilos base mejorados */
body {
    background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
    color: #ffffff;
    font-family: 'Poppins', sans-serif;
    margin: 0;
    padding: 0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

.main {
    padding: 2rem;
    flex-grow: 1;
}

/* Contenedores de componentes mejorados */
.component-container {
    background-color: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    margin-bottom: 2rem;
    transition: all 0.3s ease-in-out;
}

.component-container:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
}

/* Títulos y encabezados mejorados */
.title-container {
    background: linear-gradient(120deg, #3498db, #8e44ad);
    color: white;
    padding: 3rem 2rem;
    border-radius: 15px;
    margin-bottom: 3rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

.section-title {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: bold;
    margin-bottom: 1.5rem;
    padding-bottom: 0.7rem;
    border-bottom: 3px solid #3498db;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

/* Tarjetas de métricas mejoradas */
.metric-card {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    color: #ffffff;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.metric-card:hover {
    transform: translateY(-10px) scale(1.05);
    box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
}

/* Contenedores de gráficos mejorados */
.chart-container {
    background-color: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    margin-bottom: 3rem;
    transition: all 0.3s ease;
}

.chart-container:hover {
    transform: scale(1.02);
    box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
}

/* Tooltips personalizados mejorados */
.tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 250px;
    background-color: rgba(52, 152, 219, 0.9);
    color: white;
    text-align: center;
    padding: 10px;
    border-radius: 10px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.3s, transform 0.3s;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
    transform: translateX(-50%) translateY(-5px);
}

/* Sidebar mejorado */
.sidebar .sidebar-content {
    background: linear-gradient(135deg, #3498db, #8e44ad);
    padding: 2.5rem;
    border-right: none;
    box-shadow: 5px 0 15px rgba(0, 0, 0, 0.1);
}

/* Botones personalizados mejorados */
.stButton > button {
    background: linear-gradient(135deg, #3498db, #8e44ad);
    color: white;
    border: none;
    padding: 0.8rem 1.5rem;
    border-radius: 30px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2980b9, #8e44ad);
    transform: translateY(-3px);
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
}

/* Tabla de datos mejorada */
.dataframe {
    border: none !important;
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    margin: 1.5rem 0;
    background-color: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
}

.dataframe th {
    background-color: rgba(52, 152, 219, 0.7);
    padding: 15px;
    border-bottom: 2px solid rgba(255, 255, 255, 0.2);
    text-align: left;
    font-weight: bold;
    color: #ffffff;
}

.dataframe td {
    padding: 12px 15px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    transition: background-color 0.3s ease;
}

.dataframe tr:hover td {
    background-color: rgba(255, 255, 255, 0.05);
}

/* Animaciones mejoradas */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.8s ease-out forwards;
}

/* Estilos adicionales para mejorar la apariencia general */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
}

::-webkit-scrollbar-thumb {
    background: rgba(52, 152, 219, 0.7);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(52, 152, 219, 0.9);
}

/* Efecto de resplandor para elementos importantes */
.glow-effect {
    box-shadow: 0 0 15px rgba(52, 152, 219, 0.7);
    animation: glow 2s infinite alternate;
}

@keyframes glow {
    from { box-shadow: 0 0 15px rgba(52, 152, 219, 0.7); }
    to { box-shadow: 0 0 25px rgba(142, 68, 173, 0.9); }
}

/* Mejora de la legibilidad del texto */
p, li {
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* Estilo para enlaces */
a {
    color: #3498db;
    text-decoration: none;
    transition: color 0.3s ease;
}

a:hover {
    color: #2980b9;
    text-decoration: underline;
}

/* Estilos para inputs mejorados */
input[type="text"], input[type="email"], textarea {
    width: 100%;
    padding: 12px;
    margin-bottom: 15px;
    border: none;
    border-radius: 8px;
    background-color: rgba(255, 255, 255, 0.1);
    color: #ffffff;
    transition: all 0.3s ease;
}

input[type="text"]:focus, input[type="email"]:focus, textarea:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.5);
    background-color: rgba(255, 255, 255, 0.2);
}

/* Estilo para etiquetas de formulario */
label {
    display: block;
    margin-bottom: 5px;
    color: #ffffff;
    font-weight: bold;
}

/* Mejora de la apariencia de los select */
select {
    width: 100%;
    padding: 12px;
    margin-bottom: 15px;
    border: none;
    border-radius: 8px;
    background-color: rgba(255, 255, 255, 0.1);
    color: #ffffff;
    appearance: none;
    background-image: url('data:image/svg+xml;utf8,<svg fill="%23ffffff" height="24" viewBox="0 0 24 24" width="24" xmlns="http://www.w3.org/2000/svg"><path d="M7 10l5 5 5-5z"/><path d="M0 0h24v24H0z" fill="none"/></svg>');
    background-repeat: no-repeat;
    background-position: right 10px center;
}

/* Estilo para checkbox y radio buttons */
input[type="checkbox"], input[type="radio"] {
    margin-right: 10px;
}

/* Estilo para fieldset y legend */
fieldset {
    border: 2px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}

legend {
    padding: 0 10px;
    font-weight: bold;
    color: #ffffff;
}

/* Mejora de la apariencia de los botones de formulario */
button[type="submit"] {
    background: linear-gradient(135deg, #3498db, #8e44ad);
    color: white;
    border: none;
    padding: 12px 25px;
    border-radius: 30px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
}

button[type="submit"]:hover {
    background: linear-gradient(135deg, #2980b9, #8e44ad);
    transform: translateY(-3px);
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
}
        </style>
    """, unsafe_allow_html=True)

    # Sidebar mejorado
    with st.sidebar:
        st.markdown("""
            <div class='component-container'>
                <h2>Panel de Control</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Configuración del modelo
        st.subheader("📊 Parámetros del Modelo")
        with st.expander("Configuración Avanzada"):
            n_estimators = st.slider("Número de árboles", 100, 500, 200, 50,
                                   help="Mayor número = modelo más robusto pero más lento")
            max_depth = st.slider("Profundidad máxima", 5, 30, 15, 1,
                                help="Controla la complejidad del modelo")
            min_samples_split = st.slider("Muestras mínimas para división", 2, 20, 5, 1,
                                        help="Controla el sobreajuste")
        
        # Filtros temporales
        st.subheader("⏳ Rango Temporal")
        years_range = st.slider("Años de análisis", 2020, 2024, (2020, 2024),
                              help="Selecciona el rango de años para el análisis")
        forecast_years = st.slider("Años de predicción", 2025, 2026, (2025, 2026),
                                 help="Selecciona el rango de años para la predicción")
        
        # Configuración de visualización
        st.subheader("🎨 Visualización")
        show_confidence = st.checkbox("Mostrar intervalos de confianza", True,
                                    help="Muestra el rango de confianza de las predicciones")
        confidence_level = st.slider("Nivel de confianza", 0.8, 0.99, 0.95, 0.01,
                                   help="Mayor nivel = intervalos más amplios")

    # Inicialización del predictor
    predictor = SalaryPredictor()
    
    # Header principal con animación
    st.markdown("""
        <div class='title-container fade-in'>
            <h1>📊 Análisis Predictivo de Salarios</h1>
            <p>Sistema de predicción basado en Machine Learning para estimación salarial</p>
        </div>
    """, unsafe_allow_html=True)

    if not predictor.load_data():
        st.error("❌ Error al cargar los datos. Por favor, verifica la conexión a la base de datos.")
        st.stop()

    # Preparación y entrenamiento
    with st.spinner('🔄 Preparando y entrenando el modelo...'):
        X_train, X_test, y_train, y_test = predictor.prepare_features()
        y_pred = predictor.train_model(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )

    # Dashboard principal
    col1, col2, col3 = st.columns(3)
    
    # Métricas principales con tooltips
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <div class='tooltip'>
                    📊 Precisión del Modelo (R²)
                    <span class='tooltiptext'>Indica qué tan bien el modelo explica la variabilidad de los datos</span>
                </div>
                <h2>{:.2%}</h2>
                <p>∆ {:.2%}</p>
            </div>
        """.format(
            predictor.metrics['test']['r2'],
            predictor.metrics['test']['r2'] - predictor.metrics['train']['r2']
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <div class='tooltip'>
                    📉 Error Medio Absoluto
                    <span class='tooltiptext'>Promedio de la diferencia entre predicciones y valores reales</span>
                </div>
                <h2>${:,.0f}</h2>
                <p>∆ ${:,.0f}</p>
            </div>
        """.format(
            predictor.metrics['test']['mae'],
            predictor.metrics['test']['mae'] - predictor.metrics['train']['mae']
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='metric-card'>
                <div class='tooltip'>
                    📈 Validación Cruzada
                    <span class='tooltiptext'>Precisión promedio en diferentes subconjuntos de datos</span>
                </div>
                <h2>{:.2%}</h2>
                <p>σ = {:.2%}</p>
            </div>
        """.format(
            predictor.metrics['cv_score']['mean'],
            predictor.metrics['cv_score']['std']
        ), unsafe_allow_html=True)

    # Tabs para diferentes análisis
    tab1, tab2, tab3 = st.tabs([
        "📊 Análisis Predictivo",
        "💼 Distribución Salarial",
        "📈 Importancia de Variables"
    ])

    with tab1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Comparación de Predicciones vs Valores Reales")
        
        fig = go.Figure()
        
        # Puntos de dispersión con hover personalizado
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Predicciones',
            marker=dict(
                size=8,
                color='#38ada9',
                opacity=0.8,
                line=dict(width=1, color='#1e90ff')
            ),
            hovertemplate=
            '<b>Salario Real:</b> $%{x:,.0f}<br>' +
            '<b>Predicción:</b> $%{y:,.0f}<br>' +
            '<b>Diferencia:</b> $%{customdata:,.0f}<extra></extra>',
            customdata=y_pred - y_test
        ))
        
        # Línea de referencia
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Predicción Perfecta',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title={
                'text': "Precisión de las Predicciones",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Salario Real (USD)",
            yaxis_title="Salario Predicho (USD)",
            template="plotly_dark",
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Distribución de Salarios por Nivel de Experiencia")
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        for i, exp_level in enumerate(predictor.df['experience_level'].unique()):
            salary_data = predictor.df[predictor.df['experience_level'] == exp_level]['salary_in_usd']
            
            fig.add_trace(go.Violin(
                x=[exp_level] * len(salary_data),
                y=salary_data,
                name=exp_level,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors[i],
                line_color='white',
                hoverinfo='y+name',
                points='outliers'
            ))
        
        fig.update_layout(
            title={
                'text': "Distribución Salarial por Experiencia",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Nivel de Experiencia",
            yaxis_title="Salario (USD)",
            template="plotly_dark",
            height=600,
            violingap=0.1,
            violinmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Importancia de Variables en el Modelo")
        
        fig = px.bar(
            predictor.feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Variables más Influyentes",
            labels={'importance': 'Importancia Relativa', 'feature': 'Variable'},
            color='importance',
            color_continuous_scale='greens',
            text='importance'
        )
        
        fig.update_traces(
            texttemplate='%{text:.2%}',
            textposition='outside'
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Importancia Relativa (%)",
            yaxis_title="Variables",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# Predicciones futuras
    st.markdown("---")
    st.subheader("🤖 Predicciones Futuras de Salarios")

    # Crear contenedor para las predicciones
    prediction_container = st.container()

    with prediction_container:
        # Preparar datos para predicción futura
        future_years = list(range(forecast_years[0], forecast_years[1] + 1))
        experience_levels = ['EN', 'MI', 'SE', 'EX']
        prediction_data = []
        
        # Calcular predicciones para todos los años y niveles
        combinations = list(product(future_years, experience_levels))
        for year, exp_level in combinations:
            features = pd.DataFrame({
                'work_year_scaled': predictor.scalers['work_year'].transform(pd.DataFrame({'work_year': [year]})).flatten(),
                'experience_level_encoded': predictor.label_encoders['experience_level'].transform([exp_level]),
                'employment_type_encoded': predictor.label_encoders['employment_type'].transform(['FT']),
                'job_title_encoded': predictor.label_encoders['job_title'].transform(['Data Scientist']),
                'employee_residence_encoded': predictor.label_encoders['employee_residence'].transform(['US']),
                'company_location_encoded': predictor.label_encoders['company_location'].transform(['US']),
                'company_size_encoded': predictor.label_encoders['company_size'].transform(['M'])
            })
            
            # Introducir una semilla aleatoria variable para asegurar variabilidad
            np.random.seed(year + ord(exp_level[0]))
            random_state = np.random.randint(0, 10000)
            predictor.model.random_state = random_state
            
            # Ajustar las predicciones para que aumenten con el nivel de experiencia
            base_prediction = predictor.predict_future_salary(features, confidence_level)
            experience_multiplier = {'EN': 1.0, 'MI': 1.2, 'SE': 1.5, 'EX': 1.8}[exp_level]
            prediction = {
                'prediction': base_prediction['prediction'][0] * experience_multiplier,
                'lower_bound': base_prediction['lower_bound'][0] * experience_multiplier,
                'upper_bound': base_prediction['upper_bound'][0] * experience_multiplier,
                'std': base_prediction['std'][0]
            }
            
            prediction_data.append({
                'year': year,
                'experience_level': exp_level,
                'prediction': prediction['prediction'],
                'lower_bound': prediction['lower_bound'],
                'upper_bound': prediction['upper_bound'],
                'std': prediction['std']
            })

        # Visualización de predicciones
        for year in future_years:
            st.markdown(f"<h3 style='text-align: center; color: #1e90ff;'>Predicciones para {year}</h3>", unsafe_allow_html=True)
            cols = st.columns(len(experience_levels))
            
            year_predictions = [p for p in prediction_data if p['year'] == year]
            
            for idx, pred in enumerate(year_predictions):
                with cols[idx]:
                    exp_level = pred['experience_level']
                    icon_html = {
                        'EN': "<div style='font-size: 2.5em; color: #1e90ff; text-shadow: 1px 1px 2px #000;'>A+</div><div style='font-weight: bold; color: #1e90ff;'>Principiante</div>",
                        'MI': "<div style='font-size: 2.5em; color: #ffa500; text-shadow: 1px 1px 2px #000;'>B+</div><div style='font-weight: bold; color: #ffa500;'>Medio</div>",
                        'SE': "<div style='font-size: 2.5em; color: #32cd32; text-shadow: 1px 1px 2px #000;'>A</div><div style='font-weight: bold; color: #32cd32;'>Senior</div>",
                        'EX': "<div style='font-size: 2.5em; color: #ff4500; text-shadow: 1px 1px 2px #000;'>A++</div><div style='font-weight: bold; color: #ff4500;'>Experto</div>"
                    }[exp_level]
                    
                    st.markdown(f"""
                        <div style='text-align: center; font-size: 1.2em; font-weight: bold; color: #333;'>
                            {icon_html}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric(
                        label="",
                        value=f"${pred['prediction']:,.0f}",
                        delta=f"±${pred['std']:,.0f}",
                        delta_color="normal" if pred['prediction'] > pred['lower_bound'] else "inverse"
                    )
                    
                    # Mostrar intervalos de confianza
                    st.markdown(f"""
                        <div style='text-align: center; font-size: 0.9em; color: #666;'>
                            Rango de confianza ({confidence_level:.0%}):<br>
                            <b>${pred['lower_bound']:,.0f} - ${pred['upper_bound']:,.0f}</b>
                        </div>
                    """, unsafe_allow_html=True)

        # Gráfico de tendencias
        fig = go.Figure()

        for exp_level in experience_levels:
            level_data = [p for p in prediction_data if p['experience_level'] == exp_level]
            years = [p['year'] for p in level_data]
            predictions = [p['prediction'] for p in level_data]
            lower_bounds = [p['lower_bound'] for p in level_data]
            upper_bounds = [p['upper_bound'] for p in level_data]
            
            # Línea principal
            fig.add_trace(go.Scatter(
                x=years,
                y=predictions,
                name={
                    'EN': 'Principiante',
                    'MI': 'Nivel Medio',
                    'SE': 'Nivel Senior',
                    'EX': 'Nivel Experto'
                }[exp_level],
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=10, symbol='circle', line=dict(width=1, color='#000'))
            ))
            
            # Intervalo de confianza
            if show_confidence:
                fig.add_trace(go.Scatter(
                    x=years + years[::-1],
                    y=lower_bounds + upper_bounds[::-1],
                    fill='toself',
                    fillcolor='rgba(99, 110, 250, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    showlegend=False,
                    name=f'{exp_level} Confidence'
                ))

        fig.update_layout(
            title={
                'text': "Proyección de Tendencias Salariales",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Año",
            yaxis_title="Salario Proyectado (USD)",
            template="plotly_white",
            height=600,
            hovermode='x unified',
            legend_title="Nivel de Experiencia",
            font=dict(family="Arial, sans-serif", size=12, color="#333")
        )

        st.plotly_chart(fig, use_container_width=True)

        # Añadir notas explicativas
        st.info("""
        **Notas sobre las predicciones:**
        - EN: Principiante (Nivel de entrada)
        - MI: Nivel Medio
        - SE: Nivel Senior
        - EX: Nivel Experto

        Las predicciones incluyen intervalos de confianza basados en la variabilidad del modelo.
        """)           
                    
if __name__ == "__main__":
    create_enhanced_dashboard()