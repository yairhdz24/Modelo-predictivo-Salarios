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
        X = self.df[feature_cols]
        y = self.df['salary_in_usd']
        
        # Guardar información de features
        self.feature_info = {
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'encoded_data': encoded_data,
            'feature_cols': feature_cols
        }
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

    def train_model(self, X_train, y_train, X_test, y_test):
        """Entrena el modelo y calcula métricas detalladas."""
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
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
        page_title="🎯 Análisis Predictivo Avanzado de Salarios",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Estilos CSS avanzados
    st.markdown("""
        <style>
        /* Estilos base */
        .main {
            padding: 2rem;
            background-color: #f8f9fa;
        }
        
        /* Contenedores de componentes */
        .component-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
            transition: transform 0.2s ease-in-out;
        }
        
        .component-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        
        /* Títulos y encabezados */
        .title-container {
            background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .section-title {
            color: #2a5298;
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e5e7eb;
        }
        
        /* Tarjetas de métricas */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: scale(1.02);
        }
        
        /* Contenedores de gráficos */
        .chart-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 2rem;
        }
        
        /* Tooltips personalizados */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #333;
            color: white;
            text-align: center;
            padding: 8px;
            border-radius: 6px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Sidebar mejorado */
        .sidebar .sidebar-content {
            background-color: #f1f5f9;
            padding: 2rem;
            border-right: 1px solid #e5e7eb;
        }
        
        /* Botones personalizados */
        .stButton>button {
            background-color: #2a5298;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: all 0.2s ease;
        }
        
        .stButton>button:hover {
            background-color: #1e3c72;
            transform: translateY(-2px);
        }
        
        /* Tabla de datos */
        .dataframe {
            border: none !important;
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
            margin: 1rem 0;
        }
        
        .dataframe th {
            background-color: #f8f9fa;
            padding: 12px;
            border-bottom: 2px solid #e5e7eb;
            text-align: left;
        }
        
        .dataframe td {
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
        }
        
        /* Animaciones */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar mejorado
    with st.sidebar:
        st.markdown("""
            <div class='component-container'>
                <h2>🏯 Panel de Control</h2>
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
        years_range = st.slider("Años de análisis", 2022, 2024, (2022, 2024),
                              help="Selecciona el rango de años para el análisis")
        forecast_years = st.slider("Años de predicción", 2025, 2030, (2025, 2027),
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
            <h1>🎯 Análisis Predictivo Avanzado de Salarios</h1>
            <p>Sistema de predicción basado en Machine Learning para estimación salarial</p>
        </div>
    """, unsafe_allow_html=True)

    if not predictor.load_data():
        st.error("❌ Error al cargar los datos. Por favor, verifica la conexión a la base de datos.")
        st.stop()

    # Preparación y entrenamiento
    with st.spinner('🔄 Preparando y entrenando el modelo...'):
        X_train, X_test, y_train, y_test = predictor.prepare_features()
        y_pred = predictor.train_model(X_train, y_train, X_test, y_test)

    # Dashboard principal
    col1, col2, col3 = st.columns(3)
    
    # Métricas principales con tooltips
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <div class='tooltip'>
                    🎯 Precisión del Modelo (R²)
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
                    📊 Error Medio Absoluto
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
                    🎭 Validación Cruzada
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
        "📈 Análisis Predictivo",
        "📊 Distribución Salarial",
        "🎯 Importancia de Variables"
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
                color='rgb(99, 110, 250)',
                opacity=0.6,
                line=dict(width=1, color='white')
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
            template="plotly_white",
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
            template="plotly_white",
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
            color_continuous_scale='viridis',
            text='importance'
        )
        
        fig.update_traces(
            texttemplate='%{text:.2%}',
            textposition='outside'
        )
        
        fig.update_layout(
            template="plotly_white",
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
            
            prediction = predictor.predict_future_salary(features, confidence_level)
            prediction_data.append({
                'year': year,
                'experience_level': exp_level,
                'prediction': prediction['prediction'][0],
                'lower_bound': prediction['lower_bound'][0],
                'upper_bound': prediction['upper_bound'][0],
                'std': prediction['std'][0]
            })

        # Visualización de predicciones
        for year in future_years:
            st.write(f"### Predicciones para {year}")
            cols = st.columns(len(experience_levels))
            
            year_predictions = [p for p in prediction_data if p['year'] == year]
            
            for idx, pred in enumerate(year_predictions):
                with cols[idx]:
                    exp_level = pred['experience_level']
                    emoji = {'EN': '🥶', 'MI': '🧑‍🎓', 'SE': '🧑‍💼', 'EX': '🎯'}[exp_level]
                    
                    st.metric(
                        label=f"{emoji} {exp_level}",
                        value=f"${pred['prediction']:,.0f}",
                        delta=f"±${pred['std']:,.0f}"
                    )
                    
                    # Mostrar intervalos de confianza
                    st.caption(f"""
                    Rango de confianza ({confidence_level:.0%}):
                    ${pred['lower_bound']:,.0f} - ${pred['upper_bound']:,.0f}
                    """)

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
                name=f'{exp_level}',
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=8)
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
            title="Proyección de Tendencias Salariales",
            xaxis_title="Año",
            yaxis_title="Salario Proyectado (USD)",
            template="plotly_white",
            height=600,
            hovermode='x unified',
            legend_title="Nivel de Experiencia"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Añadir notas explicativas
        st.info("""
        **Notas sobre las predicciones:**
        - EN: Entry Level (Nivel de entrada)
        - MI: Mid Level (Nivel medio)
        - SE: Senior Level (Nivel senior)
        - EX: Expert Level (Nivel experto)
        
        Las predicciones incluyen intervalos de confianza basados en la variabilidad del modelo.
        """)
                    
                    
if __name__ == "__main__":
    create_enhanced_dashboard()
