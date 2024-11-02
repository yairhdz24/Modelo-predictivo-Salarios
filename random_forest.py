import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats

class SalaryPredictor:
    def __init__(self):
        """Inicializa el modelo, codificadores y el DataFrame."""
        self.model = None  # Aquí almacenaremos el modelo una vez entrenado
        self.label_encoders = {}  # Diccionario para almacenar los codificadores de cada columna categórica
        self.df = None  # DataFrame donde cargaremos los datos

    def load_data(self):
        """Carga los datos desde la base de datos PostgreSQL."""
        engine = create_engine('postgresql://yairhdz24:yairhdz24@localhost:5432/Salarios')
        self.df = pd.read_sql('SELECT * FROM data_science_salaries', engine)

    def preprocess_data(self):
        """Preprocesa los datos para preparar el modelo."""
        # Elimina duplicados para evitar sesgo en el modelo
        self.df.drop_duplicates(inplace=True)
        
        # Rellena valores nulos en columnas numéricas con la mediana
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())

        # Elimina outliers extremos en la columna 'salary_in_usd' usando z-score
        z_scores = stats.zscore(self.df['salary_in_usd'])
        self.df = self.df[(z_scores < 3) & (z_scores > -3)]

        # Codifica variables categóricas para que puedan usarse en el modelo
        categorical_cols = ['experience_level', 'employment_type', 'job_title', 
                            'employee_residence', 'company_location', 'company_size']
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col + '_encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le  # Guarda el codificador para uso futuro

    def train_model(self):
        """Entrena el modelo Random Forest usando GridSearch para optimizar los parámetros."""
        # Define las características (X) y el objetivo (y)
        feature_cols = ['work_year'] + [col + '_encoded' for col in ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']]
        X = self.df[feature_cols]
        y = self.df['salary_in_usd']

        # Divide los datos en un 70% para entrenamiento y un 30% para prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Define los parámetros para GridSearchCV
        param_grid = {
            'n_estimators': [100, 150],  # Número de árboles en el bosque
            'max_depth': [5, 10]         # Profundidad máxima de cada árbol
        }

        # Usa GridSearchCV para optimizar el modelo
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=3,  # Cross-validation de 3 pliegues
            n_jobs=-1,  # Usa todos los núcleos disponibles
            verbose=1   # Muestra el progreso
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_  # Modelo con los mejores parámetros

        # Evalúa el modelo con el conjunto de prueba
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Error cuadrático medio
        r2 = r2_score(y_test, y_pred)  # Coeficiente de determinación R²

        print(f"Mejores parametros: {grid_search.best_params_}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.2f}")

    def predict_future_salaries(self, future_years):
        """Predice los salarios futuros para los años dados."""
        predictions = {}
        feature_cols = ['work_year'] + [col + '_encoded' for col in ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']]

        for year in future_years:
            # Crear un DataFrame con los datos para el año futuro
            future_data = pd.DataFrame({
                'work_year': [year] * len(self.df),
                'experience_level_encoded': [self.df['experience_level_encoded'].mode()[0]] * len(self.df),
                'employment_type_encoded': [self.df['employment_type_encoded'].mode()[0]] * len(self.df),
                'job_title_encoded': [self.df['job_title_encoded'].mode()[0]] * len(self.df),
                'employee_residence_encoded': [self.df['employee_residence_encoded'].mode()[0]] * len(self.df),
                'company_location_encoded': [self.df['company_location_encoded'].mode()[0]] * len(self.df),
                'company_size_encoded': [self.df['company_size_encoded'].mode()[0]] * len(self.df),
            })

            # Realiza la predicción para el año futuro
            future_data = future_data[feature_cols]
            predicted_salaries = self.model.predict(future_data)
            predictions[year] = predicted_salaries.mean()  # Calcula el salario promedio

        return predictions

# Uso del modelo
salary_predictor = SalaryPredictor()
salary_predictor.load_data()  # Cargar datos desde la base de datos
salary_predictor.preprocess_data()  # Preprocesar los datos
salary_predictor.train_model()  # Entrenar el modelo

# Predecir salarios para los años futuros
future_years = [2025]
predictions = salary_predictor.predict_future_salaries(future_years)

# Mostrar las predicciones de salario promedio para los años futuros
for year, salary in predictions.items():
    print(f"Predicción de salario promedio para {year}: {salary:.2f} USD")
