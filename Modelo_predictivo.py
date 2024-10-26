import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import mplcursors  # Para hacer gráficos interactivos

# Configuración de estilo para gráficos
sns.set(style="whitegrid")

# 1. Conectar a la base de datos y cargar datos
engine = create_engine('postgresql://yairhdz24:yairhdz24@localhost:5432/Salarios')
df = pd.read_sql('SELECT * FROM empleos', engine)

# 2. Preparar los datos
# Convertir columnas relevantes a tipo 'category' y codificar
categorical_columns = ['experience_level', 'employment_type', 'job_title', 
                       'employee_residence', 'company_location', 'company_size']
for col in categorical_columns:
    df[col] = df[col].astype('category').cat.codes

# Separar características y objetivo
X = df.drop(['salary_in_usd', 'work_year'], axis=1)
y = df['salary_in_usd']

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Entrenar el modelo de Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Predicción para el conjunto de prueba
y_pred = rf_model.predict(X_test)

# 6. Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 7. Predicción para 2025 y 2026
mean_values = np.mean(X_scaled, axis=0)
future_data_2025 = mean_values + 0.02  # Incremento para 2025
future_data_2026 = mean_values + 0.04  # Incremento para 2026

future_salaries = rf_model.predict([future_data_2025, future_data_2026])

# Mostrar resultados de predicción
print(f"Predicción del salario para 2025: ${future_salaries[0]:,.2f}")
print(f"Predicción del salario para 2026: ${future_salaries[1]:,.2f}")
print(f"Error Cuadrático Medio (RMSE): ${rmse:,.2f}")

# 8. Visualización
plt.figure(figsize=(16, 8))

# Gráfico de los salarios reales vs. predicciones
plt.subplot(1, 2, 1)
scatter = plt.scatter(y_test, y_pred, color='dodgerblue', alpha=0.6, label='Salarios Predichos')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='darkred', linestyle='--', lw=2, label='Línea de referencia')
plt.title('Salarios Reales vs. Predicciones (Random Forest)', fontsize=15)
plt.xlabel('Salario Real (USD)', fontsize=12)
plt.ylabel('Salario Predicho (USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Hacer el gráfico interactivo
mplcursors.cursor(scatter, hover=True).connect("add", lambda sel: sel.annotation.set_text(f'Salario: ${y_pred[sel.index]:,.2f}'))

# Gráfico de barras para predicciones futuras
plt.subplot(1, 2, 2)
years = ['2025', '2026']
bar = plt.bar(years, future_salaries, color=['cornflowerblue', 'lightseagreen'], alpha=0.8)
plt.title('Predicción de Salarios para 2025 y 2026', fontsize=15)
plt.xlabel('Año', fontsize=12)
plt.ylabel('Salario Predicho en USD', fontsize=12)
plt.ylim(0, future_salaries.max() * 1.1)
plt.grid(True, linestyle='--', alpha=0.7)

# Añadir etiquetas a las barras
for b in bar:
    yval = b.get_height()
    plt.text(b.get_x() + b.get_width()/2, yval + 1000, f'${yval:,.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# 9. Mostrar tablas con resultados en formato tabular
# Predicciones de salarios para los años futuros
predictions_df = pd.DataFrame({
    'Año': ['2025', '2026'],
    'Salario Predicho (USD)': [f"${future_salaries[0]:,.2f}", f"${future_salaries[1]:,.2f}"]
})
print("\nPredicciones de Salarios para 2025 y 2026:")
print(tabulate(predictions_df, headers="keys", tablefmt="fancy_grid"))

# Error cuadrático medio en formato tabular
error_df = pd.DataFrame({
    'Métrica': ['Error Cuadrático Medio (RMSE)'],
    'Valor': [f"${rmse:,.2f}"]
})
print("\nErrores del modelo:")
print(tabulate(error_df, headers="keys", tablefmt="fancy_grid"))
