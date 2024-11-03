import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Importamos métricas adicionales
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import mplcursors  # Para hacer gráficos interactivos

# Configuración de estilo para gráficos
sns.set(style="whitegrid")

# 1. Conectar a la base de datos y cargar datos
engine = create_engine('postgresql://yairhdz24:yairhdz24@localhost:5432/Salarios')  # Establecemos la conexión con la base de datos
df = pd.read_sql('SELECT * FROM empleos', engine)  # Cargamos los datos de la tabla 'empleos' en un DataFrame

# 2. Preparar los datos
# Convertir columnas relevantes a tipo 'category' y codificar
categorical_columns = ['experience_level', 'employment_type', 'job_title', 
                       'employee_residence', 'company_location', 'company_size']
for col in categorical_columns:
    df[col] = df[col].astype('category').cat.codes  # Convertimos las columnas categóricas a códigos numéricos

# Separar características (X) y objetivo (y)
X = df.drop(['salary_in_usd', 'work_year'], axis=1)  # Eliminamos la columna de salario y la de año de trabajo de las características
y = df['salary_in_usd']  # La variable objetivo es el salario en USD

# Normalizar las características
scaler = StandardScaler()  # Inicializamos el escalador
X_scaled = scaler.fit_transform(X)  # Normalizamos las características para que tengan media 0 y varianza 1

# 3. Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  # Dividimos los datos en 80% entrenamiento y 20% prueba

# 4. Entrenar el modelo de Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)  # Inicializamos el modelo con 200 árboles
rf_model.fit(X_train, y_train)  # Ajustamos el modelo a los datos de entrenamiento

# 5. Predicción para el conjunto de prueba
y_pred = rf_model.predict(X_test)  # Realizamos predicciones en el conjunto de prueba

# 6. Calcular el error cuadrático medio (RMSE)
mse = mean_squared_error(y_test, y_pred)  # Calculamos el error cuadrático medio
rmse = np.sqrt(mse)  # RMSE es la raíz cuadrada del MSE

# Otras métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)  # Calculamos el error absoluto medio (MAE)
r2 = r2_score(y_test, y_pred)  # Calculamos el coeficiente de determinación (R²)

# 7. Predicción para 2025 y 2026 con un incremento fijo
mean_values = np.mean(X_scaled, axis=0)  # Calculamos los valores medios de las características
incremento_2025 = 0.05  # Incremento fijo para 2025
incremento_2026 = 0.10  # Incremento fijo para 2026

# Aplicamos los incrementos fijos a los valores medios
future_data_2025 = mean_values + incremento_2025  # Incremento fijo para 2025
future_data_2026 = mean_values + incremento_2026  # Incremento fijo para 2026

future_salaries = rf_model.predict([future_data_2025, future_data_2026])  # Realizamos las predicciones para los años futuros

# Mostrar resultados de predicción
# print(f"Predicción del salario para 2025: ${future_salaries[0]:,.2f}")  # Mostramos la predicción para 2025
# print(f"Predicción del salario para 2026: ${future_salaries[1]:,.2f}")  # Mostramos la predicción para 2026

# Mostrar resultados de predicción
print(f"Predicción del salario para 2025: ${future_salaries[0]:,.2f}")  # Mostramos la predicción para 2025
print(f"Predicción del salario para 2026: ${future_salaries[1]:,.2f}")  # Mostramos la predicción para 2026
print(f"Error Cuadrático Medio (RMSE): ${rmse:,.2f}")  # Mostramos el RMSE
print(f"Error Absoluto Medio (MAE): ${mae:,.2f}")  # Mostramos el MAE
print(f"Coeficiente de Determinación (R²): {r2:.4f}")  # Mostramos el R²

# 8. Visualización
plt.figure(figsize=(16, 8))  # Establecemos el tamaño de la figura

# Gráfico de los salarios reales vs. predicciones
plt.subplot(1, 2, 1)  # Configuramos la primera subgráfica
scatter = plt.scatter(y_test, y_pred, color='dodgerblue', alpha=0.6, label='Salarios Predichos')  # Graficamos los salarios reales vs. predicciones
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='darkred', linestyle='--', lw=2, label='Línea de referencia')  # Línea de referencia
plt.title('Salarios Reales vs. Predicciones (Random Forest)', fontsize=15)  # Título de la gráfica
plt.xlabel('Salario Real (USD)', fontsize=12)  # Etiqueta del eje X
plt.ylabel('Salario Predicho (USD)', fontsize=12)  # Etiqueta del eje Y
plt.grid(True, linestyle='--', alpha=0.7)  # Añadir una cuadrícula
plt.legend()  # Mostrar la leyenda

# Hacer el gráfico interactivo
mplcursors.cursor(scatter, hover=True).connect("add", lambda sel: sel.annotation.set_text(f'Salario: ${y_pred[sel.index]:,.2f}'))  # Añadir interactividad al gráfico

# Gráfico de barras para predicciones futuras
plt.subplot(1, 2, 2)  # Configuramos la segunda subgráfica
years = ['2025', '2026']  # Años de predicción
bar = plt.bar(years, future_salaries, color=['cornflowerblue', 'lightseagreen'], alpha=0.8)  # Graficamos las predicciones futuras
plt.title('Predicción de Salarios para 2025 y 2026', fontsize=15)  # Título de la gráfica
plt.xlabel('Año', fontsize=12)  # Etiqueta del eje X
plt.ylabel('Salario Predicho en USD', fontsize=12)  # Etiqueta del eje Y
plt.ylim(0, future_salaries.max() * 1.1)  # Establecemos el límite del eje Y
plt.grid(True, linestyle='--', alpha=0.7)  # Añadir una cuadrícula

# Añadir etiquetas a las barras
for b in bar:  # Iteramos sobre las barras
    yval = b.get_height()  # Obtenemos la altura de la barra
    plt.text(b.get_x() + b.get_width()/2, yval + 1000, f'${yval:,.2f}', ha='center', va='bottom', fontsize=10)  # Añadimos la etiqueta

plt.tight_layout()  # Ajustamos el layout
plt.show()  # Mostramos las gráficas

# 9. Mostrar tablas con resultados en formato tabular
# Predicciones de salarios para los años futuros
predictions_df = pd.DataFrame({
    'Año': ['2025', '2026'],
    'Salario Predicho (USD)': [f"${future_salaries[0]:,.2f}", f"${future_salaries[1]:,.2f}"]
})  # Creamos un DataFrame para mostrar las predicciones
print("\nPredicciones de Salarios para 2025 y 2026:")  # Imprimimos un encabezado
print(tabulate(predictions_df, headers="keys", tablefmt="fancy_grid"))  # Mostramos la tabla de predicciones

# Error cuadrático medio en formato tabular
error_df = pd.DataFrame({
    'Métrica': ['Error Cuadrático Medio (RMSE)', 'Error Absoluto Medio (MAE)', 'Coeficiente de Determinación (R²)'],
    'Valor': [f"${rmse:,.2f}", f"${mae:,.2f}", f"{r2:.4f}"]  # Añadimos todas las métricas calculadas
})
print("\nErrores del modelo:")  # Imprimimos un encabezado
print(tabulate(error_df, headers="keys", tablefmt="fancy_grid"))  # Mostramos la tabla de errores
