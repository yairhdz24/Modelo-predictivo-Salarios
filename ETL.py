import pandas as pd
import psycopg2
from psycopg2 import sql
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Configuración de la conexión a la base de datos PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="Salarios",
    user="yairhdz24",
    password="yairhdz24"
)
cursor = conn.cursor()

print("Conexión establecida con la base de datos.")

# Crear la tabla si no existe
create_table_query = """
CREATE TABLE IF NOT EXISTS data_science_salaries (
    id SERIAL PRIMARY KEY,
    work_year INT,
    experience_level VARCHAR(2),
    employment_type VARCHAR(2),
    job_title VARCHAR(100),
    salary FLOAT,
    salary_currency VARCHAR(3),
    salary_in_usd FLOAT,
    employee_residence VARCHAR(2),
    remote_ratio INT,
    company_location VARCHAR(2),
    company_size VARCHAR(1)
);
"""
cursor.execute(create_table_query)
conn.commit()

# Proceso ETL (Extracción, Transformación y Carga)

# Extracción
def extract_data(cursor):
    query = "SELECT * FROM data_science_salaries;"
    cursor.execute(query)
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(data, columns=columns)

# Transformación
def transform_data(data):
    # Limpiar valores nulos
    imputer = SimpleImputer(strategy='mean')
    data['salary'] = imputer.fit_transform(data[['salary']])
    data['salary_in_usd'] = imputer.fit_transform(data[['salary_in_usd']])
    
    # Predecir valores faltantes (ejemplo con LinearRegression)
    model = LinearRegression()
    not_null_data = data.dropna()
    null_data = data[data.isnull().any(axis=1)]
    
    if not null_data.empty and not null_data.shape[1] == 1:
        model.fit(not_null_data.drop(columns=['salary']), not_null_data['salary'])
        data.loc[data['salary'].isnull(), 'salary'] = model.predict(null_data.drop(columns=['salary']))
    
    # Convertir todas las columnas a minúsculas
    data.columns = [col.lower() for col in data.columns]
    return data

# Carga
def load_data(data, cursor, conn):
    insert_query = """
    INSERT INTO data_science_salaries (
        work_year, experience_level, employment_type, job_title, 
        salary, salary_currency, salary_in_usd, employee_residence, 
        remote_ratio, company_location, company_size
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT DO NOTHING;
    """
    for _, row in data.iterrows():
        cursor.execute(insert_query, (
            row['work_year'], row['experience_level'], row['employment_type'],
            row['job_title'], row['salary'], row['salary_currency'], row['salary_in_usd'],
            row['employee_residence'], row['remote_ratio'], row['company_location'], 
            row['company_size']
        ))
    conn.commit()

# Ejecutar el proceso ETL
data = extract_data(cursor)
data = transform_data(data)
load_data(data, cursor, conn)

# Confirmamos los cambios y cerraramos la conexión de la bd
cursor.close()
conn.close()

print("Proceso ETL completado correctamente.")