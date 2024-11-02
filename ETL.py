import pandas as pd
import psycopg2
from psycopg2 import sql

# Cargar los datos
file_path = './datos.csv'
data = pd.read_csv(file_path)

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

# Insertar datos
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

# Confirmar los cambios y cerrar la conexión
conn.commit()
cursor.close()
conn.close()

print("Datos insertados correctamente en la base de datos.")
