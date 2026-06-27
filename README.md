# Sistema de Predicción de Fallos en Equipos Electrónicos

Sistema de monitoreo y predicción de estado de integridad de equipos electrónicos utilizando Random Forest.

## Requisitos

- Python 3.9+
- PostgreSQL 13+
- Dependencies listed in `requirements.txt`

## Instalación

1. **Clonar el repositorio**
```bash
git clone <repo-url>
cd project
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar base de datos PostgreSQL**

Crear la base de datos:
```sql
CREATE DATABASE equipos_db;
```

Editar `.streamlit/secrets.toml` con las credenciales:
```toml
[connections.postgresql]
host = "localhost"
port = "5432"
database = "equipos_db"
username = "postgres"
password = "tu_password"
```

5. **Inicializar la base de datos**

Ejecutar `schema.sql` en PostgreSQL o ejecutar la aplicación (crea las tablas automáticamente).

## Uso

1. **Iniciar la aplicación**
```bash
streamlit run streamlit_app.py
```

2. **Credenciales por defecto**
- Usuario: `admin`
- Contraseña: `admin123`

3. **Flujo de trabajo**
   1. Importar CSV con datos históricos de equipos
   2. Entrenar el modelo Random Forest
   3. Realizar predicciones
   4. Monitorear data drift y alertas

## Estructura del Proyecto

```
project/
├── features/              # Módulos funcionales
│   ├── auth/              # Autenticación
│   ├── data/              # Carga y exportación de datos
│   ├── model/             # Entrenamiento y predicción
│   ├── dashboard/         # Visualizaciones y KPIs
│   ├── monitoring/        # Data drift y matriz de confusión
│   └── alerts/            # Sistema de alertas tempranas
├── models/
│   └── trained_models/    # Modelos guardados
├── .streamlit/
│   └── secrets.toml       # Configuración de BD
├── streamlit_app.py       # Aplicación principal
├── config.yaml            # Configuración general
├── requirements.txt       # Dependencias Python
└── schema.sql             # Schema de BD
```

## Features Implementadas

- Login/Logout con autenticación
- Dashboard de análisis de activos
- Gráfico de dona por ubicación
- Sistema de filtros dinámicos
- Importación/Exportación CSV
- Predicción con Random Forest
- Cross-validation (k≥5)
- Feature importance
- Monitoreo de Data Drift
- Matriz de confusión dinámica
- Sistema de alertas tempranas
- Arquitectura basada en características
