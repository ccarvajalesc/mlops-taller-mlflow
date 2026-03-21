# MLOps - Taller MLflow

Solución completa de un pipeline de MLOps implementando un sistema de entrenamiento, monitoreo y predicción de modelos de machine learning usando MLflow.

## Integrantes del Grupo

- **Carlos Manuel Carvajales Castrillo**
- **Mateo Ruiz Mendoza**

**Materia:** MLOps

---

## 📋 Descripción del Proyecto

Este proyecto implementa una solución completa de MLOps que incluye:

1. **Base de datos dedicada para metadata de MLflow** - Almacenamiento centralizado de experimentos y ejecuciones
2. **Instancia de MLflow** - Plataforma para tracking de experimentos y registro de modelos
3. **MinIO** - Almacenamiento de objetos S3-compatible para artefactos de MLflow
4. **JupyterLab** - Entorno interactivo para desarrollo y experimentación
5. **Notebook de Entrenamiento** - Múltiples ejecuciones (20+) con variaciones de hiperparámetros
6. **Base de datos de Aplicación** - Almacenamiento de datos para entrenamiento y validación (separada de MLflow)
7. **Modelos Registrados en MLflow** - Gestión del ciclo de vida de modelos
8. **API REST** - Servicio de inferencia que consume modelos desde MLflow

---

## 🏗️ Arquitectura del Sistema

```
┌──────────────────────────────────────────────────────────┐
│                    JupyterLab (8888)                     │
│          Notebook: Experimentación & Entrenamiento       │
└─────────────────────────┬────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ MLflow       │  │ MinIO        │  │ MySQL DB     │
│ (5000)       │  │ (9000/9001)  │  │ (3306)       │
│ Tracking &   │  │ Almacenamiento│ │ Datos App    │
│ Registry     │  │ de Artefactos │ │              │
└────────┬─────┘  └──────────────┘  └──────────────┘
         │
         │
┌────────▼─────────────────────────────┐
│    API REST - FastAPI (8000)         │
│    Inference Service                 │
│    Consume Models from MLflow        │
└──────────────────────────────────────┘
```

---

## 🚀 Componentes Principales

### 1. **Base de Datos (MySQL)**
- **Propósito:** Almacenamiento de datos de entrenamiento y datos procesados
- **Puerto:** 3306
- **Credenciales:** 
  - Usuario: `mlops_user`
  - Contraseña: `mlops_pass`
  - Base de datos: `mlops_db`

### 2. **MLflow**
- **Propósito:** Tracking de experimentos, logging de métricas y registro de modelos
- **Puerto:** 5000
- **Características:**
  - Model Registry para gestión del ciclo de vida
  - Integración con MinIO para almacenamiento de artefactos
  - Backend SQL para persistencia

### 3. **MinIO**
- **Propósito:** Almacenamiento compatible con S3 para artefactos de MLflow
- **Puertos:** 
  - API: 9000
  - Console: 9001
- **Credenciales:**
  - Usuario: `admin`
  - Contraseña: `supersecret`

### 4. **JupyterLab**
- **Propósito:** Desarrollo interactivo de modelos y experimentación
- **Puerto:** 8888
- **Contraseña:** `valentasecret`
- **Ambiente:** Despliegue de variables de entorno para integración con MLflow y bases de datos

### 5. **API REST (FastAPI)**
- **Propósito:** Servicio de inferencia consumiendo modelos desde MLflow
- **Puerto:** 8000
- **Características:**
  - Carga de modelos desde MLflow Model Registry
  - Predicción en tiempo real
  - Validación de datos de entrada

---

## 📊 Especificaciones del Taller

### ✅ Experimentación y Entrenamiento

- **Múltiples ejecuciones:** 20+ experimentos con variaciones de hiperparámetros
- **Modelos testeados:** Decision Tree, KNN, SVM
- **Tracked en MLflow:** Métricas, parámetros y artefactos de cada experimento
- **Notebook:** `train.ipynb` - Implementación del pipeline de experimentación

### ✅ Gestión de Datos

- **Datos de entrada:** Dataset penguins (características numéricas y categóricas)
- **Almacenamiento:** Base de datos MySQL dedicada para la aplicación
- **Datos procesados:** Almacenados también en base de datos
- **Separación:** BD diferente a la usada para metadata de MLflow

### ✅ Modelos y Registry

- **Registro en MLflow:** Todos los modelos disponibles en Model Registry
- **Stages:** Production, Staging, Archived
- **Versionado:** Control completo del ciclo de vida del modelo

### ✅ Inferencia

- **API Rest:** Endpoint `/predict` para clasificación de pingüinos
- **Integración MLflow:** Carga automática de modelos desde registry
- **Validación:** Verificación de formato y tipo de datos

---

## 🐳 Servicios Docker

```yaml
# Servicios principales en docker-compose.yaml:

- mysql_db:      Base de datos MySQL
- minio:         Almacenamiento S3
- mlflow:        Servidor MLflow
- jupyter:       Entorno Jupyter
- api:           Servicio FastAPI
- db_loader:     Script de carga inicial de datos
```

---

## 📁 Estructura del Proyecto

```
.
├── docker-compose.yaml           # Orquestación de servicios
├── docker/
│   ├── Dockerfile.api            # API FastAPI
│   └── Dockerfile.jupyter        # JupyterLab
├── mlflow_compose/
│   ├── docker-compose.yaml       # Compose alternativo
│   ├── mlflow/
│   │   └── Dockerfile            # MLflow
│   ├── notebooks/
│   │   └── train.ipynb           # Notebook de entrenamiento
│   └── data/
│       └── covertype/            # Datasets adicionales
├── main.py                       # API principales
├── predict.py                    # Lógica de predicción
├── train.ipynb                   # Notebook de experimentación
├── load_db.py                    # Carga de datos en BD
├── datasets/
│   └── penguins_size.csv         # Dataset principal
├── encoders/
│   └── ohe_encoder.joblib        # One-Hot Encoder entrenado
└── requirements.txt              # Dependencias Python
```

---

## 🛠️ Instalación y Uso

### Requisitos Previos

- Docker
- Docker Compose
- Python 3.8+

### Inicio de Servicios

```bash
# Navegar al directorio del proyecto
cd mlops-taller-mlflow

# Iniciar todos los servicios
docker-compose up --build

# O en segundo plano
docker-compose up --build -d
```

### Acceso a Servicios

| Servicio | URL | Puerto |
|----------|-----|---------|
| JupyterLab | `http://localhost:8888` | 8888 |
| MLflow UI | `http://localhost:5000` | 5000 |
| MinIO Console | `http://localhost:9001` | 9001 |
| API Docs | `http://localhost:8000/docs` | 8000 |
| MySQL | `localhost:3306` | 3306 |

### Flujo de Trabajo

1. **Experimentación:**
   - Acceder a JupyterLab (http://localhost:8888)
   - Ejecutar notebook `train.ipynb`
   - Experimentos automáticamente registrados en MLflow

2. **Monitoreo:**
   - Visualizar experimentos en MLflow UI (http://localhost:5000)
   - Comparar métricas y parámetros
   - Promover modelos a Production

3. **Inferencia:**
   - API disponible en http://localhost:8000
   - Documentación interactiva en http://localhost:8000/docs
   - Usar endpoint `/predict` para clasificaciones

---

## 📚 Dependencias Principales

```
fastapi              # Framework para API
mlflow              # Tracking y Model Registry
pandas              # Manipulación de datos
scikit-learn        # Modelos de ML
sqlalchemy          # ORM para base de datos
joblib              # Serialización de modelos
boto3               # Cliente S3 (MinIO)
```

Ver `requirements.txt` para versiones específicas.

---

## 🔍 Variables de Entorno Clave

```
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_ACCESS_KEY_ID=admin
AWS_SECRET_ACCESS_KEY=supersecret
MYSQL_USER=mlops_user
MYSQL_PASSWORD=mlops_pass
MYSQL_DATABASE=mlops_db
```

---

## 📈 Resultados de Experimentación

El notebook de entrenamiento implementa:

- **20+ ejecuciones** con diferentes combinaciones de hiperparámetros
- **Modelos evaluados:**
  - Decision Tree Classifier
  - K-Nearest Neighbors
  - Support Vector Machine
- **Métricas registradas:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Matriz de confusión
- **Artefactos guardados:**
  - Modelos entrenados
  - Encoder transformers
  - Gráficos de evaluación

---

## 📝 Notas

- Los datos se cargan automáticamente en la base de datos al iniciar `db_loader`
- MLflow usa MinIO como backend de almacenamiento para máxima compatibilidad
- La API consume modelos directamente desde MLflow Model Registry
- Todos los componentes están en la misma red Docker para comunicación interna

---

## 📞 Soporte

Para preguntas sobre la implementación, revisar:
- Logs en directorio `logs/`
- Documentación de MLflow: https://mlflow.org/docs
- Documentación de FastAPI: https://fastapi.tiangolo.com/
