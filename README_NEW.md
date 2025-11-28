# TrailynSafe Analytics API - K-Means Clustering

API de análisis avanzado de comportamiento de conductores usando algoritmos de Machine Learning con **scikit-learn** y visualizaciones con **matplotlib**.

## 🚀 Características

- ✅ Análisis K-Means real sobre datos de PostgreSQL
- ✅ Consulta datos de viajes, confirmaciones y asistencias
- ✅ Métricas analizadas:
  - Tasa de asistencia (%)
  - Eficiencia de rutas (%)
  - Tiempo promedio de recogida (minutos)
- ✅ Generación de gráficas matplotlib (base64):
  - Dispersión: Asistencia vs Eficiencia
  - Método del Codo + Silhouette Score
  - Distribución de clusters (Bar Chart)
  - Box plots de métricas por cluster
- ✅ Detección automática del número óptimo de clusters
- ✅ API REST con FastAPI
- ✅ CORS configurado para frontend Vue

## 📦 Instalación

```bash
pip install -r requirements.txt
```

## ⚙️ Configuración

1. Copiar `.env.example` a `.env`
2. Configurar las variables de entorno:

```env
DATABASE_URL=postgresql://user:password@host:port/database
PORT=8000
LOG_LEVEL=INFO
```

**Importante:** El `DATABASE_URL` debe apuntar a la misma base de datos PostgreSQL que usa Laravel.

## 🏃 Ejecución Local

```bash
# Opción 1: Con uvicorn directamente
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Opción 2: Con python
python app.py
```

## 📍 Endpoints

### `GET /`
Estado de la API y lista de endpoints disponibles.

### `GET /health`
Health check - verifica conexión a BD y estado del servicio.

### `POST /api/analyze/driver`
Realiza análisis K-means sobre los datos de conductores.

**Request Body:**
```json
{
  "driver_id": 1,        // Opcional: ID del chofer específico (null = todos)
  "n_samples": 1000      // Número máximo de registros a analizar
}
```

**Response:**
```json
{
  "driver_id": 1,
  "optimal_k": 3,
  "clusters": {
    "0": {
      "status": "Excelente Desempeño",
      "count": 45,
      "tasa_asistencia_mean": 95.5,
      "eficiencia_mean": 102.3,
      "tiempo_recogida_mean": 5.2
    }
  },
  "recommendations": [
    "✅ Excelente Desempeño: 45 viajes - Mantener este nivel.",
    "📈 Análisis basado en 150 viajes reales."
  ],
  "plots": {
    "scatter_asistencia_eficiencia": "data:image/png;base64,...",
    "elbow_method": "data:image/png;base64,...",
    "cluster_distribution": "data:image/png;base64,...",
    "boxplot_metrics": "data:image/png;base64,..."
  },
  "data_source": "real",
  "total_records": 150
}
```

### `GET /api/drivers`
Obtiene la lista de choferes disponibles en la BD.

**Response:**
```json
[
  {
    "id": 1,
    "nombre": "Juan Pérez"
  }
]
```

## 🗄️ Datos de la Base de Datos

El servicio consulta las siguientes tablas:
- `viajes` - Información de viajes realizados
- `choferes` - Datos de conductores
- `confirmacion_viaje` - Confirmaciones de padres
- `asistencias` - Registro de asistencias de niños

## 🔧 Stack Tecnológico

- **FastAPI** - Framework web moderno para Python
- **scikit-learn** - Machine Learning (K-Means)
- **matplotlib** - Generación de gráficas
- **seaborn** - Visualizaciones estadísticas
- **pandas** - Manipulación de datos
- **SQLAlchemy** - ORM para PostgreSQL
- **psycopg2** - Driver PostgreSQL

## 🚢 Despliegue en Railway

1. Conectar el repositorio a Railway
2. Configurar las variables de entorno:
   - `DATABASE_URL` (Railway PostgreSQL)
   - `PORT` (automático)
3. El archivo `Procfile` ya está configurado:
   ```
   web: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
4. Railway detectará `requirements.txt` y instalará dependencias automáticamente

## 🔗 Integración con Frontend

El componente `AdminEstadisticas.vue` ya está configurado para:
1. Llamar al endpoint `/api/analyze/driver`
2. Mostrar las gráficas matplotlib como imágenes
3. Presentar estadísticas detalladas por cluster
4. Mostrar recomendaciones del sistema

Variable de entorno necesaria en frontend:
```env
VITE_KMEANS_API_URL=https://kmeans-flask-production.up.railway.app
```

## 🐛 Troubleshooting

### Error de conexión a BD
Verificar que `DATABASE_URL` esté correctamente configurado y que el servicio tenga acceso a la BD.

### Gráficas no se generan
Asegurar que matplotlib esté usando el backend 'Agg' (sin GUI). Ya está configurado en `app.py`.

### CORS errors
Las URLs permitidas están en `ALLOWED_ORIGINS` en `app.py`. Actualizar según sea necesario.

### No hay datos reales
El servicio generará datos simulados si no encuentra viajes en la BD. Asegurar que existan registros en las tablas `viajes`, `confirmacion_viaje` y `asistencias`.

## 📝 Licencia

Parte del proyecto TrailynSafe.
