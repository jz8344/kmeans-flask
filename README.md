# TrailynSafe Analytics Service (K-Means ML)

🤖 Servicio de análisis de comportamiento de conductores usando Machine Learning (K-Means Clustering) para el sistema TrailynSafe.

## 🚀 Características

- **Análisis K-Means**: Clustering automático de datos de conductores
- **Detección de Patrones**: Identifica estados normales, precaución y riesgo alto
- **API REST FastAPI**: Integración fácil con frontend Vue.js y backend Laravel
- **Base de Datos Compartida**: Usa la misma PostgreSQL que Laravel
- **CORS Configurado**: Listo para producción en Railway

## 📡 Endpoints

### `GET /` - Info API
```json
{
  "message": "TrailynSafe Analytics API is running",
  "version": "1.0.0",
  "endpoints": [...]
}
```

### `GET /health` - Health Check
```json
{
  "status": "ok",
  "database": "connected",
  "timestamp": "2025-11-26T..."
}
```

### `POST /api/analyze/driver` - Análisis de Conductor
**Request:**
```json
{
  "driver_id": 1,
  "n_samples": 1000
}
```

**Response:**
```json
{
  "driver_id": 1,
  "optimal_k": 3,
  "clusters": {
    "0": {
      "status": "Normal",
      "count": 334,
      "heart_rate_mean": 70.2,
      "accel_mean": 0.4
    }
  },
  "recommendations": [
    "Grupo Normal: Continuar monitoreo regular."
  ],
  "visualization_data": {...}
}
```

### `GET /api/drivers` - Lista de Conductores
```json
[
  {"id": 1, "nombre": "Juan Pérez"},
  {"id": 2, "nombre": "María López"}
]
```

## 🌐 Deploy en Railway

Ver [DEPLOYMENT.md](./DEPLOYMENT.md) para instrucciones completas.

### Variables de Entorno Requeridas:

**En Railway - Servicio Flask:**
```bash
PORT=5000
FLASK_ENV=production
DATABASE_URL=postgresql://postgres:qDVMmSkqHnhMHRoDSfKHeijDCtKnjpkg@hopper.proxy.rlwy.net:36076/railway?sslmode=require
FRONTEND_URL=https://frontend-production-a12b.up.railway.app
LARAVEL_API_URL=https://web-production-86356.up.railway.app
```

**En Railway - Servicio Frontend:**
```bash
VITE_KMEANS_API_URL=https://kmeans-flask-production.up.railway.app
```

## 💻 Instalación Local

```bash
# 1. Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar .env
DATABASE_URL=postgresql://user:password@localhost:5432/trailynsafe
PORT=5000

# 4. Ejecutar servidor
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

## 🧪 Testing

```bash
# Health check
curl https://kmeans-flask-production.up.railway.app/health

# Análisis
curl -X POST https://kmeans-flask-production.up.railway.app/api/analyze/driver \
  -H "Content-Type: application/json" \
  -d '{"driver_id": 1, "n_samples": 100}'
```

## 📊 Algoritmo K-Means

Agrupa datos de sensores (ritmo cardíaco, aceleración) en clusters:
1. **Normal**: Valores saludables
2. **Precaución**: Valores elevados (fatiga/estrés)
3. **Riesgo Alto**: Valores críticos (intervención inmediata)

El número óptimo de clusters se calcula con Silhouette Score.

## 🔧 Troubleshooting

### Error 502 Bad Gateway
- Verificar logs en Railway
- Verificar DATABASE_URL y PORT

### Error CORS
- Verificar que FRONTEND_URL esté configurada
- Reconstruir servicio después de cambios
- Agregar VITE_KMEANS_API_URL en frontend
