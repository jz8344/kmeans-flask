# Deployment Guide - Flask K-Means Service

## 📋 Variables de Entorno Requeridas en Railway

### Servicio: kmeans-flask-production

```bash
# Puerto
PORT=5000

# Entorno
FLASK_ENV=production

# Base de Datos (compartida con Laravel)
DATABASE_URL=postgresql://postgres:qDVMmSkqHnhMHRoDSfKHeijDCtKnjpkg@hopper.proxy.rlwy.net:36076/railway?sslmode=require
PGHOST=hopper.proxy.rlwy.net
PGPORT=36076
PGDATABASE=railway
PGUSER=postgres
PGPASSWORD=qDVMmSkqHnhMHRoDSfKHeijDCtKnjpkg

# URLs de otros servicios (para CORS)
LARAVEL_API_URL=https://web-production-86356.up.railway.app
FRONTEND_URL=https://frontend-production-a12b.up.railway.app

# API Keys (opcional)
GOOGLE_MAPS_API_KEY=AIzaSyCz4QsA_tgZv3Hw3O-RZLVoKefkXX7ZNoA
```

## 🔗 Variables de Entorno en FRONTEND (Vite)

Agregar al servicio frontend:

```bash
# URL del servicio de análisis K-Means
VITE_KMEANS_API_URL=https://kmeans-flask-production.up.railway.app
```

## 🔗 Variables de Entorno en BACKEND (Laravel)

Agregar al servicio Laravel (opcional, si Laravel necesita comunicarse con Flask):

```bash
# URL del servicio de análisis K-Means
KMEANS_API_URL=https://kmeans-flask-production.up.railway.app
```

## ✅ Checklist de Deployment

- [ ] Variables de entorno configuradas en Railway
- [ ] DATABASE_URL apunta a la misma BD que Laravel
- [ ] CORS configurado con URLs correctas del frontend
- [ ] Servicio desplegado y respondiendo en `/health`
- [ ] Variable VITE_KMEANS_API_URL agregada al frontend
- [ ] Frontend reconstruido con nueva variable

## 🧪 Verificar Deployment

### 1. Health Check
```bash
curl https://kmeans-flask-production.up.railway.app/health
```

Respuesta esperada:
```json
{
  "status": "ok",
  "database": "connected",
  "timestamp": "2025-11-26T..."
}
```

### 2. Test CORS
```bash
curl -X OPTIONS https://kmeans-flask-production.up.railway.app/api/analyze/driver \
  -H "Origin: https://frontend-production-a12b.up.railway.app" \
  -H "Access-Control-Request-Method: POST" \
  -v
```

Debe retornar headers:
- `Access-Control-Allow-Origin`
- `Access-Control-Allow-Methods`

### 3. Test Análisis
```bash
curl -X POST https://kmeans-flask-production.up.railway.app/api/analyze/driver \
  -H "Content-Type: application/json" \
  -d '{"driver_id": 1, "n_samples": 100}'
```

## 🔧 Troubleshooting

### Error 502 Bad Gateway
- Verificar que el servicio está corriendo: revisar logs en Railway
- Verificar que el PORT está configurado correctamente
- Verificar que DATABASE_URL es válida

### Error CORS
- Verificar que FRONTEND_URL está en la lista de orígenes permitidos
- Reconstruir el servicio después de cambiar código CORS
- Verificar que el frontend usa la URL correcta (VITE_KMEANS_API_URL)

### Base de Datos no conecta
- Verificar que DATABASE_URL tiene el formato correcto
- Verificar que las credenciales son correctas
- Verificar que el puerto es accesible desde Railway

## 📝 Estructura de Respuesta API

### POST /api/analyze/driver

Request:
```json
{
  "driver_id": 1,
  "n_samples": 1000
}
```

Response:
```json
{
  "driver_id": 1,
  "optimal_k": 3,
  "clusters": {
    "0": {
      "status": "Normal",
      "count": 334,
      "heart_rate_mean": 70.2,
      "heart_rate_std": 5.1,
      "accel_mean": 0.4,
      "accel_std": 0.3
    },
    "1": { ... },
    "2": { ... }
  },
  "recommendations": [
    "Grupo Normal: Continuar monitoreo regular.",
    "Grupo Precaución: Sugerir descansos más frecuentes.",
    "Grupo Riesgo Alto: ALERTA - Revisar comportamiento inmediatamente."
  ],
  "visualization_data": {
    "scatter": [...],
    "centroids": [...]
  }
}
```
