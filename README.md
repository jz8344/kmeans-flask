# Flask k-Means Route Optimizer

API minimalista para optimización de rutas usando k-Means + TSP.

## Endpoints

- `GET /health` - Health check
- `POST /api/generar-ruta` - Generar ruta optimizada

## Deploy en Railway

1. Crear nuevo servicio desde este directorio
2. Railway detectará Python automáticamente
3. URL: `https://tu-servicio.up.railway.app`

## Uso

```bash
curl -X POST https://tu-servicio.up.railway.app/api/generar-ruta \
  -H "Content-Type: application/json" \
  -d '{
    "viaje_id": 1,
    "puntos": [...],
    "destino": {...},
    "hora_salida": "07:00:00"
  }'
```

## Local

```bash
pip install -r requirements.txt
python app.py
```
