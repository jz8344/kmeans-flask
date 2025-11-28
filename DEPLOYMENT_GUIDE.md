# Guía de Despliegue - Servicio K-Means Flask

## 📋 Pre-requisitos

1. Cuenta en Railway con base de datos PostgreSQL configurada
2. Repositorio Git con el código
3. Variables de entorno preparadas

## 🚀 Pasos para Desplegar en Railway

### 1. Preparar el Servicio

Asegúrate de que estos archivos existen en `/kmeans_flask/`:

- ✅ `app.py` - Aplicación principal con análisis K-means
- ✅ `requirements.txt` - Dependencias Python
- ✅ `Procfile` - Comando de inicio
- ✅ `.env.example` - Plantilla de configuración

### 2. Crear Nuevo Servicio en Railway

1. Ir a Railway Dashboard
2. Seleccionar tu proyecto TrailynSafe
3. Click en "New Service" → "GitHub Repo"
4. Seleccionar el repositorio
5. Configurar el directorio raíz: `/kmeans_flask`

### 3. Configurar Variables de Entorno

En Railway, agregar las siguientes variables:

```env
# Obligatorias
DATABASE_URL=${{Postgres.DATABASE_URL}}
PORT=${{PORT}}

# Opcionales
LOG_LEVEL=INFO
DEFAULT_N_SAMPLES=1000
MAX_CLUSTERS=6
```

**Importante:** Railway automáticamente inyecta `${{Postgres.DATABASE_URL}}` si tienes PostgreSQL en el mismo proyecto.

### 4. Verificar el Procfile

Debe contener:
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

### 5. Desplegar

Railway detectará automáticamente:
- El `requirements.txt` e instalará las dependencias
- El `Procfile` y ejecutará el comando especificado
- Las variables de entorno configuradas

El despliegue toma aproximadamente 3-5 minutos.

### 6. Verificar el Despliegue

Una vez desplegado, Railway te dará una URL como:
```
https://kmeans-flask-production.up.railway.app
```

Prueba los endpoints:

```bash
# Health check
curl https://kmeans-flask-production.up.railway.app/health

# Lista de choferes
curl https://kmeans-flask-production.up.railway.app/api/drivers

# Análisis K-means (método POST)
curl -X POST https://kmeans-flask-production.up.railway.app/api/analyze/driver \
  -H "Content-Type: application/json" \
  -d '{"driver_id": null, "n_samples": 100}'
```

## 🔧 Configurar Frontend

En el proyecto `frontend/`, actualizar la variable de entorno:

```env
# .env o .env.production
VITE_KMEANS_API_URL=https://kmeans-flask-production.up.railway.app
```

O directamente en el código de `AdminEstadisticas.vue`:
```javascript
const PYTHON_API_URL = import.meta.env.VITE_KMEANS_API_URL || 'https://kmeans-flask-production.up.railway.app'
```

## 🔗 Configurar Backend Laravel (Opcional)

Si usas el proxy en Laravel, agregar en `.env`:

```env
KMEANS_API_URL=https://kmeans-flask-production.up.railway.app
```

## ✅ Checklist Post-Despliegue

- [ ] El servicio está corriendo (status "Active" en Railway)
- [ ] `/health` retorna `{"status": "ok"}`
- [ ] `/api/drivers` retorna lista de choferes
- [ ] `/api/analyze/driver` retorna análisis con gráficas
- [ ] Frontend puede llamar al endpoint y mostrar gráficas
- [ ] No hay errores CORS

## 🐛 Troubleshooting

### Error: "Module not found"
**Causa:** Dependencias no instaladas correctamente.
**Solución:** 
- Verificar que `requirements.txt` esté completo
- Revisar logs de build en Railway
- Forzar rebuild

### Error: "Database connection failed"
**Causa:** `DATABASE_URL` no configurado o inválido.
**Solución:**
- Verificar que PostgreSQL esté en el mismo proyecto Railway
- Usar `${{Postgres.DATABASE_URL}}` en las variables de entorno
- Verificar que el formato sea `postgresql://` (no `postgres://`)

### Error CORS
**Causa:** Frontend no está en la lista de orígenes permitidos.
**Solución:**
- Actualizar `ALLOWED_ORIGINS` en `app.py`
- O usar `allow_origins=["*"]` (ya configurado)

### Gráficas no se muestran
**Causa:** Backend 'Agg' no configurado para matplotlib.
**Solución:**
- Ya está configurado en `app.py` con `matplotlib.use('Agg')`
- Verificar que `pillow` esté en `requirements.txt`

### Análisis retorna datos simulados
**Causa:** No hay datos en las tablas de viajes.
**Solución:**
- Verificar que existan registros en `viajes`, `confirmacion_viaje` y `asistencias`
- El servicio automáticamente usa datos simulados como fallback

## 📊 Monitoreo

Railway proporciona:
- **Logs en tiempo real** - Click en el servicio → "Logs"
- **Métricas de CPU/RAM** - En el dashboard del servicio
- **Health checks** - Railway hace ping automático al servicio

## 🔄 Actualizaciones

Para actualizar el servicio:
1. Hacer commit y push de los cambios
2. Railway automáticamente detecta y redespliega
3. O manualmente: Click en "Deploy" → "Redeploy"

## 📝 Notas

- El servicio genera gráficas en formato base64 para evitar problemas de almacenamiento
- Las gráficas se generan on-demand, no se guardan en disco
- El análisis puede tomar 5-15 segundos dependiendo del volumen de datos
- Railway tiene un límite de 500MB RAM en el plan gratuito

## 🎯 Próximos Pasos

1. [ ] Configurar monitoreo con alertas
2. [ ] Implementar cache para análisis frecuentes
3. [ ] Agregar más tipos de análisis (regresión, clasificación)
4. [ ] Exportar reportes en PDF
