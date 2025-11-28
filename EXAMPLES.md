# 📝 Ejemplos de Uso - API K-Means

## 🔧 Requisitos Previos

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar .env
DATABASE_URL=postgresql://user:pass@host:5432/dbname
PORT=8000
```

## 🚀 Ejecutar Localmente

```bash
# Opción 1: Desarrollo con reload
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Opción 2: Producción
python app.py

# Opción 3: Con logs detallados
uvicorn app:app --host 0.0.0.0 --port 8000 --log-level debug
```

## 📡 Ejemplos de Peticiones

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "ok",
  "database": "connected",
  "timestamp": "2025-11-27T10:30:00.123456"
}
```

---

### 2. Listar Conductores Disponibles

```bash
curl http://localhost:8000/api/drivers
```

**Respuesta:**
```json
[
  {
    "id": 1,
    "nombre": "Juan Pérez González"
  },
  {
    "id": 2,
    "nombre": "María López Hernández"
  }
]
```

---

### 3. Análisis Global (Todos los Conductores)

```bash
curl -X POST http://localhost:8000/api/analyze/driver \
  -H "Content-Type: application/json" \
  -d '{
    "driver_id": null,
    "n_samples": 500
  }'
```

**Respuesta (simplificada):**
```json
{
  "driver_id": null,
  "optimal_k": 3,
  "total_records": 450,
  "data_source": "real",
  "clusters": {
    "0": {
      "status": "Excelente Desempeño",
      "count": 180,
      "tasa_asistencia_mean": 96.8,
      "eficiencia_mean": 103.2,
      "tiempo_recogida_mean": 4.5
    },
    "1": {
      "status": "Desempeño Promedio",
      "count": 200,
      "tasa_asistencia_mean": 85.3,
      "eficiencia_mean": 95.7,
      "tiempo_recogida_mean": 7.8
    },
    "2": {
      "status": "Requiere Atención",
      "count": 70,
      "tasa_asistencia_mean": 65.2,
      "eficiencia_mean": 78.4,
      "tiempo_recogida_mean": 12.3
    }
  },
  "recommendations": [
    "✅ Excelente Desempeño: 180 viajes - Mantener este nivel.",
    "⚠️ Desempeño Promedio: 200 viajes - Oportunidades de mejora.",
    "🔴 Requiere Atención: 70 viajes - Seguimiento inmediato.",
    "📈 Análisis basado en 450 viajes reales."
  ],
  "plots": {
    "scatter_asistencia_eficiencia": "data:image/png;base64,iVBORw0KGgoAAAANS...",
    "elbow_method": "data:image/png;base64,iVBORw0KGgoAAAANS...",
    "cluster_distribution": "data:image/png;base64,iVBORw0KGgoAAAANS...",
    "boxplot_metrics": "data:image/png;base64,iVBORw0KGgoAAAANS..."
  }
}
```

---

### 4. Análisis de Conductor Específico

```bash
curl -X POST http://localhost:8000/api/analyze/driver \
  -H "Content-Type: application/json" \
  -d '{
    "driver_id": 1,
    "n_samples": 1000
  }'
```

---

### 5. Análisis con Python Requests

```python
import requests
import json
from PIL import Image
import base64
from io import BytesIO

# Configuración
API_URL = "http://localhost:8000"

# Hacer petición
response = requests.post(
    f"{API_URL}/api/analyze/driver",
    json={
        "driver_id": None,  # Análisis global
        "n_samples": 1000
    }
)

if response.status_code == 200:
    data = response.json()
    
    # Mostrar estadísticas
    print(f"Clusters detectados: {data['optimal_k']}")
    print(f"Total de registros: {data['total_records']}")
    print(f"Fuente: {data['data_source']}")
    
    # Mostrar clusters
    for cluster_id, stats in data['clusters'].items():
        print(f"\n{stats['status']}:")
        print(f"  - Viajes: {stats['count']}")
        print(f"  - Asistencia: {stats['tasa_asistencia_mean']:.1f}%")
        print(f"  - Eficiencia: {stats['eficiencia_mean']:.1f}%")
    
    # Guardar gráficas
    for plot_name, base64_data in data['plots'].items():
        # Remover el prefijo data:image/png;base64,
        img_data = base64_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        
        # Guardar como archivo
        with open(f"{plot_name}.png", "wb") as f:
            f.write(img_bytes)
        
        print(f"✅ Gráfica guardada: {plot_name}.png")
    
    # Mostrar recomendaciones
    print("\n📊 Recomendaciones:")
    for rec in data['recommendations']:
        print(f"  {rec}")
else:
    print(f"Error {response.status_code}: {response.text}")
```

---

### 6. Análisis desde JavaScript/Frontend

```javascript
// AdminEstadisticas.vue - Función runAnalysis()
async function runAnalysis() {
  analyzing.value = true
  analysisResult.value = null
  
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/analyze/driver`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        driver_id: selectedDriverId.value || null,
        n_samples: 1000
      })
    })
    
    if (response.ok) {
      analysisResult.value = await response.json()
      
      // Mostrar gráficas
      console.log('Análisis completado:', analysisResult.value)
    } else {
      throw new Error('Error en el análisis')
    }
  } catch (e) {
    console.error('Error:', e)
    error.value = 'Error al ejecutar el análisis'
  } finally {
    analyzing.value = false
  }
}
```

---

### 7. Llamar desde Laravel (Proxy)

```php
// En cualquier controlador
use Illuminate\Support\Facades\Http;

public function getDriverAnalysis(Request $request)
{
    $kmeansUrl = env('KMEANS_API_URL', 'http://localhost:8000');
    
    $response = Http::timeout(30)
        ->post($kmeansUrl . '/api/analyze/driver', [
            'driver_id' => $request->input('driver_id'),
            'n_samples' => 1000
        ]);
    
    if ($response->successful()) {
        return response()->json($response->json());
    }
    
    return response()->json([
        'error' => 'Failed to get analysis'
    ], 500);
}
```

---

## 🎨 Mostrar Gráficas en HTML

```html
<!DOCTYPE html>
<html>
<head>
    <title>K-Means Analysis</title>
</head>
<body>
    <h1>Análisis de Conductores</h1>
    
    <div id="plots"></div>
    
    <script>
        fetch('http://localhost:8000/api/analyze/driver', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                driver_id: null,
                n_samples: 500
            })
        })
        .then(res => res.json())
        .then(data => {
            const plotsDiv = document.getElementById('plots');
            
            // Iterar sobre todas las gráficas
            for (const [name, base64] of Object.entries(data.plots)) {
                const img = document.createElement('img');
                img.src = base64;  // Ya incluye el prefijo data:image/png;base64,
                img.alt = name;
                img.style.maxWidth = '100%';
                img.style.margin = '20px 0';
                
                const title = document.createElement('h2');
                title.textContent = name.replace(/_/g, ' ').toUpperCase();
                
                plotsDiv.appendChild(title);
                plotsDiv.appendChild(img);
            }
        })
        .catch(err => console.error('Error:', err));
    </script>
</body>
</html>
```

---

## 🧪 Testing Local

```bash
# Test 1: Verificar servicio
python test_service.py

# Test 2: Health check
curl http://localhost:8000/health

# Test 3: Listar conductores
curl http://localhost:8000/api/drivers

# Test 4: Análisis simple
curl -X POST http://localhost:8000/api/analyze/driver \
  -H "Content-Type: application/json" \
  -d '{"driver_id": null, "n_samples": 100}'
```

---

## 📊 Interpretar Resultados

### Tasa de Asistencia
- **> 90%**: Excelente
- **70-90%**: Aceptable
- **< 70%**: Problemático

### Eficiencia
- **> 100%**: Más rápido que lo estimado
- **90-100%**: En tiempo
- **< 90%**: Retrasos frecuentes

### Tiempo de Recogida
- **< 5 min**: Muy eficiente
- **5-10 min**: Normal
- **> 10 min**: Lento

---

## 🔒 Seguridad en Producción

```python
# Agregar autenticación (ejemplo con API Key)
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "tu-api-key-secreta"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Usar en endpoints
@app.post("/api/analyze/driver")
async def analyze_driver(
    request: AnalysisRequest, 
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    # ... código del análisis
```

---

## 📖 Referencias

- [scikit-learn K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Railway Deployment](https://docs.railway.app/)
