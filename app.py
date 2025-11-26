import os
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la aplicación FastAPI
app = FastAPI(
    title="TrailynSafe Analytics API",
    description="API para análisis avanzado de comportamiento de conductores usando K-Means",
    version="1.0.0"
)

# Configuración de CORS
# Se recomienda ser específico con los orígenes en producción
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://frontend-production-a12b.up.railway.app",
    "https://web-production-86356.up.railway.app",
    "*" # En caso de duda, permitir todo pero con credentials=False
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Para permitir todo sin credenciales
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise

# Configuración de Base de Datos
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = None
SessionLocal = None

if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Conexión a base de datos configurada correctamente.")
    except Exception as e:
        logger.error(f"Error al configurar la base de datos: {e}")

# Dependencia para obtener la sesión de BD
def get_db():
    if SessionLocal is None:
        logger.warning("SessionLocal es None. Base de datos no configurada.")
        # Si no hay BD, permitimos continuar pero las consultas fallarán controladamente
        # raise HTTPException(status_code=503, detail="Base de datos no configurada")
        yield None
        return

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Modelos Pydantic
class AnalysisRequest(BaseModel):
    driver_id: Optional[int] = None
    n_samples: int = 1000

class ClusterStats(BaseModel):
    heart_rate_mean: float
    heart_rate_std: float
    accel_mean: float
    accel_std: float
    count: int

class AnalysisResponse(BaseModel):
    driver_id: Optional[int]
    optimal_k: int
    clusters: Dict[str, Any]
    recommendations: List[str]
    visualization_data: Dict[str, Any]

# Lógica de Análisis (Adaptada de driver_kmeans_analysis.py)
class DriverBehaviorAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.df = None
        
    def generate_sample_data(self, n_samples=1000):
        """
        Genera datos simulados si no hay datos reales disponibles.
        """
        np.random.seed(42)
        
        normal = n_samples // 3
        stressed = n_samples // 3
        risk = n_samples - (normal + stressed)
        
        # Conductor normal
        heart_rate_normal = np.random.normal(70, 5, normal)
        accel_x_normal = np.random.normal(0.2, 0.3, normal)
        accel_y_normal = np.random.normal(0.3, 0.3, normal)
        accel_z_normal = np.random.normal(9.8, 0.2, normal)
        
        # Conductor estresado
        heart_rate_stressed = np.random.normal(95, 8, stressed)
        accel_x_stressed = np.random.normal(0.8, 0.5, stressed)
        accel_y_stressed = np.random.normal(1.0, 0.5, stressed)
        accel_z_stressed = np.random.normal(9.7, 0.4, stressed)
        
        # Conductor en riesgo
        heart_rate_risk = np.random.normal(125, 10, risk)
        accel_x_risk = np.random.normal(1.5, 0.8, risk)
        accel_y_risk = np.random.normal(2.0, 1.0, risk)
        accel_z_risk = np.random.normal(9.5, 0.6, risk)
        
        heart_rate = np.concatenate([heart_rate_normal, heart_rate_stressed, heart_rate_risk])
        accel_x = np.concatenate([accel_x_normal, accel_x_stressed, accel_x_risk])
        accel_y = np.concatenate([accel_y_normal, accel_y_stressed, accel_y_risk])
        accel_z = np.concatenate([accel_z_normal, accel_z_stressed, accel_z_risk])
        
        accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        self.df = pd.DataFrame({
            'heart_rate': heart_rate,
            'accel_magnitude': accel_magnitude,
            'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='30s')
        })
        return self.df

    def fetch_data_from_db(self, db: Session, driver_id: int):
        """
        Intenta obtener datos reales de la base de datos.
        Por ahora, como no tenemos tabla de sensores, usaremos UbicacionChofer para simular
        o retornaremos False para usar datos simulados.
        """
        # Aquí iría la consulta real si tuviéramos los datos
        # query = text("SELECT * FROM sensor_data WHERE driver_id = :driver_id")
        # result = db.execute(query, {"driver_id": driver_id})
        # ...
        
        # Verificamos si el chofer existe al menos
        try:
            query = text("SELECT id, nombre, apellidos FROM choferes WHERE id = :driver_id")
            result = db.execute(query, {"driver_id": driver_id}).fetchone()
            if not result:
                return False # Chofer no encontrado
            
            # Si el chofer existe pero no tenemos datos de sensores, usamos simulados
            # O podríamos intentar usar 'velocidad' de UbicacionChofer como proxy
            return False 
        except Exception as e:
            logger.error(f"Error fetching data from DB: {e}")
            return False

    def find_optimal_clusters(self, max_clusters=10):
        features = self.df[['heart_rate', 'accel_magnitude']]
        scaled_features = self.scaler.fit_transform(features)
        
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)
            silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))
            
        optimal_k = K_range[np.argmax(silhouette_scores)]
        return optimal_k

    def perform_clustering(self, n_clusters=3):
        features = self.df[['heart_rate', 'accel_magnitude']]
        scaled_features = self.scaler.fit_transform(features)
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(scaled_features)
        
        # Etiquetas
        cluster_means = self.df.groupby('cluster')['heart_rate'].mean()
        sorted_clusters = cluster_means.sort_values().index.tolist()
        
        cluster_labels = {}
        if n_clusters >= 3:
            cluster_labels[sorted_clusters[0]] = 'Normal'
            cluster_labels[sorted_clusters[1]] = 'Precaución'
            cluster_labels[sorted_clusters[2]] = 'Riesgo Alto'
            for i in range(3, n_clusters):
                cluster_labels[sorted_clusters[i]] = f'Cluster {i+1}'
        else:
            for i in range(n_clusters):
                cluster_labels[sorted_clusters[i]] = f'Grupo {i+1}'
                
        self.df['status'] = self.df['cluster'].map(cluster_labels)
        
        # Estadísticas
        stats = {}
        for cluster in sorted(self.df['cluster'].unique()):
            cluster_data = self.df[self.df['cluster'] == cluster]
            stats[str(cluster)] = {
                "status": cluster_labels.get(cluster, f"Cluster {cluster}"),
                "count": int(len(cluster_data)),
                "heart_rate_mean": float(cluster_data['heart_rate'].mean()),
                "heart_rate_std": float(cluster_data['heart_rate'].std()),
                "accel_mean": float(cluster_data['accel_magnitude'].mean()),
                "accel_std": float(cluster_data['accel_magnitude'].std())
            }
            
        return stats, cluster_labels

    def get_visualization_data(self):
        """
        Prepara datos para visualización en frontend (e.g. Chart.js o ApexCharts)
        """
        # Convertir a lista de diccionarios para JSON
        # Limitar a 1000 puntos para no saturar el frontend
        df_sample = self.df.sample(min(len(self.df), 1000))
        return {
            "scatter": df_sample[['heart_rate', 'accel_magnitude', 'cluster', 'status']].to_dict(orient='records'),
            "centroids": self.scaler.inverse_transform(self.kmeans.cluster_centers_).tolist()
        }

# Endpoints
@app.get("/")
def read_root():
    return {"message": "TrailynSafe Analytics API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/analyze/driver", response_model=AnalysisResponse)
def analyze_driver(request: AnalysisRequest, db: Session = Depends(get_db)):
    analyzer = DriverBehaviorAnalysis()
    
    # Intentar obtener datos reales
    has_real_data = False
    if request.driver_id and db is not None:
        try:
            has_real_data = analyzer.fetch_data_from_db(db, request.driver_id)
        except Exception as e:
            logger.error(f"Error al intentar obtener datos reales: {e}")
            has_real_data = False
    
    # Si no hay datos reales, generar simulados
    if not has_real_data:
        logger.info(f"Usando datos simulados para driver_id={request.driver_id}")
        analyzer.generate_sample_data(n_samples=request.n_samples)
    
    # Análisis
    try:
        optimal_k = analyzer.find_optimal_clusters(max_clusters=5)
        stats, labels = analyzer.perform_clustering(n_clusters=optimal_k)
        viz_data = analyzer.get_visualization_data()
        
        # Recomendaciones
        recommendations = []
        for cluster_id, data in stats.items():
            status = data['status']
            if status == 'Normal':
                recommendations.append(f"Grupo {status}: Continuar monitoreo regular.")
            elif status == 'Precaución':
                recommendations.append(f"Grupo {status}: Sugerir descansos más frecuentes.")
            elif status == 'Riesgo Alto':
                recommendations.append(f"Grupo {status}: ALERTA - Revisar comportamiento inmediatamente.")
                
        return AnalysisResponse(
            driver_id=request.driver_id,
            optimal_k=optimal_k,
            clusters=stats,
            recommendations=recommendations,
            visualization_data=viz_data
        )
    except Exception as e:
        logger.error(f"Error durante el análisis: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno en el análisis: {str(e)}")

@app.get("/api/drivers")
def get_drivers(db: Session = Depends(get_db)):
    """
    Obtiene lista de choferes para el selector en el frontend
    """
    try:
        logger.info("Intentando obtener choferes de la BD...")
        query = text("SELECT id, nombre, apellidos FROM choferes ORDER BY nombre")
        result = db.execute(query).fetchall()
        drivers = [{"id": row.id, "nombre": f"{row.nombre} {row.apellidos}"} for row in result]
        logger.info(f"Se encontraron {len(drivers)} choferes.")
        return drivers
    except Exception as e:
        logger.error(f"Error fetching drivers: {e}")
        # En caso de error de BD, devolver lista vacía o simulada para no romper el frontend con 500
        logger.warning("Devolviendo lista vacía debido a error de BD.")
        return []

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
