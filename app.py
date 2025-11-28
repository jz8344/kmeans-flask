import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la aplicación FastAPI
app = FastAPI(
    title="TrailynSafe Analytics API",
    description="API para análisis avanzado de comportamiento de conductores usando K-Means",
    version="1.0.1"
)

# Configuración de CORS mejorada para Railway
# IMPORTANTE: No usar allow_credentials=True con wildcard "*"
ALLOWED_ORIGINS = [
    "https://frontend-production-a12b.up.railway.app",
    "https://web-production-86356.up.railway.app",
    "http://localhost:5173",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=False,  # DEBE ser False cuando se usa wildcard
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
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
        Ahora incluye las métricas correctas para el clustering.
        """
        np.random.seed(42)
        
        excelente = n_samples // 3
        promedio = n_samples // 3
        atencion = n_samples - (excelente + promedio)
        
        # Grupo Excelente: alta asistencia, buena eficiencia, tiempo rápido
        tasa_asistencia_exc = np.random.normal(95, 3, excelente)
        eficiencia_exc = np.random.normal(105, 5, excelente)
        tiempo_recogida_exc = np.random.normal(4, 1, excelente)
        confirmaciones_exc = np.random.randint(8, 15, excelente)
        
        # Grupo Promedio: asistencia media, eficiencia normal
        tasa_asistencia_prom = np.random.normal(80, 5, promedio)
        eficiencia_prom = np.random.normal(95, 8, promedio)
        tiempo_recogida_prom = np.random.normal(8, 2, promedio)
        confirmaciones_prom = np.random.randint(5, 12, promedio)
        
        # Grupo Requiere Atención: baja asistencia, problemas de eficiencia
        tasa_asistencia_aten = np.random.normal(65, 8, atencion)
        eficiencia_aten = np.random.normal(75, 10, atencion)
        tiempo_recogida_aten = np.random.normal(15, 3, atencion)
        confirmaciones_aten = np.random.randint(3, 10, atencion)
        
        # Combinar todos los datos
        tasa_asistencia = np.concatenate([tasa_asistencia_exc, tasa_asistencia_prom, tasa_asistencia_aten])
        eficiencia = np.concatenate([eficiencia_exc, eficiencia_prom, eficiencia_aten])
        tiempo_recogida = np.concatenate([tiempo_recogida_exc, tiempo_recogida_prom, tiempo_recogida_aten])
        confirmaciones = np.concatenate([confirmaciones_exc, confirmaciones_prom, confirmaciones_aten])
        
        # Clip valores para que sean realistas
        tasa_asistencia = np.clip(tasa_asistencia, 0, 100)
        eficiencia = np.clip(eficiencia, 50, 150)
        tiempo_recogida = np.clip(tiempo_recogida, 1, 30)
        
        self.df = pd.DataFrame({
            'viaje_id': range(1, n_samples + 1),
            'chofer_id': 1,
            'tasa_asistencia': tasa_asistencia,
            'eficiencia': eficiencia,
            'tiempo_promedio_recogida': tiempo_recogida,
            'total_confirmaciones': confirmaciones,
            'duracion_real': np.random.uniform(20, 60, n_samples),
            'duracion_estimada': np.random.uniform(20, 60, n_samples)
        })
        
        logger.info(f"Generados {n_samples} registros simulados con métricas de clustering")
        return self.df

    def fetch_data_from_db(self, db: Session, driver_id: int = None):
        """
        Obtiene datos reales de la base de datos para análisis K-means.
        Usa datos de viajes, confirmaciones, asistencias y ubicaciones GPS.
        """
        try:
            # Si se especifica driver_id, filtrar por ese chofer
            if driver_id:
                # Verificar que el chofer existe
                query = text("SELECT id, nombre, apellidos FROM choferes WHERE id = :driver_id")
                result = db.execute(query, {"driver_id": driver_id}).fetchone()
                if not result:
                    logger.warning(f"Chofer {driver_id} no encontrado")
                    return False
                
                # Obtener datos de viajes del chofer
                query = text("""
                    SELECT 
                        v.id as viaje_id,
                        v.chofer_id,
                        v.fecha_inicio,
                        v.fecha_fin,
                        v.estado,
                        v.duracion_estimada,
                        COUNT(DISTINCT cv.id) as total_confirmaciones,
                        COUNT(DISTINCT a.id) as total_asistencias,
                        COALESCE(AVG(EXTRACT(EPOCH FROM (a.hora_registro - cv.created_at))/60), 0) as tiempo_promedio_recogida
                    FROM viajes v
                    LEFT JOIN confirmaciones_viaje cv ON cv.viaje_id = v.id
                    LEFT JOIN asistencias a ON a.viaje_id = v.id
                    WHERE v.chofer_id = :driver_id
                    GROUP BY v.id, v.chofer_id, v.fecha_inicio, v.fecha_fin, v.estado, v.duracion_estimada
                    ORDER BY v.fecha_inicio DESC
                    LIMIT 100
                """)
                result = db.execute(query, {"driver_id": driver_id})
            else:
                # Análisis global de todos los choferes
                query = text("""
                    SELECT 
                        v.id as viaje_id,
                        v.chofer_id,
                        c.nombre as chofer_nombre,
                        c.apellidos as chofer_apellidos,
                        v.fecha_inicio,
                        v.fecha_fin,
                        v.estado,
                        v.duracion_estimada,
                        COUNT(DISTINCT cv.id) as total_confirmaciones,
                        COUNT(DISTINCT a.id) as total_asistencias,
                        COALESCE(AVG(EXTRACT(EPOCH FROM (a.hora_registro - cv.created_at))/60), 0) as tiempo_promedio_recogida
                    FROM viajes v
                    LEFT JOIN choferes c ON c.id = v.chofer_id
                    LEFT JOIN confirmaciones_viaje cv ON cv.viaje_id = v.id
                    LEFT JOIN asistencias a ON a.viaje_id = v.id
                    WHERE v.fecha_inicio >= NOW() - INTERVAL '6 months'
                    GROUP BY v.id, v.chofer_id, c.nombre, c.apellidos, v.fecha_inicio, v.fecha_fin, v.estado, v.duracion_estimada
                    ORDER BY v.fecha_inicio DESC
                    LIMIT 500
                """)
                result = db.execute(query)
            
            rows = result.fetchall()
            
            if not rows or len(rows) == 0:
                logger.warning("No se encontraron datos de viajes")
                return False
            
            # Convertir a DataFrame
            data = []
            for row in rows:
                data.append({
                    'viaje_id': row.viaje_id,
                    'chofer_id': row.chofer_id,
                    'chofer_nombre': getattr(row, 'chofer_nombre', 'N/A'),
                    'chofer_apellidos': getattr(row, 'chofer_apellidos', 'N/A'),
                    'fecha_inicio': row.fecha_inicio,
                    'fecha_fin': row.fecha_fin,
                    'estado': row.estado,
                    'duracion_estimada': row.duracion_estimada or 0,
                    'total_confirmaciones': row.total_confirmaciones or 0,
                    'total_asistencias': row.total_asistencias or 0,
                    'tiempo_promedio_recogida': float(row.tiempo_promedio_recogida or 0)
                })
            
            self.df = pd.DataFrame(data)
            
            # Calcular métricas adicionales
            self.df['tasa_asistencia'] = np.where(
                self.df['total_confirmaciones'] > 0,
                (self.df['total_asistencias'] / self.df['total_confirmaciones'] * 100),
                0
            )
            
            # Duración real del viaje (si tiene fecha_fin)
            self.df['duracion_real'] = self.df.apply(
                lambda x: (x['fecha_fin'] - x['fecha_inicio']).total_seconds() / 60 
                if pd.notnull(x['fecha_fin']) and pd.notnull(x['fecha_inicio']) 
                else x['duracion_estimada'], 
                axis=1
            )
            
            # Eficiencia (duración real vs estimada)
            self.df['eficiencia'] = np.where(
                self.df['duracion_estimada'] > 0,
                (self.df['duracion_estimada'] / self.df['duracion_real'] * 100).clip(0, 200),
                100
            )
            
            logger.info(f"Se cargaron {len(self.df)} registros de viajes para análisis")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching data from DB: {e}")
            return False

    def find_optimal_clusters(self, max_clusters=6):
        """
        Encuentra el número óptimo de clusters usando el método del codo y silhouette score.
        Usa métricas reales de los choferes: tasa de asistencia, eficiencia, tiempo de recogida.
        """
        # Seleccionar features relevantes
        feature_cols = ['tasa_asistencia', 'eficiencia', 'tiempo_promedio_recogida', 'total_confirmaciones']
        
        # Verificar que existen las columnas
        available_features = [col for col in feature_cols if col in self.df.columns]
        
        if len(available_features) < 2:
            logger.warning("No hay suficientes features para clustering, usando valores por defecto")
            return 3
        
        features = self.df[available_features].fillna(0)
        
        # Si hay muy pocos datos, limitar clusters
        n_samples = len(features)
        max_clusters = min(max_clusters, n_samples // 2)
        
        if max_clusters < 2:
            return 2
        
        scaled_features = self.scaler.fit_transform(features)
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)
            inertias.append(kmeans.inertia_)
            
            if n_samples > k:  # Silhouette score requiere más muestras que clusters
                silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))
            else:
                silhouette_scores.append(0)
        
        # Elegir el k con mejor silhouette score (balance entre separación y cohesión)
        optimal_k = K_range[np.argmax(silhouette_scores)] if silhouette_scores else 3
        
        # Guardar métricas para gráficas
        self.elbow_data = {
            'k_values': list(K_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
        
        return optimal_k

    def perform_clustering(self, n_clusters=3):
        """
        Realiza clustering K-means sobre datos reales de choferes.
        Features: tasa_asistencia, eficiencia, tiempo_promedio_recogida
        """
        # Seleccionar features para clustering
        feature_cols = ['tasa_asistencia', 'eficiencia', 'tiempo_promedio_recogida']
        available_features = [col for col in feature_cols if col in self.df.columns]
        
        if len(available_features) == 0:
            logger.error("No hay features disponibles para clustering")
            raise ValueError("No hay columnas válidas para realizar clustering")
        
        features = self.df[available_features].fillna(0)
        
        # Verificar que hay datos
        if features.empty or len(features) == 0:
            logger.error("DataFrame de features está vacío")
            raise ValueError("No hay datos para realizar clustering")
        
        scaled_features = self.scaler.fit_transform(features)
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(scaled_features)
        
        # Etiquetar clusters basándose en tasa de asistencia (métrica principal)
        if 'tasa_asistencia' in self.df.columns:
            cluster_means = self.df.groupby('cluster')['tasa_asistencia'].mean()
        elif 'eficiencia' in self.df.columns:
            cluster_means = self.df.groupby('cluster')['eficiencia'].mean()
        else:
            cluster_means = self.df.groupby('cluster').size()
        
        sorted_clusters = cluster_means.sort_values(ascending=False).index.tolist()
        
        # Asignar etiquetas descriptivas
        cluster_labels = {}
        if n_clusters >= 3:
            cluster_labels[sorted_clusters[0]] = 'Excelente Desempeño'
            cluster_labels[sorted_clusters[1]] = 'Desempeño Promedio'
            cluster_labels[sorted_clusters[2]] = 'Requiere Atención'
            for i in range(3, n_clusters):
                cluster_labels[sorted_clusters[i]] = f'Grupo {i+1}'
        else:
            cluster_labels[sorted_clusters[0]] = 'Alto Desempeño'
            cluster_labels[sorted_clusters[-1]] = 'Bajo Desempeño'
                
        self.df['status'] = self.df['cluster'].map(cluster_labels)
        
        # Calcular estadísticas por cluster
        stats = {}
        for cluster in sorted(self.df['cluster'].unique()):
            cluster_data = self.df[self.df['cluster'] == cluster]
            
            stats[str(cluster)] = {
                "status": cluster_labels.get(cluster, f"Cluster {cluster}"),
                "count": int(len(cluster_data)),
                "tasa_asistencia_mean": float(cluster_data['tasa_asistencia'].mean()) if 'tasa_asistencia' in cluster_data else 0,
                "tasa_asistencia_std": float(cluster_data['tasa_asistencia'].std()) if 'tasa_asistencia' in cluster_data else 0,
                "eficiencia_mean": float(cluster_data['eficiencia'].mean()) if 'eficiencia' in cluster_data else 0,
                "eficiencia_std": float(cluster_data['eficiencia'].std()) if 'eficiencia' in cluster_data else 0,
                "tiempo_recogida_mean": float(cluster_data['tiempo_promedio_recogida'].mean()) if 'tiempo_promedio_recogida' in cluster_data else 0,
                "tiempo_recogida_std": float(cluster_data['tiempo_promedio_recogida'].std()) if 'tiempo_promedio_recogida' in cluster_data else 0,
                "total_confirmaciones_mean": float(cluster_data['total_confirmaciones'].mean()) if 'total_confirmaciones' in cluster_data else 0
            }
            
        return stats, cluster_labels

    def get_visualization_data(self):
        """
        Prepara datos para visualización en frontend
        """
        df_sample = self.df.sample(min(len(self.df), 500))
        
        # Seleccionar columnas relevantes que existen
        cols_to_export = ['cluster', 'status']
        for col in ['tasa_asistencia', 'eficiencia', 'tiempo_promedio_recogida', 'total_confirmaciones', 'chofer_id']:
            if col in df_sample.columns:
                cols_to_export.append(col)
        
        return {
            "scatter": df_sample[cols_to_export].to_dict(orient='records'),
            "centroids": self.scaler.inverse_transform(self.kmeans.cluster_centers_).tolist() if self.kmeans else []
        }
    
    def generate_plots(self):
        """
        Genera gráficas matplotlib y las devuelve como imágenes base64
        """
        plots = {}
        
        # Configuración de estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        try:
            # 1. Gráfica de dispersión: Tasa de Asistencia vs Eficiencia
            if 'tasa_asistencia' in self.df.columns and 'eficiencia' in self.df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for cluster in sorted(self.df['cluster'].unique()):
                    cluster_data = self.df[self.df['cluster'] == cluster]
                    ax.scatter(
                        cluster_data['tasa_asistencia'], 
                        cluster_data['eficiencia'],
                        label=cluster_data['status'].iloc[0] if len(cluster_data) > 0 else f'Cluster {cluster}',
                        alpha=0.6,
                        s=100
                    )
                
                ax.set_xlabel('Tasa de Asistencia (%)', fontsize=12)
                ax.set_ylabel('Eficiencia (%)', fontsize=12)
                ax.set_title('Clustering de Choferes: Asistencia vs Eficiencia', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plots['scatter_asistencia_eficiencia'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # 2. Método del Codo (Elbow Method)
            if hasattr(self, 'elbow_data') and self.elbow_data:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Inertia
                ax1.plot(self.elbow_data['k_values'], self.elbow_data['inertias'], 'bo-', linewidth=2, markersize=8)
                ax1.set_xlabel('Número de Clusters (K)', fontsize=12)
                ax1.set_ylabel('Inercia (Within-Cluster Sum of Squares)', fontsize=12)
                ax1.set_title('Método del Codo', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                # Silhouette Score
                ax2.plot(self.elbow_data['k_values'], self.elbow_data['silhouette_scores'], 'ro-', linewidth=2, markersize=8)
                ax2.set_xlabel('Número de Clusters (K)', fontsize=12)
                ax2.set_ylabel('Silhouette Score', fontsize=12)
                ax2.set_title('Silhouette Score por K', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plots['elbow_method'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # 3. Distribución de clusters (Bar Chart)
            fig, ax = plt.subplots(figsize=(10, 6))
            cluster_counts = self.df.groupby('status').size().sort_values(ascending=False)
            
            colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db', '#9b59b6']
            cluster_counts.plot(kind='bar', ax=ax, color=colors[:len(cluster_counts)])
            
            ax.set_xlabel('Categoría de Desempeño', fontsize=12)
            ax.set_ylabel('Número de Viajes', fontsize=12)
            ax.set_title('Distribución de Viajes por Categoría de Desempeño', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plots['cluster_distribution'] = self._fig_to_base64(fig)
            plt.close(fig)
            
            # 4. Box plot de métricas por cluster
            if 'tasa_asistencia' in self.df.columns:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Tasa de Asistencia
                self.df.boxplot(column='tasa_asistencia', by='status', ax=axes[0])
                axes[0].set_title('Tasa de Asistencia por Cluster')
                axes[0].set_xlabel('')
                axes[0].set_ylabel('Tasa de Asistencia (%)')
                
                # Eficiencia
                if 'eficiencia' in self.df.columns:
                    self.df.boxplot(column='eficiencia', by='status', ax=axes[1])
                    axes[1].set_title('Eficiencia por Cluster')
                    axes[1].set_xlabel('')
                    axes[1].set_ylabel('Eficiencia (%)')
                
                # Tiempo de Recogida
                if 'tiempo_promedio_recogida' in self.df.columns:
                    self.df.boxplot(column='tiempo_promedio_recogida', by='status', ax=axes[2])
                    axes[2].set_title('Tiempo Promedio de Recogida por Cluster')
                    axes[2].set_xlabel('')
                    axes[2].set_ylabel('Tiempo (minutos)')
                
                plt.suptitle('Análisis de Métricas por Cluster', fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                plots['boxplot_metrics'] = self._fig_to_base64(fig)
                plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error generando gráficas: {e}")
        
        return plots
    
    def _fig_to_base64(self, fig):
        """Convierte una figura matplotlib a string base64"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        return f"data:image/png;base64,{image_base64}"

# Endpoints
@app.get("/")
def read_root():
    return {
        "message": "TrailynSafe Analytics API is running",
        "version": "1.0.1",
        "status": "healthy",
        "endpoints": [
            "/api/analyze/driver (POST)",
            "/api/analyze/driver/test (GET)",
            "/api/drivers (GET)",
            "/health (GET)"
        ]
    }

@app.get("/health")
def health_check():
    db_status = "connected" if engine else "not_configured"
    return {
        "status": "ok",
        "database": db_status,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.options("/api/analyze/driver")
def options_analyze():
    """Manejar preflight requests explícitamente"""
    return {"message": "OK"}

@app.options("/api/drivers")
def options_drivers():
    """Manejar preflight requests explícitamente"""
    return {"message": "OK"}

@app.get("/api/analyze/driver/test")
def test_analyze():
    """Endpoint de prueba para verificar que el servicio funciona"""
    return {
        "message": "El servicio está funcionando correctamente",
        "instructions": "Este endpoint requiere POST con JSON: {driver_id: 1, n_samples: 1000}",
        "example_curl": "curl -X POST https://kmeans-flask-production.up.railway.app/api/analyze/driver -H 'Content-Type: application/json' -d '{\"driver_id\":1,\"n_samples\":100}'"
    }

@app.post("/api/analyze/driver")
def analyze_driver(request: AnalysisRequest, db: Session = Depends(get_db)):
    """
    Endpoint principal para análisis K-means de choferes.
    Retorna estadísticas de clusters y gráficas matplotlib en base64.
    """
    analyzer = DriverBehaviorAnalysis()
    
    # Intentar obtener datos reales de la BD
    has_real_data = False
    if db is not None:
        try:
            has_real_data = analyzer.fetch_data_from_db(db, request.driver_id)
        except Exception as e:
            logger.error(f"Error al obtener datos reales: {e}")
            has_real_data = False
    
    # Si no hay datos reales, generar simulados
    if not has_real_data:
        logger.info(f"Usando datos simulados (no se encontraron datos reales)")
        analyzer.generate_sample_data(n_samples=request.n_samples)
    
    # Realizar análisis K-means
    try:
        optimal_k = analyzer.find_optimal_clusters(max_clusters=5)
        stats, labels = analyzer.perform_clustering(n_clusters=optimal_k)
        viz_data = analyzer.get_visualization_data()
        
        # Generar gráficas matplotlib
        plots = analyzer.generate_plots()
        
        # Generar recomendaciones basadas en los clusters
        recommendations = []
        for cluster_id, data in stats.items():
            status = data['status']
            count = data['count']
            
            if 'Excelente' in status or 'Alto' in status:
                recommendations.append(f"✅ {status}: {count} viajes - Mantener este nivel de desempeño.")
            elif 'Promedio' in status:
                recommendations.append(f"⚠️ {status}: {count} viajes - Oportunidades de mejora en puntualidad y asistencia.")
            elif 'Atención' in status or 'Bajo' in status:
                recommendations.append(f"🔴 {status}: {count} viajes - Requiere capacitación y seguimiento inmediato.")
            else:
                recommendations.append(f"📊 {status}: {count} viajes registrados.")
        
        # Agregar recomendación general
        if has_real_data:
            total_viajes = len(analyzer.df)
            recommendations.append(f"\n📈 Análisis basado en {total_viajes} viajes reales de la base de datos.")
        else:
            recommendations.append(f"\n⚠️ Análisis basado en datos simulados. Conecte a la base de datos para análisis real.")
        
        return JSONResponse(content={
            "driver_id": request.driver_id,
            "optimal_k": optimal_k,
            "clusters": stats,
            "recommendations": recommendations,
            "visualization_data": viz_data,
            "plots": plots,
            "data_source": "real" if has_real_data else "simulated",
            "total_records": len(analyzer.df)
        })
        
    except Exception as e:
        logger.error(f"Error durante el análisis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno en el análisis: {str(e)}")

@app.get("/api/drivers")
def get_drivers(db: Session = Depends(get_db)):
    """
    Obtiene lista de choferes para el selector en el frontend
    """
    if db is None:
        logger.warning("Base de datos no disponible. Devolviendo lista vacía.")
        return []
        
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
