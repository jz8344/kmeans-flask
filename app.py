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
import traceback
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
        Usa MÚLTIPLES fuentes: viajes, confirmaciones, rutas, paradas, ubicaciones GPS.
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
                
                # Obtener datos completos del chofer
                query = text("""
                    SELECT 
                        v.id as viaje_id,
                        v.chofer_id,
                        v.estado as viaje_estado,
                        v.cupo_maximo,
                        v.cupo_minimo,
                        v.confirmaciones_actuales,
                        v.created_at as viaje_creado,
                        
                        -- Confirmaciones
                        COUNT(DISTINCT cv.id) as total_confirmaciones,
                        COUNT(DISTINCT CASE WHEN cv.estado = 'confirmado' THEN cv.id END) as confirmaciones_activas,
                        
                        -- Asistencias
                        COUNT(DISTINCT a.id) as total_asistencias,
                        COUNT(DISTINCT CASE WHEN a.estado = 'presente' THEN a.id END) as asistencias_presentes,
                        
                        -- Rutas
                        r.distancia_total_km,
                        r.tiempo_estimado_minutos,
                        r.estado as ruta_estado,
                        COUNT(DISTINCT pr.id) as total_paradas,
                        COUNT(DISTINCT CASE WHEN pr.estado = 'completada' THEN pr.id END) as paradas_completadas,
                        
                        -- Ubicaciones GPS (velocidad promedio y total de reportes)
                        COUNT(DISTINCT uc.id) as total_ubicaciones_gps,
                        COALESCE(AVG(uc.velocidad), 0) as velocidad_promedio_ms,
                        COALESCE(MAX(uc.velocidad), 0) as velocidad_maxima_ms
                        
                    FROM viajes v
                    LEFT JOIN confirmaciones_viaje cv ON cv.viaje_id = v.id
                    LEFT JOIN asistencias a ON a.viaje_id = v.id
                    LEFT JOIN rutas r ON r.viaje_id = v.id
                    LEFT JOIN paradas_ruta pr ON pr.ruta_id = r.id
                    LEFT JOIN ubicaciones_chofer uc ON uc.viaje_id = v.id AND uc.chofer_id = v.chofer_id
                    WHERE v.chofer_id = :driver_id
                    GROUP BY v.id, v.chofer_id, v.estado, v.cupo_maximo, v.cupo_minimo, 
                             v.confirmaciones_actuales, v.created_at, r.id, r.distancia_total_km, 
                             r.tiempo_estimado_minutos, r.estado
                    ORDER BY v.created_at DESC
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
                        v.estado as viaje_estado,
                        v.cupo_maximo,
                        v.cupo_minimo,
                        v.confirmaciones_actuales,
                        v.created_at as viaje_creado,
                        
                        -- Confirmaciones
                        COUNT(DISTINCT cv.id) as total_confirmaciones,
                        COUNT(DISTINCT CASE WHEN cv.estado = 'confirmado' THEN cv.id END) as confirmaciones_activas,
                        
                        -- Asistencias  
                        COUNT(DISTINCT a.id) as total_asistencias,
                        COUNT(DISTINCT CASE WHEN a.estado = 'presente' THEN a.id END) as asistencias_presentes,
                        
                        -- Rutas
                        r.distancia_total_km,
                        r.tiempo_estimado_minutos,
                        r.estado as ruta_estado,
                        COUNT(DISTINCT pr.id) as total_paradas,
                        COUNT(DISTINCT CASE WHEN pr.estado = 'completada' THEN pr.id END) as paradas_completadas,
                        
                        -- Ubicaciones GPS
                        COUNT(DISTINCT uc.id) as total_ubicaciones_gps,
                        COALESCE(AVG(uc.velocidad), 0) as velocidad_promedio_ms,
                        COALESCE(MAX(uc.velocidad), 0) as velocidad_maxima_ms
                        
                    FROM viajes v
                    LEFT JOIN choferes c ON c.id = v.chofer_id
                    LEFT JOIN confirmaciones_viaje cv ON cv.viaje_id = v.id
                    LEFT JOIN asistencias a ON a.viaje_id = v.id
                    LEFT JOIN rutas r ON r.viaje_id = v.id
                    LEFT JOIN paradas_ruta pr ON pr.ruta_id = r.id
                    LEFT JOIN ubicaciones_chofer uc ON uc.viaje_id = v.id AND uc.chofer_id = v.chofer_id
                    GROUP BY v.id, v.chofer_id, c.nombre, c.apellidos, v.estado, v.cupo_maximo, 
                             v.cupo_minimo, v.confirmaciones_actuales, v.created_at, r.id,
                             r.distancia_total_km, r.tiempo_estimado_minutos, r.estado
                    ORDER BY v.created_at DESC
                    LIMIT 500
                """)
                result = db.execute(query)
            
            rows = result.fetchall()
            
            if not rows or len(rows) == 0:
                logger.warning("No se encontraron datos de viajes")
                return False
            
            # Convertir a DataFrame con TODAS las métricas
            data = []
            for row in rows:
                data.append({
                    'viaje_id': row.viaje_id,
                    'chofer_id': row.chofer_id,
                    'chofer_nombre': getattr(row, 'chofer_nombre', 'N/A'),
                    'chofer_apellidos': getattr(row, 'chofer_apellidos', 'N/A'),
                    'viaje_estado': row.viaje_estado,
                    'cupo_maximo': row.cupo_maximo or 0,
                    'cupo_minimo': row.cupo_minimo or 0,
                    'confirmaciones_actuales': row.confirmaciones_actuales or 0,
                    
                    # Confirmaciones
                    'total_confirmaciones': row.total_confirmaciones or 0,
                    'confirmaciones_activas': row.confirmaciones_activas or 0,
                    
                    # Asistencias
                    'total_asistencias': row.total_asistencias or 0,
                    'asistencias_presentes': row.asistencias_presentes or 0,
                    
                    # Rutas
                    'distancia_total_km': float(row.distancia_total_km or 0),
                    'tiempo_estimado_minutos': row.tiempo_estimado_minutos or 0,
                    'ruta_estado': row.ruta_estado or 'sin_ruta',
                    'total_paradas': row.total_paradas or 0,
                    'paradas_completadas': row.paradas_completadas or 0,
                    
                    # GPS
                    'total_ubicaciones_gps': row.total_ubicaciones_gps or 0,
                    'velocidad_promedio_ms': float(row.velocidad_promedio_ms or 0),
                    'velocidad_maxima_ms': float(row.velocidad_maxima_ms or 0)
                })
            
            self.df = pd.DataFrame(data)
            
            # ===== CALCULAR MÉTRICAS DERIVADAS =====
            
            # 1. Tasa de asistencia (presentes / confirmaciones)
            self.df['tasa_asistencia'] = np.where(
                self.df['total_confirmaciones'] > 0,
                (self.df['asistencias_presentes'] / self.df['total_confirmaciones'] * 100),
                0
            )
            
            # 2. Tasa de ocupación (confirmaciones / cupo máximo)
            self.df['tasa_ocupacion'] = np.where(
                self.df['cupo_maximo'] > 0,
                (self.df['confirmaciones_actuales'] / self.df['cupo_maximo'] * 100),
                0
            )
            
            # 3. Eficiencia de ruta (paradas completadas / total paradas)
            self.df['eficiencia_ruta'] = np.where(
                self.df['total_paradas'] > 0,
                (self.df['paradas_completadas'] / self.df['total_paradas'] * 100),
                0
            )
            
            # 4. Velocidad promedio en km/h
            self.df['velocidad_promedio_kmh'] = self.df['velocidad_promedio_ms'] * 3.6
            self.df['velocidad_maxima_kmh'] = self.df['velocidad_maxima_ms'] * 3.6
            
            # 5. Actividad GPS (reportes por viaje)
            self.df['actividad_gps'] = self.df['total_ubicaciones_gps']
            
            # 6. Score de puntualidad (basado en reportes GPS y paradas completadas)
            self.df['score_puntualidad'] = np.where(
                (self.df['total_paradas'] > 0) & (self.df['actividad_gps'] > 0),
                ((self.df['eficiencia_ruta'] * 0.7) + (np.minimum(self.df['actividad_gps'] / 10, 30))),
                self.df['eficiencia_ruta']
            )
            
            # 7. Score general (combinación de múltiples métricas)
            self.df['score_general'] = (
                (self.df['tasa_asistencia'] * 0.3) +
                (self.df['tasa_ocupacion'] * 0.2) +
                (self.df['eficiencia_ruta'] * 0.3) +
                (self.df['score_puntualidad'] * 0.2)
            )
            
            logger.info(f"✅ Se cargaron {len(self.df)} registros de viajes con métricas completas")
            logger.info(f"   Métricas disponibles: {list(self.df.columns)}")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching data from DB: {e}")
            return False

    def find_optimal_clusters(self, max_clusters=6):
        """
        Encuentra el número óptimo de clusters usando el método del codo y silhouette score.
        Usa MÚLTIPLES métricas: asistencia, ocupación, eficiencia de ruta, GPS, velocidad.
        """
        # Priorizar features más importantes (ordenadas por relevancia)
        priority_features = [
            'score_general',           # Score combinado
            'tasa_asistencia',         # Asistencia real
            'tasa_ocupacion',          # Ocupación del vehículo
            'eficiencia_ruta',         # Completitud de rutas
            'score_puntualidad',       # Puntualidad
            'velocidad_promedio_kmh',  # Velocidad de conducción
            'actividad_gps',           # Actividad del dispositivo
            'total_confirmaciones',    # Volumen de trabajo
            'distancia_total_km'       # Distancia recorrida
        ]
        
        # Seleccionar las features que existen en el DataFrame
        available_features = [col for col in priority_features if col in self.df.columns]
        
        if len(available_features) < 2:
            logger.warning(f"Solo {len(available_features)} features disponibles, usando por defecto 3 clusters")
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
        Usa múltiples features para análisis robusto.
        """
        # Usar las mismas features que en find_optimal_clusters
        priority_features = [
            'score_general', 'tasa_asistencia', 'tasa_ocupacion', 'eficiencia_ruta',
            'score_puntualidad', 'velocidad_promedio_kmh', 'actividad_gps',
            'total_confirmaciones', 'distancia_total_km'
        ]
        
        available_features = [col for col in priority_features if col in self.df.columns]
        
        if len(available_features) == 0:
            logger.error("No hay features disponibles para clustering")
            raise ValueError("No hay columnas válidas para realizar clustering")
        
        logger.info(f"Usando {len(available_features)} features para clustering: {available_features}")
        
        features = self.df[available_features].fillna(0)
        
        # Verificar que hay datos
        if features.empty or len(features) == 0:
            logger.error("DataFrame de features está vacío")
            raise ValueError("No hay datos para realizar clustering")
        
        # VALIDACIÓN CRÍTICA: n_samples debe ser >= n_clusters
        n_samples = len(features)
        if n_samples < n_clusters:
            logger.warning(f"⚠️ n_samples={n_samples} < n_clusters={n_clusters}. Ajustando n_clusters a {n_samples}.")
            n_clusters = max(1, n_samples - 1)  # Usar n_samples - 1 clusters
            if n_clusters < 2:
                logger.error("Insuficientes datos para clustering (< 2 registros)")
                raise ValueError(f"Se requieren al menos 2 registros para K-means. Solo hay {n_samples}.")
        
        scaled_features = self.scaler.fit_transform(features)
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(scaled_features)
        
        # Etiquetar clusters basándose en score_general (métrica combinada)
        if 'score_general' in self.df.columns:
            cluster_means = self.df.groupby('cluster')['score_general'].mean()
        elif 'tasa_asistencia' in self.df.columns:
            cluster_means = self.df.groupby('cluster')['tasa_asistencia'].mean()
        elif 'tasa_ocupacion' in self.df.columns:
            cluster_means = self.df.groupby('cluster')['tasa_ocupacion'].mean()
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
        
        # Calcular estadísticas COMPLETAS por cluster
        stats = {}
        for cluster in sorted(self.df['cluster'].unique()):
            cluster_data = self.df[self.df['cluster'] == cluster]
            
            # Función auxiliar para convertir valores a float sin NaN (JSON compliant)
            def safe_float(value, default=0.0):
                try:
                    result = float(value)
                    return default if pd.isna(result) or np.isnan(result) or np.isinf(result) else result
                except (ValueError, TypeError):
                    return default
            
            stats[str(cluster)] = {
                "status": cluster_labels.get(cluster, f"Cluster {cluster}"),
                "count": int(len(cluster_data)),
                
                # Score general
                "score_general_mean": safe_float(cluster_data['score_general'].mean() if 'score_general' in cluster_data else 0),
                "score_general_std": safe_float(cluster_data['score_general'].std() if 'score_general' in cluster_data else 0),
                
                # Asistencias
                "tasa_asistencia_mean": safe_float(cluster_data['tasa_asistencia'].mean() if 'tasa_asistencia' in cluster_data else 0),
                "tasa_asistencia_std": safe_float(cluster_data['tasa_asistencia'].std() if 'tasa_asistencia' in cluster_data else 0),
                
                # Ocupación
                "tasa_ocupacion_mean": safe_float(cluster_data['tasa_ocupacion'].mean() if 'tasa_ocupacion' in cluster_data else 0),
                "tasa_ocupacion_std": safe_float(cluster_data['tasa_ocupacion'].std() if 'tasa_ocupacion' in cluster_data else 0),
                
                # Eficiencia de ruta
                "eficiencia_ruta_mean": safe_float(cluster_data['eficiencia_ruta'].mean() if 'eficiencia_ruta' in cluster_data else 0),
                "eficiencia_ruta_std": safe_float(cluster_data['eficiencia_ruta'].std() if 'eficiencia_ruta' in cluster_data else 0),
                
                # Puntualidad
                "score_puntualidad_mean": safe_float(cluster_data['score_puntualidad'].mean() if 'score_puntualidad' in cluster_data else 0),
                
                # Velocidad
                "velocidad_promedio_kmh_mean": safe_float(cluster_data['velocidad_promedio_kmh'].mean() if 'velocidad_promedio_kmh' in cluster_data else 0),
                "velocidad_maxima_kmh_mean": safe_float(cluster_data['velocidad_maxima_kmh'].mean() if 'velocidad_maxima_kmh' in cluster_data else 0),
                
                # Actividad
                "actividad_gps_mean": safe_float(cluster_data['actividad_gps'].mean() if 'actividad_gps' in cluster_data else 0),
                "total_confirmaciones_mean": safe_float(cluster_data['total_confirmaciones'].mean() if 'total_confirmaciones' in cluster_data else 0),
                "distancia_total_km_mean": safe_float(cluster_data['distancia_total_km'].mean() if 'distancia_total_km' in cluster_data else 0)
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
            # 1. Gráfica de dispersión: Score General vs Eficiencia de Ruta
            if 'score_general' in self.df.columns and 'eficiencia_ruta' in self.df.columns:
                fig, ax = plt.subplots(figsize=(12, 7))
                
                for cluster in sorted(self.df['cluster'].unique()):
                    cluster_data = self.df[self.df['cluster'] == cluster]
                    scatter = ax.scatter(
                        cluster_data['score_general'], 
                        cluster_data['eficiencia_ruta'],
                        label=cluster_data['status'].iloc[0] if len(cluster_data) > 0 else f'Cluster {cluster}',
                        alpha=0.7,
                        s=150,
                        edgecolors='black',
                        linewidth=0.5
                    )
                
                ax.set_xlabel('Score General (Combinado)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Eficiencia de Ruta (%)', fontsize=13, fontweight='bold')
                ax.set_title('Clustering de Conductores: Desempeño General vs Eficiencia', 
                            fontsize=15, fontweight='bold', pad=20)
                ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
                
                plots['scatter_score_eficiencia'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # 1b. Scatter alternativo: Ocupación vs Asistencia
            elif 'tasa_ocupacion' in self.df.columns and 'tasa_asistencia' in self.df.columns:
                fig, ax = plt.subplots(figsize=(12, 7))
                
                for cluster in sorted(self.df['cluster'].unique()):
                    cluster_data = self.df[self.df['cluster'] == cluster]
                    ax.scatter(
                        cluster_data['tasa_ocupacion'], 
                        cluster_data['tasa_asistencia'],
                        label=cluster_data['status'].iloc[0] if len(cluster_data) > 0 else f'Cluster {cluster}',
                        alpha=0.7,
                        s=150,
                        edgecolors='black',
                        linewidth=0.5
                    )
                
                ax.set_xlabel('Tasa de Ocupación (%)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Tasa de Asistencia (%)', fontsize=13, fontweight='bold')
                ax.set_title('Clustering de Conductores: Ocupación vs Asistencia', 
                            fontsize=15, fontweight='bold', pad=20)
                ax.legend(loc='best', frameon=True, shadow=True)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                plots['scatter_ocupacion_asistencia'] = self._fig_to_base64(fig)
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
            
            # 4. Box plots de TODAS las métricas principales
            metrics_to_plot = []
            titles = []
            ylabels = []
            
            if 'score_general' in self.df.columns:
                metrics_to_plot.append('score_general')
                titles.append('Score General')
                ylabels.append('Score (0-100)')
            
            if 'tasa_ocupacion' in self.df.columns:
                metrics_to_plot.append('tasa_ocupacion')
                titles.append('Tasa de Ocupación')
                ylabels.append('Ocupación (%)')
            
            if 'eficiencia_ruta' in self.df.columns:
                metrics_to_plot.append('eficiencia_ruta')
                titles.append('Eficiencia de Ruta')
                ylabels.append('Eficiencia (%)')
            
            if 'velocidad_promedio_kmh' in self.df.columns:
                metrics_to_plot.append('velocidad_promedio_kmh')
                titles.append('Velocidad Promedio')
                ylabels.append('Velocidad (km/h)')
            
            # Si hay métricas para graficar
            if len(metrics_to_plot) > 0:
                n_plots = min(len(metrics_to_plot), 4)  # Máximo 4 gráficas
                fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 6))
                
                # Si solo hay 1 gráfica, axes no es array
                if n_plots == 1:
                    axes = [axes]
                
                for i, (metric, title, ylabel) in enumerate(zip(metrics_to_plot[:n_plots], titles[:n_plots], ylabels[:n_plots])):
                    self.df.boxplot(column=metric, by='status', ax=axes[i], patch_artist=True)
                    axes[i].set_title(title, fontsize=12, fontweight='bold')
                    axes[i].set_xlabel('')
                    axes[i].set_ylabel(ylabel, fontsize=11)
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(True, alpha=0.3, linestyle='--')
                
                plt.suptitle('Análisis de Métricas por Cluster (Datos Reales)', 
                            fontsize=14, fontweight='bold', y=1.02)
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
    
    # Si no hay datos reales O hay muy pocos, generar simulados
    if not has_real_data:
        logger.info(f"Usando datos simulados (no se encontraron datos reales)")
        analyzer.generate_sample_data(n_samples=request.n_samples)
    elif len(analyzer.df) < 3:
        logger.warning(f"⚠️ Solo {len(analyzer.df)} registros reales. Generando datos simulados complementarios para análisis robusto.")
        # Guardar datos reales
        real_data_backup = analyzer.df.copy()
        # Generar datos simulados
        analyzer.generate_sample_data(n_samples=max(50, request.n_samples))
        # Agregar nota en el análisis
        has_real_data = False  # Marcar como simulado para la recomendación
    
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


@app.get("/api/analyze/system-stats")
async def analyze_system_stats(db: Session = Depends(get_db)):
    """
    Analiza estadísticas generales del sistema y genera visualizaciones matplotlib
    Incluye: usuarios, estudiantes, choferes, unidades, escuelas, actividad temporal
    """
    if db is None:
        logger.warning("Base de datos no disponible para análisis del sistema.")
        return {
            "status": "error",
            "message": "Base de datos no configurada. Verifique las variables de entorno."
        }
    
    try:
        plots = {}
        stats = {}
        
        # 1. Obtener totales por entidad
        entities_query = text("""
            SELECT 
                'Usuarios' as categoria, COUNT(*) as total FROM usuarios
            UNION ALL
            SELECT 'Estudiantes' as categoria, COUNT(*) as total FROM hijos
            UNION ALL
            SELECT 'Choferes' as categoria, COUNT(*) as total FROM choferes
            UNION ALL
            SELECT 'Unidades' as categoria, COUNT(*) as total FROM unidades
            UNION ALL
            SELECT 'Escuelas' as categoria, COUNT(*) as total FROM escuelas
            UNION ALL
            SELECT 'Viajes' as categoria, COUNT(*) as total FROM viajes
        """)
        entities_result = db.execute(entities_query)
        entities_data = [{"categoria": row[0], "total": row[1]} for row in entities_result]
        df_entities = pd.DataFrame(entities_data)
        stats['entities'] = entities_data
        
        # 2. Obtener escuelas por nivel
        schools_query = text("""
            SELECT nivel, COUNT(*) as cantidad 
            FROM escuelas 
            GROUP BY nivel
            ORDER BY cantidad DESC
        """)
        schools_result = db.execute(schools_query)
        schools_data = [{"nivel": row[0] or "Sin Nivel", "cantidad": row[1]} for row in schools_result]
        df_schools = pd.DataFrame(schools_data)
        stats['schools_by_level'] = schools_data
        
        # 3. Obtener registros por día (últimos 30 días)
        registros_query = text("""
            SELECT 
                DATE(fecha_registro) as fecha,
                COUNT(*) as registros
            FROM usuarios
            WHERE fecha_registro >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(fecha_registro)
            ORDER BY fecha
        """)
        registros_result = db.execute(registros_query)
        registros_data = [{"fecha": str(row[0]), "registros": row[1]} for row in registros_result]
        df_registros = pd.DataFrame(registros_data) if registros_data else pd.DataFrame(columns=['fecha', 'registros'])
        stats['daily_registrations'] = registros_data
        
        # 4. Obtener actividad por entidad (viajes, confirmaciones, asistencias)
        actividad_query = text("""
            SELECT 
                'Viajes' as tipo, DATE(created_at) as fecha, COUNT(*) as cantidad
            FROM viajes
            WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            
            UNION ALL
            
            SELECT 
                'Confirmaciones' as tipo, DATE(created_at) as fecha, COUNT(*) as cantidad
            FROM confirmaciones_viaje
            WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            
            UNION ALL
            
            SELECT 
                'Asistencias' as tipo, DATE(created_at) as fecha, COUNT(*) as cantidad
            FROM asistencias
            WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            
            ORDER BY fecha, tipo
        """)
        actividad_result = db.execute(actividad_query)
        actividad_data = [{"tipo": row[0], "fecha": str(row[1]), "cantidad": row[2]} for row in actividad_result]
        stats['activity_timeline'] = actividad_data
        
        # 5. Estado de viajes
        viajes_estado_query = text("""
            SELECT estado, COUNT(*) as cantidad
            FROM viajes
            GROUP BY estado
        """)
        viajes_estado_result = db.execute(viajes_estado_query)
        viajes_estado_data = [{"estado": row[0], "cantidad": row[1]} for row in viajes_estado_result]
        df_viajes_estado = pd.DataFrame(viajes_estado_data) if viajes_estado_data else pd.DataFrame(columns=['estado', 'cantidad'])
        stats['trips_by_status'] = viajes_estado_data
        
        # ===== GENERAR GRÁFICAS MATPLOTLIB =====
        sns.set_style("whitegrid")
        
        # GRÁFICA 1: Bar Chart - Totales por Entidad
        if not df_entities.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            colors = ['#007bff', '#28a745', '#ffc107', '#17a2b8', '#6c757d', '#6f42c1']
            bars = ax.bar(df_entities['categoria'], df_entities['total'], color=colors, edgecolor='black', linewidth=1.2)
            
            # Agregar valores encima de las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax.set_xlabel('Categoría', fontsize=13, fontweight='bold')
            ax.set_ylabel('Cantidad Total', fontsize=13, fontweight='bold')
            ax.set_title('Resumen General del Sistema TrailynSafe', fontsize=15, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.xticks(rotation=0, ha='center')
            plt.tight_layout()
            
            # Convertir a base64
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plots['entities_overview'] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
            plt.close(fig)
        
        # GRÁFICA 2: Pie Chart - Escuelas por Nivel
        if not df_schools.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            colors_pie = ['#ff6384', '#36a2eb', '#ffce56', '#4bc0c0', '#9966ff', '#ff9f40']
            wedges, texts, autotexts = ax.pie(
                df_schools['cantidad'], 
                labels=df_schools['nivel'],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors_pie,
                explode=[0.05] * len(df_schools),
                shadow=True,
                textprops={'fontsize': 12, 'fontweight': 'bold'}
            )
            
            # Mejorar legibilidad
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Distribución de Escuelas por Nivel Educativo', fontsize=15, fontweight='bold', pad=20)
            plt.tight_layout()
            
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plots['schools_distribution'] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
            plt.close(fig)
        
        # GRÁFICA 3: Line Chart - Registros por Día (últimos 30 días)
        if not df_registros.empty:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Convertir fecha a datetime
            df_registros['fecha'] = pd.to_datetime(df_registros['fecha'])
            df_registros = df_registros.sort_values('fecha')
            
            ax.plot(df_registros['fecha'], df_registros['registros'], 
                   marker='o', linewidth=2.5, markersize=8, color='#007bff', 
                   markerfacecolor='#ffffff', markeredgewidth=2, markeredgecolor='#007bff')
            ax.fill_between(df_registros['fecha'], df_registros['registros'], alpha=0.3, color='#007bff')
            
            ax.set_xlabel('Fecha', fontsize=13, fontweight='bold')
            ax.set_ylabel('Nuevos Registros', fontsize=13, fontweight='bold')
            ax.set_title('Tendencia de Registros de Usuarios (Últimos 30 Días)', fontsize=15, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plots['registration_trend'] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
            plt.close(fig)
        
        # GRÁFICA 4: Heatmap de Actividad (Viajes, Confirmaciones, Asistencias)
        if actividad_data:
            df_actividad = pd.DataFrame(actividad_data)
            df_actividad['fecha'] = pd.to_datetime(df_actividad['fecha'])
            
            # Pivot para heatmap
            df_pivot = df_actividad.pivot_table(
                values='cantidad', 
                index='tipo', 
                columns='fecha', 
                fill_value=0
            )
            
            if not df_pivot.empty:
                fig, ax = plt.subplots(figsize=(16, 6))
                sns.heatmap(df_pivot, annot=True, fmt='g', cmap='YlGnBu', 
                           linewidths=0.5, linecolor='gray', ax=ax, 
                           cbar_kws={'label': 'Cantidad'})
                ax.set_title('Mapa de Calor: Actividad del Sistema (Últimos 30 Días)', 
                            fontsize=15, fontweight='bold', pad=20)
                ax.set_xlabel('Fecha', fontsize=13, fontweight='bold')
                ax.set_ylabel('Tipo de Actividad', fontsize=13, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                buffer = BytesIO()
                fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                plots['activity_heatmap'] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
                plt.close(fig)
        
        # GRÁFICA 5: Pie Chart - Viajes por Estado
        if not df_viajes_estado.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            colors_viajes = ['#28a745', '#ffc107', '#17a2b8', '#dc3545', '#6c757d']
            wedges, texts, autotexts = ax.pie(
                df_viajes_estado['cantidad'], 
                labels=df_viajes_estado['estado'],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors_viajes[:len(df_viajes_estado)],
                explode=[0.05] * len(df_viajes_estado),
                shadow=True,
                textprops={'fontsize': 12, 'fontweight': 'bold'}
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Distribución de Viajes por Estado', fontsize=15, fontweight='bold', pad=20)
            plt.tight_layout()
            
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plots['trips_by_status_chart'] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
            plt.close(fig)
        
        # ===== GRÁFICAS AVANZADAS =====
        
        # GRÁFICA 6: Matriz de Correlación entre Entidades
        try:
            correlation_query = text("""
                SELECT 
                    DATE(u.fecha_registro) as fecha,
                    COUNT(DISTINCT u.id) as usuarios,
                    COUNT(DISTINCT h.id) as estudiantes,
                    COUNT(DISTINCT v.id) as viajes,
                    COUNT(DISTINCT cv.id) as confirmaciones
                FROM usuarios u
                LEFT JOIN hijos h ON DATE(h.created_at) = DATE(u.fecha_registro)
                LEFT JOIN viajes v ON DATE(v.created_at) = DATE(u.fecha_registro)
                LEFT JOIN confirmaciones_viaje cv ON DATE(cv.created_at) = DATE(u.fecha_registro)
                WHERE u.fecha_registro >= CURRENT_DATE - INTERVAL '90 days'
                GROUP BY DATE(u.fecha_registro)
                ORDER BY fecha
            """)
            corr_result = db.execute(correlation_query)
            corr_data = [{"fecha": str(row[0]), "usuarios": row[1], "estudiantes": row[2], "viajes": row[3], "confirmaciones": row[4]} for row in corr_result]
            
            if corr_data:
                df_corr = pd.DataFrame(corr_data)
                df_corr_numeric = df_corr[['usuarios', 'estudiantes', 'viajes', 'confirmaciones']]
                
                if not df_corr_numeric.empty and len(df_corr_numeric) > 1:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    correlation_matrix = df_corr_numeric.corr()
                    
                    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                               center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                               ax=ax, vmin=-1, vmax=1)
                    
                    ax.set_title('Matriz de Correlación entre Entidades del Sistema', 
                                fontsize=15, fontweight='bold', pad=20)
                    plt.tight_layout()
                    
                    buffer = BytesIO()
                    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                    buffer.seek(0)
                    plots['correlation_matrix'] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
                    plt.close(fig)
        except Exception as e:
            logger.warning(f"No se pudo generar matriz de correlación: {e}")
        
        # GRÁFICA 7: Crecimiento Comparativo Temporal (Multi-línea)
        try:
            growth_query = text("""
                SELECT 
                    'Usuarios' as tipo,
                    DATE_TRUNC('week', fecha_registro) as periodo,
                    COUNT(*) as cantidad
                FROM usuarios
                WHERE fecha_registro >= CURRENT_DATE - INTERVAL '12 weeks'
                GROUP BY DATE_TRUNC('week', fecha_registro)
                
                UNION ALL
                
                SELECT 
                    'Estudiantes' as tipo,
                    DATE_TRUNC('week', created_at) as periodo,
                    COUNT(*) as cantidad
                FROM hijos
                WHERE created_at >= CURRENT_DATE - INTERVAL '12 weeks'
                GROUP BY DATE_TRUNC('week', created_at)
                
                UNION ALL
                
                SELECT 
                    'Viajes' as tipo,
                    DATE_TRUNC('week', created_at) as periodo,
                    COUNT(*) as cantidad
                FROM viajes
                WHERE created_at >= CURRENT_DATE - INTERVAL '12 weeks'
                GROUP BY DATE_TRUNC('week', created_at)
                
                ORDER BY periodo, tipo
            """)
            growth_result = db.execute(growth_query)
            growth_data = [{"tipo": row[0], "periodo": str(row[1]), "cantidad": row[2]} for row in growth_result]
            
            if growth_data:
                df_growth = pd.DataFrame(growth_data)
                df_growth['periodo'] = pd.to_datetime(df_growth['periodo'])
                
                fig, ax = plt.subplots(figsize=(14, 7))
                
                for tipo in df_growth['tipo'].unique():
                    df_tipo = df_growth[df_growth['tipo'] == tipo]
                    ax.plot(df_tipo['periodo'], df_tipo['cantidad'], 
                           marker='o', linewidth=2.5, markersize=8, label=tipo)
                
                ax.set_xlabel('Periodo (Semanas)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Cantidad de Registros', fontsize=13, fontweight='bold')
                ax.set_title('Crecimiento Comparativo del Sistema (Últimas 12 Semanas)', 
                            fontsize=15, fontweight='bold', pad=20)
                ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                buffer = BytesIO()
                fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                plots['growth_comparison'] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
                plt.close(fig)
        except Exception as e:
            logger.warning(f"No se pudo generar gráfica de crecimiento: {e}")
        
        # GRÁFICA 8: Top Choferes por Desempeño (Viajes Completados)
        try:
            top_drivers_query = text("""
                SELECT 
                    CONCAT(c.nombre, ' ', c.apellidos) as chofer,
                    COUNT(DISTINCT v.id) as total_viajes,
                    COUNT(DISTINCT CASE WHEN v.estado = 'completado' THEN v.id END) as viajes_completados,
                    COUNT(DISTINCT cv.id) as total_confirmaciones,
                    COALESCE(AVG(uc.velocidad) * 3.6, 0) as velocidad_promedio_kmh
                FROM choferes c
                LEFT JOIN viajes v ON v.chofer_id = c.id
                LEFT JOIN confirmaciones_viaje cv ON cv.viaje_id = v.id
                LEFT JOIN ubicaciones_chofer uc ON uc.chofer_id = c.id AND uc.viaje_id = v.id
                GROUP BY c.id, c.nombre, c.apellidos
                HAVING COUNT(DISTINCT v.id) > 0
                ORDER BY viajes_completados DESC, total_confirmaciones DESC
                LIMIT 10
            """)
            top_drivers_result = db.execute(top_drivers_query)
            top_drivers_data = [{"chofer": row[0], "total_viajes": row[1], "viajes_completados": row[2], 
                                "confirmaciones": row[3], "velocidad": row[4]} for row in top_drivers_result]
            
            if top_drivers_data:
                df_top_drivers = pd.DataFrame(top_drivers_data)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                # Subplot 1: Viajes completados
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_top_drivers)))
                ax1.barh(df_top_drivers['chofer'], df_top_drivers['viajes_completados'], color=colors, edgecolor='black')
                ax1.set_xlabel('Viajes Completados', fontsize=12, fontweight='bold')
                ax1.set_title('Top Choferes por Viajes Completados', fontsize=14, fontweight='bold')
                ax1.grid(axis='x', alpha=0.3, linestyle='--')
                
                # Subplot 2: Confirmaciones
                ax2.barh(df_top_drivers['chofer'], df_top_drivers['confirmaciones'], color=colors, edgecolor='black')
                ax2.set_xlabel('Total Confirmaciones', fontsize=12, fontweight='bold')
                ax2.set_title('Top Choferes por Confirmaciones', fontsize=14, fontweight='bold')
                ax2.grid(axis='x', alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                
                buffer = BytesIO()
                fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                plots['top_drivers_performance'] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
                plt.close(fig)
        except Exception as e:
            logger.warning(f"No se pudo generar gráfica de top choferes: {e}")
        
        # GRÁFICA 9: Análisis de Ocupación de Unidades (Box Plot)
        try:
            ocupacion_query = text("""
                SELECT 
                    u.placa,
                    v.cupo_maximo,
                    v.confirmaciones_actuales,
                    CASE 
                        WHEN v.cupo_maximo > 0 THEN (v.confirmaciones_actuales::float / v.cupo_maximo) * 100
                        ELSE 0
                    END as porcentaje_ocupacion
                FROM unidades u
                LEFT JOIN viajes v ON v.unidad_id = u.id
                WHERE v.id IS NOT NULL AND v.cupo_maximo > 0
            """)
            ocupacion_result = db.execute(ocupacion_query)
            ocupacion_data = [{"placa": row[0], "cupo_maximo": row[1], "confirmaciones": row[2], 
                              "ocupacion": row[3]} for row in ocupacion_result]
            
            if ocupacion_data:
                df_ocupacion = pd.DataFrame(ocupacion_data)
                
                fig, ax = plt.subplots(figsize=(12, 7))
                
                # Crear box plot por unidad
                placas_unicas = df_ocupacion['placa'].unique()[:15]  # Limitar a 15 unidades
                data_for_boxplot = [df_ocupacion[df_ocupacion['placa'] == placa]['ocupacion'].values 
                                   for placa in placas_unicas]
                
                bp = ax.boxplot(data_for_boxplot, labels=placas_unicas, patch_artist=True,
                               boxprops=dict(facecolor='#36a2eb', alpha=0.7),
                               medianprops=dict(color='red', linewidth=2),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5))
                
                ax.set_xlabel('Unidad (Placa)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Porcentaje de Ocupación (%)', fontsize=13, fontweight='bold')
                ax.set_title('Análisis de Ocupación de Unidades', fontsize=15, fontweight='bold', pad=20)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.axhline(y=75, color='orange', linestyle='--', linewidth=2, label='Objetivo 75%')
                ax.legend()
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                buffer = BytesIO()
                fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                plots['units_occupancy_analysis'] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
                plt.close(fig)
        except Exception as e:
            logger.warning(f"No se pudo generar análisis de ocupación: {e}")
        
        # GRÁFICA 10: Clustering de Escuelas (K-Means por características)
        try:
            escuelas_clustering_query = text("""
                SELECT 
                    e.nombre,
                    e.nivel,
                    COUNT(DISTINCT h.id) as total_estudiantes,
                    COUNT(DISTINCT u.id) as total_usuarios,
                    COUNT(DISTINCT pr.id) as total_paradas
                FROM escuelas e
                LEFT JOIN hijos h ON h.escuela_id = e.id
                LEFT JOIN usuarios u ON u.id = h.usuario_id
                LEFT JOIN paradas_ruta pr ON pr.escuela_id = e.id
                GROUP BY e.id, e.nombre, e.nivel
                HAVING COUNT(DISTINCT h.id) > 0
            """)
            esc_cluster_result = db.execute(escuelas_clustering_query)
            esc_cluster_data = [{"nombre": row[0], "nivel": row[1], "estudiantes": row[2], 
                                "usuarios": row[3], "paradas": row[4]} for row in esc_cluster_result]
            
            if len(esc_cluster_data) >= 3:  # K-means requiere al menos 3 muestras
                df_esc = pd.DataFrame(esc_cluster_data)
                
                # Features para clustering
                features_esc = df_esc[['estudiantes', 'usuarios', 'paradas']].fillna(0)
                scaler_esc = StandardScaler()
                features_scaled = scaler_esc.fit_transform(features_esc)
                
                # K-means con 3 clusters
                n_clusters_esc = min(3, len(df_esc) - 1)
                kmeans_esc = KMeans(n_clusters=n_clusters_esc, random_state=42, n_init=10)
                df_esc['cluster'] = kmeans_esc.fit_predict(features_scaled)
                
                # Visualización
                fig, ax = plt.subplots(figsize=(12, 8))
                
                scatter = ax.scatter(df_esc['estudiantes'], df_esc['usuarios'], 
                                    c=df_esc['cluster'], s=df_esc['paradas']*50 + 100,
                                    cmap='viridis', alpha=0.6, edgecolors='black', linewidth=1.5)
                
                # Añadir nombres de escuelas
                for idx, row in df_esc.iterrows():
                    ax.annotate(row['nombre'][:15], (row['estudiantes'], row['usuarios']),
                               fontsize=8, alpha=0.7, ha='center')
                
                ax.set_xlabel('Total de Estudiantes', fontsize=13, fontweight='bold')
                ax.set_ylabel('Total de Usuarios', fontsize=13, fontweight='bold')
                ax.set_title('Clustering de Escuelas por Características\n(Tamaño = Número de Paradas)', 
                            fontsize=15, fontweight='bold', pad=20)
                plt.colorbar(scatter, ax=ax, label='Cluster')
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                
                buffer = BytesIO()
                fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                plots['schools_clustering'] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
                plt.close(fig)
        except Exception as e:
            logger.warning(f"No se pudo generar clustering de escuelas: {e}")
        
        return {
            "status": "success",
            "stats": stats,
            "plots": plots,
            "summary": {
                "total_entities": int(df_entities['total'].sum()) if not df_entities.empty else 0,
                "total_schools": int(df_schools['cantidad'].sum()) if not df_schools.empty else 0,
                "recent_registrations": int(df_registros['registros'].sum()) if not df_registros.empty else 0,
                "visualization_count": len(plots)
            }
        }
        
    except Exception as e:
        print(f"Error en análisis de estadísticas del sistema: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error al generar estadísticas: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
