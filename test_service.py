"""
Script de prueba para el servicio K-means Flask
Prueba la conexión a BD y el análisis K-means
"""

import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def test_imports():
    """Verifica que todas las librerías estén instaladas"""
    print("🔍 Verificando imports...")
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import seaborn
        import psycopg2
        import sqlalchemy
        print("✅ Todos los imports exitosos")
        return True
    except ImportError as e:
        print(f"❌ Error en imports: {e}")
        return False

def test_database_connection():
    """Verifica la conexión a la base de datos"""
    print("\n🔍 Verificando conexión a PostgreSQL...")
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL no configurado")
        return False
    
    # Convertir postgres:// a postgresql://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Probar consulta simple
            result = conn.execute(text("SELECT COUNT(*) as total FROM choferes"))
            count = result.fetchone()[0]
            print(f"✅ Conexión exitosa - {count} choferes en la BD")
            
            # Verificar que hay datos para análisis
            result = conn.execute(text("SELECT COUNT(*) as total FROM viajes"))
            viajes_count = result.fetchone()[0]
            print(f"✅ {viajes_count} viajes encontrados")
            
            return True
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False

def test_kmeans_analysis():
    """Prueba el análisis K-means con datos simulados"""
    print("\n🔍 Probando análisis K-means...")
    
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from app import DriverBehaviorAnalysis
        
        analyzer = DriverBehaviorAnalysis()
        
        # Generar datos de prueba
        print("  - Generando datos simulados...")
        analyzer.generate_sample_data(n_samples=500)
        
        # Encontrar clusters óptimos
        print("  - Encontrando número óptimo de clusters...")
        optimal_k = analyzer.find_optimal_clusters(max_clusters=5)
        print(f"  - K óptimo: {optimal_k}")
        
        # Realizar clustering
        print("  - Realizando clustering...")
        stats, labels = analyzer.perform_clustering(n_clusters=optimal_k)
        
        print(f"✅ Análisis completado - {len(stats)} clusters detectados")
        
        # Probar generación de gráficas
        print("  - Generando gráficas matplotlib...")
        plots = analyzer.generate_plots()
        print(f"✅ {len(plots)} gráficas generadas")
        
        return True
    except Exception as e:
        print(f"❌ Error en análisis K-means: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecuta todas las pruebas"""
    print("=" * 60)
    print("🧪 PRUEBAS DEL SERVICIO K-MEANS FLASK")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Base de datos
    results.append(("Conexión BD", test_database_connection()))
    
    # Test 3: Análisis K-means
    results.append(("Análisis K-means", test_kmeans_analysis()))
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("🎉 TODAS LAS PRUEBAS PASARON")
        print("\n💡 El servicio está listo para ejecutarse:")
        print("   uvicorn app:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print("⚠️  ALGUNAS PRUEBAS FALLARON")
        print("\n🔧 Revisar los errores anteriores y:")
        print("   1. Verificar que DATABASE_URL esté configurado")
        print("   2. Instalar dependencias: pip install -r requirements.txt")
        print("   3. Verificar conectividad a PostgreSQL")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
