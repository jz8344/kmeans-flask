"""
Flask API para generación de rutas con k-Means + TSP
Minimalista y directo
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import math
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
print(">>> app.py importado correctamente", flush=True)
app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'k-Means Route Optimizer'})

@app.route('/api/generar-ruta', methods=['POST'])
def generar_ruta():
    try:
        data = request.get_json(silent=True) or {}
        
        viaje_id = data.get('viaje_id')
        puntos = data.get('puntos', [])
        destino = data.get('destino')
        hora_salida = data.get('hora_salida', '07:00:00')
        
        if not puntos or not destino:
            return jsonify({'error': 'Datos incompletos'}), 400
        
        # Ejecutar k-Means + TSP
        resultado = optimizar_ruta(puntos, destino, hora_salida)
        
        return jsonify({
            'success': True,
            'viaje_id': viaje_id,
            'ruta': resultado
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def optimizar_ruta(puntos, destino, hora_salida):
    """k-Means clustering + TSP vecino más cercano"""
    
    # 1. Coordenadas
    coords = np.array([[p['latitud'], p['longitud']] for p in puntos])
    
    # 2. k-Means
    n_clusters = max(1, min(math.ceil(len(puntos) / 10), len(puntos)))
    
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        centros = kmeans.cluster_centers_
    else:
        labels = np.zeros(len(puntos), dtype=int)
        centros = np.array([np.mean(coords, axis=0)])
    
    # 3. Ordenar clusters (más lejano primero)
    destino_coord = np.array([destino['latitud'], destino['longitud']])
    distancias = [(i, haversine(centro, destino_coord)) 
                  for i, centro in enumerate(centros)]
    clusters_ordenados = [i for i, _ in sorted(distancias, key=lambda x: x[1], reverse=True)]
    
    # 4. TSP por cluster
    ruta = []
    punto_anterior = None
    
    for cluster_id in clusters_ordenados:
        indices = np.where(labels == cluster_id)[0]
        puntos_cluster = [puntos[i] for i in indices]
        
        for p in puntos_cluster:
            p['cluster'] = int(cluster_id)
        
        ordenados = tsp_vecino_cercano(puntos_cluster, punto_anterior)
        ruta.extend(ordenados)
        punto_anterior = ordenados[-1] if ordenados else None
    
    # 5. Calcular tiempos
    hora = datetime.strptime(hora_salida, '%H:%M:%S')
    resultado = []
    
    for i, parada in enumerate(ruta):
        if i == 0:
            dist_km = 0
            tiempo_min = 5
        else:
            anterior = ruta[i-1]
            dist_km = haversine(
                [anterior['latitud'], anterior['longitud']],
                [parada['latitud'], parada['longitud']]
            )
            tiempo_min = (dist_km / 20) * 60 + 2  # 20 km/h + 2 min parada
        
        hora += timedelta(minutes=tiempo_min)
        
        resultado.append({
            'confirmacion_id': parada['confirmacion_id'],
            'hijo_id': parada['hijo_id'],
            'direccion': parada['direccion'],
            'latitud': parada['latitud'],
            'longitud': parada['longitud'],
            'hora_estimada': hora.strftime('%H:%M:%S'),
            'distancia_desde_anterior_km': round(dist_km, 2),
            'tiempo_desde_anterior_min': round(tiempo_min),
            'cluster': parada.get('cluster', 0)
        })
    
    distancia_total = sum(p['distancia_desde_anterior_km'] for p in resultado)
    tiempo_total = sum(p['tiempo_desde_anterior_min'] for p in resultado)
    
    return {
        'paradas': resultado,
        'distancia_total_km': round(distancia_total, 2),
        'tiempo_total_min': round(tiempo_total),
        'parametros': {
            'n_clusters': n_clusters,
            'total_puntos': len(puntos)
        }
    }

def haversine(coord1, coord2):
    """Distancia en km entre coordenadas"""
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return 6371 * c

def tsp_vecino_cercano(puntos, punto_inicial=None):
    """TSP greedy"""
    if not puntos:
        return []
    if len(puntos) == 1:
        return puntos
    
    restantes = puntos.copy()
    ruta = []
    
    if punto_inicial:
        actual = min(restantes, key=lambda p: haversine(
            [punto_inicial['latitud'], punto_inicial['longitud']],
            [p['latitud'], p['longitud']]
        ))
    else:
        actual = max(restantes, key=lambda p: p['latitud'])
    
    ruta.append(actual)
    restantes.remove(actual)
    
    while restantes:
        siguiente = min(restantes, key=lambda p: haversine(
            [actual['latitud'], actual['longitud']],
            [p['latitud'], p['longitud']]
        ))
        ruta.append(siguiente)
        restantes.remove(siguiente)
        actual = siguiente
    
    return ruta

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
