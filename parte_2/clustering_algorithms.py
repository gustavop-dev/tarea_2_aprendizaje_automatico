import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import defaultdict
import time

class KMeansFromScratch:
    """
    Implementación de K-Means desde cero
    """
    
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        
    def fit(self, X):
        """
        Entrenar el modelo K-Means
        """
        if self.random_state:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Inicializar centroides aleatoriamente
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        
        self.history = []
        
        for iteration in range(self.max_iters):
            # Asignar cada punto al centroide más cercano
            distances = self._calculate_distances(X)
            labels = np.argmin(distances, axis=1)
            
            # Actualizar centroides
            new_centroids = np.zeros((self.k, n_features))
            for i in range(self.k):
                if np.sum(labels == i) > 0:
                    new_centroids[i] = X[labels == i].mean(axis=0)
                else:
                    new_centroids[i] = self.centroids[i]
            
            # Verificar convergencia
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                print(f"K-Means convergió en {iteration + 1} iteraciones")
                break
                
            self.centroids = new_centroids
            self.history.append(self.centroids.copy())
            
        self.labels_ = labels
        self.inertia_ = self._calculate_inertia(X, labels)
        
        return self
    
    def _calculate_distances(self, X):
        """Calcular distancias euclidiana a todos los centroides"""
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances
    
    def _calculate_inertia(self, X, labels):
        """Calcular suma de cuadrados intra-cluster (WCSS)"""
        inertia = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia
    
    def predict(self, X):
        """Predecir cluster para nuevos datos"""
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)

class DBSCANFromScratch:
    """
    Implementación de DBSCAN desde cero
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        
    def fit(self, X):
        """
        Entrenar el modelo DBSCAN
        """
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)  # -1 significa ruido
        
        cluster_id = 0
        
        for i in range(n_samples):
            # Saltar si ya está asignado
            if self.labels_[i] != -1:
                continue
                
            # Encontrar vecinos
            neighbors = self._get_neighbors(X, i)
            
            # Si no tiene suficientes vecinos, es ruido
            if len(neighbors) < self.min_samples:
                continue
                
            # Crear nuevo cluster
            self.labels_[i] = cluster_id
            
            # Expandir cluster
            seed_set = neighbors.copy()
            j = 0
            while j < len(seed_set):
                neighbor_idx = seed_set[j]
                
                # Si es ruido, cambiarlo al cluster actual
                if self.labels_[neighbor_idx] == -1:
                    self.labels_[neighbor_idx] = cluster_id
                
                # Si no está asignado, asignarlo y encontrar sus vecinos
                elif self.labels_[neighbor_idx] == -1:
                    self.labels_[neighbor_idx] = cluster_id
                    new_neighbors = self._get_neighbors(X, neighbor_idx)
                    
                    if len(new_neighbors) >= self.min_samples:
                        seed_set.extend(new_neighbors)
                
                j += 1
            
            cluster_id += 1
        
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        return self
    
    def _get_neighbors(self, X, point_idx):
        """Encontrar todos los vecinos dentro de eps"""
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return np.where(distances <= self.eps)[0].tolist()

class BFRFromScratch:
    """
    Implementación del algoritmo BFR (Bradley-Fayyad-Reina) desde cero
    
    BFR es un algoritmo de clustering para datasets grandes que no caben en memoria.
    Mantiene estadísticas resumen de los clusters en lugar de todos los puntos.
    """
    
    def __init__(self, k=3, chunk_size=100, threshold_factor=2.0):
        self.k = k
        self.chunk_size = chunk_size
        self.threshold_factor = threshold_factor
        
    def fit(self, X):
        """
        Entrenar el modelo BFR procesando datos en chunks
        """
        n_samples, n_features = X.shape
        
        # Inicializar clusters principales (Discard Set)
        self.clusters = {}
        for i in range(self.k):
            self.clusters[i] = {
                'N': 0,  # Número de puntos
                'SUM': np.zeros(n_features),  # Suma de coordenadas
                'SUMSQ': np.zeros(n_features),  # Suma de cuadrados
            }
        
        # Compression Set (mini-clusters)
        self.compression_set = {}
        self.compression_id = 0
        
        # Retained Set (puntos outliers)
        self.retained_set = []
        
        # Procesar datos en chunks
        num_chunks = (n_samples + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, n_samples)
            chunk = X[start_idx:end_idx]
            
            self._process_chunk(chunk, chunk_idx == 0)
        
        # Asignar etiquetas finales
        self.labels_ = self._assign_final_labels(X)
        self.n_clusters_ = self.k
        
        return self
    
    def _process_chunk(self, chunk, is_first_chunk):
        """Procesar un chunk de datos"""
        
        if is_first_chunk:
            # Primer chunk: usar K-means para inicializar
            kmeans = KMeansFromScratch(k=self.k, random_state=42)
            kmeans.fit(chunk)
            
            # Inicializar estadísticas de clusters
            for i in range(self.k):
                mask = kmeans.labels_ == i
                points = chunk[mask]
                if len(points) > 0:
                    self.clusters[i]['N'] = len(points)
                    self.clusters[i]['SUM'] = np.sum(points, axis=0)
                    self.clusters[i]['SUMSQ'] = np.sum(points ** 2, axis=0)
        else:
            # Chunks siguientes: asignar puntos a clusters existentes
            for point in chunk:
                assigned = False
                
                # Intentar asignar a cluster principal
                best_cluster = self._find_closest_cluster(point)
                if best_cluster is not None and self._within_threshold(point, best_cluster):
                    self._add_to_cluster(point, best_cluster)
                    assigned = True
                
                # Si no se asigna, intentar con compression set
                if not assigned:
                    best_compression = self._find_closest_compression(point)
                    if best_compression is not None and self._within_compression_threshold(point, best_compression):
                        self._add_to_compression(point, best_compression)
                        assigned = True
                
                # Si aún no se asigna, agregar a retained set
                if not assigned:
                    self.retained_set.append(point)
            
            # Intentar crear nuevos mini-clusters con retained set
            self._update_compression_set()
    
    def _find_closest_cluster(self, point):
        """Encontrar el cluster principal más cercano"""
        min_distance = float('inf')
        closest_cluster = None
        
        for cluster_id, stats in self.clusters.items():
            if stats['N'] > 0:
                centroid = stats['SUM'] / stats['N']
                distance = np.sqrt(np.sum((point - centroid) ** 2))
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster_id
        
        return closest_cluster
    
    def _within_threshold(self, point, cluster_id):
        """Verificar si un punto está dentro del threshold del cluster"""
        stats = self.clusters[cluster_id]
        if stats['N'] == 0:
            return False
        
        centroid = stats['SUM'] / stats['N']
        variance = (stats['SUMSQ'] / stats['N']) - (centroid ** 2)
        std = np.sqrt(np.maximum(variance, 1e-10))  # Evitar división por cero
        
        distance = np.sqrt(np.sum((point - centroid) ** 2))
        threshold = self.threshold_factor * np.mean(std)
        
        return distance <= threshold
    
    def _add_to_cluster(self, point, cluster_id):
        """Agregar punto a cluster principal"""
        self.clusters[cluster_id]['N'] += 1
        self.clusters[cluster_id]['SUM'] += point
        self.clusters[cluster_id]['SUMSQ'] += point ** 2
    
    def _find_closest_compression(self, point):
        """Encontrar mini-cluster más cercano"""
        min_distance = float('inf')
        closest_compression = None
        
        for comp_id, stats in self.compression_set.items():
            centroid = stats['SUM'] / stats['N']
            distance = np.sqrt(np.sum((point - centroid) ** 2))
            if distance < min_distance:
                min_distance = distance
                closest_compression = comp_id
        
        return closest_compression
    
    def _within_compression_threshold(self, point, comp_id):
        """Verificar threshold para compression set"""
        stats = self.compression_set[comp_id]
        centroid = stats['SUM'] / stats['N']
        variance = (stats['SUMSQ'] / stats['N']) - (centroid ** 2)
        std = np.sqrt(np.maximum(variance, 1e-10))
        
        distance = np.sqrt(np.sum((point - centroid) ** 2))
        threshold = self.threshold_factor * np.mean(std)
        
        return distance <= threshold
    
    def _add_to_compression(self, point, comp_id):
        """Agregar punto a compression set"""
        self.compression_set[comp_id]['N'] += 1
        self.compression_set[comp_id]['SUM'] += point
        self.compression_set[comp_id]['SUMSQ'] += point ** 2
    
    def _update_compression_set(self):
        """Actualizar compression set con retained set"""
        if len(self.retained_set) >= 10:  # Mínimo de puntos para crear mini-cluster
            # Convertir a array
            retained_array = np.array(self.retained_set)
            
            # Usar K-means para crear mini-clusters
            mini_k = min(3, len(self.retained_set) // 5)  # Número adaptativo de mini-clusters
            if mini_k >= 1:
                kmeans = KMeansFromScratch(k=mini_k, random_state=42)
                kmeans.fit(retained_array)
                
                # Crear nuevos mini-clusters
                for i in range(mini_k):
                    mask = kmeans.labels_ == i
                    points = retained_array[mask]
                    if len(points) >= 3:  # Mínimo para crear mini-cluster
                        comp_id = self.compression_id
                        self.compression_set[comp_id] = {
                            'N': len(points),
                            'SUM': np.sum(points, axis=0),
                            'SUMSQ': np.sum(points ** 2, axis=0)
                        }
                        self.compression_id += 1
                
                # Limpiar retained set
                self.retained_set = []
    
    def _assign_final_labels(self, X):
        """Asignar etiquetas finales a todos los puntos"""
        labels = np.zeros(len(X))
        
        for i, point in enumerate(X):
            # Buscar en clusters principales
            closest_cluster = self._find_closest_cluster(point)
            if closest_cluster is not None:
                labels[i] = closest_cluster
            else:
                # Si no encuentra cluster principal, asignar al más cercano por fuerza
                min_distance = float('inf')
                for cluster_id, stats in self.clusters.items():
                    if stats['N'] > 0:
                        centroid = stats['SUM'] / stats['N']
                        distance = np.sqrt(np.sum((point - centroid) ** 2))
                        if distance < min_distance:
                            min_distance = distance
                            labels[i] = cluster_id
        
        return labels.astype(int)

def load_tabular_dataset():
    """
    Cargar dataset tabular para clustering
    Usaremos el dataset Iris como ejemplo
    """
    # Cargar Iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, iris.feature_names, iris.target_names

def create_synthetic_dataset():
    """
    Crear dataset sintético para pruebas más robustas
    """
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, 
                      random_state=42, n_features=2)
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Ejemplo de uso
if __name__ == "__main__":
    # Crear dataset de prueba
    X, true_labels = create_synthetic_dataset()
    
    print("Probando algoritmos de clustering...")
    print(f"Dataset: {X.shape[0]} puntos, {X.shape[1]} dimensiones")
    
    # K-Means
    print("\n1. K-Means:")
    kmeans = KMeansFromScratch(k=4, random_state=42)
    kmeans.fit(X)
    print(f"Inertia: {kmeans.inertia_:.3f}")
    
    # DBSCAN
    print("\n2. DBSCAN:")
    dbscan = DBSCANFromScratch(eps=0.3, min_samples=5)
    dbscan.fit(X)
    print(f"Número de clusters: {dbscan.n_clusters_}")
    print(f"Puntos de ruido: {np.sum(dbscan.labels_ == -1)}")
    
    # BFR
    print("\n3. BFR:")
    bfr = BFRFromScratch(k=4, chunk_size=50)
    bfr.fit(X)
    print(f"Número de clusters: {bfr.n_clusters_}")
    
    # Visualización simple
    if X.shape[1] == 2:  # Solo si es 2D
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Dataset original
        axes[0,0].scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis')
        axes[0,0].set_title('Dataset Original')
        
        # K-Means
        axes[0,1].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
        axes[0,1].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                         c='red', marker='x', s=100)
        axes[0,1].set_title('K-Means')
        
        # DBSCAN
        axes[1,0].scatter(X[:, 0], X[:, 1], c=dbscan.labels_, cmap='viridis')
        axes[1,0].set_title('DBSCAN')
        
        # BFR
        axes[1,1].scatter(X[:, 0], X[:, 1], c=bfr.labels_, cmap='viridis')
        axes[1,1].set_title('BFR')
        
        plt.tight_layout()
        plt.savefig('clustering_comparison.png', dpi=150, bbox_inches='tight')
        plt.show() 