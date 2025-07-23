import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.datasets import make_blobs, load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from clustering_algorithms import KMeansFromScratch, DBSCANFromScratch, BFRFromScratch

class ClusteringEvaluator:
    """
    Evaluador completo para algoritmos de clustering
    """
    
    def __init__(self):
        self.results = {}
        
    def evaluate_clustering(self, X, true_labels, predicted_labels, algorithm_name, 
                           centroids=None, exec_time=None):
        """
        Evaluar clustering con múltiples métricas
        
        Métricas utilizadas:
        1. Adjusted Rand Index (ARI): Mide similitud con etiquetas verdaderas
        2. Normalized Mutual Information (NMI): Información mutua normalizada  
        3. Silhouette Score: Cohesión intra-cluster vs separación inter-cluster
        4. Homogeneity: Cada cluster contiene solo miembros de una clase
        5. Completeness: Todos miembros de una clase están en el mismo cluster
        6. V-measure: Media armónica de homogeneity y completeness
        """
        
        # Manejar etiquetas de ruido en DBSCAN
        valid_mask = predicted_labels != -1
        if np.sum(valid_mask) < 2:  # Muy pocos puntos válidos
            print(f"Advertencia: {algorithm_name} produjo muy pocos clusters válidos")
            return None
        
        X_valid = X[valid_mask]
        true_valid = true_labels[valid_mask] if true_labels is not None else None
        pred_valid = predicted_labels[valid_mask]
        
        metrics = {}
        
        # Métricas que requieren etiquetas verdaderas
        if true_labels is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_valid, pred_valid)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_valid, pred_valid)
            metrics['homogeneity'] = homogeneity_score(true_valid, pred_valid)
            metrics['completeness'] = completeness_score(true_valid, pred_valid)
            metrics['v_measure'] = v_measure_score(true_valid, pred_valid)
        
        # Métricas intrínsecas (no requieren etiquetas verdaderas)
        if len(np.unique(pred_valid)) > 1:  # Más de un cluster
            metrics['silhouette_score'] = silhouette_score(X_valid, pred_valid)
        else:
            metrics['silhouette_score'] = -1  # Solo un cluster
        
        # Métricas adicionales
        metrics['n_clusters'] = len(np.unique(pred_valid))
        metrics['n_noise_points'] = np.sum(predicted_labels == -1)
        metrics['noise_ratio'] = metrics['n_noise_points'] / len(predicted_labels)
        
        if exec_time is not None:
            metrics['execution_time'] = exec_time
        
        # Calcular inercia si hay centroides
        if centroids is not None:
            metrics['inertia'] = self._calculate_inertia(X_valid, pred_valid, centroids)
        
        self.results[algorithm_name] = metrics
        return metrics
    
    def _calculate_inertia(self, X, labels, centroids):
        """Calcular inercia (WCSS)"""
        inertia = 0
        for i, centroid in enumerate(centroids):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroid) ** 2)
        return inertia
    
    def print_comparison(self):
        """Imprimir comparación detallada"""
        if not self.results:
            print("No hay resultados para comparar")
            return
        
        print("\n" + "="*80)
        print("COMPARACIÓN DE ALGORITMOS DE CLUSTERING")
        print("="*80)
        
        # Crear DataFrame para mejor visualización
        df = pd.DataFrame(self.results).T
        
        print("\n📊 MÉTRICAS DE EVALUACIÓN:")
        print("-" * 50)
        
        # Métricas principales
        metrics_to_show = [
            'adjusted_rand_score', 'normalized_mutual_info', 'silhouette_score',
            'homogeneity', 'completeness', 'v_measure', 'n_clusters', 
            'noise_ratio', 'execution_time'
        ]
        
        for metric in metrics_to_show:
            if metric in df.columns:
                print(f"\n{metric.upper().replace('_', ' ')}:")
                for alg in df.index:
                    value = df.loc[alg, metric]
                    if isinstance(value, float):
                        print(f"  {alg:<12}: {value:.4f}")
                    else:
                        print(f"  {alg:<12}: {value}")
        
        # Encontrar mejor algoritmo por métrica
        print("\n🏆 MEJOR ALGORITMO POR MÉTRICA:")
        print("-" * 40)
        
        best_algorithms = {}
        for metric in ['adjusted_rand_score', 'normalized_mutual_info', 'silhouette_score']:
            if metric in df.columns:
                best_alg = df[metric].idxmax()
                best_value = df.loc[best_alg, metric]
                best_algorithms[metric] = (best_alg, best_value)
                print(f"{metric:<25}: {best_alg} ({best_value:.4f})")
        
        return df
    
    def plot_comparison(self, save_plot=True):
        """Crear gráficos de comparación"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results).T
        
        # Métricas para graficar
        metrics = ['adjusted_rand_score', 'normalized_mutual_info', 'silhouette_score', 'v_measure']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if not available_metrics:
            print("No hay métricas disponibles para graficar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:4]):
            if i < len(axes):
                values = df[metric].values
                algorithms = df.index.tolist()
                
                bars = axes[i].bar(algorithms, values, alpha=0.7, 
                                 color=['skyblue', 'lightcoral', 'lightgreen'])
                axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('Score')
                axes[i].set_ylim(0, 1) if metric != 'silhouette_score' else axes[i].set_ylim(-1, 1)
                
                # Añadir valores en las barras
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Ocultar subplots no utilizados
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Comparación de Algoritmos de Clustering', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('clustering_metrics_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()

def evaluate_on_multiple_datasets():
    """
    Evaluar algoritmos en múltiples datasets
    """
    evaluator = ClusteringEvaluator()
    
    datasets = {
        'Synthetic Blobs': create_synthetic_blobs(),
        'Iris': load_iris_dataset(), 
        'Wine': load_wine_dataset()
    }
    
    results_summary = {}
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n🔍 EVALUANDO DATASET: {dataset_name}")
        print("=" * 60)
        print(f"Datos: {X.shape[0]} muestras, {X.shape[1]} características")
        print(f"Clusters verdaderos: {len(np.unique(y))}")
        
        # Configurar parámetros por dataset
        if dataset_name == 'Synthetic Blobs':
            k = 4
            eps, min_samples = 0.3, 5
        else:
            k = len(np.unique(y))
            eps, min_samples = 0.5, 3
        
        dataset_results = {}
        
        # K-Means
        print("\n⚙️  Ejecutando K-Means...")
        start_time = time.time()
        kmeans = KMeansFromScratch(k=k, random_state=42)
        kmeans.fit(X)
        kmeans_time = time.time() - start_time
        
        metrics = evaluator.evaluate_clustering(
            X, y, kmeans.labels_, f'K-Means ({dataset_name})', 
            kmeans.centroids, kmeans_time
        )
        dataset_results['K-Means'] = metrics
        
        # DBSCAN
        print("⚙️  Ejecutando DBSCAN...")
        start_time = time.time()
        dbscan = DBSCANFromScratch(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        dbscan_time = time.time() - start_time
        
        metrics = evaluator.evaluate_clustering(
            X, y, dbscan.labels_, f'DBSCAN ({dataset_name})', 
            None, dbscan_time
        )
        dataset_results['DBSCAN'] = metrics
        
        # BFR
        print("⚙️  Ejecutando BFR...")
        start_time = time.time()
        bfr = BFRFromScratch(k=k, chunk_size=min(50, len(X)//3))
        bfr.fit(X)
        bfr_time = time.time() - start_time
        
        metrics = evaluator.evaluate_clustering(
            X, y, bfr.labels_, f'BFR ({dataset_name})', 
            None, bfr_time
        )
        dataset_results['BFR'] = metrics
        
        results_summary[dataset_name] = dataset_results
        
        # Visualizar si es 2D
        if X.shape[1] == 2:
            plot_clustering_results(X, y, kmeans.labels_, dbscan.labels_, 
                                  bfr.labels_, kmeans.centroids, dataset_name)
    
    return results_summary, evaluator

def create_synthetic_blobs():
    """Crear dataset sintético con blobs"""
    X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.60, 
                      random_state=42, n_features=2)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def load_iris_dataset():
    """Cargar dataset Iris"""
    iris = load_iris()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(iris.data)
    return X_scaled, iris.target

def load_wine_dataset():
    """Cargar dataset Wine"""
    wine = load_wine()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(wine.data)
    return X_scaled, wine.target

def plot_clustering_results(X, true_labels, kmeans_labels, dbscan_labels, 
                          bfr_labels, centroids, dataset_name):
    """Visualizar resultados de clustering para datos 2D"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Dataset original
    scatter = axes[0,0].scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', alpha=0.7)
    axes[0,0].set_title(f'{dataset_name} - Etiquetas Verdaderas')
    axes[0,0].grid(True, alpha=0.3)
    
    # K-Means
    axes[0,1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
    if centroids is not None:
        axes[0,1].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
    axes[0,1].set_title('K-Means')
    axes[0,1].grid(True, alpha=0.3)
    
    # DBSCAN
    axes[1,0].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.7)
    axes[1,0].set_title('DBSCAN')
    axes[1,0].grid(True, alpha=0.3)
    
    # BFR
    axes[1,1].scatter(X[:, 0], X[:, 1], c=bfr_labels, cmap='viridis', alpha=0.7)
    axes[1,1].set_title('BFR')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Resultados de Clustering - {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'clustering_results_{dataset_name.lower().replace(" ", "_")}.png', 
               dpi=150, bbox_inches='tight')
    plt.show()

def analyze_clustering_recommendations():
    """
    Análisis de cuándo usar cada algoritmo
    """
    
    print("\n" + "="*80)
    print("📋 RECOMENDACIONES PARA USO DE ALGORITMOS DE CLUSTERING")
    print("="*80)
    
    recommendations = {
        "K-Means": {
            "Ventajas": [
                "Simple y eficiente computacionalmente",
                "Funciona bien con clusters esféricos",
                "Predecible y determinista (con misma inicialización)",
                "Buen rendimiento en datasets grandes"
            ],
            "Desventajas": [
                "Requiere especificar K de antemano",
                "Sensible a inicialización y outliers", 
                "Asume clusters esféricos de tamaño similar",
                "No maneja bien formas complejas"
            ],
            "Cuándo usar": [
                "Clusters esféricos y bien separados",
                "Tamaño de clusters similar",
                "Dataset sin muchos outliers",
                "Necesitas eficiencia computacional"
            ]
        },
        
        "DBSCAN": {
            "Ventajas": [
                "No requiere especificar número de clusters",
                "Encuentra clusters de forma arbitraria",
                "Robusto a outliers (los marca como ruido)",
                "Puede encontrar clusters de densidad variable"
            ],
            "Desventajas": [
                "Sensible a parámetros eps y min_samples",
                "Dificultad con clusters de densidad muy diferente",
                "Puede ser costoso computacionalmente",
                "No funciona bien en alta dimensionalidad"
            ],
            "Cuándo usar": [
                "Clusters de forma irregular",
                "Presencia de outliers/ruido",
                "No sabes cuántos clusters hay",
                "Clusters basados en densidad"
            ]
        },
        
        "BFR": {
            "Ventajas": [
                "Diseñado para datasets muy grandes",
                "Eficiente en memoria (no carga todo)",
                "Mantiene estadísticas resumidas",
                "Bueno para clustering incremental"
            ],
            "Desventajas": [
                "Más complejo de implementar y ajustar",
                "Asume distribución gaussiana",
                "Requiere especificar K",
                "Puede ser menos preciso que otros métodos"
            ],
            "Cuándo usar": [
                "Datasets extremadamente grandes",
                "Memoria limitada",
                "Datos llegando en streaming",
                "Clusters aproximadamente gaussianos"
            ]
        }
    }
    
    for algorithm, details in recommendations.items():
        print(f"\n🔧 {algorithm}:")
        print("-" * 20)
        
        print("✅ Ventajas:")
        for advantage in details["Ventajas"]:
            print(f"   • {advantage}")
        
        print("\n❌ Desventajas:")  
        for disadvantage in details["Desventajas"]:
            print(f"   • {disadvantage}")
        
        print("\n🎯 Cuándo usar:")
        for use_case in details["Cuándo usar"]:
            print(f"   • {use_case}")
        
        print()

def main():
    """Función principal de evaluación"""
    
    print("🚀 INICIANDO EVALUACIÓN COMPLETA DE CLUSTERING")
    print("="*60)
    
    # Evaluar en múltiples datasets
    results_summary, evaluator = evaluate_on_multiple_datasets()
    
    # Imprimir comparación general
    evaluator.print_comparison()
    
    # Crear gráficos de comparación
    evaluator.plot_comparison()
    
    # Análisis y recomendaciones
    analyze_clustering_recommendations()
    
    # Conclusiones
    print("\n" + "="*80)
    print("💡 CONCLUSIONES Y MÉTRICAS RECOMENDADAS")
    print("="*80)
    
    print("""
📊 MÉTRICAS RECOMENDADAS PARA CLUSTERING:

1. **Adjusted Rand Index (ARI)**: 
   - Rango: [-1, 1], mejor = 1
   - Usa cuando tienes etiquetas verdaderas
   - Corrige por casualidad

2. **Silhouette Score**: 
   - Rango: [-1, 1], mejor = 1  
   - Métrica intrínseca (no necesita etiquetas verdaderas)
   - Mide cohesión vs separación

3. **Normalized Mutual Information (NMI)**:
   - Rango: [0, 1], mejor = 1
   - Mide información compartida entre clusters verdaderos y predichos

4. **V-measure**:
   - Rango: [0, 1], mejor = 1
   - Combina homogeneidad y completeness

🎯 MÉTRICA PRINCIPAL RECOMENDADA:
- **Con etiquetas verdaderas**: Adjusted Rand Index (ARI)
- **Sin etiquetas verdaderas**: Silhouette Score

💡 Para tu tarea específica, recomiendo usar **ARI** como métrica principal 
   porque evalúa qué tan bien los algoritmos descubren la estructura real de los datos.
""")

if __name__ == "__main__":
    main() 