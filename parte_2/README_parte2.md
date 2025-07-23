# PARTE 2 - EXTRAS DE APRENDIZAJE AUTOMÁTICO

## 📋 Contenido Implementado

Esta carpeta contiene la implementación de **dos extras** solicitados por el profesor:

### 🎯 **EXTRA 1: Vision Transformers para Imágenes**
- **Objetivo**: Adaptar autoencoder para funcionar con imágenes usando Vision Transformers
- **Investigación**: Diferencias entre Vision Transformers y CNNs
- **Implementación**: Autoencoder basado en ViT con gradient accumulation

### 🎯 **EXTRA 2: Algoritmos de Clustering**
- **Objetivo**: Implementar K-means, DBSCAN y BFR desde cero
- **Comparación**: Evaluar los tres métodos con múltiples métricas
- **Dataset**: Datasets tabulares (Iris, Wine, sintéticos)

---

## 📁 Estructura de Archivos

```
parte_2/
├── vision_transformers_vs_cnn.py    # 📖 Teoría: ViT vs CNNs
├── vision_autoencoder.py            # 🏗️ Implementación ViT Autoencoder  
├── vision_training.py               # 🚀 Training loop para imágenes
├── clustering_algorithms.py         # 🧮 K-means, DBSCAN, BFR desde cero
├── clustering_evaluation.py         # 📊 Evaluación y comparación completa
└── README_parte2.md                # 📋 Este archivo
```

---

## 🔍 **EXTRA 1: VISION TRANSFORMERS**

### **Archivos principales:**
- `vision_transformers_vs_cnn.py`: Explicación teórica completa
- `vision_autoencoder.py`: Implementación desde cero del Vision Transformer Autoencoder
- `vision_training.py`: Training loop adaptado para imágenes con CIFAR-10

### **Características implementadas:**
✅ **Vision Transformer completo** con:
- Patch Embedding
- Multi-Head Self-Attention
- Transformer Blocks
- Positional Encoding

✅ **Autoencoder Architecture**:
- Encoder basado en ViT
- Decoder para reconstrucción de imágenes
- Espacio latente configurable

✅ **Training adaptado**:
- Gradient Accumulation implementado
- Dataset CIFAR-10 automático
- Visualización de reconstrucciones
- Métricas de evaluación (MSE, MAE)

### **Principales diferencias ViT vs CNN:**

| Aspecto | CNNs | Vision Transformers |
|---------|------|-------------------|
| **Arquitectura** | Convoluciones + Pooling | Self-Attention + MLP |
| **Inductive Bias** | Localidad espacial | Ninguno (aprende relaciones) |
| **Campo Receptivo** | Limitado inicialmente | Global desde el inicio |
| **Datos Requeridos** | Funcionan con pocos datos | Requieren datasets grandes |
| **Eficiencia** | Muy eficientes | Más costosos computacionalmente |

### **Cuándo usar cada uno:**
- **CNNs**: Datasets pequeños, recursos limitados, necesitas interpretabilidad
- **ViTs**: Datasets grandes, recursos abundantes, relaciones globales importantes

---

## 🔍 **EXTRA 2: ALGORITMOS DE CLUSTERING**

### **Archivos principales:**
- `clustering_algorithms.py`: Implementaciones desde cero
- `clustering_evaluation.py`: Evaluación completa con métricas

### **Algoritmos implementados:**

#### 🎯 **1. K-Means (desde cero)**
- Inicialización aleatoria de centroides
- Iteración hasta convergencia
- Cálculo de inercia (WCSS)
- Optimizado y eficiente

#### 🎯 **2. DBSCAN (desde cero)**
- Algoritmo basado en densidad
- Detección automática de outliers
- No requiere especificar número de clusters
- Manejo de clusters de forma irregular

#### 🎯 **3. BFR - Bradley-Fayyad-Reina (desde cero)**
- **Novedad**: Algoritmo para datasets muy grandes
- Procesamiento en chunks (memoria limitada)
- Mantiene estadísticas resumidas (N, SUM, SUMSQ)
- Tres conjuntos: Discard Set, Compression Set, Retained Set
- Ideal para streaming de datos

### **Métricas de evaluación utilizadas:**

| Métrica | Rango | Descripción | Cuándo usar |
|---------|-------|-------------|-------------|
| **Adjusted Rand Index (ARI)** | [-1, 1] | Similitud con etiquetas verdaderas | Con ground truth |
| **Silhouette Score** | [-1, 1] | Cohesión vs separación | Sin ground truth |
| **Normalized Mutual Info** | [0, 1] | Información mutua normalizada | Con ground truth |
| **V-measure** | [0, 1] | Combina homogeneidad y completeness | Con ground truth |

### **Datasets evaluados:**
- **Synthetic Blobs**: Dataset sintético 2D para visualización
- **Iris**: Dataset clásico de flores (4 características, 3 clases)
- **Wine**: Dataset de vinos (13 características, 3 clases)

### **Recomendaciones de uso:**

#### 🔧 **K-Means**
✅ **Usar cuando:**
- Clusters esféricos y bien separados
- Dataset sin muchos outliers
- Necesitas eficiencia computacional

#### 🔧 **DBSCAN**
✅ **Usar cuando:**
- Clusters de forma irregular
- Presencia de outliers/ruido
- No sabes cuántos clusters hay

#### 🔧 **BFR**
✅ **Usar cuando:**
- Datasets extremadamente grandes
- Memoria limitada
- Datos llegando en streaming

---

## 🏆 **MÉTRICA RECOMENDADA**

### **Para tu tarea específica:**
**Adjusted Rand Index (ARI)** es la métrica principal recomendada porque:
- Evalúa qué tan bien los algoritmos descubren la estructura real
- Corrige por asignaciones casuales
- Funciona bien para comparar diferentes algoritmos
- Rango intuitivo [-1, 1] donde 1 = perfecto

### **Métrica secundaria:**
**Silhouette Score** como métrica intrínseca (no requiere etiquetas verdaderas)

---

## 🚀 **Cómo ejecutar:**

### **Extra 1 - Vision Transformers:**
```bash
cd parte_2

# Ver explicación teórica
python vision_transformers_vs_cnn.py

# Probar arquitectura
python vision_autoencoder.py

# Entrenar con CIFAR-10 (toma unos minutos)
python vision_training.py
```

### **Extra 2 - Clustering:**
```bash
cd parte_2

# Probar algoritmos básicos
python clustering_algorithms.py

# Evaluación completa (recomendado)
python clustering_evaluation.py
```

---

## 📊 **Resultados esperados:**

### **Vision Transformers:**
- Modelo con ~6.4M parámetros
- Reconstrucciones de imágenes CIFAR-10
- Curvas de entrenamiento
- Comparación visual original vs reconstruido

### **Clustering:**
- Comparación de métricas entre algoritmos
- Visualizaciones 2D de resultados
- Análisis de fortalezas/debilidades
- Recomendaciones de uso

---

## ✨ **Características destacadas:**

1. **Implementaciones desde cero** - No usar librerías, solo conceptos fundamentales
2. **Gradient Accumulation** - Implementado en ambos extras
3. **Evaluación exhaustiva** - Múltiples métricas y datasets
4. **Visualizaciones** - Gráficos y comparaciones visuales
5. **Documentación completa** - Explicaciones teóricas y prácticas

---

¡Tarea completada con éxito! 🎉 