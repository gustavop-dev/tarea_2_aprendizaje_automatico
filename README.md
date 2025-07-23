# 🧠 TAREA 2 - APRENDIZAJE AUTOMÁTICO

## 📋 Descripción General

Este proyecto implementa una **tarea completa de aprendizaje automático** que incluye:
- **Tarea Original**: Autoencoder con Gradient Accumulation  
- **Extra 1**: Vision Transformers para imágenes
- **Extra 2**: Algoritmos de clustering desde cero (K-means, DBSCAN, BFR)

Todas las implementaciones están desarrolladas **desde cero** usando conceptos fundamentales, con evaluaciones exhaustivas y documentación completa.

---

## 🏗️ Estructura del Proyecto

```
📦 tarea_2_aprendizaje_automatico/
├── 📁 venv/                          # Entorno virtual Python
├── 📁 parte_1/                       # 🎯 TAREA ORIGINAL
│   ├── autoencoder.py                # Arquitectura Autoencoder en PyTorch
│   └── training.py                   # Training loop + Gradient Accumulation
├── 📁 parte_2/                       # ⭐ EXTRAS
│   ├── vision_transformers_vs_cnn.py # 📖 Teoría ViT vs CNNs
│   ├── vision_autoencoder.py         # 🏗️ ViT Autoencoder desde cero
│   ├── vision_training.py            # 🚀 Training para imágenes (CIFAR-10)
│   ├── clustering_algorithms.py      # 🧮 K-means, DBSCAN, BFR
│   ├── clustering_evaluation.py      # 📊 Evaluación y comparación
│   └── README_parte2.md             # 📋 Documentación detallada
├── 📄 requirements.txt               # Dependencias del proyecto
├── 📄 .gitignore                     # Archivos a ignorar en Git
└── 📄 README.md                      # 📋 Este archivo
```

---

## 🎯 **PARTE 1: TAREA ORIGINAL**

### 📝 **Objetivos Cumplidos:**
✅ **Construir arquitectura de red neuronal Autoencoder en PyTorch**  
✅ **Escribir training loop con técnica de Gradient Accumulation**

### 🔧 **Implementación:**

#### `autoencoder.py`
- **Clase `Autoencoder`** completa con encoder y decoder
- Arquitectura: `784 → 128 → 64 → 32 → 64 → 128 → 784`
- Métodos `encode()`, `decode()` y `forward()`
- Activaciones ReLU y Sigmoid para normalización

#### `training.py`  
- **Gradient Accumulation** implementado correctamente
- Dataset sintético con patrones reconocibles
- Monitoreo de pérdidas por época
- Visualización de curvas de entrenamiento
- Guardado automático del modelo

### 🏃‍♂️ **Cómo ejecutar:**
```bash
cd parte_1
python training.py
```

### 📊 **Resultados esperados:**
- Entrenamiento por 20 épocas
- Loss disminuye de ~0.33 → ~0.25
- Modelo guardado como `autoencoder_model.pth`
- Gráfico de curva de entrenamiento

---

## ⭐ **PARTE 2: EXTRAS**

### 🎯 **EXTRA 1: VISION TRANSFORMERS PARA IMÁGENES**

#### 📖 **Investigación y Teoría:**
- **Diferencias fundamentales** entre Vision Transformers y CNNs
- **Análisis comparativo** de arquitecturas, procesamiento y uso
- **Recomendaciones** de cuándo usar cada método

#### 🏗️ **Implementación desde cero:**
- **Vision Transformer Autoencoder** completo
- Componentes: Patch Embedding, Multi-Head Attention, Transformer Blocks
- **Training loop adaptado** para imágenes con CIFAR-10
- **Gradient Accumulation** implementado para ViT
- Visualización de reconstrucciones de imágenes

#### 🏃‍♂️ **Cómo ejecutar:**
```bash
cd parte_2

# Ver teoría ViT vs CNNs
python vision_transformers_vs_cnn.py

# Probar arquitectura ViT
python vision_autoencoder.py

# Entrenar con CIFAR-10 (recomendado)
python vision_training.py
```

#### 📊 **Resultados esperados:**
- Modelo con ~6.4M parámetros  
- Descarga automática de CIFAR-10
- Reconstrucciones visuales cada 5 épocas
- Métricas MSE, MAE, RMSE en test set

---

### 🎯 **EXTRA 2: ALGORITMOS DE CLUSTERING**

#### 🧮 **Implementaciones desde cero:**

##### 1. **K-Means**
- Inicialización aleatoria de centroides
- Iteración hasta convergencia  
- Cálculo de inercia (WCSS)

##### 2. **DBSCAN**
- Algoritmo basado en densidad
- Detección automática de outliers
- Clusters de forma arbitraria

##### 3. **BFR (Bradley-Fayyad-Reina)**
- **Novedad**: Para datasets muy grandes
- Procesamiento en chunks (memoria limitada)
- Tres conjuntos: Discard, Compression, Retained

#### 📊 **Evaluación completa:**
- **Datasets**: Iris, Wine, Synthetic Blobs
- **Métricas**: ARI, Silhouette Score, NMI, V-measure
- **Visualizaciones** 2D de resultados
- **Análisis comparativo** detallado

#### 🏃‍♂️ **Cómo ejecutar:**
```bash
cd parte_2

# Prueba básica de algoritmos
python clustering_algorithms.py

# Evaluación completa (RECOMENDADO)
python clustering_evaluation.py
```

#### 📊 **Resultados esperados:**
- Comparación de métricas entre algoritmos
- Gráficos de barras comparativos
- Visualizaciones 2D de clustering
- Recomendaciones de uso por algoritmo

---

## 🚀 **INSTRUCCIONES DE EJECUCIÓN COMPLETAS**

### 1️⃣ **Configuración inicial:**
```bash
# Clonar o descargar el proyecto
cd tarea_2_aprendizaje_automatico

# Crear y activar entorno virtual (ya creado)
source venv/bin/activate

# Verificar dependencias (ya instaladas)
pip list
```

### 2️⃣ **Ejecutar Parte 1 (Tarea Original):**
```bash
# Autoencoder básico con Gradient Accumulation
cd parte_1
python training.py
cd ..
```

### 3️⃣ **Ejecutar Parte 2 - Extra 1 (Vision Transformers):**
```bash
cd parte_2

# Teoría y comparación ViT vs CNNs
python vision_transformers_vs_cnn.py

# Probar arquitectura ViT
python vision_autoencoder.py

# Entrenar con imágenes reales (CIFAR-10)
python vision_training.py
```

### 4️⃣ **Ejecutar Parte 2 - Extra 2 (Clustering):**
```bash
# Desde parte_2/

# Algoritmos básicos
python clustering_algorithms.py

# Evaluación completa con métricas
python clustering_evaluation.py
```

---

## 📦 **Dependencias**

```txt
torch>=2.0.0           # PyTorch para redes neuronales
torchvision>=0.15.0    # Datasets e imágenes  
numpy>=1.20.0          # Computación numérica
matplotlib>=3.5.0      # Visualizaciones
scikit-learn>=1.0.0    # Métricas de evaluación
pandas>=1.3.0          # Manipulación de datos
```

Instalación automática:
```bash
pip install -r requirements.txt
```

---

## 🏆 **Características Destacadas**

### ✨ **Implementaciones desde cero:**
- No uso librerías pre-existentes para algoritmos principales
- Conceptos fundamentales implementados manualmente
- Código educativo y bien documentado

### ✨ **Gradient Accumulation:**
- Implementado en **ambas partes** del proyecto
- Técnica para simular batches más grandes
- Optimización de memoria y estabilidad

### ✨ **Evaluación exhaustiva:**
- **Múltiples métricas** de evaluación
- **Datasets reales** (CIFAR-10, Iris, Wine)
- **Visualizaciones** y gráficos comparativos

### ✨ **Documentación completa:**
- READMEs detallados en cada parte
- Explicaciones teóricas incluidas  
- Instrucciones paso a paso

---

## 🎯 **Respuestas a Preguntas Clave**

### **"¿Qué métrica usaría para clustering?"**
**Adjusted Rand Index (ARI)** como métrica principal:
- Evalúa similitud con estructura real de datos
- Corrige asignaciones casuales  
- Rango intuitivo [-1, 1] donde 1 = perfecto
- Ideal para comparar algoritmos diferentes

### **"Diferencias ViT vs CNNs?"**
| Aspecto | CNNs | Vision Transformers |
|---------|------|-------------------|
| **Arquitectura** | Convoluciones | Self-Attention |
| **Campo Receptivo** | Local → Global | Global desde inicio |
| **Datos Necesarios** | Pocos | Muchos |
| **Eficiencia** | Alta | Media |

---

## 💡 **Resultados y Conclusiones**

### **Autoencoder (Parte 1):**
- ✅ Convergencia exitosa en ~20 épocas
- ✅ Loss: 0.33 → 0.25
- ✅ Gradient Accumulation funcional

### **Vision Transformers (Extra 1):**
- ✅ Modelo funcional con 6.4M parámetros
- ✅ Reconstrucciones visuales de CIFAR-10
- ✅ Training estable con ViT

### **Clustering (Extra 2):**
- ✅ K-Means: Mejor para clusters esféricos
- ✅ DBSCAN: Mejor para formas irregulares  
- ✅ BFR: Mejor para datasets grandes
- ✅ ARI como métrica recomendada

---