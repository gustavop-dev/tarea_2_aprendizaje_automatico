"""
VISION TRANSFORMERS vs REDES CONVOLUCIONALES

==============================================================
DIFERENCIAS PRINCIPALES
==============================================================

1. ARQUITECTURA FUNDAMENTAL:

CNNs (Redes Convolucionales):
- Basadas en operaciones de convolución
- Utilizan filtros/kernels que se deslizan sobre la imagen
- Explotan la localidad espacial y translación invariante
- Arquitectura jerárquica: extracción de características locales → globales
- Inductive bias: asumen que píxeles cercanos están relacionados

Vision Transformers (ViTs):
- Basados en mecanismo de atención (attention mechanism)
- Dividen imagen en patches y los tratan como secuencias
- No tienen inductive bias espacial inherente
- Aprenden relaciones globales desde el inicio
- Basados en arquitectura Transformer (originalmente para NLP)

==============================================================
2. PROCESAMIENTO DE DATOS:

CNNs:
- Input: Imagen completa (H x W x C)
- Procesan píxel por píxel con ventanas locales
- Pooling para reducir dimensionalidad
- Feature maps de diferentes escalas

Vision Transformers:
- Input: Secuencia de patches (N x D)
- Cada patch se linealiza y proyecta a embedding
- Añaden positional encoding para mantener información espacial
- Self-attention para relacionar todos los patches

==============================================================
3. VENTAJAS Y DESVENTAJAS:

CNNs:
✅ Ventajas:
- Eficientes computacionalmente
- Buen inductive bias para imágenes
- Funcionan bien con pocos datos
- Interpretables (visualización de filtros)

❌ Desventajas:
- Campo receptivo limitado
- Dificultad para capturar dependencias de largo alcance
- Arquitectura fija

Vision Transformers:
✅ Ventajas:
- Capturan dependencias globales desde el inicio
- Escalables a imágenes grandes
- Flexibles (mismo mecanismo para diferentes tareas)
- Excelente rendimiento con grandes datasets

❌ Desventajas:
- Requieren grandes cantidades de datos
- Computacionalmente más costosos
- Menos interpretables
- Falta de inductive bias espacial

==============================================================
4. CUÁNDO USAR CADA UNO:

Usar CNNs cuando:
- Dataset pequeño/mediano
- Recursos computacionales limitados
- Necesitas interpretabilidad
- Tareas específicas de visión (detección de bordes, texturas)

Usar Vision Transformers cuando:
- Dataset muy grande disponible
- Recursos computacionales abundantes
- Necesitas capturar relaciones globales
- Tareas que requieren comprensión de contexto completo

==============================================================
""" 