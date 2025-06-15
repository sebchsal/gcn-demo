# GCN Karate Club Demo
Esta demo implementa una Red Neuronal de Grafos Convolucionales (GCN) para clasificar los nodos del famoso grafo del Karate Club de Zachary usando PyTorch Geometric. Fue desarollada en python y utilzando Visual Studio Code como editor de texto

## Estructura del Proyecto
```
├── main.py                      # Script principal: entrenamiento, visualizaciones, análisis
├── model.py                     # Definición del modelo GCN
├── data.py                      # Carga del dataset Karate Club
├── training.py                  # Función de entrenamiento
├── analysis.py                  # Análisis de importancia de nodos
├── visualization/               #Aporte visual del codigo
│   ├── dashboard.py             # Dashboard interactivo con métricas
│   ├── graph_plotly.py          # Visualización del grafo con Plotly
│   ├── radar_analysis.py        # Radar de métricas estructurales
│   └── traditional_plots.py     # Visualizaciones con Matplotlib
```

## Requisitos
Instala las dependencias con:
```bash
pip install -r requirements.txt #
```
- En requirements.txt se encuentran las bibliotecas utilizadas
(Recuerda tener instalado `torch-geometric` correctamente según tu entorno.)

## ¿Qué hace?
- Entrena una GCN sobre el grafo de karate
- Visualiza etiquetas reales y predicciones
- Crea un dashboard con métricas de confianza y centralidad
- Proyecta embeddings en 2D a través del tiempo

## Resultados
Visualizaciones interactivas y análisis embebido de la red entrenada.

## Bibliotecas Utilizadas
Este proyecto usa las siguientes bibliotecas principales:

- **torch**: para construir y entrenar el modelo GCN.
- **torch_geometric**: operaciones sobre grafos para redes neuronales.
- **networkx**: manejo y análisis de grafos.
- **matplotlib**: visualización tradicional.
- **plotly**: visualización interactiva de métricas y grafos.
- **scikit-learn**:
  - `classification_report`: para evaluar predicciones.
  - `PCA`: para reducir embeddings a 2D.
- **pandas** y **numpy**: manipulación de datos.

---

Desarrollado por [TuNombre]