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

## Documentacion utilizada de Plolty para los aportes
1. plotly.subplots.make_subplots:
Crea un objeto Figure con un layout de subplots (filas, columnas, tipos de ejes, títulos, etc.).
Link documentación: https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots
2. plotly.graph_objects.Figure:
Contenedor principal de Plotly: almacena las “trazas” (gráficos) y el layout (títulos, márgenes, ejes).
Link documentación: https://plotly.com/python/graph-objects/
3. plotly.graph_objects.Scatter:
Traza para gráficos de dispersión (puntos, líneas, texto). Se usan en:
- Líneas de conexión en el grafo interactivo
- Scatter centralidad vs confianza
- Scatter grado vs confianza
Link documentación: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter
4. plotly.graph_objects.Histogram:
Traza para histogramas de distribución (barras verticales u horizontales).
Link documentación: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Histogram
5. plotly.graph_objects.Heatmap:
Traza para mapas de calor (tiles coloreados según valores en matriz).
Link documentación: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Heatmap
6. plotly.graph_objects.Table
Traza para tablas interactivas (celdas y encabezados formateables).
Link documentación: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Table
7. plotly.graph_objects.Scatterpolar
Traza para diagramas radiales (radar charts), usada en create_network_statistics_radar.
Link documentación: https://plotly.com/python-api-reference/
---

Desarrollado por:
- Gloriana Mojica Rojas
- Sebastian Chaves Salazar
- Priscilla Murillo Romero
- Naara Menjívar Ramirez