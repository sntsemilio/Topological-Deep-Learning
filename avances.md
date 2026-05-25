### Aspectos Clave de la Implementación y Validación:                                                                                   
                                                                                                                                          
  1. Entorno Aislado y Estable: Para prevenir los conflictos de versiones entre  scikit-learn  y  giotto-tda  que ocurrían debido al cruce
  de librerías globales del sistema, he forzado un aislamiento absoluto en Anaconda configurando  os.environ["PYTHONNOUSERSITE"] = "1"  al
  inicio de la libreta. Asimismo, he instalado  torch ,  kmapper  y  giotto-tda  de forma nativa en tu Anaconda, lo que resolvió          
  definitivamente el error de  c10.dll  y el conflicto de  numpy .                                                                        
  2. Parámetros de Takens: Los algoritmos implementados de AMI (Información Mutua Promedio) y FNN (Falsos Vecinos Más Cercanos) calcularon
  dinámicamente que el retardo óptimo global es $\tau = 5$ y la dimensión es $d = 2$, inmergiendo la dinámica conjunta de Valencia y      
  Activación en un espacio de fase en $\mathbb{R}^4$.                                                                                     
  3. Grafo de Reeb Segregado: Se construyó el Grafo de Reeb global utilizando una lente compuesta de PCA más la excentricidad Euclídea    
  $L_2$ (para medir distancias topológicas). El grafo interactivo resultante se ha exportado a reeb_graph.html.                            
  4. Homología en Ventanas: Segmentamos cada serie en 5 ventanas de longitud 30 y stride 17, calculando complejos de Vietoris-Rips sobre  
  cada atractor local para generar tensores secuenciales de paisajes de persistencia ($H_0, H_1$) de forma estable.                       
  5. Transformer e Interpretabilidad: Diseñamos una arquitectura en PyTorch con una capa personalizada  AttentionExtractionLayer . El     
  modelo entrenó con éxito obteniendo las siguientes métricas en el conjunto de prueba (validation split):                                
      • ROC-AUC en Test: 0.7721                                                                                                           
      • F1-Score en Test: 0.7593                                                                                                          
      • Forma de la Matriz de Atención:  (91, 5, 5) , permitiendo extraer y visualizar directamente los heatmaps para evaluar el colapso  
      topológico en los checkpoints de estrés.                                                                                            
                                                                                                                                          
                                                                                                                                          
  La libreta sensory_habituation_tdl.ipynb y el grafo interactivo reeb_graph.html están listos para ser abiertos y ejecutados en tu servidor
de Jupyter.  