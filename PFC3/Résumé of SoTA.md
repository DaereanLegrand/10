Tabla de extractores de características según paper: 
	*Shou, Y., Meng, T., Ai, W., Fu, F., Yin, N., & Li, K. (2026). A Comprehensive Survey on Multi-modal Conversational Emotion Recognition with Deep Learning. In ACM Transactions on Information Systems (Vol. 44, Issue 2, pp. 1–48). Association for Computing Machinery (ACM).*

![[Pasted image 20260331154718.png]]

También es recomendado el uso de OpenFace debido a que este puede obtener características faciales, hasta 68 de ellas. Se hallan frame por frame, luego se combinan a través de un LSTM o similar, a mon avis se puede usar con Mamba directamente.

Según el survey, seguimos con fusión temprana. $$h_{e}=x^{t}+x^{a}+x^{v}$$
Concatenación $$h_{e}=Concat([x^{t},x^{a},x^{v}])$$, concatenación y fusión tardía fueron usados en mi implementación previa. 

SVM también es una opción pero que no escala con el dataset. $$f(x) = \text{sign} \left( \sum_{i=1}^{N} \alpha_i^* y_i \exp \left( -\frac{\|x - z\|^2}{2\sigma^2} \right) + b^* \right)$$
	*Pérez-Rosas, V., Mihalcea, R., & Morency, L. P. (2013, August). Utterance-level multimodal sentiment analysis. In _Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ (pp. 973-982).*
$$\max_{\alpha, \beta} \left[ \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{N} \alpha_i \alpha_j y_i y_j \text{K}_{mkl}(x_i, x_j) \right] $$
$$\text{subject to:}$$
$$\sum_{i} \alpha_i y_i = 0$$
$$0 \leq \alpha_i \leq C$$
$$\text{K}_{mkl} = \sum_{k}^{M} \beta_k K_k > 0,$$
.
	*Poria, S., Cambria, E., & Gelbukh, A. (2015, September). Deep convolutional neural network textual features and multiple kernel learning for utterance-level multimodal sentiment analysis. In _Proceedings of the 2015 conference on empirical methods in natural language processing_ (pp. 2539-2544).*

Select Additive Learning with CNN's. Usando las CNN y una técnica llamada SAL que permite Seleccionar y Agregar. 
![[Pasted image 20260331160904.png]]
	*Wang, H., Meghawat, A., Morency, L. P., & Xing, E. P. (2017, July). Select-additive learning: Improving generalization in multimodal sentiment analysis. In _2017 IEEE International Conference on Multimedia and Expo (ICME)_ (pp. 949-954). IEEE.*

Tensor based networks, tanto los TFN normales como los Low-rank TFN. Las desventajas del primero es que no escala con respecto al contexto, pero logra fusionar las características extraídas. Usa el producto externo en un espacio tridimensional, audio, texto y video.
![[Pasted image 20260331161624.png]]
Crea un espacio tridimensional basado en características extraídas.
![[Pasted image 20260331161715.png]]
En cambio el LFM o Low-ranked TFN descompone las características en factores. Utiliza el producto de Hadamard, multiplicación elemento a elemento. 
	*Zadeh, A., Chen, M., Poria, S., Cambria, E., & Morency, L. P. (2017, September). Tensor fusion network for multimodal sentiment analysis. In _Proceedings of the 2017 conference on empirical methods in natural language processing_ (pp. 1103-1114).*
	*Liu, Z., Shen, Y., Lakshminarasimhan, V. B., Liang, P. P., Zadeh, A. B., & Morency, L. P. (2018, July). Efficient low-rank multimodal fusion with modality-specific factors. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ (pp. 2247-2256).*

Modelos GAN que generen datos para el entrenamiento fueron propuestos, no hubo mejoras significativas relevantes, puesto que las muestras generadas pueden llegar a ser de baja calidad.

Modelos de contextualización secuenciales. Modelos propuestos usaron HMM, Hidden Markov Models.
	*Morency, L. P., Mihalcea, R., & Doshi, P. (2011, November). Towards multimodal sentiment analysis: Harvesting opinions from the web. In _Proceedings of the 13th international conference on multimodal interfaces_ (pp. 169-176).*

![[Pasted image 20260331162557.png]]
Modelos más recientes usan LSTM, en combinación con otros modelos basados en atención como Transformers, en el caso de la implementación realizada en 2024 se llegó a una implementación pura con fusión final en Mamba.
![[Pasted image 20260331162756.png]]
Una idea para mejorar el sistema podría ser el uso de Mamba para el aprovechamiento de contexto para entrenar un modelo final de HMMs. O crear un MoE basado por características.

En el caso de LSTM, en el estado del arte se uso el siguiente esquema para la detección de características faciales en videos. No mostrando un resultado mayor a 50% de acc. 
![[Pasted image 20260331163048.png]]
	*Singh, R., Saurav, S., Kumar, T., Saini, R., Vohra, A., & Singh, S. (2023). Facial expression recognition in videos using hybrid CNN & ConvLSTM. In International Journal of Information Technology (Vol. 15, Issue 4, pp. 1819–1830). Springer Science and Business Media LLC.*

También se uso Transformers puro y el método con mayor resultados obtenidos. Siendo uno de los primero capaces de obtener representaciones usando información contextual.
	*Yang, D., Huang, S., Liu, Y., & Zhang, L. (2022). Contextual and cross-modal interaction for multi-modal speech emotion recognition. _IEEE Signal Processing Letters_, _29_, 2093-2097.*

![[Pasted image 20260401012229.png]]
Mamba 3 fue lanzado hace 2 semanas, reemplazando la red convolucional por un RoPE que permitiría cierta no linearidad, en capas de activación, normalización y multiplicación. Demostrando una mayor potencia en el apartado de recurrencia, también una mayor eficiencia en inferencia lo que permitiría su aplicación en sistemas embebidos.
	*Lahoti, Aakash, et al. "Mamba-3: Improved sequence modeling using state space principles." _arXiv preprint arXiv:2603.15569_ (2026).*

Para la extracción de características de texto se están usando en el estado del arte, nuevas ideas tales como el uso modelos generales o los de tipo Omni. Como gemini 3 y GPT omini. Emotion-Llamav2 llegó hasta un 55%, demostrando la poca eficacia de dichos modelos en su estado puro, sin embargo como se verá al final en el estado del arte, sirven de mucho en la extracción. Esta basado en MiniGPT-v2. 
	*Peng, X., Chen, J., Cheng, Z., Peng, B., Wu, F., Dong, Y., ... & Cheng, Z. Q. (2026). Emotion-LLaMAv2 and MMEVerse: A New Framework and Benchmark for Multimodal Emotion Understanding. _arXiv preprint arXiv:2601.16449_.*

Con respecto a Audio se paso de WavLM a modelos fundacionales, y también el uso de modelos omni como los mencionados en la extracción de características del texto. También se usó SALMONN siendo una LLM basada en audio, obteniendo un resultado de hasta 45.62% de acc, teniendo en cuenta de que solo procesa audio, es para tener en cuenta.
	*Yang, Chih-Kai, Neo S. Ho, and Hung-yi Lee. "Towards holistic evaluation of large audio-language models: A comprehensive survey." _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_. 2025.*
	*Shou, Yuntao, et al. "Multimodal large language models meet multimodal emotion recognition and reasoning: A survey." _arXiv preprint arXiv:2509.24322_ (2025).*

Y para la extracción de características en el video Divided Space-Time Mamba, pero no tuve acceso para ver las 2 investigaciones al respecto. Hay una tercera, que usa DST para un modelo con self-supervision, no es tanto de detección de emociones en conversaciones, pero se debe de tener en cuenta. Por lo demás, ViT es una opción viable pero no existen records de su uso en el tema, mucho más que eso TinyViM parece ser una buena opción por el contexto. ViViT también.
![[Pasted image 20260401021405.png]]
	*Ben Salah, M. K., Jouvet, P., & Noumeir, R. (2025). PICU Face and Thoracoabdominal Detection Using Self-Supervised Divided Space–Time Mamba. _Life_, _15_(11), 1706.*
	*Ma, Xiaowen, Zhenliang Ni, and Xinghao Chen. "Tinyvim: Frequency decoupling for tiny hybrid vision mamba." _Proceedings of the IEEE/CVF International Conference on Computer Vision_. 2025.*
	*Arnab, Anurag, et al. "Vivit: A video vision transformer." _Proceedings of the IEEE/CVF international conference on computer vision_. 2021.*

El estado del arte actual usa la siguiente arquitectura:
![[Pasted image 20260401022240.png]]
En el apartado de la fusión es donde se diferencia de los demás por ser completamente Transformers, es decir que usa una implementación única por cada fuente. Usa también un model gráfico y un Relational Graph Convolutional Network para finalmente pasar por un modelo de Transformers para grafos. Sería interesante combinar el mecanismo de atención puro y los grafos con con el contexto largo de Mamba, el problema sería la reimplementación de cada una de las extracciones de cada fuente. En el caso de las características faciales y corpóreas la situación es peor.

Este modelo alcanza la increíble acc de 65.32%
	*Jin, H., Yang, T., Yan, L., Wang, C., & Song, X. (2025). Multimodal Emotion Recognition in Conversations Using Transformer and Graph Neural Networks. _Applied Sciences_, _15_(22)*

El otro approach es el uso de MoE, Mixture of Experts. Mixture of Speech-Text Experts for Recognition of Emotions (MiSTER-E), especializandose en audio y texto. Llegando a un 69.5%, usando SALMONN y LLaMa. Una idea sería hacer MoE con Visión y características faciales, pero pierde sentido, siento la complejidad computational se hace más grande.
	*Dutta, Soumya, Smruthi Balaji, and Sriram Ganapathy. "A Mixture-of-Experts model for multimodal emotion recognition in conversations." _Computer Speech & Language_ 100 (2026): 101965.*

SALM llega a una acc de 67.13%. Usando Global Liquid Neural Network en combinación con Mamba. El procedimiento es mucho más complejo. Analizar en más detalle.
![[Pasted image 20260401023927.png]]
	*Chen, G.; Liao, Y.; Zhang, D.; Yang, W.; Mai, Z.; Xu, C. Multimodal Emotion Recognition via the Fusion of Mamba and Liquid Neural Networks with Cross-Modal Alignment. _Electronics_ **2025**, _14_, 3638.*

Con respecto al dataset, se puede incursionar el task más complejo que hace diferencia por cada uno de los personajes.
No me parece atractiva la idea, pues desde el baseline propuesto en el paper la acc parece inflada en comparación a MELD. Además no se evalúa la conversación en si, sino a el diálogo de cada personaje.
	*Li, Deng, et al. "Deemo: De-identity multimodal emotion recognition and reasoning." _Proceedings of the 33rd ACM International Conference on Multimedia_. 2025.*
