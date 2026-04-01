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

