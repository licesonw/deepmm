# DeepMM
Multimodal deep learning package that uses both categorical and text-based features in a single deep architecture for regression and binary classification use cases.

This model employs the idea of categorical entitiy embeddings (see [https://arxiv.org/abs/1604.06737]) for mapping highly sparse one-hot encoded categorical features into a latent low-dimensional features space, where similaries between features are properly encoded. The bi-interaction pooling operation (see [https://arxiv.org/abs/1708.05027]) is incorporated to account for second-order feature interactions. An LSTM-based sub-network is used to process the sequential text features.

The architecture is oriented on other deep learning approaches for processing sparse features, such as:
* He et al. (2017) *Neural factorization machines for sparse predictive Analysis* [https://arxiv.org/abs/1708.05027]
* Cheng et al. (2016) *Wide and deep learning for recommender systems* [https://arxiv.org/abs/1606.07792]
* Guo et al. (2017) *DeepFM: A Factorization Machine-based neural network for CTR prediction* [https://arxiv.org/abs/1703.04247]
* Wang et al. (2017) *Deep & Cross Network for ad click predictions* [https://arxiv.org/abs/1708.05123]


General outline of the multimodal model architecture (with concatenation of categorical embedding vectors) with four categorical features (C1-C4):
![image](img/multimodal_model.png)

