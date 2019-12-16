import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import (Zeros, glorot_normal,
                                                  glorot_uniform)
from keras.layers import Layer


class BiInteractionPooling(Layer):
    """
    Implementation of the bi-interaction pooling layer to capture second-order interactions between sparse features.

    Paper: [ [SIGIR 2017] Neural Factorization Machines for Sparse Predictive Analytics, He et al. (2017) ]

    Link: https://arxiv.org/abs/1708.05027
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(BiInteractionPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # if len(input_shape) != 2:
        #     raise ValueError('Unexpected input dimensions %d, expected to be 2 dimensions' % (len(input_shape)))

        super(BiInteractionPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):

        assert isinstance(inputs, list)

        # Stack embedding vectors together
        stacked_embeddings = tf.stack(inputs, axis=1)

        summed_features_emb = tf.reduce_sum(stacked_embeddings, 1)  # None * K
        # get the element-multiplication
        summed_features_emb_square = tf.square(summed_features_emb)  # None * K

        # _________ square_sum part _____________
        squared_features_emb = tf.square(stacked_embeddings)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K

        # ________ FM __________
        output = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K

        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (None, self.output_dim)