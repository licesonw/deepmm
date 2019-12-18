import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, Input, Embedding, concatenate, Reshape, Concatenate, LSTM
import keras.backend as K

import tensorflow as tf

from .layers import BiInteractionPooling


def CategoricalEmbeddingLayer(cardinality, embedding_size):
    """Creates an embeddings input branch for a single categorical variable.

    Arguments:
        cardinality {int} -- How many cateogories does the variable have.
        embedding_size {int} -- Size of the embedding vector.

    Returns:
        {keras.models.model} -- A functional API keras model containing the Embedding layer together with a flattening/reshape layer.
    """
    inputs = Input(shape=[1])
    layer = Embedding(input_dim=cardinality, output_dim=embedding_size,
                      input_length=1)(inputs)
    layer = Reshape((embedding_size,))(layer)
    model = Model(inputs, layer)
    return model


def CategoricalSubnetwork(category_cardinality, embedding_size, hidden_layers, dropout_layers=[], use_bi_interaction=False):
    """Creates an keras functional API model with several different input branches for the categorical variables. If desired,
    uses Bi-Interaction Pooling layer to capture seconds order feature interactions between sparse features better.
    
    Paper: [ [CoRR 2016] Entity Embeddings of Categorical Variables, Guo and Berkhahn (2016), https://arxiv.org/abs/1604.06737 ]
    Paper: [ [SIGIR 2017] Neural Factorization Machines for Sparse Predictive Analytics, He et al. (2017), https://arxiv.org/abs/1708.05027 ]

    Arguments:
        category_cardinality {list of int} -- The cardinality of each category to be embedded.
        embedding_size {int} -- The desired dimension of the embedding vectors.

    Keyword Arguments:
        hidden_layers {list} -- A list of the neurons in the hidden layers. Length of the list determines the amount of hidden layers. (default: {[128]})
        dropout_layers {list} -- A list of the dropout rates in the hidden layers. Leave empty for no dropout. If used, dimension has to agree with hidden_layers arguement. (default: {[]})
        use_bi_interaction {bool} -- Wether to use Bi-Interaction pooling for modeling interactions between sparse features. [[SIGIR 2017]Neural Factorization Machines for Sparse Predictive Analytics, He et al. (2017)] (default: {False})

    Returns:
        keras.models.model -- Functional API keras model.
    """
    # Validate input
    if len(hidden_layers) != len(dropout_layers):
        if len(dropout_layers) == 0:
            print('Proceedig without dropout.')
            pass
        else:
            raise ValueError('Categorical Sub-Network: Dimensions of \'hidden_layers\' and \'dropout_layers\' do not agree. \
                Either leave \'dropout_layers\' list empty ([]) or use the same dimension as argument \'hidden_layers\'.')

    # Create the embeddings sub branches
    embeddings_branches = []
    for indx, cardinality in enumerate(category_cardinality):
        branch = CategoricalEmbeddingLayer(cardinality, embedding_size)
        embeddings_branches.append(branch)

    # Employ bi-interaction pooling if desired by user
    if use_bi_interaction:
        layer = BiInteractionPooling(embedding_size)([branch.output for branch in embeddings_branches])
        layer = Dropout(0.3)(layer)
    else:
        layer = concatenate([branch.output for branch in embeddings_branches])

    # Add dense layers and dropout layers
    for indx, hidden_layer_size in enumerate(hidden_layers):
        layer = Dense(hidden_layer_size, activation='relu', name='cat_dense_layer_'+str(indx))(layer)
        if len(dropout_layers) != 0:
            layer = Dropout(dropout_layers[indx], name='cat_dropout_layer_'+str(indx))(layer)

    model = Model(inputs=[branch.input for branch in embeddings_branches], outputs=layer)
    return model


def TextualSubnetwork(vocab_size, embedding_dim, max_len, weights, lstm_neurons=128, dropout_rate=0.5):
    """Creates a keras functional API model for the text-based subordinate network.

    Arguments:
        vocab_size {int} -- Size of the vocabulary based on tokenization.
        embedding_dim {int} -- Embedding dimension of the word vectors.
        max_len {int} -- Maximum length of the sequences.
        weights {numpy array} -- The embedding matrix, i.e. the weights of the pretrained embedding layer.

    Keyword Arguments:
        lstm_neurons {int} -- Number of neurons in the LSTM cell (default: {128})
        dropout_rate {float} -- The dropout rate after the LSTM cell (default: {0.5})

    Returns:
        keras.models.model -- The resulting model object as a keras functional API model
    """
    inputs = Input(name='txt_input', shape=[max_len])
    layer = Embedding(vocab_size, embedding_dim, input_length=max_len, trainable=False, weights=[weights], name='txt_embedding_layer')(inputs)
    layer = LSTM(lstm_neurons, name='txt_lstm_layer')(layer)
    layer = Dropout(dropout_rate, name='txt_dropout_layer')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


def DeepMultimodalModel(task, num_unique_categories, cat_embedding_dim, txt_vocab_size, txt_embedding_dim, txt_max_len, txt_weights, cat_hidden_neurons=[512, 512, 256],
                        cat_dropout=[0.5, 0.5, 0.5], cat_bi_interaction=False, txt_lstm_neurons=128, txt_dropout=0.5, final_hidden_neurons=[256, 128],
                        final_dropout=[0.3, 0.3]):
    """The multimodal neural network for processing a mixture of categorical and text-based features. Each modality has a
    subordinate network with its respective input. The categorical subnet employs the concept of categorical entity embeddings
    to map sparse categorical features into a latent low-dimensional feature space followed by a block of fully conncted dense
    layers to model feature interactions. The text-based subnet uses LSTM cells for processing the text sequences. The tensors
    of each modality are concatenated and fed through another block of fully connected dense layers.

    Arguments:
        task {string} -- Either 'regression' or 'binary' prediction task.
        num_unique_categories {list of ints} -- The cardinaliy of each category in a list.
        cat_embedding_dim {int} -- Dimension of embedding vectors for the categorical features.
        txt_vocab_size {int} -- The vocabulary size of the text features.
        txt_embedding_dim {int} -- The dimension of word embedding vectors.
        txt_max_len {int} -- The maximum length of the sequences.
        txt_weights {numpy array} -- The 2D matrix of pretrained word embedding vectors.

    Keyword Arguments:
        cat_hidden_neurons {list} -- Number of neurons in each hidden layer of the categorical subnet. (default: {[512, 512, 256]})
        cat_dropout {list} -- The dropout rate of the block of dense layers in the categorical subnet. (default: {[0.5, 0.5, 0.5]})
        cat_bi_interaction {bool} -- Whether to use bi-interaction pooling layer. (default: {False})
        txt_lstm_neurons {int} -- Number of neurons in the LSTM cell. (default: {128})
        txt_dropout {float} -- Dropout rate after the LSTM cell in the textual subnet. (default: {0.5})
        final_hidden_neurons {list} -- Number of neurons in each hidden layer after modality concatenation. (default: {[256, 128]})
        final_dropout {list} -- The dropout rate in the final dense layer block. (default: {[0.3, 0.3]})

    Returns:
        keras.models.model -- The keras model object.
    """

    # Validate input
    if len(final_hidden_neurons) != len(final_dropout):
        if len(final_dropout) == 0:
            print('Proceedig without dropout in final dense layer block.')
            pass
        else:
            raise ValueError('Dimensions of \'final_hidden_neurons\' and \'final_dropout\' do not agree. \
                Either leave \'final_dropout\' list empty ([]) or use the same dimension as argument \'final_hidden_neurons\'.')

    # Check task parameter
    if task == 'regression':
        final_activation = 'linear'
    elif task == 'binary':
        final_activation = 'sigmoid'
    else:
        raise ValueError('Parameter \'task\' has to be either \'regression\' of \'binary\'.')

    # Set up subordinate networks for each modality
    cat_subnet = CategoricalSubnetwork(num_unique_categories, cat_embedding_dim, cat_hidden_neurons, cat_dropout, cat_bi_interaction)
    txt_subnet = TextualSubnetwork(txt_vocab_size, txt_embedding_dim, txt_max_len, txt_weights, txt_lstm_neurons, txt_dropout)

    # Concatenate the modalities (i.e. the output tensors of each model)
    x = concatenate([cat_subnet.output, txt_subnet.output])

    # Add dense layers and dropout layers before the final prediction layer
    for indx, hidden_layer_size in enumerate(final_hidden_neurons):
        x = Dense(hidden_layer_size, activation='relu', name='final_dense_layer_'+str(indx))(x)
        if len(final_dropout) != 0:
            x = Dropout(final_dropout[indx], name='final_dropout_layer_'+str(indx))(x)

    x = Dense(1, activation=final_activation, name='output_layer')(x)

    model = Model(inputs=cat_subnet.inputs + txt_subnet.inputs, outputs=x)
    return model
