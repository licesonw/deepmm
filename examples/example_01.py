import pandas as pd
import numpy as np
import pickle
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from gensim.models import Word2Vec, KeyedVectors

from deepmm.models import DeepMultimodalModel

# Read some data that contains a mix of categorical and text-based features
#   - e.g. Mercari Price Suggestion Challenge https://www.kaggle.com/c/mercari-price-suggestion-challenge/data
df = pd.read_csv('data.csv')


# Load pretrained embeddings
w2v = KeyedVectors.load_word2vec_format('embeddings_w2v.txt')

# Hyperparameters for text tokenization
EMBEDDING_DIM = 100
NUM_MAX_WORDS = 500
MAX_LEN = 150
X_nlp = df['TEXT']


# Tokenize text documents via keras tokenizer
tok = Tokenizer(num_words=NUM_MAX_WORDS)
tok.fit_on_texts(X_nlp)
sequences = tok.texts_to_sequences(X_nlp)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=MAX_LEN)

word_index = tok.word_index
print('Found %s unique tokens.' % len(word_index))
vocabulary_size = min(len(word_index)+1, NUM_MAX_WORDS)
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

# Preparing the embedding matrix:
#   We only take the embeddings that are neccessarry for the given vocabulary
num_none = 0
for word, i in word_index.items():
    if i>=NUM_MAX_WORDS:
        continue
    try:
        embedding_vector = w2v[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)
        num_none = num_none+1
        

# Define categorical features and target
cat_features = ['C1', 'C2', 'C3', 'C4']
target = ['TARGET']

# Label encode the categories for each categorical feature (numeric value is needed for feeding into keras model)
X_categorical = []
label_encoder = []

for feature in cat_features:
    le = LabelEncoder()
    X_categorical.append(pd.DataFrame(le.fit_transform(df[feature]), columns=[feature]))
    label_encoder.append(le)

Y = df[target]


# Split all data into training and test chunks
# IMPORTANT: Make sure that textual and categorical data is properly aligned (e.g. here choose same random_state)!

X_nlp_train_all, X_nlp_test_all, y_train_all, y_test_all = train_test_split(sequences_matrix, Y, random_state=42)

# Split sparse part into train and test
X_categorical_train = []
X_categorical_test = []
for X_category in X_categorical:
    tr, te, y_train_catembeddings, y_test_catembeddings =  train_test_split(X_category, Y, random_state=42)
    X_categorical_train.append(tr)
    X_categorical_test.append(te)

X_train_catembeddings = X_categorical_train
X_train_all = X_categorical_train
X_train_all.append(X_nlp_train_all)

X_test_all = X_categorical_test
X_test_catembeddings = X_categorical_test
X_test_all.append(X_nlp_test_all)

# Get cardinality of each categorical variable
num_unique_categories = [df[cat].nunique() for cat in cat_features]

# Setup model object
model = DeepMultimodalModel(task='regression', num_unique_categories=num_unique_categories, cat_embedding_dim=16,
                            txt_vocab_size=vocabulary_size, txt_embedding_dim=EMBEDDING_DIM, txt_max_len=MAX_LEN,
                           txt_weights=embedding_matrix,
                           cat_hidden_neurons=[100,50,10], cat_dropout=[0.1, 0.2, 0.2], cat_bi_interaction=True,
                           txt_lstm_neurons=32, txt_dropout=0.2, final_hidden_neurons=[64, 32], final_dropout=[0.3, 0.3])

model.compile("adam", "mse", metrics=['mse', 'mae'], )

# Fit model
hist = model.fit(X_train_all, y_train_all, epochs=100, batch_size=256, validation_split=0.2)