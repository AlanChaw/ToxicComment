import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping
sys.path.append('../')
from NN import Settings
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from keras.utils import plot_model # for visulize model


visual = True              # trigger of printing model figure

EMBEDDING_FILE = Settings.glove_model_path
TRAIN_DATA_FILE = Settings.train_file_path
TEST_DATA_FILE = Settings.test_file_path

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)


# parameters class
class Para(object):
    # basic parameters
    word_vec_size = 200      # word embedding size
    max_features = 50000    # how many unique words to use
    sentence_length = 100   # how many words in a sentence
    # RNN parameters
    lstm_unit_size = 50     # the unit numbers (equal to the output vector length) of RNN
    dropout = 0             # dropout on input linear neuron
    recurrent_dropout = 0   # dropout on recurrent linear neuron
    optimizer = 'adam'      # optimizer function
    merge_mode = 'concat'   # bidirectional lstm merge method

    batch_size = 32         # input batch size
    epchos = 10             # maximum eochos
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=0, verbose=1, mode='auto')]
                            # callback list for model.fit()


# cleaning data
list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

# tokenize sentences, generate word dictionary, and pad sentences to fixed length
tokenizer = Tokenizer(num_words=Para.max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=Para.sentence_length, padding='post', truncating='post')
X_te = pad_sequences(list_tokenized_test, maxlen=Para.sentence_length, padding='post', truncating='post')


# get the embedding (mu, sigma) value, for the guassian initialization
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()


# initial the word embedding matrix (dictionary) with guassian random

word_index = tokenizer.word_index
nb_words = min(Para.max_features, len(word_index))
seed = 7
np.random.seed(seed)  # for reproducbility
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, Para.word_vec_size))

# for the top (max_features) words, find it's embedding vector, add to the embedding_matrix
for word, i in word_index.items():
    if i >= Para.max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# different kinds of LSTM structures, return the final tensor x
inp = Input(shape=(Para.sentence_length,))


def create_model(learn_rate=0.001, dropout=0, recurrent_dropout=0):
    # create model
    model = Sequential()
    model.add(Embedding(Para.max_features, Para.word_vec_size, weights=[embedding_matrix]))
    model.add(Bidirectional(LSTM(Para.lstm_unit_size, return_sequences=True,
                                 dropout=dropout, recurrent_dropout=recurrent_dropout)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(6, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learn_rate), metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, epochs=2, batch_size=32, verbose=1)

# define grid search parameters
dropout = [0, 0.2, 0.4, 0.5]
# recurrent_dropout = [0, 0.25, 0.5]
lr = [0.001, 0.01, 0.1, 1]

param_grid = dict(learn_rate=lr, dropout=dropout)
cv = ShuffleSplit(n_splits=2, test_size=0.1, random_state=1)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)
grid_result = grid.fit(X_t, y)

with open('result.txt', 'w') as result:
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    result.write("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        result.write("%f (%f) with: %r \n" % (mean, stdev, param))
