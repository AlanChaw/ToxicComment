import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
sys.path.append('../')   # for excecutable on command line
from NN import Settings
from keras.utils import plot_model  # for visulize model
import pandas as pd
import numpy as np
from NN.Settings import *
# for reproducbility
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


visual = True
class_name = "toxic"


# roc callback class
class roc_callback(Callback):
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x = x_train
        self.y = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.best_roc_val = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))),
              end=100 * ' ' + '\n')

        if roc_val > self.best_roc_val:
            self.best_roc_val = roc_val
        else:
            self.stopped_epoch = epoch
            self.model.stop_training = True
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# parameters class
class Para(object):
    # basic parameters
    word_vec_size = 100     # word embedding size
    max_features = 20000    # how many unique words to use
    sentence_length = 100   # how many words in a sentence
    # RNN parameters
    lstm_unit_size = 50     # the unit numbers (equal to the output vector length) of RNN
    dropout = 0             # dropout on input linear neuron
    recurrent_dropout = 0   # dropout on recurrent linear neuron
    # optimizer = 'adam'    # optimizer function
    merge_mode = 'concat'   # bidirectional lstm merge method
    learning_rate = 0.01
    batch_size = 128         # input batch size
    epchos = 10             # maximum eochos


# construct train, validation , test set
train_file = pd.read_csv(train_cleaned_file_path)
trains = []
for i in range(0, 6):
    df = pd.concat([train_file.iloc[:, 0], train_file.iloc[:, 1], train_file.iloc[:, i+2]], axis=1)
    trains.append({
        'class': df.columns[2],
        'df': df
    })
train = pd.DataFrame()
for t in trains:
    if t['class'] == class_name:
        train = t['df']
test = pd.read_csv(test_cleaned_file_path)
embedding_file = glove_model_path

list_sentences_train = train["comment_text"].fillna("_na_").values
y = train.iloc[:, 2].values
# list_sentences_train, list_sentences_val, y_train, y_val = train_test_split(list_sentences_train, y,
#                                                                             test_size=0.1, shuffle=False, stratify=None)
list_sentences_test = test["comment_text"].fillna("_na_").values


# tokenize sentences, generate word dictionary, and pad sentences to fixed length
tokenizer = Tokenizer(num_words=Para.max_features)
tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_tr = pad_sequences(list_tokenized_train, maxlen=Para.sentence_length, padding='post', truncating='post')
X_te = pad_sequences(list_tokenized_test, maxlen=Para.sentence_length, padding='post', truncating='post')


# get the embedding (mu, sigma) value, for the guassian initialization
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file))
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()


# initial the word embedding matrix (dictionary) with guassian random
word_index = tokenizer.word_index
nb_words = min(Para.max_features, len(word_index))
# embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, Para.word_vec_size))
embedding_matrix = np.zeros((nb_words, Para.word_vec_size))

# for the top (max_features) words, find it's embedding vector, add to the embedding_matrix
for word, i in word_index.items():
    if i >= Para.max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

# tensor construction
inp = Input(shape=(Para.sentence_length,))
x = Embedding(Para.max_features, Para.word_vec_size, weights=[embedding_matrix], trainable=True)(inp)
x = Bidirectional(LSTM(Para.lstm_unit_size, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)

# construct and fix model
model = Model(inputs=inp, outputs=x)
optimizer = Adam(lr=Para.learning_rate)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
if visual:
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

#call back functions
early_stop = EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=0, verbose=1, mode='auto')
X_train, X_val, y_train, y_val = train_test_split(X_tr, y, test_size=0.1, shuffle=False)
roc = roc_callback(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val)
callbacks = [roc]          # callback list for model.fit()
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=Para.batch_size, epochs=10,
          callbacks=callbacks, verbose=1)

y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(Settings.sample_sub_path)
sample_submission[class_name] = y_test
sample_submission.to_csv(Settings.sub_path, index=False)
