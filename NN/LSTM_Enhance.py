import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import TimeDistributed
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping
sys.path.append('../')
from NN import Settings

from keras.utils import plot_model  # for visulize model


visual = True              # trigger of printing model figure

EMBEDDING_FILE = Settings.glove_model_path
TRAIN_DATA_FILE = Settings.train_file_path
TEST_DATA_FILE = Settings.test_file_path

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)


# parameters class
class Para(object):
    # basic parameters
    word_vec_size = 300     # word embedding size
    max_features = 20000    # how many unique words to use
    sentence_length = 100   # how many words in a sentence
    # RNN parameters
    lstm_unit_size = 50     # the unit numbers (equal to the output vector length) of RNN
    dropout = 0             # dropout on input linear neuron
    recurrent_dropout = 0   # dropout on recurrent linear neuron
    # optimizer = 'adam'      # optimizer function
    merge_mode = 'concat'   # bidirectional lstm merge method
    learning_rate = 0.001

    batch_size = 128         # input batch size
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


embeddings_index = {}
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
word_index = tokenizer.word_index
num_words = min(Para.max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, Para.word_vec_size))

# for the top (max_features) words, find it's embedding vector, add to the embedding_matrix
for word, i in word_index.items():
    if i >= Para.max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# different kinds of LSTM structures, return the final tensor x
inp = Input(shape=(Para.sentence_length,))

# 1 hidden layer LSTM
def pure_LSTM():
    x = Embedding(Para.max_features, Para.word_vec_size, weights=[embedding_matrix])(inp)
    x = LSTM(Para.lstm_unit_size, return_sequences=False)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x


# 2 hidden layer LSTM
def double_LSTM():
    x = Embedding(Para.max_features, Para.word_vec_size, weights=[embedding_matrix])(inp)
    x = LSTM(Para.lstm_unit_size, return_sequences=True,
             dropout=Para.dropout, recurrent_dropout=Para.recurrent_dropout)(x)
    x = LSTM(Para.lstm_unit_size, return_sequences=False,
             dropout=Para.dropout, recurrent_dropout=Para.recurrent_dropout)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x


# 1 hidden layer bidirectional LSTM
def pure_bi_LSTM():
    x = Embedding(Para.max_features, Para.word_vec_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(Para.lstm_unit_size, return_sequences=False,
                           dropout=Para.dropout, recurrent_dropout=Para.recurrent_dropout))(x)
    x = Dense(6, activation="sigmoid")(x)
    return x


# 1 hidden layer bi LSTM, global maxpool
def bi_LSTM_GMP():
    x = Embedding(Para.max_features, Para.word_vec_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(Para.lstm_unit_size, return_sequences=True,
                           dropout=Para.dropout, recurrent_dropout=Para.recurrent_dropout))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(6, activation="sigmoid")(x)
    return x


# 2 hidden layer bi LSTM, global maxpool
def double_bi_LSTM():
    x = Embedding(Para.max_features, Para.word_vec_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(Para.lstm_unit_size, return_sequences=True,
                           dropout=Para.dropout, recurrent_dropout=Para.recurrent_dropout), merge_mode=Para.merge_mode)(x)
    x = Bidirectional(LSTM(Para.lstm_unit_size, return_sequences=True,
                           dropout=Para.dropout, recurrent_dropout=Para.recurrent_dropout), merge_mode=Para.merge_mode)(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dense(6, activation="sigmoid")(x)
    return x


# original method - 1 hidden layer bi LSTM, maxpool, 1 dense layer
def bi_LSTM_GMP_Dense():
    x = Embedding(Para.max_features, Para.word_vec_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(Para.lstm_unit_size, return_sequences=True,
                           dropout=Para.dropout, recurrent_dropout=Para.recurrent_dropout))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation='relu')(x)
    # x = Dropout(Para.dropout)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x


# original method - sequential version
def sequentila_original():
     model = Sequential()


# construct, compile and fit model
x = bi_LSTM_GMP_Dense()
model = Model(inputs=inp, outputs=x)
optimizer = Adam(lr=Para.learning_rate)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
if visual:
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.fit(X_t, y, batch_size=Para.batch_size, epochs=Para.epchos, validation_split=0.1, callbacks=[], verbose=1)

y_test = model.predict([X_te], batch_size=1024, verbose=2)
sample_submission = pd.read_csv(Settings.sample_sub_path)
sample_submission[list_classes] = y_test
sample_submission.to_csv(Settings.sub_path, index=False)

