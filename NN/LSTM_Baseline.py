import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, SimpleRNN, concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.layers import TimeDistributed
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
sys.path.append('../')
from NN import Settings

from keras.utils import plot_model # for visulize model
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

#
# path = '../input/'
# comp = 'jigsaw-toxic-comment-classification-challenge/'


EMBEDDING_FILE = Settings.glove_model_path
TRAIN_DATA_FILE = Settings.train_file_path
TEST_DATA_FILE = Settings.test_file_path

embed_size = 300  # how big is each word vector
max_features = 100000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 150  # max number of words in a comment to use
lstm_unit_size = 128
visual = True  # trigger of plot

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post', truncating='post')
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post', truncating='post')


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
word_index = tokenizer.word_index
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# simple RNN
def simpleRNN():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = SimpleRNN(lstm_unit_size, return_sequences=False)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    return inp, x

# LSTM
def lstm():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = LSTM(lstm_unit_size, return_sequences=False)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    return inp, x

# bi lstm
def bi_lstm():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(lstm_unit_size, return_sequences=False))(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    return inp, x

# bi lstm pooling
def bi_lstm_pool():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(lstm_unit_size, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    # x = Dense(50, activation="relu")(x)
    # x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    return inp, x

# bi 2*lstm pooling
def bi_2lstm_pool():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(lstm_unit_size, return_sequences=True))(x)
    x = Bidirectional(LSTM(lstm_unit_size, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    return inp, x

# bi GRU pooling
def bi_gru_pool():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(lstm_unit_size, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    return inp, x

# bi 2*GRU pooling
def bi_2lstm_pool():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(lstm_unit_size, return_sequences=True))(x)
    x = Bidirectional(GRU(lstm_unit_size, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    return inp, x



# original method
def bi_LSTM_GMP_Dense():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(lstm_unit_size, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    return inp, x


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


inp, x = bi_lstm_pool()
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


if visual:
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


start = time.time()

batch_size = 128
epochs = 5
X_tra, X_val, y_tra, y_val = train_test_split(X_t, y, train_size=0.9)

filepath = "weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)
callbacks_list = [ra_val, checkpoint, early]
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = callbacks_list, verbose=1)
# # Loading model weights
model.load_weights(filepath)
print('Predicting....')
y_pred = model.predict(X_te, batch_size=1024, verbose=1)
submission = pd.read_csv(Settings.sample_sub_path)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv(Settings.sub_path, index=False)



# ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
# callbacks_list = [ra_val]
# model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks_list, verbose=1)
#
# interval = time.time() - start
# print("total time: " + str(interval))
#
# y_test = model.predict([X_te], batch_size=1024, verbose=1)
# sample_submission = pd.read_csv(Settings.sample_sub_path)
# sample_submission[list_classes] = y_test
# sample_submission.to_csv(Settings.sub_path, index=False)
#


# inp = Input(shape=(maxlen, ))
# x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
# # x = Bidirectional(LSTM(50, return_sequences=False, dropout=0, recurrent_dropout=0))(x)
# # x = GlobalMaxPool1D()(x)
# # x = Dense(50, activation="relu")(x)
# # x = Dropout(0)(x)
# x = Dense(6, activation="sigmoid")(x)
# model = Model(inputs=inp, outputs=x)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)




# def pure_LSTM():
#     inp = Input(shape=(maxlen,))
#     x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#     x = LSTM(lstm_unit_size, return_sequences=False)(x)
#     x = Dense(6, activation="sigmoid")(x)
#     return inp, x
#
#
# def double_LSTM():
#     inp = Input(shape=(maxlen,))
#     x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#     x = LSTM(lstm_unit_size, return_sequences=True, dropout=0, recurrent_dropout=0)(x)
#     x = LSTM(lstm_unit_size, return_sequences=False, dropout=0, recurrent_dropout=0)(x)
#     x = Dense(6, activation="sigmoid")(x)
#     return inp, x
#
#
# def pure_bi_LSTM():
#     inp = Input(shape=(maxlen,))
#     x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#     x = Bidirectional(LSTM(lstm_unit_size, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))(x)
#     x = Dense(6, activation="sigmoid")(x)
#     return inp, x
#
#
# def double_bi_LSTM():
#     inp = Input(shape=(maxlen,))
#     x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#     x = Bidirectional(LSTM(lstm_unit_size, return_sequences=True), merge_mode='concat')(x)
#     # x = Bidirectional(LSTM(lstm_unit_size, return_sequences=True, dropout=0, recurrent_dropout=0), merge_mode='concat')(x)
#     x = Bidirectional(LSTM(lstm_unit_size, return_sequences=True,dropout=0, recurrent_dropout=0), merge_mode='concat')(x)
#     x = GlobalMaxPool1D()(x)
#     x = Dense(50, activation="relu")(x)
#     x = Dense(6, activation="sigmoid")(x)
#     return inp, x
#
#
# # set return_sequences=True, and add a GlobalMaxPool1D()(x) layer after bi_LSTM layer
# def bi_LSTM_GMP():
#     inp = Input(shape=(maxlen, ))
#     x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#     x = Bidirectional(LSTM(lstm_unit_size, return_sequences=True, dropout=0, recurrent_dropout=0))(x)
#     x = GlobalMaxPool1D()(x)
#     x = Dense(6, activation="sigmoid")(x)
#     return inp, x
