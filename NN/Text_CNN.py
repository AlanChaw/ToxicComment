import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, Conv1D, MaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D, concatenate, Flatten, PReLU, BatchNormalization
from keras.models import Model
from keras.layers import TimeDistributed
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
sys.path.append('../')
from NN import Settings

from keras.utils import plot_model # for visulize model

#
# path = '../input/'
# comp = 'jigsaw-toxic-comment-classification-challenge/'


EMBEDDING_FILE = Settings.glove_model_path
TRAIN_DATA_FILE = Settings.train_file_path
TEST_DATA_FILE = Settings.test_file_path

embed_size = 300  # how big is each word vector
max_features = 50000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 300  # max number of words in a comment to use
lstm_unit_size = 50
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

sequence_input = Input(shape=(maxlen, ))
x_3 = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(sequence_input)
# inp = Input(shape=(maxlen, ))
# x_3 = Embedding(max_features,
#                 embed_size,
#                 weights=[embedding_matrix],
#                 input_length=maxlen,
#                 trainable=False)(inp)
x_3 = SpatialDropout1D(0.2)(x_3)

cnn1 = Conv1D(256, 2, padding='same', strides=1, activation='relu')(x_3)
cnn2 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(x_3)
cnn3 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(x_3)
cnn4 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(x_3)
cnn5 = Conv1D(256, 6, padding='same', strides=1, activation='relu')(x_3)
cnn = concatenate([cnn1, cnn2, cnn3, cnn4, cnn5], axis=-1)

cnn1 = Conv1D(128, 3, padding='same', strides=1, activation='relu')(cnn)
cnn1 = MaxPooling1D(pool_size=200)(cnn1)
cnn2 = Conv1D(128, 4, padding='same', strides=1, activation='relu')(cnn)
cnn2 = MaxPooling1D(pool_size=200)(cnn2)
cnn3 = Conv1D(128, 5, padding='same', strides=1, activation='relu')(cnn)
cnn3 = MaxPooling1D(pool_size=200)(cnn3)
cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)

x = Flatten()(cnn)
x = Dropout(0.2)(x)

x = Dense(128, kernel_initializer='he_normal')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(6, activation="sigmoid")(x)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
sgd = SGD(lr=0.001)

model = Model(inputs=sequence_input, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy', 'binary_crossentropy'])

if visual:
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0,
                                   patience=0, verbose=1, mode='auto')]
model.fit(X_t, y, batch_size=128, epochs=10, validation_split=0.1, callbacks=callbacks)

y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(Settings.sample_sub_path)
sample_submission[list_classes] = y_test
sample_submission.to_csv(Settings.sub_path, index=False)

