import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, Conv1D, MaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D, concatenate, Flatten, PReLU, BatchNormalization
from keras.layers import GRU, GlobalAveragePooling1D, GlobalMaxPooling1D
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

embed_size = 50  # how big is each word vector
max_features = 50000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100  # max number of words in a comment to use
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


embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


inp = Input(shape=(maxlen, ))
x_4 = Embedding(max_features,
                embed_size,
                weights=[embedding_matrix],
                input_length=maxlen,
                trainable=False)(inp)
x_3 = SpatialDropout1D(0.2)(x_4)
x_3 = Bidirectional(GRU(196, return_sequences=True, dropout=0.2, kernel_initializer='he_normal'),
                    merge_mode='concat')(x_3)
x_3 = Conv1D(96, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x_3)
avg_pool_3 = GlobalAveragePooling1D()(x_3)
max_pool_3 = GlobalMaxPooling1D()(x_3)
# att_3 = Attention()(x_3)
x = concatenate([avg_pool_3, max_pool_3])
x = Dense(6, activation="sigmoid")(x)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
sgd = SGD(lr=0.001)

model = Model(inputs=inp, outputs=x)
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

