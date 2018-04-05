import pandas as pd
from SettingsS import *
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import re
from nltk.tokenize import TweetTokenizer
import numpy as np
import gensim
import time
from scipy import sparse

model_path = Settings.word2vec_model_path
model = gensim.models.KeyedVectors.load(model_path)

APPO = Settings.APPO
train_file = Settings.train_file_path
test_file = Settings.test_file_path
train_cleaned_file = Settings.train_cleaned_file_path
test_cleaned_file = Settings.test_cleaned_file_path
features_file = Settings.features_file_path
tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
normalization = Settings.doc_vec_normalization


def clean(comment):
    # to lower case
    comment = comment.lower()
    # delete "/n"
    comment = re.sub("\\n", " ", comment)
    # split into words
    words = tokenizer.tokenize(comment)
    # cleared_comment = " ".join(words)
    return words


def construct_doc_vec(sentence):
    global model
    doc_vec = np.zeros(model.get_vector('a').shape)
    for word in sentence:
        doc_vec += model.get_vector(word)
    return doc_vec


def generate_doc_vec():
    start = time.time()

    # pd.set_option('display.max_colwidth', 20000)
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # generate word2vec features
    merged = pd.concat([train.iloc[:, 0:2], test.iloc[:, 0:2]])
    merged = merged.reset_index(drop=True)
    all_comments = merged.comment_text
    sentences = all_comments.apply(lambda x: clean(x))
    doc_vec = [construct_doc_vec(sen) for sen in sentences]
    if normalization:
        matrix = np.matrix(doc_vec)
        normal_vec = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0))
        all_features = sparse.csc_matrix(normal_vec)
    else:
        all_features = sparse.csc_matrix(doc_vec)
    # sparse.save_npz(features_file, all_features)
    elapsed = time.time() - start
    print("time for generate features: ", elapsed, "\n")

    return all_features
