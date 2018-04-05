from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from scipy import sparse
from Settings import *
import numpy as np
import scipy as sc
from Traditional.Preprocess.Features import *
import time


start = time.time()

train = pd.read_csv(Settings.train_cleaned_file_path)
test = pd.read_csv(Settings.test_cleaned_file_path)
features = generate_doc_vec()

pd.set_option('display.max_rows', 20000)
train['comment_text'].fillna('null', inplace=True)
test['comment_text'].fillna('null', inplace=True)
merge = pd.concat([train.iloc[:, 0:2], test.iloc[:, 0:2]])
corpus = merge.comment_text

tfidf_word = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", min_df=3, max_df=0.95, use_idf=True,
                             smooth_idf=True, sublinear_tf=True, analyzer='word')
tfidf_word.fit(corpus)
word_tfidf_vec = tfidf_word.transform(corpus)

tfidf_char = TfidfVectorizer(ngram_range=(4, 6), strip_accents="unicode", analyzer='char', sublinear_tf=True,
                             use_idf=True, smooth_idf=True, max_features=50000)
tfidf_char.fit(corpus)
char_tfidf_vec = tfidf_char.transform(corpus)

final_VSM = sparse.hstack((word_tfidf_vec, char_tfidf_vec, features), format('csr'))
sparse.save_npz(features_file, final_VSM)

elapsed = time.time() - start
print("time for generate VSM: ", elapsed, "\n")


