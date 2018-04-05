import gensim
import logging
from SettingsS import *
import pandas as pd
import re
from nltk.tokenize import TweetTokenizer
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

tokenizer = TweetTokenizer()
train_file = Settings.train_file_path
test_file = Settings.test_file_path
model_path = Settings.word2vec_model_path


def clean(comment):
    # to lower case
    comment = comment.lower()
    # delete "/n"
    comment = re.sub("\\n", " ", comment)
    # split into words
    words = tokenizer.tokenize(comment)
    # cleared_comment = " ".join(words)
    return words


train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
merged = pd.concat([train.iloc[:, 0:2], test.iloc[:, 0:2]])
merged = merged.reset_index(drop=True)
all_comments = merged.comment_text

sentences = all_comments.apply(lambda x: clean(x))


# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1, size=100, workers=4)
# min_count是最少的单词数量，默认值5, size是特征向量的大小 默认100, workers并行数，只有有Cython才有效

# model.save(model_path)

word_vectors = model.wv
del model
word_vectors.save(model_path)
