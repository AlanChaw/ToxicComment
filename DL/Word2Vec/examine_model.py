import gensim
from SettingsS import *


model_path = Settings.word2vec_model_path

# model = gensim.models.Word2Vec.load_word2vec_format(model_path)

# model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

model = gensim.models.KeyedVectors.load(model_path)
