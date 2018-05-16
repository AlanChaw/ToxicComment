from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import time
import numpy as np
from Traditional import Settings
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


train = pd.read_csv(Settings.train_cleaned_file_path)
test = pd.read_csv(Settings.test_cleaned_file_path)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
sub_id = pd.read_csv(Settings.sample_sub_path).id
VSM_vec = sparse.load_npz(Settings.features_file_path)
train_features, test_features = VSM_vec[0:train.__len__()], VSM_vec[train.__len__():]

# train_dense = csr_matrix.todense(train_features)
# pca = SparsePCA(n_components=2, verbose=2, n_jobs=-1)
# pca.fit(train_dense)
svd = TruncatedSVD(n_components=1000, n_iter=5)
# svd.fit(train_features)
print(svd.explained_variance_ratio_.sum())

