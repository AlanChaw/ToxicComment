from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
from time import time
import numpy as np
from Settings import *
from scipy import sparse

train = pd.read_csv(Settings.train_cleaned_file_path)
test = pd.read_csv(Settings.test_cleaned_file_path)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
sub_id = pd.read_csv(Settings.sample_sub_path).id

VSM_vec = sparse.load_npz(Settings.features_file_path)
x, x_test = VSM_vec[0:train.__len__()], VSM_vec[train.__len__():]


def pr(y_i, y):
    p = x[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)


def get_mdl(y):
    y = y.values
    y1 = pr(1, y)
    y2 = pr(0, y)
    r = np.log(pr(1, y), pr(0, y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)

    # print(np.isnan(x.data).any())
    # print(np.isnan(r.data).any())

    return m.fit(x_nb, y), r


pred_results = np.zeros((len(test), len(label_cols)))
for i, j in enumerate(label_cols):
    start = time()
    print('fit', j)
    m, r = get_mdl(train[j])
    pred_results[:, i] = m.predict_proba(x_test.multiply(r))[:, 1]
    elapsed = time() - start
    print("time for fit", label_cols[i], " : ", elapsed)
    # cv_score = np.mean(cross_val_score(m, x, train[j], cv=3, scoring='roc_auc'))
    # print("cv score for ", label_cols[i], " : ", cv_score)

submission = pd.concat([sub_id, pd.DataFrame(pred_results, columns=label_cols)], axis=1)
submission.to_csv(Settings.sub_path, index=False)
