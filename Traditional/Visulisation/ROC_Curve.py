from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from Traditional import Settings
from scipy import sparse
from sklearn.model_selection import ShuffleSplit
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from itertools import cycle



train = pd.read_csv(Settings.train_cleaned_file_path)
test = pd.read_csv(Settings.test_cleaned_file_path)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
sub_id = pd.read_csv(Settings.sample_sub_path).id
VSM_vec = sparse.load_npz(Settings.features_file_path)
train_features, test_features = VSM_vec[0:train.__len__()], VSM_vec[train.__len__():]
submission = pd.DataFrame.from_dict({'id': test['id']})

fpr = dict()
tpr = dict()
roc_auc = dict()

for class_name in label_cols:
    train_target = train[class_name]
    model = LogisticRegression(C=1, solver='sag', verbose=1, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.2)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr[class_name], tpr[class_name], _ = roc_curve(y_test, y_proba)
    roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])

plt.figure()
lw = 1
colors = cycle(['navy', 'deeppink', 'aqua', 'darkorange', 'cornflowerblue', 'darkred'])
for label, color in zip(label_cols, colors):
    plt.plot(fpr[label], tpr[label], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(label, roc_auc[label]))

plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve (for LR classifier)')
plt.legend(loc="lower right")
plt.show()
