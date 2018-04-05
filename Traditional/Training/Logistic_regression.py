

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
from time import time
import numpy as np
from Traditional.Preprocess.VSM import VSM
from Settings import *

train = pd.read_csv(Settings.train_cleaned_file_path)
test = pd.read_csv(Settings.test_cleaned_file_path)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
sub_id = pd.read_csv(Settings.sample_sub_path).id
VSM_vec = sparse.load_npz(Settings.features_file_path)
train_features, test_features = VSM_vec[0:train.__len__()], VSM_vec[train.__len__():]

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# train_features, test_features = VSM.create_vsm()

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')

    # cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    # scores.append(cv_score)
    # print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv(Settings.sub_path, index=False)

