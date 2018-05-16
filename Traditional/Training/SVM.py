from sklearn.svm import SVC
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
from sklearn.ensemble import BaggingClassifier



train = pd.read_csv(Settings.train_cleaned_file_path)
test = pd.read_csv(Settings.test_cleaned_file_path)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
sub_id = pd.read_csv(Settings.sample_sub_path).id
VSM_vec = sparse.load_npz(Settings.features_file_path)
train_features, test_features = VSM_vec[0:train.__len__()], VSM_vec[train.__len__():]
submission = pd.DataFrame.from_dict({'id': test['id']})


def create_model(C = 1):
    scores = []
    for class_name in label_cols:
        train_target = train[class_name]
        # model = SVC(C=C, kernel='linear', probability=True, class_weight='balanced', verbose=True)
        n_estimators = 100
        model = BaggingClassifier(SVC(kernel='linear', C=C, probability=True, class_weight='balanced', verbose=1),
                                  n_estimators=n_estimators, max_samples=1.0 / n_estimators,
                                  n_jobs=-1, verbose=1, bootstrap=False)
        X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.33)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        cv_score = roc_auc_score(y_test, y_proba)
        scores.append(cv_score)
    return np.mean(scores)


def generate_combinations(param_grid):
    combinations = []
    # combinations = list(itertools.product(param_grid['C'], param_grid['solver']))
    for key in param_grid.keys():
        if combinations.__len__() == 0:
            combinations = param_grid[key]
        else:
            combinations = list(itertools.product(combinations, param_grid[key]))
    return combinations


# param_grid = dict(test=['t1', 't2'], C=[0.001, 0.01, 0.1, 1, 10, 100], solver=['newton-cg', 'lbfgs', 'sag'])
param_grid = dict(C=[0.01, 0.1, 1, 10])
all_combinations = generate_combinations(param_grid)
best_score = 0
best_param = 0
for param in all_combinations:
    start = time.time()
    score = create_model(C=param)
    elapsed = time.time() - start
    print("CV score for params: " + str(param) + "  is: " + str(score) + "  time: " + str(elapsed))
    if score > best_score:
        best_score = score
        best_param = param

print("Best roc auc score: " + str(best_score) + "  With the params of: " + str(best_param))

for class_name in label_cols:
    train_target = train[class_name]
    model = SVC(C=best_param, kernel='linear', probability=True, class_weight=['balanced'], verbose=True)
    model.fit(train_features, train_target)
    submission[class_name] = model.predict_proba(test_features)[:, 1]
submission.to_csv(Settings.sub_path, index=False)

