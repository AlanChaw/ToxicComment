'''
The core idea behind all blends is "diversity".
By blending some moderate model results (with some weights), we can create a more "diverse" stable results.
Errors from one model will be covered by others. Same goes for all the models.
So, we get more stable results after blendings.
'''
import pandas as pd
import numpy as np

# grucnn = pd.read_csv('../input/bi-gru-cnn-poolings/submission_RCNN.csv')
# gruglo = pd.read_csv("../input/pooled-gru-glove-with-preprocessing/submission_RCNN.csv")
# ave = pd.read_csv("../input/toxic-avenger/submission_RCNN.csv")
# supbl = pd.read_csv('../input/blend-of-blends-1/superblend_1.csv')
# best = pd.read_csv('../input/toxic-hight-of-blending/hight_of_blending.csv')
# lgbm = pd.read_csv('../input/lgbm-with-words-and-chars-n-gram/lvl0_lgbm_clean_sub.csv')
# wordbtch = pd.read_csv('../input/wordbatch-fm-ftrl-using-mse-lb-0-9804/lvl0_wordbatch_clean_sub.csv')
# tidy = pd.read_csv('../input/tidy-xgboost-glmnet-text2vec-lsa/tidy_glm.csv')
# fast = pd.read_csv('../input/pooled-gru-fasttext-6c07c9/submission_RCNN.csv')
# bilst = pd.read_csv('../input/bidirectional-lstm-with-convolution/submission_RCNN.csv')
# oofs = pd.read_csv('../input/oof-stacking-regime/submission_RCNN.csv')
# corrbl = pd.read_csv('../input/another-blend-tinkered-by-correlation/corr_blend.csv')
# rkera = pd.read_csv('../input/why-a-such-low-score-with-r-and-keras/submission_RCNN.csv')

LR = pd.read_csv('../File/submission_LR.csv')
CNN = pd.read_csv('../File/submission CNN.csv')
RCNN = pd.read_csv('../File/submission_RCNN.csv')
LSTM = pd.read_csv('../File/submission_LSTM.csv')

b1 = RCNN.copy()
col = RCNN.columns

col = col.tolist()
col.remove('id')
for i in col:
    # b1[i] = (2 * fast[i] + 2 * gruglo[i] + grucnn[i] * 4 + ave[i] + supbl[i] * 2 + best[i] * 4 + wordbtch[i] * 2 + lgbm[
        # i] * 2 + tidy[i] + bilst[i] * 4 + oofs[i] * 5 + corrbl[i] * 4) / 33
    b1[i] = (0.5 * RCNN[i] + 0.2 * LSTM[i] + 0.2 * CNN[i] + 0.1 * LR[i])

b1.to_csv('../File/blend_it_all.csv', index=False)
