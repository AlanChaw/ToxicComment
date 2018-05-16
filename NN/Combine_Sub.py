import pandas as pd


new_sub = pd.read_csv('../File/submission_RCNN.csv')
best_sub = pd.read_csv('../File/submission_LSTM.csv')
sample_sub = pd.read_csv('../File/sample_submission.csv')

sample_sub['identity_hate'] = best_sub['identity_hate']
sample_sub.to_csv("../File/sub_one_column.csv", index=False)

#
# best_sub['toxic'] = new_sub['toxic']
# best_sub.to_csv("../File/submission_RCNN.csv", index=False)


