# file process
import pandas as pd
import numpy as np

# preprocess

#visual
from matplotlib import pyplot as plt
import seaborn as sns

# nlp
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

APPO = Settings.APPO
train_file = Settings.train_file_path
test_file = Settings.test_file_path
train_cleaned_file = Settings.train_cleaned_file_path
test_cleaned_file = Settings.test_cleaned_file_path
tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
counter = 0


train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# # 各个类别的数量
# rowsums = train.iloc[:, 2:].sum(axis=1)
# train['clean'] = (rowsums == 0)
# colsums = train.iloc[:, 2:].sum(axis=0)
# plt.bar(colsums.index, colsums.values, alpha=0.9, width = 0.5, facecolor = 'lightskyblue', edgecolor = 'white', label='one', lw=1)
# plt.show()
#
# # 拥有多个标签的数量
# multitag = rowsums.value_counts()
# plt.bar(multitag.index, multitag.values, alpha=0.9, width = 0.5, facecolor = 'lightskyblue', edgecolor = 'white', label='one', lw=1)
# plt.show()


# Correlation heatmap
sns.set(style="white")
data = train.iloc[:, 2: 8]
corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.show()
