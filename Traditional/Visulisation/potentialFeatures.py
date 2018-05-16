# file process
import pandas as pd
import numpy as np
import re
import string

# preprocess

#visual
from matplotlib import pyplot as plt
import seaborn as sns

# nlp
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

# from Traditional.Settings import *
import Settings


APPO = Settings.APPO
train_file = Settings.train_file_path
test_file = Settings.test_file_path
train_cleaned_file = Settings.train_cleaned_file_path
test_cleaned_file = Settings.test_cleaned_file_path
tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
counter = 0


df = pd.read_csv(train_file)
rowsums = df.iloc[:, 2:].sum(axis=1)
df['nontoxic'] = (rowsums == 0)

# number of sentences
df['sentence_num'] = df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
# number of words
df['word_num'] = df["comment_text"].apply(lambda x: len(str(x).split()))
# number of unique_words
df['unique_num'] = df["comment_text"].apply(lambda x: len(set(str(x).split())))
# percentage of unique_words
df['unique_perc'] = df['unique_num'] / df['word_num']
# number of capital words
df["capital_num"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
# percentage of capital words
df['capital_perc'] = (df['capital_num'] / df['word_num']) * 100
# number of puncts
df['punct_num'] = df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
# percentage of puncts
df['punct_perc'] = df['punct_num'] / df['word_num']
# number of !
df['excla_perc'] = df['comment_text'].apply(lambda comment: comment.count('!'))

features = ('sentence_num', 'word_num', 'unique_num', 'unique_perc',
            'capital_num', 'capital_perc', 'punct_num', 'punct_perc',
            'excla_perc')
columns = ('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')

rows = [{c: df[f].corr(df[c]) for c in columns} for f in features]
df_correlations = pd.DataFrame(rows, index=features)
df_correlations = abs(df_correlations)
df_correlations
ax = sns.heatmap(df_correlations,cmap='YlGnBu', linewidths='1')
plt.show()

sns.boxplot(y='capital_perc',x='nontoxic', data=df, fliersize=1, whis=5)
plt.show()

#
temp_df = pd.melt(df, value_vars=['unique_num', 'capital_num'], id_vars='nontoxic')
sns.boxplot(x='variable', y='value', hue='nontoxic', data=temp_df)
plot.show()