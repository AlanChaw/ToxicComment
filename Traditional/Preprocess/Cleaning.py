# file process
import pandas as pd
from SettingsS import *
# preprocess
import re

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


def main():
    # pd.set_option('display.max_colwidth', 20000)

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    merged_comments = pd.concat([train.iloc[:, 0:2], test.iloc[:, 0:2]])
    # df = merged_comments.reset_index(drop=True)
    corpus = merged_comments.comment_text
    cleaned_corpus = corpus.apply(lambda x: clean(x))

    train.comment_text = cleaned_corpus.iloc[0: len(train), ]
    test.comment_text = cleaned_corpus.iloc[len(train):, ]

    train.to_csv(train_cleaned_file, index=False)
    test.to_csv(test_cleaned_file, index=False)


def clean(comment):
    # global counter
    # print(counter)
    # counter = counter + 1

    # print("before clean : ", comment)
    # to lower case
    comment = comment.lower()
    # delete "/n"
    comment = re.sub("\\n", " ", comment)
    # delete ip address
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)
    # delete username
    comment = re.sub("\[\[.*\]", "", comment)
    # split into words
    words = tokenizer.tokenize(comment)
    # change suoxie
    words = [APPO[w] if w in APPO else w for w in words]
    middle_comment = " ".join(words)
    words = middle_comment.split(" ")
    # lemmatize
    words = [lemmatizer.lemmatize(w, pos='v') for w in words]
    # delete stopwords
    # words = [w for w in words if not w in set(stopwords.words("english"))]
    # delete punctuations
    pattern = re.compile("[^a-z]")
    words = [w for w in words if not pattern.match(w)]

    cleared_comment = " ".join(words)
    return cleared_comment


if __name__ == '__main__':
    main()
