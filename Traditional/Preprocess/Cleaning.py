# file process
import pandas as pd
from Traditional import Settings
# preprocess
import re
import enchant
from nltk.corpus import stopwords

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
# eng_stopwords = set(stopwords.words("english"))
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
    global counter
    print(counter)
    counter = counter + 1

    # print("before clean : ", comment)
    # to lower case
    comment = comment.lower()
    # delete "/n"
    comment = re.sub("\\n", " ", comment)
    # delete ip address
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)
    # delete username
    comment = re.sub("\[\[.*\]", "", comment)
    # deal with "fuck"
    comment = re.sub("f+u+c+k+", "fuck", comment)
    comment = re.sub("f\su\sc\sk", "fuck", comment)
    # deal with "bitch"
    comment = re.sub("b+i+t+c+h+", "bitch", comment)
    comment = re.sub("b\si\st\sc\sh ", "bitch", comment)
    # deal with "suck"
    comment = re.sub("s+u+c+k+", "suck", comment)
    comment = re.sub("s\su\sc\sk", "suck", comment)
    # deal with "you"
    comment = re.sub("y+o+u+", "you", comment)
    comment = re.sub("y\so\su", "you", comment)

    # split into words
    words = tokenizer.tokenize(comment)
    # change suoxie
    words = [APPO[w] if w in APPO else w for w in words]
    middle_comment = " ".join(words)
    words = middle_comment.split(" ")
    # lemmatize
    words = [lemmatizer.lemmatize(w, pos='v') for w in words]
    # delete stopwords
    words = [w for w in words if not w in set(stopwords.words("english"))]

    # delete punctuations
    pattern = re.compile("[^a-z]")
    words = [w for w in words if not pattern.match(w)]

    # correct spelling
    # d = enchant.Dict("en_US")
    # words = [d.suggest(w)[0] if not d.check(w) else w for w in words]
    # words = [d.suggest(w)[0] for w in words if d.check(w)]

    cleared_comment = " ".join(words)
    return cleared_comment


def correct(word, d):
    if not d.check(word):
        return d.suggest(word)[0]
    else:
        return word


if __name__ == '__main__':
    main()
