

# https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view
# Aphost lookup dict


test_trigger = False

sample_sub_path = '../File/sample_submission.csv'
sub_path = '../File/submission.csv'
word2vec_model_path = '..//File/Word2VecModel/model'
glove_model_path = '../File/Glove/glove.6B.50d.txt'
doc_vec_normalization = True

if test_trigger:
    train_file_path = '../File/Slice/train_first100.csv'
    test_file_path = '../File/Slice/test_first100.csv'
    train_cleaned_file_path = '../File/Middle/train_cleaned_slice.csv'
    test_cleaned_file_path = '../File/Middle/test_cleaned_slice.csv'
    features_file_path = '../File/Middle/features_slice.npz'
else:
    train_file_path = '../File/train.csv'
    test_file_path = '../File/test.csv'
    train_cleaned_file_path = '../File/Middle/train_cleaned.csv'
    test_cleaned_file_path = '../File/Middle/test_cleaned.csv'
    features_file_path = '../File/Middle/features.npz'

APPO = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'd": "i had",
    "i'll": "i will",
    "i'm": "i am",
    "isn't": "is not",
    "it's": "it is",
    "it'll": "it will",
    "i've": "i have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll": " will",
    "didn't": "did not",
    "tryin'": "trying"
}

