import os
import spacy
import pickle
import nltk
import re
import unicodedata
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from random import shuffle
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP

stopword_list = nltk.corpus.stopwords.words('english')
important_words = ['above', 'below', 'off', 'over', 'under', 'more', 'most', 'such', 'no', 'nor', 'not', 'only', 'so', 'than', 'too', 'very', 'just', 'but']
necessary_stop_words=list(set(stopword_list)-set(important_words))
np.set_printoptions(precision=2, linewidth=80)
nlp = spacy.load("en_core_web_sm")
text_tokenizer = ToktokTokenizer()


def load_data() :
    pos_reviews = []
    limit = 0
    directory = "aclImdb/train/pos"
    for review in os.listdir(directory):
        if limit < 5000:
            if review.endswith(".txt"):
                f = open(f"{directory}/{review}", encoding="utf8")
                text = f.read()
                pos_reviews.append((text,"Positive"))
                limit += 1
    # print(pos_reviews[0])

    neg_reviews = []
    limit = 0
    directory = "aclImdb/train/neg"
    for review in os.listdir(directory):
        if limit < 5000:
            if review.endswith(".txt"):
                f = open(f"{directory}/{review}", encoding="utf8")
                text = f.read()
                neg_reviews.append((text,"Negative"))
                limit += 1
    # print(neg_reviews[0])
    dataset=[]
    dataset = dataset+pos_reviews+neg_reviews
    shuffle(dataset)
    reviews=[]
    sentiments=[]
    for data in dataset:
        reviews.append(data[0])
        sentiments.append(data[1])
    return reviews,sentiments;


def preprocessing(reviews,sentiments) :
	# Encoding Sentiment column
	label = LabelEncoder()
	sentiments = label.fit_transform(sentiments)

	reviews = np.array(reviews)
	sentiments = np.array(sentiments)

	# build train and test datasets
	train_reviews = reviews[:8000]
	train_sentiments = sentiments[:8000]
	test_reviews = reviews[8001:10000]
	test_sentiments = sentiments[8001:10000]

	# normalize datasets
	normalize_train_reviews = normalize_corpus(train_reviews)
	normalize_test_reviews = normalize_corpus(test_reviews)
	return normalize_train_reviews,normalize_test_reviews,train_sentiments,test_sentiments;

# # Cleaning Text - strip HTML
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


# # Removing accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# # Expanding Contractions
def expand(text, contraction_mapping=CONTRACTION_MAP):
    pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded = first_char + expanded[1:]
        return expanded

    expanded_text = pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# # Removing Special Characters
def remove_special_char(text):
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    return text


# # Lemmatizing text
def lemmatization(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# # Removing Stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = text_tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        cleaned_tokens = [token for token in tokens if token not in stopword_list]
    else:
        cleaned_tokens = [token for token in tokens if token.lower() not in stopword_list]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text


# # Normalize text corpus - tying it all together
def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True, accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True, stopword_removal=True):
    normalized_corpus = []

    for document in corpus:

        if html_stripping:
            document = remove_html_tags(document)

        if accented_char_removal:
            document = remove_accented_chars(document)

        if contraction_expansion:
            document = expand(document)

        if text_lower_case:
            document = document.lower()

        # remove extra newlines
        document = re.sub(r'[\r|\n|\r\n]+', ' ', document)
        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        document = special_char_pattern.sub(" \\1 ", document)

        if text_lemmatization:
            document = lemmatization(document)

        if special_char_removal:
            document = remove_special_char(document)

            # remove extra whitespace
        document = re.sub(' +', ' ', document)

        if stopword_removal:
            document = remove_stopwords(document, is_lower_case=text_lower_case)

        normalized_corpus.append(document)

    return normalized_corpus

def bag_of_words(normalize_train_reviews,normalize_test_reviews) :
    # build BOW features on train reviews
    cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0, ngram_range=(1, 2))
    cv_train_features = cv.fit_transform(normalize_train_reviews)
    # transform test reviews into features
    cv_test_features = cv.transform(normalize_test_reviews)
    print('BOW model: Train features shape:', cv_train_features.shape, ' Test features shape:', cv_test_features.shape)
    return cv,cv_train_features,cv_test_features;

def tf_idf(normalize_train_reviews,normalize_test_reviews):
    # build TFIDF features on train reviews
    tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1, 2), sublinear_tf=True)
    tv_train_features = tv.fit_transform(normalize_train_reviews)
    # transform test reviews into features
    tv_test_features = tv.transform(normalize_test_reviews)
    print('TFIDF model: Train features shape:', tv_train_features.shape, ' Test features shape:',tv_test_features.shape)
    return tv,tv_train_features,tv_test_features;
def save(classifier,cv) :
    f = open('my_model.pickle', 'wb')
    pickle.dump((classifier,cv), f)
    f.close()

def load() :
    f = open('my_model.pickle', 'rb')
    classifier,cv = pickle.load(f)
    f.close()
    return classifier,cv;

def logistic_regression(cv,cv_train_features,cv_test_features,train_sentiments,test_sentiments) :
    lr = LogisticRegression(penalty='l2', max_iter=200, C=1)

    # Logistic Regression model on BOW features
    lr.fit(cv_train_features, train_sentiments)
    # predict using model
    predictions = lr.predict(cv_test_features)
    score = lr.score(cv_test_features, test_sentiments)
    print("Accuracy score:", np.round(score, 2) * 100, "%")
    save(lr,cv)

    return predictions,score;

def Naive_Bayes(cv,cv_train_features,cv_test_features,train_sentiments,test_sentiments) :
    lr = MultinomialNB()
    # Logistic Regression model on BOW features
    lr.fit(cv_train_features, train_sentiments)
    # predict using model
    predictions = lr.predict(cv_test_features)
    score = lr.score(cv_test_features, test_sentiments)
    print("Accuracy score:", np.round(score, 2) * 100, "%")
    save(lr,cv)

    return predictions,score;

def accuracy_precision_recall_fscore(test_sentiments,predictions,score):
    cm = metrics.confusion_matrix(test_sentiments, predictions)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    print("Confusion matrix:")
    all_sample_title = 'Accuracy Score: {0}%'.format(np.round(score,2)*100)
    plt.title(all_sample_title, size = 12)
    plt.show()

    print("Classification report\n")
    print(classification_report(test_sentiments, predictions))

    print('Accuracy:  {:2.2%} '.format(metrics.accuracy_score(test_sentiments, predictions)))
    print('Precision: {:2.2%} '.format(metrics.precision_score(test_sentiments, predictions, average='weighted')))
    print('Recall:    {:2.2%} '.format(metrics.recall_score(test_sentiments, predictions, average='weighted')))
    print('F1 Score:  {:2.2%} '.format(metrics.f1_score(test_sentiments, predictions, average='weighted')))

def custom_input(review,cv,classifier) :
        clean_custom_review = normalize_corpus([review])
        cv_test_features = cv.transform(clean_custom_review)
        result = classifier.predict_proba(cv_test_features)[0]
        if (result[0] > 0.5):
            return 0
        else:
            return 1

def main() :
    print("\nTraining the model\n")
    reviews, sentiments = load_data()
    normalize_train_reviews, normalize_test_reviews, train_sentiments, test_sentiments = preprocessing(reviews,
                                                                                                       sentiments)
    print("Select one of the Feature extraction method\n")
    a = int(input("1. Bag of words\n2. tf-idf\n"))
    if a == 1:
        cv, cv_train_features, cv_test_features = bag_of_words(normalize_train_reviews, normalize_test_reviews)
        print("Select one of the classification method\n")
        print("1.Naive-Bayes\n2.Logistic Regression\n")
        b = int(input("Enter your choice : \n"))
        if b == 1:
            print("Testing the model\n")
            predictions, score = Naive_Bayes(cv, cv_train_features, cv_test_features, train_sentiments, test_sentiments)
            print("Classification Report of the model\n")
            accuracy_precision_recall_fscore(test_sentiments, predictions, score)
        else:
            print("Testing the model\n")
            predictions, score = logistic_regression(cv, cv_train_features, cv_test_features, train_sentiments,
                                                     test_sentiments)
            print("Classification Report of the model\n")
            accuracy_precision_recall_fscore(test_sentiments, predictions, score)
    else:
        cv, cv_train_features, cv_test_features = tf_idf(normalize_train_reviews, normalize_test_reviews)
        print("Select one of the classification method\n")
        print("1.Naive-Bayes\n2.Logistic Regression\n")
        c = int(input("Enter your choice : \n"))
        if c == 1:
            print("Testing the model\n")
            predictions, score = Naive_Bayes(cv, cv_train_features, cv_test_features, train_sentiments, test_sentiments)
            print("Classification Report of the model\n")
            accuracy_precision_recall_fscore(test_sentiments, predictions, score)
        else:
            print("Testing the model\n")
            predictions, score = logistic_regression(cv, cv_train_features, cv_test_features, train_sentiments,
                                                     test_sentiments)
            print("Classification Report of the model\n")
            accuracy_precision_recall_fscore(test_sentiments, predictions, score)

