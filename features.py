import re
import string
import numpy as np
from scipy.stats import nanmean
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


def get_all_text(x):
    return get_cleaned_question_text(x) + " " + get_topic_text(x)

def get_cleaned_question_text(x):
    question_text = x['question_text']
    question_text = _remove_puncation(question_text)
    question_text = _remove_multiple_spaces(question_text)
    return _remove_common_words(question_text.lower())

def get_topic_text(x):
    return ' '.join([_concat_name(i['name']) for i in x['topics']])

def _concat_name(s):
    """
    Remove the space between words.

    Eg: Computer Science -> ComputerDummyScience

    So that different topics with some identical don't mix.
    """
    return s.replace(" ", "Dummy")

def _remove_puncation(s):
    # From
    # http://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
    return re.sub('[%s]' % re.escape(string.punctuation), '', s)

def _remove_multiple_spaces(s):
    return re.sub(' +', ' ', s)

def _remove_common_words(s):
    text_list = s.split(' ')
    common_words = set(["what", "how", "why", "if", "when"])
    return ' '.join(filter(lambda x: x not in common_words, text_list))



class TopicNum(BaseEstimator, TransformerMixin):
    """The number of topic a sample has.
    """
    def __init__(self):
        pass

    def _get_topic_num(self, x):
        """Number of topics.
        """
        if x.get('topics'):
            return [len(x['topics'])]
        else:
            return [0]

    def fit(self, X, y):
        """Dummy implementation.
        """
        return self

    def transform(self, X):
        features =  np.asarray(map(self._get_topic_num, X))
        return features


class TopicBOW(BaseEstimator, TransformerMixin):
    "Bag of representation of Question Text"
    def __init__(self):
        self.tfidf_ = TfidfVectorizer(ngram_range=(1, 1))


    def fit(self, X, y):
        features = np.asarray([get_all_text(x) for x in X])
        self.tfidf_.fit(features)
        return self

    def transform(self, X):
        features = np.asarray([get_all_text(x) for x in X])
        return self.tfidf_.transform(features)


class AverageTopicFollowers(BaseEstimator, TransformerMixin):
    def cal_mean_followers(self, x):
        if x.get(u'topics'):
            return nanmean([i[u'followers'] for i in x[u'topics']])
        else:
            return np.nan

    def fit(self, X, y):
        self.mean_ = nanmean(map(self.cal_mean_followers, X))
        return self

    def transform(self, X):
        def _func(x):
            if x.get(u'topics'):
                return nanmean([i[u'followers'] for i in x[u'topics']])
            else:
                return self.mean_

        return np.asarray(map(_func, X))[:, np.newaxis]


class ContextTopicFollowers(AverageTopicFollowers):
    def fit(self, X, y):
        self.mean_ = nanmean(map(self.cal_mean_followers, X))
        return self

    def transform(self, X):
        def _func(x):
            if x.get('context_topics') and x.get('context_topics').get('followers'):
                return x['context_topics']['followers']
            elif x.get('topics'):
                return nanmean([i['followers'] for i in x['topics']])
            else:
                return self.mean_

        return np.asarray(map(_func, X))[:, np.newaxis]


class Anonymous(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.asarray(map(lambda x: x[u'anonymous'], X))[:, np.newaxis]


class NumAnswers(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.asarray(map(lambda x: x[u'num_answers'], X))[:, np.newaxis]


class PromotedTo(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.asarray(map(lambda x: x[u'promoted_to'], X))[:, np.newaxis]
