from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

from jsonio import read_infile_as_json, read_outfile_as_json
from features import ContextTopicFollowers, TopicNum, NumAnswers, PromotedTo
from evaluation import evaluate_rmsle


if __name__ == '__main__':
    train_json, test_json = read_infile_as_json('./data/input3.in')
    train_y = np.asarray(map(lambda x: x['__ans__'], train_json))

    test_labels_json = read_outfile_as_json('./data/output3.out')
    test_labels_dict = {i['question_key']: i['__ans__'] for i in test_labels_json}

    test_y = np.asarray([test_labels_dict[x['question_key']] for x in test_json])

    feature_extractors = FeatureUnion([('ContextTopicFollowers', ContextTopicFollowers()),
                                       ('TopicNum', TopicNum()),
                                       ('NumAnswers', NumAnswers()),
                                       ('PromotedTo', PromotedTo())])
    m = Pipeline(steps=[("features", feature_extractors),
                        ('LR', GradientBoostingRegressor())])

    m.fit(train_json, np.log(train_y + 1))
    pred = np.exp(m.predict(test_json)) - 1

    print(evaluate_rmsle(pred, test_y))
