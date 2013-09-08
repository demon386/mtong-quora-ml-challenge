from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np

from jsonio import read_infile_as_json, read_outfile_as_json
from features import TopicBOW, TopicNum
from evaluation import evaluate_classification


if __name__ == '__main__':
    train_json, test_json = read_infile_as_json('./data/answered_data_10k.in')
    train_y = np.asarray(map(lambda x: x['__ans__'], train_json))

    test_labels_json = read_outfile_as_json('./data/answered_data_10k.out')
    test_labels_dict = {i['question_key']: i['__ans__'] for i in test_labels_json}

    # Output should be ordered according to test_json
    test_y = np.asarray([test_labels_dict[x['question_key']] for x in test_json])

    feature_extractors = FeatureUnion([("TopicBOW", TopicBOW()), ("TopicNum", TopicNum())])
    m = Pipeline(steps=[("features", feature_extractors),
                        ('LR', LogisticRegression(penalty="l1"))])

    m.fit(train_json, train_y)
    pred = m.predict(test_json)

    print(evaluate_classification(pred, test_y))
